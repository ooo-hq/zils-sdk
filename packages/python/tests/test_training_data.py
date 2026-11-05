import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zils_sdk import training_data


def datasets(root):
    files = {}
    for split in ("train", "calibration", "test"):
        files[split] = root / (split + ".jsonl")
        rows = [
            {
                "id": f"{split}-{i}",
                "group_id": f"{split}-{i}",
                "family": "routing",
                "state": {"text": f"{split} request {i}"},
                "question": {"type": "choice", "criteria": {"yes": "Accept", "no": "Decline"}},
                "label": label,
            }
            for i, label in enumerate(("yes", "no"))
        ]
        files[split].write_text("".join(json.dumps(row) + "\n" for row in rows))
    return files


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = datasets(self.root)

    def prepare(self):
        return training_data.prepare(self.files, self.root / "snapshots")

    def test_cost_ceiling_prices_every_train_prompt_at_the_token_cap(self):
        self.assertEqual(
            training_data.cost_ceiling(2),
            {"train_cases": 2, "max_training_tokens": 16384, "ceiling_usd": 2.0},
        )
        self.assertEqual(training_data.cost_ceiling(2000)["max_training_tokens"], 16_384_000)
        self.assertEqual(training_data.cost_ceiling(2000)["ceiling_usd"], 3.28)

    def test_check_validates_without_keeping_a_snapshot(self):
        result = training_data.check(self.files)
        self.assertEqual(
            {split: info["count"] for split, info in result["files"].items()},
            {"train": 2, "calibration": 2, "test": 2},
        )
        self.assertEqual(result["ceiling_usd"], 2.0)
        self.assertNotIn("path", json.dumps(result))
        self.files["test"].write_text("private-invalid-json")
        with self.assertRaisesRegex(ValueError, "Invalid text dataset case"):
            training_data.check(self.files)

    def test_snapshots_are_private_and_preserve_exact_bytes(self):
        original = self.files["train"].read_bytes()
        result = self.prepare()
        self.files["train"].write_text("changed after validation")
        saved = Path(result["files"]["train"]["path"])
        self.assertEqual(saved.read_bytes(), original)
        self.assertEqual(result["files"]["train"]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(result["files"]["train"]["count"], 2)
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        self.assertEqual(saved.parent.stat().st_mode & 0o777, 0o700)

    def test_invalid_records_do_not_leave_snapshots(self):
        good = json.loads(self.files["train"].read_text().splitlines()[0])
        changes = [
            {"id": ""},
            {"group_id": ""},
            {"family": ""},
            {"label": "missing"},
            {"label": True},
            {"image": {"asset_id": "private"}},
            {"question": {"type": "choice", "criteria": {"one": "Only"}}},
            {"question": {"type": "score", "criteria": ["low", "high"]}, "label": "2"},
            {"question": {"type": "noul", "criteria": {"typo": "private"}}},
            {"state": float("inf")},
            {"state": "x" * (128 * 1024)},
        ]
        raw = [
            b"",
            b"\xff\n",
            b'{"id":"a","id":"private"}\n',
            b'{"state":NaN}\n',
            b'{"state":1e999}\n',
        ]
        raw += [(json.dumps({**good, **change}) + "\n").encode() for change in changes]
        raw += [(json.dumps(good) + "\n").encode() * 2]
        for data in raw:
            with self.subTest(data=data[:40]):
                self.files["train"].write_bytes(data)
                with self.assertRaises(ValueError) as error:
                    self.prepare()
                self.assertNotIn("private", str(error.exception))
                self.assertEqual(list((self.root / "snapshots").glob("*")), [])

    def test_split_overlap_and_family_contract(self):
        train = json.loads(self.files["train"].read_text().splitlines()[0])
        original = self.files["test"].read_text()
        test = json.loads(original.splitlines()[0])
        for changes in [
            {"id": train["id"]},
            {"group_id": train["group_id"]},
            {"state": train["state"], "question": train["question"]},
            {"family": "other"},
        ]:
            self.files["test"].write_text(json.dumps({**test, **changes}) + "\n")
            with self.assertRaises(ValueError):
                self.prepare()
        self.files["test"].write_text(original)

    def test_file_and_case_limits(self):
        with patch.object(training_data, "MAX_SPLIT", 20):
            with self.assertRaisesRegex(ValueError, "size"):
                self.prepare()
        with patch.object(training_data, "MAX_CASE", 20):
            with self.assertRaises(ValueError):
                self.prepare()

    def test_noul_score_and_json_states_are_preserved(self):
        for kind, criteria, label in [("noul", None, "true"), ("score", ["low", "high"], "1")]:
            for split, file in self.files.items():
                row = {
                    "id": split,
                    "group_id": split,
                    "family": "same",
                    "state": [split, None, False, 0],
                    "question": {"type": kind, "criteria": criteria},
                    "label": label,
                }
                file.write_text(json.dumps(row) + "\n")
            self.assertEqual(self.prepare()["files"]["test"]["count"], 1)

    def test_policy_requires_finite_thresholds_and_explicit_consent(self):
        self.assertEqual(
            training_data.validate_policy("routing", 0.8, 0.01, True),
            {"min_accuracy": 0.8, "min_brier_improvement": 0.01},
        )
        for args in [
            ("Bad name", 0.8, 0.1, True),
            ("ok", True, 0.1, True),
            ("ok", 0.8, float("nan"), True),
            ("ok", 0.8, 0.1, False),
            ("ok", 0.8, 3, True),
            ("ok", -1, 0, True),
        ]:
            with self.assertRaises(ValueError):
                training_data.validate_policy(*args)

    def test_native_outcome_limit(self):
        case = json.loads(self.files["train"].read_text().splitlines()[0])
        for kind in ("choice", "score"):
            for count in (255, 256):
                with self.subTest(kind=kind, count=count):
                    criteria = (
                        {str(i): "Option" for i in range(count)}
                        if kind == "choice"
                        else ["Option"] * count
                    )
                    row = {**case, "question": {"type": kind, "criteria": criteria}, "label": "254"}
                    if count == 255:
                        training_data._case(row)
                    else:
                        with self.assertRaises(ValueError):
                            training_data._case(row)


if __name__ == "__main__":
    unittest.main()
