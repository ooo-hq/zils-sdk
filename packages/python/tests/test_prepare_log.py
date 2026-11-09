"""Training data from decision logs (the SDK's `record`), as `zils fit train` and `zils train prepare <logs>` use it."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zils_sdk import training_data, training_prepare

QUESTIONS = {
    "next_action": {
        "type": "choice",
        "instructions": "Reach the armor without dying. Choose the next action.",
        "criteria": {
            "forward": "Step toward the armor.",
            "shoot": "Fire ahead.",
            "backward": "Step back.",
        },
    },
    "urgent": {"type": "noul", "instructions": "Is this ticket urgent?"},
}


def action(health, distance):
    return "backward" if distance <= 3 and health <= 30 else "shoot" if distance <= 3 else "forward"


def game_logs(folder, games=12, turns=10):
    """One log per game, as `record` writes them; game 5 starts in exactly the situation game 0 started in."""
    folder.mkdir()
    for game in range(games):
        lines = []
        for turn in range(turns):
            health = 100 - 8 * turn - game
            distance = (game + turn) % 6
            if game == 5 and turn == 0:
                health, distance = 100, 0  # the repeat: identical to game 0, turn 0
            state = {
                "health": health,
                "enemies": [{"kind": "imp", "distance": distance}],
                "turn": turn,
            }
            lines.append(
                {
                    "question": "next_action",
                    "state": state,
                    "answer": action(health, distance),
                    "ts": 1.0,
                }
            )
        (folder / f"game-{game:02d}.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines)
        )
    return folder


def read_split(directory, split):
    return [
        json.loads(line)
        for line in (directory / f"{split}.jsonl").read_text().splitlines()
        if line.strip()
    ]


class PrepareLogTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.logs = game_logs(self.root / "logs")

    def test_whole_episodes_per_split_with_the_programs_state_kept(self):
        result = training_prepare.prepare_log(self.logs, questions=QUESTIONS, out=self.root / "out")
        out = self.root / "out"
        cases = {split: read_split(out, split) for split in training_data.SPLITS}
        groups = {split: {c["group_id"] for c in rows} for split, rows in cases.items()}
        self.assertFalse(
            groups["train"] & groups["calibration"]
            or groups["train"] & groups["test"]
            or groups["calibration"] & groups["test"]
        )
        self.assertEqual(sum(len(g) for g in groups.values()), 12)  # one group per game
        first = cases["train"][0]
        # The current training layout: the wording in state.decision, the program's own state kept as JSON.
        self.assertEqual(first["state"]["decision"], QUESTIONS["next_action"]["instructions"])
        self.assertIsInstance(first["state"]["information"]["enemies"], list)
        self.assertEqual(
            first["question"],
            {"type": "choice", "criteria": {k: k for k in QUESTIONS["next_action"]["criteria"]}},
        )
        self.assertEqual(result["repeats_left_out"], 1)
        self.assertEqual(sum(result["counts"].values()), 12 * 10 - 1)
        self.assertEqual((result["decide"], result["episodes"]), ("next_action", 12))
        self.assertIn("zils train submit", result["next"])
        training_data.check(
            {s: out / f"{s}.jsonl" for s in training_data.SPLITS}
        )  # the service's own check

    def test_without_questions_the_answers_seen_are_the_options(self):
        result = training_prepare.prepare_log(self.logs, out=self.root / "out")
        case = read_split(self.root / "out", "train")[0]
        self.assertEqual(set(case["question"]["criteria"]), {"forward", "shoot", "backward"})
        self.assertEqual(case["state"]["decision"], "next_action")
        self.assertEqual(result["name"], "next-action")

    def test_one_file_groups_by_its_episode_field_or_by_row(self):
        rows = [
            json.loads(line)
            for f in sorted(self.logs.glob("*.jsonl"))
            for line in f.read_text().splitlines()
        ]
        tagged = self.root / "tagged.jsonl"
        tagged.write_text(
            "".join(json.dumps({**r, "episode": f"g{i // 10}"}) + "\n" for i, r in enumerate(rows))
        )
        self.assertEqual(training_prepare.prepare_log(tagged, out=self.root / "a")["episodes"], 12)
        flat = self.root / "flat.jsonl"
        flat.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertEqual(
            training_prepare.prepare_log(flat, out=self.root / "b")["episodes"], 12 * 10 - 1
        )

    def test_a_question_logged_as_a_sentence_gets_a_short_name(self):
        sentence = self.root / "sentence.jsonl"
        rows = [
            json.loads(line)
            for f in sorted(self.logs.glob("*.jsonl"))
            for line in f.read_text().splitlines()
        ]
        long = "Walk the corridor to the armor without dying and choose the next action"
        sentence.write_text(
            "".join(
                json.dumps({**r, "question": long, "episode": i // 10}) + "\n"
                for i, r in enumerate(rows)
            )
        )
        result = training_prepare.prepare_log(sentence, out=self.root / "out")
        self.assertEqual(result["name"], "walk-the-corridor-to-the-armor-without")
        self.assertEqual(result["question"], long)

    def test_yes_no_questions_train_as_noul(self):
        log = self.root / "urgent.jsonl"
        log.write_text(
            "".join(
                json.dumps(
                    {
                        "question": "urgent",
                        "state": {"text": f"ticket {i}"},
                        "answer": [True, "yes", False, "no"][i % 4],
                    }
                )
                + "\n"
                for i in range(40)
            )
        )
        training_prepare.prepare_log(log, questions=QUESTIONS, out=self.root / "out")
        case = read_split(self.root / "out", "train")[0]
        self.assertEqual(case["question"], {"type": "noul"})
        self.assertIn(case["label"], ("true", "false"))
        self.assertEqual(case["state"]["decision"], "Is this ticket urgent?")

    def test_refusals_explain_what_to_fix(self):
        mixed = self.root / "mixed.jsonl"
        mixed.write_text(
            "".join(
                json.dumps({"question": q, "state": {"i": i}, "answer": "x" if i % 2 else "y"})
                + "\n"
                for i, q in enumerate(["a", "b"] * 20)
            )
        )
        with self.assertRaisesRegex(ValueError, r"a, b.*--decide"):
            training_prepare.prepare_log(mixed, out=self.root / "1")
        with self.assertRaisesRegex(ValueError, "teleport"):
            bad = self.root / "bad"
            bad.mkdir()
            (bad / "g.jsonl").write_text(
                json.dumps({"question": "next_action", "state": {"x": 1}, "answer": "teleport"})
                + "\n"
            )
            training_prepare.prepare_log(bad, questions=QUESTIONS, out=self.root / "2")
        with self.assertRaisesRegex(ValueError, "score"):
            training_prepare.prepare_log(
                self.logs,
                questions={"next_action": {"type": "score", "criteria": ["low", "high"]}},
                out=self.root / "3",
            )
        broken = self.root / "broken.jsonl"
        broken.write_text('{"question": "a"}\n')
        with self.assertRaisesRegex(ValueError, "broken.jsonl:1"):
            training_prepare.prepare_log(broken, out=self.root / "4")


class CommandTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.logs = game_logs(self.root / "logs")
        self.questions = self.root / "questions.json"
        self.questions.write_text(json.dumps(QUESTIONS))

    def cli(self, *argv):
        """Run the CLI in-process; returns (exit code, what it printed to stderr)."""
        import io
        from contextlib import redirect_stderr

        from zils_sdk import cli

        errors = io.StringIO()
        with patch.object(sys, "argv", ["zils", *argv]), redirect_stderr(errors):
            return cli.main(), errors.getvalue()

    def test_train_prepare_reads_a_log_folder(self):
        with patch("builtins.print") as printed:
            self.cli(
                "train",
                "prepare",
                str(self.logs),
                "--questions",
                str(self.questions),
                "--decide",
                "next_action",
                "--out",
                str(self.root / "out"),
                "--json",
            )
        result = json.loads(printed.call_args.args[0])
        self.assertEqual(result["episodes"], 12)
        code, said = self.cli(
            "train", "prepare", str(self.logs), "--read", "health", "--out", str(self.root / "x")
        )
        self.assertEqual(code, 2)
        self.assertIn("--read and --group are for tables", said)

    def test_fit_train_prepares_and_submits_with_the_usual_options(self):
        submitted = {}

        def submit(self, name, files, policy, consent, **options):
            submitted.update(name=name, files=files, policy=policy, consent=consent, **options)
            return {
                "id": "00000000-0000-4000-8000-000000000001",
                "name": name,
                "status": "validating",
            }

        with patch("zils_sdk.training.TrainingClient.submit", submit), patch("builtins.print"):
            code, said = self.cli(
                "fit",
                "train",
                "--log",
                str(self.logs),
                "--questions",
                str(self.questions),
                "--decide",
                "next_action",
                "--out",
                str(self.root / "out"),
                "--min-accuracy",
                "0.8",
                "--min-brier-improvement",
                "0.01",
                "--tier",
                "open",
                "--allow-open-tier-training",
                "--allow-training-data-export",
            )
        self.assertEqual(code, 0, said)
        self.assertEqual(submitted["name"], "next-action")
        self.assertEqual(submitted["policy"], {"min_accuracy": 0.8, "min_brier_improvement": 0.01})
        self.assertEqual(
            (submitted["consent"], submitted["tier"], submitted["allow_open_tier_training"]),
            (True, "open", True),
        )
        self.assertEqual(
            {s: Path(p) for s, p in submitted["files"].items()},
            {s: self.root / "out" / f"{s}.jsonl" for s in training_data.SPLITS},
        )


if __name__ == "__main__":
    unittest.main()
