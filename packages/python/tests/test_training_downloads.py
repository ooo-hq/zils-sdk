import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_fixture import H2O_MODEL, JEVK5_MODEL, MODEL, OWNER, ROUND, training_server
from zils_sdk import training
from zils_sdk._training_http import TrainingError


def accepted(fixture, model=MODEL):
    job_id = "44444444-4444-4444-8444-444444444444"
    fixture.artifacts = {
        "adapter_config.json": b'{"peft_type":"LORA"}',
        "adapter_model.safetensors": b"synthetic adapter bytes",
        "model.json": json.dumps(
            {
                "version": "zils-checkpoint/v1",
                "model": model["id"],
                "kind": "adapter",
                "temperature": 1.25,
                **({"profile": model} if model["id"] in training.PROFILED else {}),
            }
        ).encode(),
    }
    records = [
        [name, hashlib.sha256(fixture.artifacts[name]).hexdigest()]
        for name in training.ARTIFACTS[:-1]
    ]
    digest = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    delivery = {"status": "accepted", "sha256": digest, "uid": 7}
    release = {
        "job_id": job_id,
        "round_id": ROUND,
        "model": model,
        "base": model["base"],
        "base_revision": model["base_revision"],
        **delivery,
    }
    fixture.artifacts["release.json"] = json.dumps(release).encode()
    job = fixture.jobs[job_id] = {
        "id": job_id,
        "name": "routing",
        "status": "completed",
        "model": model,
        "result": {"delivery": delivery},
        "workflow": {"state": "activating", "model_id": "premature-id"},
    }
    return job, digest


class DownloadsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "model"

    def client(self, fixture):
        return training.TrainingClient(session=fixture.session, root=self.root / "receipts")

    def test_d3_download_requires_pinned_profile_and_older_results_still_download(self):
        with training_server() as fixture:
            d3_job, digest = accepted(fixture, MODEL)
            result = self.client(fixture).download(d3_job["id"], self.out)
            self.assertEqual(result["sha256"], digest)
            self.assertEqual(json.loads((self.out / "model.json").read_text())["profile"], MODEL)
            for index, model in enumerate((H2O_MODEL, JEVK5_MODEL)):
                job, digest = accepted(fixture, model)
                out = self.root / f"older-{index}"
                self.assertEqual(self.client(fixture).download(job["id"], out)["sha256"], digest)
            metadata = json.loads((self.out / "model.json").read_text())
            del metadata["profile"]
            (self.out / "model.json").write_text(json.dumps(metadata))
            records = [
                [name, hashlib.sha256((self.out / name).read_bytes()).hexdigest()]
                for name in training.ARTIFACTS[:-1]
            ]
            forged = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
            d3_job["result"]["delivery"]["sha256"] = forged
            release = json.loads((self.out / "release.json").read_text())
            (self.out / "release.json").write_text(json.dumps({**release, "sha256": forged}))
            with self.assertRaisesRegex(ValueError, "accepted job"):
                training.verify_checkpoint(self.out, d3_job, records, ROUND)

    def test_h2o_download_preserves_native_profile(self):
        with training_server() as fixture:
            job, digest = accepted(fixture, H2O_MODEL)
            result = self.client(fixture).download(job["id"], self.out)
            self.assertEqual(result["sha256"], digest)
            self.assertEqual(
                json.loads((self.out / "model.json").read_text())["profile"], H2O_MODEL
            )
            metadata = json.loads((self.out / "model.json").read_text())
            metadata["profile"]["base_revision"] = MODEL["base_revision"]
            (self.out / "model.json").write_text(json.dumps(metadata))
            records = [
                [name, hashlib.sha256((self.out / name).read_bytes()).hexdigest()]
                for name in training.ARTIFACTS[:-1]
            ]
            forged = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
            job["result"]["delivery"]["sha256"] = forged
            release = json.loads((self.out / "release.json").read_text())
            release["sha256"] = forged
            (self.out / "release.json").write_text(json.dumps(release))
            with self.assertRaisesRegex(ValueError, "accepted job"):
                training.verify_checkpoint(self.out, job, records, ROUND)

    def test_download_verifies_release_and_publishes_private_files(self):
        with training_server() as fixture:
            job, digest = accepted(fixture)
            result = self.client(fixture).download(job["id"], self.out)
            self.assertEqual(result["sha256"], digest)
            self.assertEqual(set(p.name for p in self.out.iterdir()), set(training.ARTIFACTS))
            self.assertEqual(self.out.stat().st_mode & 0o777, 0o700)
            for name in training.ARTIFACTS:
                self.assertEqual((self.out / name).read_bytes(), fixture.artifacts[name])
                self.assertEqual((self.out / name).stat().st_mode & 0o777, 0o600)
            for _, path, headers, _ in fixture.calls:
                if path.startswith("/storage/"):
                    self.assertNotIn("Authorization", headers)
            self.assertNotIn("private-grant", json.dumps(result))

    def test_corruption_identity_and_release_mismatch_publish_nothing(self):
        for wrong in (
            "bytes",
            "job",
            "model",
            "base",
            "release-hash",
            "not-accepted",
            "metadata",
            "extra-file",
            "traversal",
        ):
            with self.subTest(wrong=wrong), training_server() as fixture:
                job, _ = accepted(fixture)
                if wrong == "bytes":
                    fixture.artifacts["adapter_model.safetensors"] += b"corruption"
                elif wrong == "metadata":
                    fixture.artifacts["model.json"] = b'{"kind":"base"}'
                elif wrong in ("extra-file", "traversal"):
                    fixture.artifacts["extra.txt" if wrong == "extra-file" else "../outside"] = (
                        b"private"
                    )
                else:
                    release = json.loads(fixture.artifacts["release.json"])
                    if wrong == "job":
                        release["job_id"] = OWNER
                    if wrong == "model":
                        release["model"] = {"id": "imajev-4b-v1"}
                    if wrong == "base":
                        release["base_revision"] = "wrong"
                    if wrong == "release-hash":
                        release["sha256"] = "0" * 64
                    if wrong == "not-accepted":
                        release["status"] = "no_qualifying_model"
                    fixture.artifacts["release.json"] = json.dumps(release).encode()
                with self.assertRaises((ValueError, TrainingError)):
                    self.client(fixture).download(job["id"], self.out)
                self.assertFalse(self.out.exists())
                self.assertFalse(list(self.root.glob(".zils-download-*")))

    def test_truncation_redirect_oversize_and_missing_length_are_bounded(self):
        for mode in ("truncated", "redirect", "no-length", "oversized"):
            with self.subTest(mode=mode), training_server() as fixture:
                job, _ = accepted(fixture)
                fixture.download_mode = mode
                client = self.client(fixture)
                if mode == "no-length":
                    client.download(job["id"], self.out)
                    for file in self.out.iterdir():
                        file.unlink()
                    self.out.rmdir()
                else:
                    with patch.object(
                        training, "MAX_ARTIFACT_BYTES", 8 if mode == "oversized" else 512 * 1024**2
                    ):
                        with self.assertRaises((ValueError, TrainingError)):
                            client.download(job["id"], self.out)
                    self.assertFalse(self.out.exists())

    def test_existing_outputs_and_unaccepted_jobs_never_download(self):
        with training_server() as fixture:
            job, _ = accepted(fixture)
            client = self.client(fixture)
            self.out.mkdir()
            sentinel = self.out / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises((ValueError, OSError)):
                client.download(job["id"], self.out)
            self.assertEqual(sentinel.read_text(), "keep")
            link = self.root / "link"
            link.symlink_to(self.out, target_is_directory=True)
            with self.assertRaises((ValueError, OSError)):
                client.download(job["id"], link)
            job["result"]["delivery"]["status"] = "no_qualifying_model"
            with self.assertRaises(ValueError):
                client.download(job["id"], self.root / "absent")
            self.assertFalse(any("/storage/" in call[1] for call in fixture.calls))

    def test_status_separates_training_acceptance_activation_and_readiness(self):
        with training_server() as fixture:
            job, _ = accepted(fixture)
            client = self.client(fixture)
            for status in (
                "uploading",
                "validating",
                "awaiting_approval",
                "queued",
                "running",
                "evaluating",
                "failed",
            ):
                job["status"] = status
                shown = client.get(job["id"])
                self.assertEqual(shown["status"], status)
                self.assertNotIn("model_id", shown)
                self.assertFalse(shown["download_available"])
            job["status"] = "completed"
            for state in ("activating", "activation_failed", "needs_review", "ready"):
                job["workflow"]["state"] = state
                shown = client.get(job["id"])
                self.assertEqual(shown["delivery"], "accepted")
                self.assertEqual(shown["workflow"], state)
                self.assertEqual("model_id" in shown, state == "ready")
            job["result"]["delivery"]["status"] = "no_qualifying_model"
            shown = client.get(job["id"])
            self.assertEqual(shown["delivery"], "no_qualifying_model")
            self.assertFalse(shown["download_available"])
            self.assertNotIn("model_id", shown)

    def test_cancel_changes_same_job_and_ownership_failures_never_retry(self):
        with training_server() as fixture:
            job, _ = accepted(fixture)
            job["status"] = "queued"
            result = self.client(fixture).cancel(job["id"])
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["cancelled"])
            self.assertEqual(len([c for c in fixture.calls if c[0] == "POST"]), 1)
            fixture.session["access_token"] = "other-token"
            for action in ("get", "cancel"):
                count = len(fixture.calls)
                with self.assertRaises(TrainingError) as error:
                    getattr(self.client(fixture), action)(job["id"])
                self.assertEqual(error.exception.status, 401)
                self.assertEqual(len(fixture.calls), count + 1)

    def test_evaluation_metrics_are_selected_and_malformed_status_fails_safely(self):
        with training_server() as fixture:
            job, _ = accepted(fixture)
            job["result"]["baseline"] = {"accuracy": 0.7, "brier": 0.4, "private": "secret"}
            job["result"]["miners"] = [
                {"uid": 7, "accuracy": 0.9, "brier": 0.2, "checkpoint": "private-path"}
            ]
            result = training.public_status(job)
            self.assertEqual(result["evaluation"]["candidate"]["accuracy"], 0.9)
            self.assertNotIn("private", json.dumps(result))
            for field, value in [
                ("result", []),
                ("result", {"delivery": "secret"}),
                ("workflow", "secret"),
            ]:
                bad = {**job, field: value}
                with self.assertRaises(ValueError):
                    training.public_status(bad)
