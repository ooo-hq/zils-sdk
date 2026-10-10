import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

from test_account import callback_request
from zils_sdk import account
from zils_sdk._training_http import TrainingError
from zils_sdk.training import TrainingClient


@unittest.skipUnless(
    os.environ.get("ZILS_TEST_PLATFORM") == "1", "optional real-platform integration"
)
class PlatformTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        from tests.fixture_jobs import examples

        self.files = {}
        for split, cases in examples().items():
            file = self.root / (split + ".jsonl")
            file.write_text("".join(json.dumps(case) + "\n" for case in cases))
            self.files[split] = file

    def sign_in(self, service):
        def browser(url):
            state = parse_qs(urlsplit(url).query)["state"][0]
            thread = threading.Thread(
                target=callback_request,
                args=("/callback?" + urlencode({"state": state, "code": "fixture-code"}),),
            )
            thread.start()
            self.addCleanup(thread.join, 5)
            return True

        with patch("webbrowser.open", browser):
            account.login()
        return TrainingClient(root=self.root / "training")

    def test_browser_account_submit_resume_and_tenant_boundaries(self):
        from platform_training import training_service
        from tests.test_queue import OTHER, OWNER
        from zils.cloud import DATA_BUCKET

        with (
            training_service() as service,
            patch.dict(
                os.environ,
                {
                    "ZILS_CONFIG_PATH": str(self.root / "config.json"),
                    "ZILS_TRAINING_URL": service.url,
                },
            ),
            redirect_stderr(io.StringIO()),
        ):
            client = self.sign_in(service)
            self.assertEqual(account.credentials()["account_id"], OWNER)
            service.store.fail_split = "calibration"
            with self.assertRaises(TrainingError):
                client.submit(
                    "routing",
                    self.files,
                    {"min_accuracy": 0.8, "min_brier_improvement": 0.01},
                    True,
                )
            job = service.store.tables["fez_training_jobs"][0]
            job_id = job["id"]
            first = service.store.objects[DATA_BUCKET, f"{job_id}/inputs/train.jsonl"]
            service.store.fail_split = None
            self.assertEqual(client.resume(job_id, self.files)["status"], "validating")
            self.assertEqual(
                service.store.objects[DATA_BUCKET, f"{job_id}/inputs/train.jsonl"], first
            )
            self.assertEqual(
                len(
                    [
                        path
                        for method, path, _ in service.calls
                        if method == "PUT" and "/inputs/train.jsonl?" in path
                    ]
                ),
                1,
            )
            self.assertEqual(len(client.list()["jobs"]), 1)
            session = {**account.credentials(), "account_id": OTHER, "access_token": "other-token"}
            foreign = TrainingClient(session=session, root=self.root / "foreign")
            for method in ("get", "cancel", "download"):
                with self.assertRaises(TrainingError) as caught:
                    getattr(foreign, method)(
                        job_id, self.root / "foreign-model"
                    ) if method == "download" else getattr(foreign, method)(job_id)
                self.assertEqual(caught.exception.status, 404)
            self.assertEqual(client.cancel(job_id)["status"], "failed")
            self.assertFalse(any("Authorization" in headers for _, _, headers in service.calls))

    def test_real_checkpoint_hash_and_readiness_contract(self):
        import zils
        from platform_training import training_service
        from zils import models
        from zils.cloud import MODEL_BUCKET

        with (
            training_service() as service,
            patch.dict(
                os.environ,
                {
                    "ZILS_CONFIG_PATH": str(self.root / "config.json"),
                    "ZILS_TRAINING_URL": service.url,
                },
            ),
            redirect_stderr(io.StringIO()),
        ):
            client = self.sign_in(service)
            submitted = client.submit(
                "routing", self.files, {"min_accuracy": 0.8, "min_brier_improvement": 0.01}, True
            )
            job = service.store.tables["fez_training_jobs"][0]
            job["status"] = "completed"
            job["result"] = {"delivery": {"status": "no_qualifying_model"}}
            self.assertFalse(client.get(job["id"])["download_available"])
            with self.assertRaises(ValueError):
                client.download(job["id"], self.root / "not-accepted")
            checkpoint = self.root / "reference"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture adapter")
            models.write_metadata(checkpoint, temperature=1.3)
            digest = zils.checkpoint_hash(checkpoint)
            profile = models.spec(models.JEVK5)
            round_id = "33333333-3333-4333-8333-333333333333"
            delivery = {"status": "accepted", "sha256": digest, "uid": 1}
            release = {
                "job_id": job["id"],
                "round_id": round_id,
                "model": profile,
                "base": profile["base"],
                "base_revision": profile["base_revision"],
                **delivery,
            }
            (checkpoint / "release.json").write_text(json.dumps(release))
            job["release_prefix"] = f"{job['id']}/releases/{round_id}"
            job["result"] = {"delivery": delivery, "workflow": {"state": "activating"}}
            for file in checkpoint.iterdir():
                service.store.objects[MODEL_BUCKET, job["release_prefix"] + "/" + file.name] = (
                    file.read_bytes()
                )
            self.assertNotIn("model_id", client.get(job["id"]))
            destination = self.root / "accepted"
            result = client.download(submitted["id"], destination)
            self.assertEqual(result["sha256"], zils.checkpoint_hash(destination))
            job["result"]["workflow"] = {"state": "ready", "model_id": "zils-adapter-fixture"}
            self.assertEqual(client.get(job["id"])["model_id"], "zils-adapter-fixture")
