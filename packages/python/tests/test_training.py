import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from test_training_data import datasets
from training_fixture import H2O_MODEL, JEVK5_MODEL, MODEL, OWNER, training_server
from zils_sdk import training
from zils_sdk._training_http import TrainingError


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = datasets(self.root)
        self.receipts = self.root / "receipts"

    def client(self, fixture):
        return training.TrainingClient(session=fixture.session, root=self.receipts)

    def submit(self, client):
        return client.submit(
            "routing", self.files, {"min_accuracy": 0.8, "min_brier_improvement": 0.01}, True
        )

    def test_submit_uses_d3_config_and_rejects_unpinned_or_retired_profiles(self):
        with training_server() as fixture:
            result = self.submit(self.client(fixture))
            self.assertEqual(fixture.jobs[result["id"]]["model"], MODEL)
            for model in (
                {**MODEL, "base_revision": "0" * 40},
                {**MODEL, "base": "vllm-sr/d3-mini"},
                H2O_MODEL,
                JEVK5_MODEL,
            ):
                fixture.session["config"]["model"] = model
                with self.assertRaisesRegex(ValueError, "pinned d3-mini"):
                    self.submit(self.client(fixture))
            self.assertEqual(len(fixture.jobs), 1)

    def test_submit_creates_once_uploads_snapshots_without_bearer_and_submits(self):
        with training_server() as fixture, redirect_stderr(io.StringIO()) as output:
            result = self.submit(self.client(fixture))
            self.assertEqual(result["status"], "validating")
            self.assertEqual(len(fixture.jobs), 1)
            creates = [call for call in fixture.calls if call[:2] == ("POST", "/v1/jobs")]
            self.assertEqual(len(creates), 1)
            self.assertTrue(json.loads(creates[0][3])["allow_training_data_export"])
            for split, source in self.files.items():
                self.assertEqual(fixture.objects[result["id"], split], source.read_bytes())
            for method, path, headers, body in fixture.calls:
                if method == "PUT":
                    self.assertNotIn("Authorization", headers)
                    self.assertEqual(int(headers["Content-Length"]), len(body))
            receipt = (self.receipts / (result["id"] + ".json")).read_text()
            self.assertNotIn("account-token", receipt)
            self.assertNotIn("private-grant", receipt + output.getvalue() + json.dumps(result))
            self.assertEqual(list((self.receipts / "snapshots").glob("*")), [])

    def test_out_of_credit_at_submit_says_to_add_credit_and_resume(self):
        said = "Add credit or resolve your payment issue before submitting this job."
        with training_server() as fixture, redirect_stderr(io.StringIO()):
            fixture.submit_error = (402, {"error": said})
            client = self.client(fixture)
            with self.assertRaises(TrainingError) as caught:
                self.submit(client)
            job_id = next(iter(fixture.jobs))
            message = str(caught.exception)
            self.assertEqual(caught.exception.status, 402)
            self.assertIn("not enough prepaid credit", message)
            self.assertIn("https://zils.ai/billing", message)
            self.assertNotIn(said, message)  # training errors never echo remote text
            self.assertIn(f"zils train resume {job_id}", message)
            for split, path in self.files.items():
                self.assertIn(f"--{split} {path}", message)
            self.assertNotIn("Check the job and account", message)
            self.assertEqual(fixture.jobs[job_id]["status"], "uploading")
            puts = sum(call[0] == "PUT" for call in fixture.calls)
            fixture.submit_error = None
            result = client.resume(job_id, self.files)
            self.assertEqual(result["status"], "validating")
            self.assertEqual(sum(call[0] == "PUT" for call in fixture.calls), puts)

    def test_a_failed_jobs_reason_is_shown(self):
        from zils_sdk.training_cli import render

        reason = (
            "Training this dataset costs $3.40 (17000000 training tokens at $0.20 per million)."
            " Add credit and submit it again."
        )
        with training_server() as fixture, redirect_stderr(io.StringIO()):
            client = self.client(fixture)
            result = self.submit(client)
            fixture.jobs[result["id"]].update(status="failed", error=reason)
            status = client.get(result["id"])
            self.assertEqual(status["error"], reason)
            self.assertIn("Reason: " + reason, render(status))
            fixture.jobs[result["id"]]["error"] = "Bad\x1b[31m data\n" + "x" * 900
            shown = client.get(result["id"])["error"]
            self.assertTrue(shown.startswith("Bad[31m data"))
            self.assertTrue(shown.isprintable())
            self.assertLessEqual(len(shown), 500)
            fixture.jobs[result["id"]]["error"] = "Cancelled by customer."
            status = client.get(result["id"])
            self.assertNotIn("error", status)
            self.assertIn("Cancellation recorded", render(status))

    def test_lost_create_reply_retains_pending_receipt_and_never_retries(self):
        with training_server() as fixture:
            fixture.drop_create = True
            client = self.client(fixture)
            with self.assertRaisesRegex(TrainingError, "zils train list"):
                self.submit(client)
            self.assertEqual(len(fixture.jobs), 1)
            self.assertEqual(len(client.list()["jobs"]), 1)
            pending = list(self.receipts.glob("pending-*.json"))
            self.assertEqual(len(pending), 1)
            self.assertIsNone(json.loads(pending[0].read_text())["job_id"])

    def test_submission_context_is_visible_before_job_creation(self):
        with training_server() as fixture, redirect_stderr(io.StringIO()) as output:
            client = self.client(fixture)
            request = client._request

            def checked_request(session, method, path, body=None):
                if method == "POST" and path == "/v1/jobs":
                    context = output.getvalue()
                    for value in (OWNER, fixture.url, "routing", "0.8", "0.01"):
                        self.assertIn(value, context)
                    for split, file in self.files.items():
                        self.assertIn(split, context)
                        self.assertIn(str(file), context)
                    self.assertIn("Training-data export: allowed", context)
                    self.assertIn("training allowance/credit", context)
                    self.assertNotIn("account-token", context)
                return request(session, method, path, body)

            with patch.object(client, "_request", side_effect=checked_request):
                self.submit(client)

    def test_partial_failure_resumes_same_job_and_rejects_changed_files(self):
        with training_server() as fixture, redirect_stderr(io.StringIO()):
            fixture.fail_split = "calibration"
            client = self.client(fixture)
            with self.assertRaises(TrainingError):
                self.submit(client)
            job_id = next(iter(fixture.jobs))
            calls = len(fixture.calls)
            original = self.files["test"].read_bytes()
            self.files["test"].write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                client.resume(job_id, self.files)
            self.assertEqual(len(fixture.calls), calls)
            self.files["test"].write_bytes(original)
            fixture.fail_split = None
            result = client.resume(job_id, self.files)
            self.assertEqual(result["status"], "validating")
            self.assertEqual(len(fixture.jobs), 1)
            train_puts = [
                call
                for call in fixture.calls
                if call[0] == "PUT" and call[1].endswith("/train.jsonl")
            ]
            self.assertEqual(len(train_puts), 1)
            before = len(fixture.calls)
            client.resume(job_id, self.files)
            self.assertEqual([call[0] for call in fixture.calls[before:]], ["GET"])

    def test_interrupted_resume_keeps_its_frozen_bytes(self):
        with training_server() as fixture, redirect_stderr(io.StringIO()):
            fixture.fail_split = "test"
            client = self.client(fixture)
            with self.assertRaises(TrainingError):
                self.submit(client)
            job_id = next(iter(fixture.jobs))
            with self.assertRaises(TrainingError):
                client.resume(job_id, self.files)
            receipt = json.loads((self.receipts / (job_id + ".json")).read_text())
            for split, info in receipt["files"].items():
                self.assertEqual(Path(info["path"]).read_bytes(), self.files[split].read_bytes())

    def test_receipt_binding_blocks_other_accounts_and_services(self):
        with training_server() as fixture, redirect_stderr(io.StringIO()):
            fixture.fail_split = "train"
            with self.assertRaises(TrainingError):
                self.submit(self.client(fixture))
            job_id = next(iter(fixture.jobs))
            for field, value in [
                ("account_id", "22222222-2222-4222-8222-222222222222"),
                ("base_url", "https://other.example"),
            ]:
                session = {**fixture.session, field: value}
                client = training.TrainingClient(session=session, root=self.receipts)
                with self.assertRaisesRegex(ValueError, "receipt"):
                    client.resume(job_id, self.files)

    def test_invalid_grants_never_send_dataset_bytes(self):
        with training_server() as fixture:
            job_id = "44444444-4444-4444-8444-444444444444"
            grant = fixture.grant(job_id, "train.jsonl")
            variants = [
                {**grant, "provider": "unknown"},
                {**grant, "expires_at": "2000-01-01T00:00:00Z"},
                {**grant, "headers": {**grant["headers"], "Authorization": "private-token"}},
                {**grant, "url": grant["url"].replace(fixture.url, "https://attacker.example")},
                {**grant, "url": grant["url"].replace(job_id, OWNER)},
                {**grant, "url": grant["url"].replace("train.jsonl", "test.jsonl")},
                {**grant, "url": grant["url"].replace("/inputs/", "/inputs/../inputs/")},
                {**grant, "url": grant["url"].replace("/inputs/", "/inputs/%2e%2e/")},
                {**grant, "url": grant["url"].replace("/inputs/", "/inputs/\\")},
                {**grant, "url": grant["url"] + "&token=second"},
                {**grant, "url": grant["url"] + "#fragment"},
            ]
            for bad in variants:
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        training.validate_grant(
                            bad,
                            fixture.session["config"],
                            job_id=job_id,
                            filename="train.jsonl",
                            use="dataset-upload",
                        )
            fixture.tamper_grant = lambda value: {
                **value,
                "url": value["url"].replace("/inputs/", "/inputs/%2e%2e/"),
            }
            with self.assertRaises(ValueError):
                self.submit(self.client(fixture))
            self.assertFalse(any(call[0] == "PUT" for call in fixture.calls))

    def test_spaces_requires_matching_provider_origin_and_multipart_parameters(self):
        with training_server() as fixture:
            config = copy.deepcopy(fixture.session["config"])
            config["storage"]["spaces_origin"] = "https://bucket.example"
            job_id = "44444444-4444-4444-8444-444444444444"
            grant = fixture.grant(job_id, "train.jsonl")
            grant.update(
                provider="spaces",
                headers={"Content-Type": "application/octet-stream"},
                url=f"https://bucket.example/objects/{OWNER}/fez-training-data/{job_id}/inputs/train.jsonl?partNumber=1&uploadId=private-grant",
            )
            self.assertEqual(
                training.validate_grant(
                    grant, config, job_id=job_id, filename="train.jsonl", use="dataset-upload"
                ),
                grant,
            )
            for url in [
                grant["url"].replace("partNumber=1", "partNumber=2"),
                grant["url"].split("?")[0],
            ]:
                with self.assertRaises(ValueError):
                    training.validate_grant(
                        {**grant, "url": url},
                        config,
                        job_id=job_id,
                        filename="train.jsonl",
                        use="dataset-upload",
                    )


if __name__ == "__main__":
    unittest.main()
