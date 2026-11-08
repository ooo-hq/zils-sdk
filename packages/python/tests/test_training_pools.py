"""Pool selection crosses the HTTP boundary once and stays pinned through resume."""

import copy
import hashlib
import json
import unittest

import test_training as workflows
import test_training_cli as terminal
import test_training_downloads as downloads
from training_fixture import MODEL, training_server
from zils_sdk import training
from zils_sdk._training_http import TrainingError

LITE = {
    **MODEL,
    "id": "d3-lite-0.8b-v1",
    "name": "d3-lite",
    "base": "vllm-sr/d3-lite",
    "base_revision": "03560c26f5199c3c04129a2ccb38960c913635e0",
    "runtime_revision": "d3-vendor-runtime-adf7f7aa0856/v1",
}


def catalog():
    return {
        pool: {
            "id": pool,
            "enabled": True,
            "channels": {"stable": model, "latest": model},
            "availability": {
                c: {"training_enabled": True, "inputs": {"text": True}}
                for c in ("stable", "latest")
            },
        }
        for pool, model in (("d3-mini", MODEL), ("d3-lite", LITE))
    }


class PoolTests(unittest.TestCase):
    setUp = workflows.TrainingTests.setUp
    client = workflows.TrainingTests.client

    def test_selection_and_receipt_pin_the_same_profile(self):
        with training_server() as fixture:
            fixture.session["config"]["pools"] = catalog()
            client = self.client(fixture)
            result = client.submit(
                "lite",
                self.files,
                {"min_accuracy": 0.8, "min_brier_improvement": 0.01},
                True,
                pool="d3-lite",
                base_channel="latest",
            )
            request = json.loads(next(c[3] for c in fixture.calls if c[:2] == ("POST", "/v1/jobs")))
            self.assertEqual(
                {k: request[k] for k in ("pool", "base_channel", "model")},
                {"pool": "d3-lite", "base_channel": "latest", "model": LITE},
            )
            receipt = json.loads((self.receipts / (result["id"] + ".json")).read_text())
            self.assertEqual(receipt["model"], LITE)
            fixture.session["config"]["pools"]["d3-lite"]["enabled"] = False
            self.assertEqual(client.resume(result["id"], self.files)["status"], "validating")
            fixture.jobs[result["id"]]["model"] = MODEL
            with self.assertRaisesRegex(ValueError, "profile"):
                client.resume(result["id"], self.files)

    def test_unavailable_conflicting_or_unpinned_selection_never_creates(self):
        with training_server() as fixture:
            fixture.session["config"]["pools"] = catalog()
            client = self.client(fixture)
            for selection in (
                {"pool": "unknown"},
                {"pool": "d3-lite", "model": MODEL["id"]},
                {"base_channel": "bad"},
                {"model": "arbitrary"},
            ):
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    client.submit(
                        "lite",
                        self.files,
                        {"min_accuracy": 0.8, "min_brier_improvement": 0.01},
                        True,
                        **selection,
                    )
            for field, value in (
                ("enabled", False),
                ("availability", {"stable": {"training_enabled": False}}),
                ("channels", {"stable": {**LITE, "base_revision": "0" * 40}}),
            ):
                fixture.session["config"]["pools"] = catalog()
                fixture.session["config"]["pools"]["d3-lite"][field] = value
                with self.assertRaises(ValueError):
                    client.submit(
                        "lite",
                        self.files,
                        {"min_accuracy": 0.8, "min_brier_improvement": 0.01},
                        True,
                        pool="d3-lite",
                    )
            self.assertFalse(fixture.jobs)

    def test_create_reply_cannot_change_selected_model(self):
        with training_server() as fixture:
            fixture.session["config"]["pools"] = catalog()
            fixture.returned_model = MODEL
            with self.assertRaises(TrainingError):
                self.client(fixture).submit(
                    "lite",
                    self.files,
                    {"min_accuracy": 0.8, "min_brier_improvement": 0.01},
                    True,
                    model=LITE["id"],
                )
            self.assertEqual(len(fixture.jobs), 1)
            self.assertFalse(fixture.objects)


class PoolCliTests(unittest.TestCase):
    setUp = terminal.TrainingCliTests.setUp
    signin = terminal.TrainingCliTests.signin
    run_cli = terminal.TrainingCliTests.run_cli
    file_args = terminal.TrainingCliTests.file_args
    submit_args = terminal.TrainingCliTests.submit_args

    def test_pool_flags_reach_http_without_credentials_in_arguments(self):
        with training_server() as fixture:
            fixture.session["config"]["pools"] = catalog()
            self.signin(fixture)
            result = self.run_cli(
                *self.submit_args(),
                "--pool",
                "d3-lite",
                "--base-channel",
                "latest",
                "--model",
                LITE["id"],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(fixture.jobs[json.loads(result.stdout)["id"]]["model"], LITE)


class PoolDownloadTests(unittest.TestCase):
    setUp = downloads.DownloadsTests.setUp
    client = downloads.DownloadsTests.client

    def test_new_pool_download_preserves_full_profile_and_rejects_other_profile(self):
        with training_server() as fixture:
            job, digest = downloads.accepted(fixture, LITE)
            result = self.client(fixture).download(job["id"], self.out)
            self.assertEqual(result["sha256"], digest)
            self.assertEqual(json.loads((self.out / "model.json").read_text())["profile"], LITE)
            bad = copy.deepcopy(job)
            bad["model"] = MODEL
            records = [
                [name, hashlib.sha256((self.out / name).read_bytes()).hexdigest()]
                for name in training.ARTIFACTS[:-1]
            ]
            with self.assertRaises(ValueError):
                training.verify_checkpoint(
                    self.out, bad, records, "33333333-3333-4333-8333-333333333333"
                )


class ServiceWithoutPoolsTests(unittest.TestCase):
    setUp = workflows.TrainingTests.setUp
    client = workflows.TrainingTests.client
    POLICY = {"min_accuracy": 0.8, "min_brier_improvement": 0.01}

    def test_the_default_pool_works_on_a_service_without_pools(self):
        # training.zils.ai on 8 Nov 2026: no pools catalog, so naming d3-mini (the only choice) was refused.
        with training_server() as fixture:
            self.assertNotIn("pools", fixture.session["config"])
            result = self.client(fixture).submit(
                "mini", self.files, self.POLICY, True, pool="d3-mini"
            )
            request = json.loads(next(c[3] for c in fixture.calls if c[:2] == ("POST", "/v1/jobs")))
            self.assertNotIn("pool", request)
            self.assertEqual(request["model"], MODEL)
            self.assertIn(result["id"], fixture.jobs)

    def test_a_refused_create_keeps_the_reason_the_sdk_knows(self):
        for status, expected in ((400, "HTTP 400"), (401, "Run zils login"), (402, "Add credit")):
            with training_server() as fixture:
                fixture.create_error = (status, {"error": "remote text the SDK never shows"})
                with self.assertRaises(TrainingError) as raised:
                    self.client(fixture).submit("mini", self.files, self.POLICY, True)
                message = str(raised.exception)
                self.assertIn(expected, message)
                self.assertIn("Run zils train list before submitting again", message)
                self.assertNotIn("remote text", message)
