import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from test_training_data import datasets
from test_training_downloads import accepted
from training_fixture import OWNER, training_server
from zils_sdk import cli


class TrainingCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = datasets(self.root)
        self.key = self.root / "config.json"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("ZILS_", "TYPESAFE_"))}
        self.env["ZILS_CONFIG_PATH"] = str(self.key)

    def signin(self, fixture):
        session = {
            **fixture.session,
            "refresh_token": "fixture-refresh",
            "expires_at": time.time() + 3600,
        }
        session["config"] = {
            **session["config"],
            "version": 1,
            "oauth": {
                "issuer": fixture.url + "/auth/v1",
                "client_id": OWNER,
                "redirect_uri": "http://127.0.0.1:43187/callback",
            },
        }
        self.key.with_suffix(".account.json").write_text(json.dumps(session))
        self.env["ZILS_TRAINING_URL"] = fixture.url

    def run_cli(self, *args, input=None):
        return subprocess.run(
            [sys.executable, "-m", "zils_sdk.cli", *args],
            env=self.env,
            input=input,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def file_args(self):
        return [arg for split, path in self.files.items() for arg in ("--" + split, str(path))]

    def submit_args(self):
        return [
            "train",
            "submit",
            "--name",
            "routing",
            *self.file_args(),
            "--min-accuracy",
            ".8",
            "--min-brier-improvement",
            ".01",
            "--allow-training-data-export",
            "--json",
        ]

    def test_subprocess_submit_list_status_resume_cancel_and_download(self):
        with training_server() as fixture:
            self.signin(fixture)
            submitted = self.run_cli(*self.submit_args())
            self.assertEqual(submitted.returncode, 0, submitted.stderr)
            job_id = json.loads(submitted.stdout)["id"]
            for args in [
                ("list",),
                ("status", job_id),
                ("resume", job_id, *self.file_args()),
                ("cancel", job_id, "--yes"),
            ]:
                result = self.run_cli("train", *args, "--json")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsInstance(json.loads(result.stdout), dict)
                self.assertNotIn("account-token", result.stdout + result.stderr)
                self.assertNotIn("private-grant", result.stdout + result.stderr)
            job, _ = accepted(fixture)
            result = self.run_cli(
                "train", "download", job["id"], "--out", str(self.root / "model"), "--json"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((self.root / "model" / "release.json").exists())

    def test_invalid_local_input_and_cancel_without_yes_never_contacts_service(self):
        with training_server() as fixture:
            self.signin(fixture)
            for args in [
                ("train", "status", "../private-secret"),
                ("train", "list", "--timeout", "nan"),
                ("train", "cancel", OWNER),
                ("train", "submit", "--name", "routing"),
                ("train", "unsupported"),
            ]:
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("private-secret", result.stdout + result.stderr)
            self.assertEqual(fixture.calls, [])
            self.files["train"].write_text("private-invalid-json")
            result = self.run_cli(*self.submit_args())
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("private-invalid-json", result.stdout + result.stderr)
            self.assertEqual(fixture.calls, [])

    def test_max_cost_below_the_ceiling_never_contacts_service(self):
        with training_server() as fixture:
            self.signin(fixture)
            result = self.run_cli(*self.submit_args(), "--max-cost", "1.5")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("$2.00", result.stderr)
            self.assertEqual(fixture.calls, [])
            result = self.run_cli(*self.submit_args(), "--max-cost", "2")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_account_is_clear_and_never_opens_browser(self):
        result = self.run_cli("train", "list")
        self.assertEqual(result.returncode, 2)
        self.assertIn("zils login", result.stderr)
        self.assertNotIn("Opening", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_interactive_decline_does_not_cancel(self):
        from zils_sdk import training_cli

        args = cli.parser().parse_args(["train", "cancel", OWNER])
        with (
            patch("sys.stdin", io.StringIO("no\n")),
            patch("sys.stdin.isatty", return_value=True),
            patch.object(training_cli, "TrainingClient") as client,
            redirect_stderr(io.StringIO()),
        ):
            cli.run(args)
        client.return_value.cancel.assert_not_called()

    @unittest.skipIf(os.name == "nt", "POSIX signal behavior")
    def test_watch_interrupt_exits_without_cancelling(self):
        with training_server() as fixture:
            self.signin(fixture)
            job, _ = accepted(fixture)
            job["status"] = "running"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "zils_sdk.cli",
                    "train",
                    "status",
                    job["id"],
                    "--watch",
                    "--json",
                ],
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while (
                    not any(call[1] == "/v1/jobs/" + job["id"] for call in fixture.calls)
                    and time.monotonic() < deadline
                    and process.poll() is None
                ):
                    time.sleep(0.02)
                process.send_signal(signal.SIGINT)
                out, err = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 130, err)
                self.assertEqual(out, "")
                self.assertFalse(any(call[0] == "POST" for call in fixture.calls))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_terminal_watch_returns_pending_activation_without_extra_calls(self):
        with training_server() as fixture:
            self.signin(fixture)
            job, _ = accepted(fixture)
            result = self.run_cli("train", "status", job["id"], "--watch", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["workflow"], "activating")
            self.assertNotIn("model_id", json.loads(result.stdout))
            self.assertEqual(len([c for c in fixture.calls if c[1] == "/v1/jobs/" + job["id"]]), 1)
