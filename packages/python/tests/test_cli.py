import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sdk_helpers import QUESTIONS, api


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / "credentials.json"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("ZILS_", "TYPESAFE_"))}
        self.env["ZILS_CONFIG_PATH"] = str(self.config)

    def run_cli(self, *args, input=None):
        return subprocess.run(
            [sys.executable, "-m", "zils_sdk.cli", *args],
            env=self.env,
            input=input,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_login_models_and_logout_share_private_config(self):
        with api() as service:
            result = self.run_cli(
                "login", "--base-url", service.url, "--api-key", "-", input="valid-key\n"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("valid-key", result.stdout + result.stderr)
            self.assertEqual(self.config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(self.config.read_text())["base_url"], service.url)
            models = self.run_cli("models", "--json")
            self.assertEqual(models.returncode, 0, models.stderr)
            self.assertEqual(len(json.loads(models.stdout)["models"]), 2)
            self.assertEqual(self.run_cli("logout").returncode, 0)
            self.assertFalse(self.config.exists())
            self.assertEqual(self.run_cli("models").returncode, 2)

    def test_failed_login_does_not_replace_existing_credentials(self):
        original = '{"api_key":"old-key","base_url":"https://example.com"}'
        self.config.write_text(original)
        with api() as service:
            result = self.run_cli(
                "login", "--base-url", service.url, "--api-key", "-", input="bad-secret"
            )
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("bad-secret", result.stdout + result.stderr)
        self.assertEqual(self.config.read_text(), original)

    def test_decide_from_stdin_and_body_file_preserves_usage(self):
        body = Path(self.temp.name) / "request.json"
        body.write_text(json.dumps({"state": {"ticket": "Refund"}, "questions": QUESTIONS}))
        with api() as service:
            self.env.update(ZILS_API_KEY="valid-key", ZILS_BASE_URL=service.url)
            for args, stdin in [
                (("--state", "-", "--questions", json.dumps(QUESTIONS)), "Refund"),
                (("--body", "@" + str(body)), None),
            ]:
                result = self.run_cli("decide", *args, "--json", input=stdin)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assertEqual(data["usage"]["billable_input_tokens"], 73)
                self.assertEqual(data["answers"]["urgency"]["legend"]["0"], {"level": "low"})

    def test_invalid_inputs_do_not_send_or_echo_private_content(self):
        with api() as service:
            self.env.update(ZILS_API_KEY="valid-key", ZILS_BASE_URL=service.url)
            for body in (
                '{"state":"private-text", bad json}',
                '{"state":"private-text","state":"second","questions":{}}',
                json.dumps({"state": "private-text", "questions": QUESTIONS, "typo": True}),
                json.dumps({"state": "private-text", "questions": {"q": {"type": "invalid"}}}),
            ):
                result = self.run_cli("decide", "--body", body)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("private-text", result.stdout + result.stderr)
            self.assertEqual(service.calls, [])

    def test_rate_limit_is_nonzero_and_not_retried(self):
        with api() as service:
            service.store.limit = True
            self.env.update(ZILS_API_KEY="valid-key", ZILS_BASE_URL=service.url)
            result = self.run_cli(
                "decide", "--state", "Refund", "--questions", json.dumps(QUESTIONS)
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("429", result.stderr)
            self.assertEqual(len(service.calls), 1)

    def test_help_and_version_work_without_credentials(self):
        for flag in ("--help", "--version"):
            result = self.run_cli(flag)
            self.assertEqual(result.returncode, 0, result.stderr)


class MinerForwardTests(unittest.TestCase):
    """`zils <miner command>` runs `zils-miner` from the miner install (ZIL-127)."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.path_dir = self.root / "path"
        self.path_dir.mkdir()
        self.record = self.root / "args.json"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("ZILS_", "TYPESAFE_"))}
        self.env.update(
            HOME=str(self.home), PATH=f"{self.path_dir}{os.pathsep}/usr/bin{os.pathsep}/bin"
        )

    def fake_miner(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        script = folder / "zils-miner"
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, sys\n"
            f"open({str(self.record)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            "print('miner says hi')\n"
            "sys.exit(7)\n"
        )
        script.chmod(0o755)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "zils_sdk.cli", *args],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_miner_commands_run_zils_miner_with_the_same_arguments(self):
        self.fake_miner(self.path_dir)
        for command in (
            "setup",
            "doctor",
            "update",
            "commit-candidate",
            "commit-body",
            "announce-serving",
            "serve",
        ):
            args = [command, "--network", "test", "--model", "d3-mini 4.5b"]
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(result.stdout, "miner says hi\n")
            self.assertEqual(json.loads(self.record.read_text()), args)

    def test_the_installer_bin_folder_is_found_off_path(self):
        self.fake_miner(self.home / ".local/bin")
        result = self.run_cli("doctor")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(json.loads(self.record.read_text()), ["doctor"])

    def test_without_the_miner_install_it_says_how_to_get_it(self):
        result = self.run_cli("doctor", "--model", "x")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("zils-miner", result.stderr)
        self.assertIn("curl -fsSL https://zils.ai/install.sh | sh", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_help_lists_both_command_groups(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("decide", result.stdout)
        self.assertIn("Miner commands", result.stdout)
        for command in ("setup", "doctor", "commit-candidate", "serve"):
            self.assertIn(command, result.stdout.split("Miner commands", 1)[1])

    def test_customer_commands_are_not_forwarded(self):
        self.fake_miner(self.path_dir)
        result = self.run_cli("models", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
