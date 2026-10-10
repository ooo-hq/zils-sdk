"""Browser login and refresh against local OAuth/coordinator HTTP endpoints."""

import base64
import hashlib
import http.client
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

from zils_sdk import cli

OWNER = "11111111-1111-4111-8111-111111111111"
CLIENT = "22222222-2222-4222-8222-222222222222"
CALLBACK = "http://127.0.0.1:43187/callback"


@contextmanager
def auth_server():
    calls = []
    state = {"drop_refresh": False, "deny_identity": False, "owner": OWNER}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, data):
            raw = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization")))
            if self.path == "/v1/config":
                self.reply(
                    200,
                    {
                        "model": {"id": "jevk5-4b-v0.3"},
                        "cli": {
                            "version": 1,
                            "oauth": {
                                "issuer": url + "/auth/v1",
                                "client_id": CLIENT,
                                "redirect_uri": CALLBACK,
                            },
                            "storage": {"supabase_origin": url, "spaces_origin": None},
                        },
                    },
                )
            elif self.path == "/v1/account":
                self.reply(401 if state["deny_identity"] else 200, {"account_id": state["owner"]})
            elif self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", url + "/private-token")
                self.end_headers()
            else:
                self.reply(400, {"error": "private-secret"})

        def do_POST(self):
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
            calls.append((self.path, form))
            if state["drop_refresh"] and form.get("grant_type") == ["refresh_token"]:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.reply(
                200,
                {
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh",
                    "token_type": "bearer",
                    "expires_in": 3600,
                },
            )

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield url, calls, state
        finally:
            server.shutdown()
            thread.join(5)


def callback_request(target, host="127.0.0.1:43187", method="GET"):
    conn = http.client.HTTPConnection("127.0.0.1", 43187, timeout=3)
    try:
        conn.request(method, target, headers={"Host": host})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.key = Path(self.temp.name) / "config.json"
        self.session = self.key.with_suffix(".account.json")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("ZILS_", "TYPESAFE_"))}
        self.env["ZILS_CONFIG_PATH"] = str(self.key)
        self.environment = patch.dict(os.environ, self.env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def browser(self, visited, *, denied=False, invalid_first=False):
        def open_browser(url):
            visited.append(url)
            query = parse_qs(urlsplit(url).query)
            target = "/callback?" + urlencode(
                {
                    "state": query["state"][0],
                    "error" if denied else "code": "access_denied" if denied else "private-code",
                }
            )

            def complete():
                if invalid_first:
                    self.assertEqual(
                        callback_request("/callback?code=private-code&state=wrong")[0], 400
                    )
                    self.assertEqual(callback_request(target, host="evil.example")[0], 400)
                    self.assertEqual(callback_request(target, method="POST")[0], 405)
                callback_request(target)

            thread = threading.Thread(target=complete)
            thread.start()
            self.addCleanup(thread.join, 5)
            return True

        return open_browser

    def login(self, url, **kwargs):
        from zils_sdk import account

        visited = []
        with (
            patch.dict(os.environ, {"ZILS_TRAINING_URL": url}),
            patch("webbrowser.open", self.browser(visited, **kwargs)),
        ):
            result = account.login()
        return result, visited

    def test_default_login_browser_and_key_mode_are_explicit(self):
        from zils_sdk import account

        self.assertIsNone(cli.parser().parse_args(["login"]).api_key)
        self.assertIs(cli.parser().parse_args(["login", "--api-key"]).api_key, True)
        with (
            patch.object(
                account,
                "login",
                return_value={"account_id": OWNER, "base_url": "https://training.zils.ai"},
            ) as login,
            redirect_stdout(io.StringIO()),
        ):
            cli.run(cli.parser().parse_args(["login"]))
            login.assert_called_once()
        with patch.object(account, "login") as login:
            with self.assertRaisesRegex(ValueError, "--api-key"):
                cli.run(cli.parser().parse_args(["login", "--base-url", "https://example.com"]))
            login.assert_not_called()
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(SystemExit):
            cli.parser().parse_args(["login", "--api-key", "private-secret"])
        self.assertNotIn("private-secret", output.getvalue())

    def test_explicit_api_key_prompts_privately_and_browser_sessions_are_not_keys(self):
        from sdk_helpers import api
        from zils_sdk.credentials import load_credentials

        with (
            api() as service,
            patch("sys.stdin.isatty", return_value=True),
            patch("getpass.getpass", return_value="valid-key") as prompt,
            redirect_stderr(io.StringIO()),
        ):
            cli.run(cli.parser().parse_args(["login", "--api-key", "--base-url", service.url]))
            prompt.assert_called_once()
            self.assertEqual(load_credentials()["api_key"], "valid-key")
        self.key.unlink()
        self.session.write_text('{"access_token":"private-token"}')
        with self.assertRaisesRegex(ValueError, "zils login --api-key"):
            load_credentials()

    def test_config_and_tokens_fail_closed(self):
        from zils_sdk import account

        with auth_server() as (url, _, _):
            config = account.load_config(url)
        for field, value in [
            ("issuer", "https://attacker.example/auth/v1"),
            ("client_id", "private-token"),
            ("redirect_uri", "http://localhost:43187/callback"),
        ]:
            changed = {**config, "oauth": {**config["oauth"], field: value}}
            with self.assertRaisesRegex(ValueError, "configuration"):
                account.validate_config(url, changed)
        for expires in (0, -1, True, float("inf"), "3600"):
            with self.assertRaisesRegex(ValueError, "sign-in response"):
                account.tokens(
                    {
                        "token_type": "bearer",
                        "access_token": "a",
                        "refresh_token": "r",
                        "expires_in": expires,
                    }
                )

    def test_success_uses_pkce_and_saves_only_after_verified_identity(self):
        from zils_sdk import account
        from zils_sdk.credentials import load_credentials

        self.key.write_text('{"api_key":"prior-key","base_url":"https://example.com"}')
        with auth_server() as (url, calls, _):
            result, visited = self.login(url, invalid_first=True)
            self.assertEqual(result, {"account_id": OWNER, "base_url": url})
            query = parse_qs(urlsplit(visited[0]).query)
            self.assertEqual(query["scope"], ["email"])
            self.assertEqual(query["code_challenge_method"], ["S256"])
            self.assertEqual(query["redirect_uri"], [CALLBACK])
            form = next(body for path, body in calls if path.endswith("/oauth/token"))
            challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"][0].encode()).digest())
                .rstrip(b"=")
                .decode()
            )
            self.assertEqual(query["code_challenge"], [challenge])
            self.assertEqual(form["code"], ["private-code"])
            self.assertEqual(self.session.stat().st_mode & 0o777, 0o600)
            self.assertEqual(load_credentials()["api_key"], "prior-key")
            with patch.dict(os.environ, {"ZILS_TRAINING_URL": url}):
                self.assertEqual(account.credentials()["access_token"], "rotated-access")
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", 43187))

    def test_callback_bounds_duplicates_state_and_denial(self):
        from zils_sdk.account import callback_code

        self.assertEqual(callback_code("/callback?code=ok&state=expected", "expected"), "ok")
        for target in [
            "/callback?code=private-code&state=wrong",
            "/callback?code=a&code=b&state=expected",
            "/other?code=a&state=expected",
            "/callback?code=" + "x" * 8192,
            "/callback?code=%00&state=expected",
            "/callback?code=a&state=expected#fragment",
            "/callback?error=access_denied&state=expected",
        ]:
            with self.subTest(target=target[:60]), self.assertRaises(ValueError) as caught:
                callback_code(target, "expected")
            self.assertNotIn("private-code", str(caught.exception))

    def test_denied_timeout_occupied_port_and_failed_identity_preserve_credentials(self):
        from zils_sdk import account

        self.key.write_text("prior-key-file")
        self.session.write_text("prior-session-file")
        with auth_server() as (url, _, state):
            with self.assertRaisesRegex(ValueError, "denied"):
                self.login(url, denied=True)
            with (
                patch.dict(os.environ, {"ZILS_TRAINING_URL": url}),
                patch("webbrowser.open", return_value=True),
                patch.object(account, "LOGIN_TIMEOUT", 0.03),
            ):
                with self.assertRaisesRegex(ValueError, "timed out"):
                    account.login()
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", 43187))
                sock.listen()
                with (
                    patch.dict(os.environ, {"ZILS_TRAINING_URL": url}),
                    patch("webbrowser.open") as browser,
                ):
                    with self.assertRaisesRegex(ValueError, "43187"):
                        account.login()
                    browser.assert_not_called()
            state["deny_identity"] = True
            with self.assertRaises(Exception):
                self.login(url)
        self.assertEqual(self.key.read_text(), "prior-key-file")
        self.assertEqual(self.session.read_text(), "prior-session-file")

    def test_two_processes_rotate_once_and_service_override_never_gets_a_token(self):
        from zils_sdk import account

        with auth_server() as (url, calls, _), auth_server() as (other, other_calls, _):
            self.login(url)
            saved = json.loads(self.session.read_text())
            saved["expires_at"] = time.time() - 10
            self.session.write_text(json.dumps(saved))
            code = (
                "from zils_sdk.account import credentials; s=credentials(); print(s['account_id'])"
            )
            env = {**self.env, "ZILS_TRAINING_URL": url}
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", code],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            for process in processes:
                out, err = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, err)
                self.assertEqual(out.strip(), OWNER)
                self.assertNotIn("rotated", out + err)
            refreshes = [
                form
                for path, form in calls
                if path.endswith("/oauth/token") and form["grant_type"] == ["refresh_token"]
            ]
            self.assertEqual(len(refreshes), 1)
            with patch.dict(os.environ, {"ZILS_TRAINING_URL": other}):
                with self.assertRaisesRegex(ValueError, "different"):
                    account.credentials()
            self.assertEqual(other_calls, [])

    def test_lost_refresh_reply_is_not_replayed_even_by_next_command(self):
        from zils_sdk import account

        with auth_server() as (url, calls, state):
            self.login(url)
            saved = json.loads(self.session.read_text())
            saved["expires_at"] = 0
            self.session.write_text(json.dumps(saved))
            state["drop_refresh"] = True
            with patch.dict(os.environ, {"ZILS_TRAINING_URL": url}):
                for _ in range(2):
                    with self.assertRaisesRegex(Exception, "zils login") as caught:
                        account.credentials()
                    self.assertNotIn("rotated", str(caught.exception))
            refreshes = [
                form
                for path, form in calls
                if path.endswith("/oauth/token") and form["grant_type"] == ["refresh_token"]
            ]
            self.assertEqual(len(refreshes), 1)

    def test_logout_removes_both_credentials_but_not_receipts(self):
        self.key.write_text("key")
        self.session.write_text("session")
        receipt = self.key.parent / "training" / "job.json"
        receipt.parent.mkdir()
        receipt.write_text("receipt")
        with redirect_stderr(io.StringIO()):
            cli.run(cli.parser().parse_args(["logout"]))
        self.assertFalse(self.key.exists())
        self.assertFalse(self.session.exists())
        self.assertTrue(receipt.exists())

    def test_transport_redacts_errors_and_refuses_redirects(self):
        from zils_sdk._training_http import TrainingError, read_json

        with auth_server() as (url, calls, _):
            for path in ("/bad?private-secret", "/redirect"):
                with self.assertRaises(TrainingError) as caught:
                    read_json("GET", url + path, token="private-token")
                self.assertNotIn("private", str(caught.exception))
            self.assertFalse(any(path == "/private-token" for path, _ in calls))


if __name__ == "__main__":
    unittest.main()
