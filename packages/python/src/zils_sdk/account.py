"""Browser account sessions for training, separate from inference API keys."""

import base64
import hashlib
import math
import os
import re
import secrets
import sys
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qsl, urlencode, urlsplit

from ._config import validate_url
from ._files import atomic_json, file_lock
from ._input import parse_json
from ._training_http import TrainingError, read_json
from .credentials import config_path

DEFAULT_TRAINING_URL = "https://training.zils.ai"
CALLBACK = "http://127.0.0.1:43187/callback"
LOGIN_TIMEOUT = 600


def identifier(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("Invalid account or job identifier.")
    return value


def origin(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"https?://[A-Za-z0-9.-]+(?::[0-9]+)?", value
    ):
        raise ValueError("Invalid service origin.")
    return validate_url(value)


def training_url():
    return origin(os.environ.get("ZILS_TRAINING_URL", "").strip() or DEFAULT_TRAINING_URL)


def validate_config(base_url, data):
    try:
        base_url = origin(base_url)
        oauth, storage = data["oauth"], data["storage"]
        supabase = origin(storage["supabase_origin"])
        spaces = origin(storage["spaces_origin"]) if storage.get("spaces_origin") else None
        if (
            data["version"] != 1
            or isinstance(data["version"], bool)
            or oauth["issuer"] != supabase + "/auth/v1"
            or oauth["redirect_uri"] != CALLBACK
            or spaces == supabase
        ):
            raise ValueError("Invalid config")
        identifier(oauth["client_id"])
        return {
            "base_url": base_url,
            "model": data.get("model"),
            "version": 1,
            "oauth": {
                "issuer": oauth["issuer"],
                "client_id": oauth["client_id"],
                "redirect_uri": CALLBACK,
            },
            "storage": {"supabase_origin": supabase, "spaces_origin": spaces},
        }
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError(
            "CLI sign-in configuration is unavailable or invalid; contact the service operator."
        ) from None


def load_config(base_url: str) -> dict:
    base_url = origin(base_url)
    response = read_json("GET", base_url + "/v1/config")
    data = response.get("cli")
    if not isinstance(data, dict):
        raise ValueError("This service has not enabled CLI browser sign-in.")
    return validate_config(base_url, {**data, "model": response.get("model")})


def session_path():
    return config_path().with_suffix(".account.json")


def session_lock():
    return file_lock(session_path().with_suffix(".lock"))


class LoginDenied(ValueError):
    pass


def callback_code(target: str, expected_state: str) -> str:
    try:
        if len(target.encode("utf-8")) > 8192 or re.search(
            r"[\x00-\x20\x7f]|%(?![0-9a-fA-F]{2})", target
        ):
            raise ValueError("Invalid callback")
        parsed = urlsplit(target)
        if parsed.path != "/callback" or parsed.scheme or parsed.netloc or "#" in target:
            raise ValueError("Invalid callback")
        pairs = parse_qsl(
            parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=8
        )
        query = dict(pairs)
        if len(query) != len(pairs) or set(query) - {"code", "state", "error", "error_description"}:
            raise ValueError("Invalid callback")
        if not secrets.compare_digest(query.get("state", "").encode(), expected_state.encode()):
            raise ValueError("Invalid callback")
        if "error" in query:
            raise LoginDenied("Browser login was denied or expired. Run zils login again.")
        code = query["code"]
        if not 1 <= len(code) <= 4096 or any(not 33 <= ord(c) <= 126 for c in code):
            raise ValueError("Invalid callback")
        return code
    except LoginDenied:
        raise
    except (ValueError, KeyError, UnicodeError):
        raise ValueError("Invalid login callback.") from None


def authorization_code(config, verifier, state):
    outcome = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(1)

        def reply(self, status):
            body = b"Return to your terminal to finish Zils sign-in."
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get_all("Host") != ["127.0.0.1:43187"]:
                self.reply(400)
                return
            try:
                code = callback_code(self.path, state)
            except LoginDenied as error:
                outcome.append(error)
                self.reply(400)
            except ValueError:
                self.reply(400)
            else:
                outcome.append(code)
                self.reply(200)

        def send_error(self, code, message=None, explain=None):
            self.reply(405 if code == 501 else code)

    try:
        listener = HTTPServer(("127.0.0.1", 43187), Handler)
    except OSError:
        raise ValueError(
            "Cannot open login callback port 43187. Close the other listener and run zils login."
        ) from None
    with listener:
        listener.timeout = min(0.2, LOGIN_TIMEOUT)
        deadline = time.monotonic() + LOGIN_TIMEOUT
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        oauth = config["oauth"]
        url = (
            oauth["issuer"]
            + "/oauth/authorize?"
            + urlencode(
                {
                    "response_type": "code",
                    "client_id": oauth["client_id"],
                    "redirect_uri": CALLBACK,
                    "state": state,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "scope": "email",
                }
            )
        )
        print("Opening your browser for Zils sign-in. Keep this terminal open.", file=sys.stderr)
        if not webbrowser.open(url):
            print("Open this sign-in URL in a browser on this computer:\n" + url, file=sys.stderr)
        while not outcome and time.monotonic() < deadline:
            listener.handle_request()
        if not outcome:
            raise ValueError("Browser login timed out. Run zils login again.")
        if isinstance(outcome[0], Exception):
            raise outcome[0]
        return outcome[0]


def tokens(value):
    try:
        expires = value["expires_in"]
        if (
            value["token_type"].lower() != "bearer"
            or isinstance(expires, bool)
            or not isinstance(expires, (int, float))
            or not math.isfinite(expires)
            or not 0 < expires <= 365 * 86400
        ):
            raise ValueError("Invalid expiry")
        for key in ("access_token", "refresh_token"):
            token = value[key]
            if (
                not isinstance(token, str)
                or not 1 <= len(token) <= 8192
                or any(not 33 <= ord(c) <= 126 for c in token)
            ):
                raise ValueError("Invalid token")
        return {
            "access_token": value["access_token"],
            "refresh_token": value["refresh_token"],
            "expires_at": time.time() + expires,
        }
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("Invalid sign-in response. Run zils login again.") from None


def identity(base_url, token):
    result = read_json("GET", base_url + "/v1/account", token=token)
    try:
        return identifier(result.get("account_id"))
    except (ValueError, TypeError):
        raise ValueError(
            "The training service could not verify this account. Run zils login."
        ) from None


def login() -> dict:
    config = load_config(training_url())
    verifier, state = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    code = authorization_code(config, verifier, state)
    oauth = config["oauth"]
    try:
        session = tokens(
            read_json(
                "POST",
                oauth["issuer"] + "/oauth/token",
                form={
                    "grant_type": "authorization_code",
                    "client_id": oauth["client_id"],
                    "redirect_uri": CALLBACK,
                    "code": code,
                    "code_verifier": verifier,
                },
            )
        )
        account_id = identity(config["base_url"], session["access_token"])
    except TrainingError:
        raise ValueError(
            "Browser sign-in could not be verified. Run zils login again; no exchange was retried."
        ) from None
    session.update(account_id=account_id, base_url=config["base_url"], config=config)
    with session_lock():
        atomic_json(session_path(), session)
    return {"account_id": account_id, "base_url": config["base_url"]}


def credentials() -> dict:
    with session_lock():
        try:
            with session_path().open("rb") as stream:
                raw = stream.read(64 * 1024 + 1)
            if len(raw) > 64 * 1024:
                raise ValueError("Session too large")
            saved = parse_json(raw.decode("utf-8"))
            base_url = origin(saved["base_url"])
            config = validate_config(base_url, saved["config"])
            account_id = identifier(saved["account_id"])
            expires = saved["expires_at"]
            if not isinstance(expires, (int, float)) or not math.isfinite(expires):
                raise ValueError("Invalid expiry")
            for key in ("access_token", "refresh_token"):
                token = saved[key]
                if (
                    not isinstance(token, str)
                    or not 1 <= len(token) <= 8192
                    or any(not 33 <= ord(c) <= 126 for c in token)
                ):
                    raise ValueError("Invalid token")
        except (OSError, ValueError, TypeError, KeyError):
            raise ValueError("No valid account session. Run zils login.") from None
        if os.environ.get("ZILS_TRAINING_URL", "").strip() and training_url() != base_url:
            raise ValueError(
                "Saved account belongs to a different training service. Run zils login for the selected service."
            )
        if saved.get("refresh_pending"):
            raise ValueError(
                "The previous session refresh could not be verified. Run zils login; it will not be replayed."
            )
        if expires <= time.time() + 60:
            # A lost refresh reply can consume a rotating token; never replay it on the next command.
            atomic_json(session_path(), {**saved, "refresh_pending": True})
            try:
                rotated = tokens(
                    read_json(
                        "POST",
                        config["oauth"]["issuer"] + "/oauth/token",
                        form={
                            "grant_type": "refresh_token",
                            "client_id": config["oauth"]["client_id"],
                            "refresh_token": saved["refresh_token"],
                        },
                    )
                )
                if identity(base_url, rotated["access_token"]) != account_id:
                    raise ValueError("Account changed")
            except (TrainingError, ValueError):
                raise ValueError(
                    "Session refresh could not be verified. Run zils login; no refresh was retried."
                ) from None
            saved.update(rotated)
            atomic_json(session_path(), saved)
        elif identity(base_url, saved["access_token"]) != account_id:
            raise ValueError("Account identity changed. Run zils login.")
        return {
            "account_id": account_id,
            "base_url": base_url,
            "access_token": saved["access_token"],
            "config": config,
        }


def logout() -> None:
    with session_lock():
        session_path().unlink(missing_ok=True)
