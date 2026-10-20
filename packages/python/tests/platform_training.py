"""Real coordinator HTTP routes with the platform's isolated queue/storage Store."""

import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from tests.test_queue import Store, server
from zils import models
from zils_platform import coordinator


@contextmanager
def training_service(*, pools=None):
    calls = []
    storage_type = Store
    if pools is not None:
        from tests.test_model_pools import PoolStore

        storage_type = PoolStore

    class Storage(storage_type):
        fail_split = None

        def signed(self, bucket, path, *, upload=False):
            original = super().signed(bucket, path, upload=upload)
            ticket = original["url"].rsplit("/", 1)[1]
            grant = {
                "url": self.url
                + "/storage/v1/object/"
                + ("upload/sign/" if upload else "sign/")
                + bucket
                + "/"
                + path
                + "?token="
                + ticket,
                "provider": "supabase",
                "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            }
            if upload:
                grant.update(
                    method="PUT",
                    headers={"Content-Type": "application/octet-stream", "x-upsert": "false"},
                )
            return grant

        def handler(self):
            parent = super().handler()
            store = self

            class Handler(parent):
                def handle_file(self):
                    calls.append((self.command, self.path, dict(self.headers)))
                    parsed = urlsplit(self.path)
                    ticket = parse_qs(parsed.query).get("token", [""])[0]
                    entry = store.tickets.get(ticket)
                    if not entry:
                        return self.send_error(403)
                    bucket, path, upload = entry
                    expected = (
                        "/storage/v1/object/"
                        + ("upload/sign/" if upload else "sign/")
                        + bucket
                        + "/"
                        + path
                    )
                    if parsed.path != expected:
                        return self.send_error(403)
                    if (
                        upload
                        and store.fail_split
                        and path.endswith("/" + store.fail_split + ".jsonl")
                    ):
                        return self.send_error(503)
                    self.path = "/" + ticket
                    return super().handle_file()

                def do_POST(self):
                    import json

                    if self.path != "/auth/v1/oauth/token":
                        return self.send_error(404)
                    body = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
                    if body.get("grant_type") not in (["authorization_code"], ["refresh_token"]):
                        return self.send_error(400)
                    raw = json.dumps(
                        {
                            "access_token": "owner-token",
                            "refresh_token": "fixture-refresh",
                            "token_type": "bearer",
                            "expires_in": 3600,
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(raw)

                do_GET = do_PUT = handle_file

            return Handler

    store = Storage()
    service = coordinator.Service(store, "http://127.0.0.1:8910", models.D3, pools=pools)
    with server(store.handler()) as storage_url:
        store.url = storage_url
        with (
            patch.dict(
                os.environ,
                {
                    "SUPABASE_URL": storage_url,
                    "ZILS_CLI_OAUTH_CLIENT_ID": "22222222-2222-4222-8222-222222222222",
                    "ZILS_CLI_SPACES_ORIGIN": "",
                },
            ),
            server(coordinator.handler(service, "https://zils.example")) as url,
        ):
            yield SimpleNamespace(url=url, store=store, service=service, calls=calls)
