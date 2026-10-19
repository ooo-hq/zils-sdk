"""Local coordinator/storage boundary; no training, credits or hosted accounts."""

import copy
import json
import socket
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

OWNER = "11111111-1111-4111-8111-111111111111"
# The profile served by GET /v1/config since d3-mini was elected.
MODEL = {
    "id": "d3-mini-4.5b-v1",
    "name": "d3-mini 4.5B",
    "base": "zils-ai/d3-mini-4.5b-v1",
    "base_revision": "b54402a963373cc44713f18c9596c580e31cc68b",
    "runtime_revision": "d3-vendor-runtime-a475367/v1",
    "prompt": "d3-native-v1",
    "max_input_tokens": 4096,
    "recipe": "d3-language-lora16-adamw/v1",
    "calibration": "native-then-heldout-temperature/v1",
    "inputs": {"images": 4, "video": True},
}
JEVK5_MODEL = {
    "id": "jevk5-4b-v0.3",
    "name": "JevK5 4B",
    "base": "alibiserikbay/JevK5",
    "base_revision": "c4f7fdb3aeab5582336406e78d3bef11bf98833d",
}
H2O_MODEL = {
    "id": "h2o-lightning-4b-v1.2.3",
    "name": "H2O Lightning 4B",
    "base": "h2oai/h2o-lightning-4b",
    "base_revision": "acaf0d4ea251e54de928c75ef4352670d33192d3",
    "runtime_revision": "h2o-torch210-cu130-fla052/v1",
    "prompt": "h2o-native-v1.2.3",
    "max_input_tokens": 2048,
    "recipe": "h2o-language-lora16-adamw/v1",
    "calibration": "native-then-heldout-temperature/v1",
}
ROUND = "33333333-3333-4333-8333-333333333333"


@contextmanager
def training_server():
    class Fixture:
        jobs = {}
        objects = {}
        calls = []
        drop_create = False
        fail_split = None
        tamper_grant = None
        artifacts = {}
        download_mode = None

        def grant(self, job_id, name, upload=True):
            bucket = "fez-training-data" if upload else "fez-training-models"
            prefix = "inputs" if upload else "releases/" + ROUND
            path = f"{bucket}/{job_id}/{prefix}/{name}"
            result = {
                "url": self.url
                + f"/storage/v1/object/{'upload/sign' if upload else 'sign'}/{path}?token=private-grant",
                "provider": "supabase",
                "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            }
            if upload:
                result.update(
                    method="PUT",
                    headers={"Content-Type": "application/octet-stream", "x-upsert": "false"},
                )
            return self.tamper_grant(result) if self.tamper_grant else result

        def uploads(self, job):
            return {
                "job": copy.deepcopy(job),
                "uploads": {
                    split: {"uploaded": True}
                    if (job["id"], split) in self.objects
                    else self.grant(job["id"], split + ".jsonl")
                    for split in ("train", "calibration", "test")
                },
            }

    fixture = Fixture()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def dispatch(self):
            path = urlsplit(self.path).path
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            fixture.calls.append((self.command, path, dict(self.headers), raw))
            if (
                path.startswith("/v1/")
                and self.headers.get("Authorization") != "Bearer account-token"
            ):
                return self.reply(401, {})
            if self.command == "GET" and path == "/v1/account":
                return self.reply(200, {"account_id": OWNER})
            if self.command == "GET" and path == "/v1/jobs":
                return self.reply(200, {"jobs": list(fixture.jobs.values()), "models": []})
            if self.command == "POST" and path == "/v1/jobs":
                body = json.loads(raw)
                job_id = str(uuid.uuid4())
                job = fixture.jobs[job_id] = {
                    "id": job_id,
                    "name": body["name"],
                    "status": "uploading",
                    "model": MODEL,
                    "workflow": None,
                    "result": None,
                }
                if fixture.drop_create:
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                return self.reply(200, fixture.uploads(job))
            if path.startswith("/storage/"):
                parts = path.split("/")
                if self.command == "PUT":
                    job_id, split = parts[-3], parts[-1].removesuffix(".jsonl")
                    if split == fixture.fail_split:
                        return self.reply(503, {"error": "private-grant"})
                    fixture.objects[job_id, split] = raw
                    return self.reply(200, {})
                name = parts[-1]
                body = fixture.artifacts[name]
                if fixture.download_mode == "redirect":
                    self.send_response(302)
                    self.send_header("Location", fixture.url + "/unexpected")
                    self.end_headers()
                    return
                self.send_response(200)
                if fixture.download_mode != "no-length":
                    self.send_header(
                        "Content-Length",
                        str(len(body) + (1 if fixture.download_mode == "truncated" else 0)),
                    )
                self.end_headers()
                self.wfile.write(body)
                return
            parts = path.split("/")
            if len(parts) < 4 or parts[3] not in fixture.jobs:
                return self.reply(404, {})
            job = fixture.jobs[parts[3]]
            action = parts[4] if len(parts) > 4 else None
            if action == "uploads":
                return self.reply(200, fixture.uploads(job))
            if action == "submit":
                job["status"] = "validating"
            if action == "cancel":
                job["status"] = "failed"
                job["error"] = "Cancelled by customer."
            if action == "downloads":
                return self.reply(
                    200,
                    {
                        "downloads": {
                            name: fixture.grant(job["id"], name, False)
                            for name in fixture.artifacts
                        }
                    },
                )
            return self.reply(200, {"job": job})

        do_GET = do_POST = do_PUT = dispatch

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        fixture.url = f"http://127.0.0.1:{server.server_port}"
        fixture.session = {
            "account_id": OWNER,
            "base_url": fixture.url,
            "access_token": "account-token",
            "config": {
                "model": MODEL,
                "storage": {"supabase_origin": fixture.url, "spaces_origin": None},
            },
        }
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield fixture
        finally:
            server.shutdown()
            thread.join(5)
