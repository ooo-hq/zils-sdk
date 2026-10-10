"""Local HTTP contract fixture; integration mode uses the separate platform."""

import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

QUESTIONS = {
    "eligible": {"type": "noul", "instructions": "Is a refund requested?"},
    "team": {"type": "choice", "criteria": {"billing": None, "support": None}},
    "urgency": {"type": "score", "criteria": [{"level": "low"}, ["high"]]},
}


@contextmanager
def api():
    if os.environ.get("ZILS_TEST_PLATFORM") == "1":
        from platform_gateway import api as gateway

        with gateway() as service:
            yield service
        return
    calls = []
    store = SimpleNamespace(limit=False)

    class Handler(BaseHTTPRequestHandler):
        def dispatch(self):
            body = (
                json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.command == "POST"
                else None
            )
            calls.append((self.command, self.path, dict(self.headers), body))
            status = 200
            if self.headers.get("Authorization") != "Bearer valid-key":
                status, value = 401, {"error": {"code": "unauthorized", "message": "Invalid key"}}
            elif store.limit:
                status, value = 429, {"error": {"code": "rate_limit", "message": "Limit reached"}}
            elif self.path == "/v1/models":
                value = {
                    "models": [
                        {"name": name, "description": "Fixture", "release_date": "2026-10-04"}
                        for name in ("zils-test-r1", "zils-shared")
                    ]
                }
            else:
                answers = {}
                for name, question in body["questions"].items():
                    kind = question["type"]
                    if kind == "noul":
                        answer = {"type": kind, "noul": 0.5}
                    elif kind == "choice":
                        keys = list(question["criteria"])
                        answer = {
                            "type": kind,
                            "choice": keys[0],
                            "confidence": 1 / len(keys),
                            "probabilities": {k: 1 / len(keys) for k in keys},
                        }
                    else:
                        criteria = question["criteria"]
                        answer = {
                            "type": kind,
                            "score": 0.5,
                            "confidence": 1 / len(criteria),
                            "probabilities": {
                                str(i): 1 / len(criteria) for i in range(len(criteria))
                            },
                            "legend": {str(i): v for i, v in enumerate(criteria)},
                        }
                    answers[name] = answer
                value = {
                    "model": "zils-test-r1",
                    "answers": answers,
                    "usage": {"input_tokens": 30, "output_tokens": 0, "billable_input_tokens": 73},
                }
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = dispatch
        do_POST = dispatch

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield SimpleNamespace(
                url=f"http://127.0.0.1:{server.server_port}", calls=calls, store=store
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
