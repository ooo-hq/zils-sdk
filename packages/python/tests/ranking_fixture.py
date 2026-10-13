"""Local decision endpoint with independently specified pair preferences."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

CASES = json.loads(
    (Path(__file__).resolve().parents[3] / "tests/fixtures/ranking.json").read_text()
)
GROUPED_CASES = json.loads(
    (Path(__file__).resolve().parents[3] / "tests/fixtures/grouped-ranking.json").read_text()
)


@contextmanager
def ranking_api(case, transform=None, status=200):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), body))
            answers = {}
            for name, question in body["questions"].items():
                if "strengths" in case or "position_probabilities" in case:
                    weights = case.get("position_probabilities") or [
                        case["strengths"][case["items"].index(option["item"])]
                        for option in question["criteria"].values()
                    ]
                    probabilities = {
                        key: weight / sum(weights)
                        for key, weight in zip(question["criteria"], weights, strict=True)
                    }
                    choice = max(probabilities, key=probabilities.get)
                    answers[name] = {
                        "type": "choice",
                        "choice": choice,
                        "probabilities": probabilities,
                        "confidence": probabilities[choice],
                    }
                    continue
                a = case["items"].index(question["criteria"]["a"]["item"])
                b = case["items"].index(question["criteria"]["b"]["item"])
                p = case["probabilities"][a][b]
                answers[name] = {
                    "type": "choice",
                    "choice": "a" if p >= 0.5 else "b",
                    "probabilities": {"a": p, "b": 1 - p},
                    "confidence": abs(2 * p - 1),
                }
            value = {
                "model": "zils-test-r1",
                "answers": answers,
                "usage": {"input_tokens": 1234, "output_tokens": 0, "billable_input_tokens": 456},
            }
            if transform:
                transform(value)
            if status != 200:
                value = {"error": {"code": "rate_limit", "message": "Limit reached"}}
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", calls=calls)
        finally:
            server.shutdown()
            thread.join(timeout=5)
