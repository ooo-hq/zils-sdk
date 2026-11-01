import hashlib
import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from zils_sdk import _hf, merge

SHA = "0123456789abcdef0123456789abcdef01234567"
FILES = {
    "config.json": b'{"model_type": "qwen3_5"}',
    "model.safetensors": b"W" * 5000,
    "model.gguf": b"skip me",
    "sub/notes.txt": b"nested",
    "assets/banner.png": b"image",
    "demo.mp4": b"video",
    ".gitattributes": b"*.safetensors filter=lfs",
    ".eval_results/scores.yaml": b"x: 1",
}
LFS = {"model.safetensors"}


class Hub:
    """A local stand-in for the Hugging Face API and file server; weights redirect to a second origin (a CDN)."""

    def __init__(self, files=FILES, bad_hash=False, sibling_names=None):
        self.requests, self.files = [], files
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                hub.requests.append((self.server.name, self.path, dict(self.headers)))
                if self.server.name == "cdn":
                    return hub.serve(self, self.path.removeprefix("/blob/"))
                if self.path.startswith("/api/models/owner/model/revision/main"):
                    if "blobs=true" not in self.path:
                        return self.send_error(400)
                    names = sibling_names or list(hub.files)
                    siblings = []
                    for name in names:
                        entry = {"rfilename": name, "size": len(hub.files.get(name, b""))}
                        if name in LFS:
                            digest = hashlib.sha256(hub.files[name]).hexdigest()
                            entry["lfs"] = {
                                "sha256": "0" * 64 if bad_hash else digest,
                                "size": entry["size"],
                            }
                        siblings.append(entry)
                    body = json.dumps(
                        {"id": "owner/model", "sha": SHA, "siblings": siblings}
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
                if self.path.startswith("/api/models/"):
                    return self.send_error(401 if "gated" in self.path else 404)
                prefix = f"/owner/model/resolve/{SHA}/"
                if not self.path.startswith(prefix):
                    return self.send_error(404)
                name = self.path.removeprefix(prefix)
                if name in LFS:
                    self.send_response(302)
                    self.send_header("Location", f"{hub.cdn_url}/blob/{name}")
                    self.send_header("Content-Length", "0")
                    return self.end_headers()
                return hub.serve(self, name)

        self.servers = []
        for name in ("hub", "cdn"):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.name = name
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers.append(server)
        self.url, self.cdn_url = (f"http://127.0.0.1:{s.server_address[1]}" for s in self.servers)

    def serve(self, handler, name):
        data = self.files[name]
        start = 0
        if (ranged := handler.headers.get("Range")) and ranged.startswith("bytes="):
            start = int(ranged.removeprefix("bytes=").split("-")[0])
            handler.send_response(206)
            handler.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            handler.send_response(200)
        handler.send_header("Content-Length", str(len(data) - start))
        handler.end_headers()
        handler.wfile.write(data[start:])

    def close(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()


class HubDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "cache"
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HF_ENDPOINT")}
        env["XDG_CACHE_HOME"] = str(self.cache)
        patcher = patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def hub(self, **options):
        hub = Hub(**options)
        self.addCleanup(hub.close)
        os.environ["HF_ENDPOINT"] = hub.url
        return hub

    def test_a_repo_downloads_its_files_at_the_pinned_revision(self):
        hub = self.hub()
        folder = _hf.download("owner/model")
        self.assertEqual(
            folder, self.cache / "zils" / "models" / "owner--model" / "snapshots" / SHA
        )
        self.assertEqual((folder / "model.safetensors").read_bytes(), FILES["model.safetensors"])
        self.assertEqual((folder / "sub" / "notes.txt").read_bytes(), b"nested")
        for skipped in (
            "model.gguf",
            "assets/banner.png",
            "demo.mp4",
            ".gitattributes",
            ".eval_results",
        ):
            self.assertFalse((folder / skipped).exists(), skipped)
        self.assertFalse(any(p.name.endswith(".partial") for p in folder.rglob("*")))
        self.assertTrue(any(name == "cdn" for name, _, _ in hub.requests))

    def test_progress_is_one_line_per_model_and_silent_when_cached(self):
        self.hub()
        first, second = io.StringIO(), io.StringIO()
        with redirect_stderr(first):
            _hf.download("owner/model")
        with redirect_stderr(second):
            _hf.download("owner/model")
        self.assertEqual(first.getvalue(), "Downloading owner/model: 3 files, 5.0 kB\n")
        self.assertEqual(second.getvalue(), "")

    def test_a_second_call_downloads_nothing(self):
        hub = self.hub()
        _hf.download("owner/model")
        count = len(hub.requests)
        _hf.download("owner/model")
        self.assertEqual([path for _, path, _ in hub.requests[count:]], [hub.requests[0][1]])

    def test_a_partial_download_resumes_with_a_range_request(self):
        hub = self.hub()
        folder = self.cache / "zils" / "models" / "owner--model" / "snapshots" / SHA
        folder.mkdir(parents=True)
        (folder / "model.safetensors.partial").write_bytes(FILES["model.safetensors"][:1200])
        _hf.download("owner/model")
        self.assertEqual((folder / "model.safetensors").read_bytes(), FILES["model.safetensors"])
        ranged = [h for name, _, h in hub.requests if name == "cdn"]
        self.assertEqual(ranged[-1].get("Range"), "bytes=1200-")

    def test_a_wrong_checksum_is_refused_and_not_kept(self):
        self.hub(bad_hash=True)
        with self.assertRaisesRegex(ValueError, "checksum"):
            _hf.download("owner/model")
        self.assertFalse(list(self.cache.rglob("model.safetensors")))

    def test_the_token_goes_to_hugging_face_only(self):
        hub = self.hub()
        os.environ["HF_TOKEN"] = "hf_secret"
        _hf.download("owner/model")
        hub_headers = [h for name, _, h in hub.requests if name == "hub"]
        cdn_headers = [h for name, _, h in hub.requests if name == "cdn"]
        self.assertTrue(all(h.get("Authorization") == "Bearer hf_secret" for h in hub_headers))
        self.assertTrue(cdn_headers and all("Authorization" not in h for h in cdn_headers))

    def test_missing_or_gated_repos_explain_themselves(self):
        self.hub()
        with self.assertRaisesRegex(ValueError, "not found"):
            _hf.download("owner/missing")
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            _hf.download("owner/gated")

    def test_file_names_that_leave_the_folder_are_refused(self):
        files = {**FILES, "../escape.json": b"x"}
        self.hub(files=files)
        with self.assertRaisesRegex(ValueError, "unsafe file name"):
            _hf.download("owner/model")
        self.assertFalse(
            (self.cache / "zils" / "models" / "owner--model" / "snapshots" / "escape.json").exists()
        )

    def test_merge_resolves_repo_ids_through_the_downloader(self):
        self.hub()
        folder = merge.resolve("owner/model")
        self.assertEqual(folder.name, SHA)
        self.assertEqual(
            folder.parent.name, "snapshots"
        )  # merge.json records the revision from this


if __name__ == "__main__":
    unittest.main()
