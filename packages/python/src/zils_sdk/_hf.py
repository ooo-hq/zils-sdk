"""Download a Hugging Face model repo with httpx, for `zils merge`.

Files land in `<cache>/zils/models/<owner>--<name>/snapshots/<commit>/`, pinned to the commit the
revision pointed at. A partial file resumes with a Range request, weight files are checked against
the SHA-256 the Hub lists, and HF_TOKEN (for gated or private repos) is sent to the Hub only: httpx
drops it when a weight download redirects to another host."""

import fnmatch
import hashlib
import os
import sys
from pathlib import Path, PurePosixPath

import httpx

# Other weight formats and repo media a merge never reads; dotfiles (.gitattributes, .eval_results/) too.
IGNORE = ("*.gguf", "*.bin", "*.pt", "*.pth", "*.onnx", "*.h5", "*.msgpack", "*.ckpt")
IGNORE += ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg", "*.mp4", "*.webm", "*.mov")
CHUNK = 1 << 20


def cache_root():
    base = os.environ.get("XDG_CACHE_HOME")
    if not base:
        if os.name == "nt":
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        else:
            base = str(Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache"))
    return Path(base) / "zils" / "models"


def wanted(name):
    return (
        not name.startswith(".")
        and "/." not in name
        and not any(fnmatch.fnmatch(name, pattern) for pattern in IGNORE)
    )


def human(size):
    for unit, scale in (("GB", 1e9), ("MB", 1e6)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size / 1e3:.1f} kB"


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"The repo lists an unsafe file name: {name!r}.")
    return path


def fetch(client, url, target, size, sha256):
    """Download `url` to `target`, resuming `<target>.partial`; skip a finished file of the right size."""
    if target.is_file() and (size is None or target.stat().st_size == size):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    start = partial.stat().st_size if partial.is_file() else 0
    digest = hashlib.sha256()
    headers = {"Range": f"bytes={start}-"} if start else {}
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code != 206:
            start = 0
        response.raise_for_status()
        if start:
            with partial.open("rb") as stream:
                for block in iter(lambda: stream.read(CHUNK), b""):
                    digest.update(block)
        with partial.open("ab" if start else "wb") as stream:
            for block in response.iter_bytes(CHUNK):
                digest.update(block)
                stream.write(block)
    if size is not None and partial.stat().st_size != size:
        partial.unlink()
        raise ValueError(f"{target.name} downloaded with the wrong size; run the command again.")
    if sha256 and digest.hexdigest() != sha256:
        partial.unlink()
        raise ValueError(f"{target.name} failed its checksum; run the command again.")
    partial.replace(target)


def download(repo, revision="main"):
    """The local folder holding `repo` at `revision` (weights, config, tokenizer), downloading what's missing."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    token = os.environ.get("HF_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = httpx.Timeout(60.0, connect=30.0)
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        response = client.get(
            f"{endpoint}/api/models/{repo}/revision/{revision}", params={"blobs": "true"}
        )
        if response.status_code in (401, 403):
            raise ValueError(
                f"{repo} is gated or private; accept its terms on Hugging Face and set HF_TOKEN."
            )
        if response.status_code == 404:
            raise ValueError(f"{repo} was not found on Hugging Face.")
        response.raise_for_status()
        info = response.json()
        commit = info["sha"]
        folder = cache_root() / repo.replace("/", "--") / "snapshots" / commit
        listed = [(safe_name(item["rfilename"]), item) for item in info.get("siblings", [])]
        files = [(name, item) for name, item in listed if wanted(item["rfilename"])]
        missing = [item for name, item in files if not folder.joinpath(*name.parts).is_file()]
        if missing:
            size = sum(item.get("size") or 0 for item in missing)
            print(f"Downloading {repo}: {len(missing)} files, {human(size)}", file=sys.stderr)
        for name, item in files:
            fetch(
                client,
                f"{endpoint}/{repo}/resolve/{commit}/{name}",
                folder.joinpath(*name.parts),
                item.get("size"),
                (item.get("lfs") or {}).get("sha256"),
            )
    return folder
