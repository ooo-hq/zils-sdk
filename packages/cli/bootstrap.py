"""Run the bundled CLI wheel in a private, versioned Python environment."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def installation_lock(path):
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                lock.write(b"0")
                lock.flush()
            deadline = time.monotonic() + 300
            while True:
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Another Zils setup is still running. Try again later.")
                    time.sleep(0.1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def pip_install(python, target, **options):
    return subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "install",
            "--no-input",
            "--only-binary=:all:",
            "--index-url",
            "https://pypi.org/simple",
            target,
        ],
        **options,
    )


def install(python, wheel):
    """The CLI must install; `zils merge`'s numpy and `zils mcp`'s MCP library are best effort,
    since a new Python can lack a binary build of them, and that must not block other commands."""
    pip_install(python, str(wheel), stdout=sys.stderr, check=True)
    for extra in ("merge", "mcp"):
        if pip_install(python, f"{wheel}[{extra}]", capture_output=True, text=True).returncode:
            version = "%d.%d" % sys.version_info[:2]
            print(
                f"zils {extra} is unavailable on Python {version}: its dependencies have no build for it yet."
                " Every other command works.",
                file=sys.stderr,
            )
    subprocess.run(
        [str(python), "-I", "-c", "import zils_sdk; from zils_sdk.cli import main"],
        stdout=sys.stderr,
        check=True,
    )


def create_runtime(environment):
    """`python -m venv`. Debian and Ubuntu ship python3 without ensurepip, so this is the step
    that fails on a fresh box; say which package fixes it instead of a bare exit status."""
    try:
        subprocess.run(
            [sys.executable, "-I", "-m", "venv", str(environment)],
            stdout=sys.stderr,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{sys.executable} cannot create a virtual environment (python3 -m venv failed). On"
            " Debian or Ubuntu run: sudo apt install python3-venv (or python3.X-venv for your"
            " Python), then retry"
        ) from error


def runtime():
    package = Path(__file__).resolve().parent
    version = json.loads((package / "package.json").read_text())["version"]
    wheel = package / "vendor" / f"zils_sdk-{version}-py3-none-any.whl"
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    cache = os.environ.get("XDG_CACHE_HOME")
    if not cache:
        if os.name == "nt":
            cache = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        else:
            cache = str(Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache"))
    root = (
        Path(cache)
        / "zils"
        / "cli"
        / f"{version}-{digest[:16]}-py{sys.version_info.major}.{sys.version_info.minor}"
    )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    environment = root / "runtime"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    ready = root / "ready"
    with installation_lock(root / "install.lock"):
        if not (ready.is_file() and python.is_file()):
            print("Setting up Zils CLI dependencies (first use only)...", file=sys.stderr)
            ready.unlink(missing_ok=True)
            if environment.exists():
                shutil.rmtree(environment)
            create_runtime(environment)
            install(python, wheel)
            ready.write_text(digest + "\n")
    return python


if __name__ == "__main__":
    try:
        executable = runtime()
        command = [str(executable), "-I", "-m", "zils_sdk.cli", *sys.argv[1:]]
        if os.name == "nt":
            # Windows exec does not preserve the caller's process/exit-code chain.
            raise SystemExit(subprocess.run(command).returncode)
        os.execv(str(executable), command)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(
            f"Zils setup failed: {error}. Check Python's venv/pip installation and network, then retry.",
            file=sys.stderr,
        )
        raise SystemExit(2)
