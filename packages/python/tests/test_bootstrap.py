import argparse
import importlib.util
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from zils_sdk import merge_cli

BOOTSTRAP = Path(__file__).resolve().parents[2] / "cli" / "bootstrap.py"
spec = importlib.util.spec_from_file_location("zils_cli_bootstrap", BOOTSTRAP)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)

PYTHON, WHEEL = Path("/runtime/bin/python"), Path("/vendor/zils_sdk-0-py3-none-any.whl")


def fake_pip(merge_code=0, core_code=0, mcp_code=0):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        code = (
            merge_code
            if command[-1].endswith("[merge]")
            else mcp_code
            if command[-1].endswith("[mcp]")
            else core_code
        )
        if code and kwargs.get("check"):
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code, "", "pip error text")

    return calls, run


class BootstrapInstallTests(unittest.TestCase):
    def install(self, **codes):
        calls, run = fake_pip(**codes)
        stderr = io.StringIO()
        with patch.object(bootstrap.subprocess, "run", run), redirect_stderr(stderr):
            bootstrap.install(PYTHON, WHEEL)
        return calls, stderr.getvalue()

    def test_core_installs_first_and_the_merge_and_mcp_extras_after(self):
        calls, stderr = self.install()
        self.assertEqual(
            [c[-1] for c in calls[:3]], [str(WHEEL), f"{WHEEL}[merge]", f"{WHEEL}[mcp]"]
        )
        self.assertTrue(all("--only-binary=:all:" in c for c in calls[:3]))
        self.assertEqual(stderr, "")

    def test_a_failed_mcp_extra_leaves_every_other_command_working(self):
        calls, stderr = self.install(mcp_code=1)
        self.assertEqual(calls[-1][-1], "import zils_sdk; from zils_sdk.cli import main")
        version = "%d.%d" % sys.version_info[:2]
        self.assertIn(f"zils mcp is unavailable on Python {version}", stderr)
        self.assertNotIn("zils merge is unavailable", stderr)
        self.assertNotIn("pip error text", stderr)

    def test_a_failed_merge_extra_leaves_every_other_command_working(self):
        calls, stderr = self.install(merge_code=1)
        self.assertEqual(calls[-1][-1], "import zils_sdk; from zils_sdk.cli import main")
        version = "%d.%d" % sys.version_info[:2]
        self.assertIn(f"zils merge is unavailable on Python {version}", stderr)
        self.assertIn("Every other command works", stderr)
        self.assertNotIn("pip error text", stderr)

    def test_a_failed_core_install_still_fails_setup(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.install(core_code=1)


class CreateRuntimeTests(unittest.TestCase):
    def test_a_missing_venv_module_names_the_package_to_install(self):
        # Debian/Ubuntu ship python3 without ensurepip; `python -m venv` then exits 1 and the
        # bare CalledProcessError told users nothing. Name the fix.
        def run(command, **kwargs):
            raise subprocess.CalledProcessError(1, command)

        with patch.object(bootstrap.subprocess, "run", run):
            with self.assertRaises(RuntimeError) as caught:
                bootstrap.create_runtime(Path("/tmp/zils-runtime"))
        self.assertIn("python3 -m venv", str(caught.exception))
        self.assertIn("apt install python3-venv", str(caught.exception))

    def test_a_working_venv_module_is_used_as_is(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(bootstrap.subprocess, "run", run):
            bootstrap.create_runtime(Path("/tmp/zils-runtime"))
        self.assertEqual(calls[0][1:4], ["-I", "-m", "venv"])


class MergeUnavailableTests(unittest.TestCase):
    def test_zils_merge_explains_how_to_get_its_dependencies(self):
        args = argparse.Namespace(
            first="a", second="b", out=Path("out"), method="linear", weight=0.5
        )
        # As when numpy is missing: the submodule can't be imported and was never set on the package.
        package = sys.modules["zils_sdk"]
        loaded = package.__dict__.pop("merge", None)
        self.addCleanup(lambda: loaded and setattr(package, "merge", loaded))
        with patch.dict(sys.modules, {"zils_sdk.merge": None}):
            with self.assertRaises(ValueError) as caught:
                merge_cli.run(args)
        message = str(caught.exception)
        self.assertIn("pip install 'zils-sdk[merge]'", message)
        self.assertIn("ZILS_PYTHON", message)
        self.assertIn("needs numpy,", message)
        self.assertNotIn("huggingface_hub", message)


if __name__ == "__main__":
    unittest.main()
