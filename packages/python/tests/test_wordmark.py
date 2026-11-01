import os
import subprocess
import sys
import unittest

from zils_sdk.fit_guide import LOGO

FIRST_ROW = LOGO.splitlines()[1].strip()  # a row of the block wordmark


def zils(*args):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ZILS_", "TYPESAFE_"))}
    return subprocess.run(
        [sys.executable, "-m", "zils_sdk.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


class WordmarkTests(unittest.TestCase):
    def test_bare_zils_shows_the_wordmark_and_the_help(self):
        result = zils()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(FIRST_ROW, result.stdout)
        self.assertIn("usage: zils", result.stdout)
        self.assertLess(result.stdout.index(FIRST_ROW), result.stdout.index("usage: zils"))

    def test_zils_help_shows_the_wordmark(self):
        for flag in ("--help", "-h"):
            with self.subTest(flag):
                result = zils(flag)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(FIRST_ROW, result.stdout)
                self.assertIn("usage: zils", result.stdout)

    def test_other_commands_stay_quiet(self):
        for args in (("--version",), ("models", "--help"), ("merge", "--help")):
            with self.subTest(args):
                result = zils(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn(FIRST_ROW, result.stdout + result.stderr)

    def test_no_colour_codes_when_output_is_not_a_terminal(self):
        self.assertNotIn("\033[", zils().stdout)


if __name__ == "__main__":
    unittest.main()
