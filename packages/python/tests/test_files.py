import ctypes
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zils_sdk._files import atomic_json


class WindowsStateTests(unittest.TestCase):
    def test_replace_requests_write_through_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "account.json"
            path.write_text("previous")
            with (
                patch("sys.platform", "win32"),
                patch.object(ctypes, "WinDLL", create=True) as library,
                patch.object(ctypes, "WinError", create=True, return_value=OSError("Disk error")),
            ):
                move = library.return_value.MoveFileExW
                move.return_value = 0
                with self.assertRaises(OSError):
                    atomic_json(path, {"refresh_pending": True})
                self.assertEqual(move.call_count, 1)
                self.assertEqual(move.call_args.args[1:], (str(path), 0x1 | 0x8))
            self.assertEqual(path.read_text(), "previous")
            self.assertEqual(list(path.parent.iterdir()), [path])
