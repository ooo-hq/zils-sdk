import json
import unittest
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[2]


class NpmMetadataTests(unittest.TestCase):
    """npm search ranks on keywords, and the package page links issues from `bugs`."""

    def test_both_packages_are_findable_and_link_issues(self):
        for folder in ("cli", "typescript"):
            package = json.loads((PACKAGES / folder / "package.json").read_text(encoding="utf-8"))
            with self.subTest(package=package["name"]):
                self.assertIn("zils", package["keywords"])
                self.assertEqual(package["bugs"], {"url": "https://github.com/ooo-hq/zils-sdk/issues"})


if __name__ == "__main__":
    unittest.main()
