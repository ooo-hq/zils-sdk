import subprocess
import unittest
from pathlib import Path

from sdk_helpers import api


class TypeScriptGatewayTests(unittest.TestCase):
    def test_typescript_package_calls_http_contract(self):
        script = Path(__file__).resolve().parents[2] / "typescript" / "test" / "gateway.mjs"
        with api() as service:
            result = subprocess.run(
                ["node", str(script), service.url], capture_output=True, text=True, timeout=15
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("passed", result.stdout)
