import json
import re
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parents[2] / "cli"


def load(name):
    return json.loads((CLI / name).read_text(encoding="utf-8"))


class McpRegistryListingTests(unittest.TestCase):
    """packages/cli/server.json is what `mcp-publisher publish` sends to the official MCP Registry."""

    def test_the_registry_name_is_the_one_the_npm_package_declares(self):
        # The Registry checks that the published npm package's mcpName equals the server name.
        server, package = load("server.json"), load("package.json")
        self.assertEqual(server["name"], "ai.zils/zils")
        self.assertEqual(package["mcpName"], server["name"])
        self.assertRegex(server["name"], r"^[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+$")

    def test_the_description_fits_the_registry_limit(self):
        description = load("server.json")["description"]
        self.assertTrue(1 <= len(description) <= 100, f"{len(description)} characters")

    def test_the_listing_runs_the_published_package_as_claude_mcp_add_does(self):
        server, package = load("server.json"), load("package.json")
        (entry,) = server["packages"]
        self.assertEqual(entry["registryType"], "npm")
        self.assertEqual(entry["identifier"], package["name"])
        self.assertEqual(entry["runtimeHint"], "npx")
        self.assertEqual(entry["transport"], {"type": "stdio"})
        # npx -y @zils/cli mcp
        self.assertEqual(entry["packageArguments"], [{"type": "positional", "value": "mcp"}])

    def test_the_versions_follow_the_package(self):
        server, package = load("server.json"), load("package.json")
        self.assertEqual(server["version"], package["version"])
        self.assertEqual(server["packages"][0]["version"], package["version"])

    def test_no_secret_is_asked_for_in_the_listing(self):
        # Keys come from `zils login`, never from an MCP client's configuration.
        text = (CLI / "server.json").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"environmentVariables|api[_-]?key|secret", text, re.I))


if __name__ == "__main__":
    unittest.main()
