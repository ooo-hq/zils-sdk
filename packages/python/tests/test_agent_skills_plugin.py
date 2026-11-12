"""Checks the public plugin repository ooo-hq/zils-agent-skills.

The test lives here because Anthropic's plugin directory refuses script files in
that repository. It runs when a checkout sits beside this one (or at
ZILS_AGENT_SKILLS_DIR) and is skipped otherwise.
"""

import json
import os
import re
import unittest
from pathlib import Path

ROOT = Path(
    os.environ.get("ZILS_AGENT_SKILLS_DIR")
    or Path(__file__).resolve().parents[4] / "zils-agent-skills"
)
HAVE_PLUGIN = (ROOT / ".claude-plugin" / "plugin.json").is_file()
CLI = "0.4.21"  # the @zils/cli release this plugin names; raise it with each CLI release
RUN = {"command": "npx", "args": ["-y", f"@zils/cli@{CLI}", "mcp"]}


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def frontmatter(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, f"{path} has no frontmatter"
    return dict(line.split(": ", 1) for line in match.group(1).splitlines())


@unittest.skipUnless(HAVE_PLUGIN, "no zils-agent-skills checkout")
class ManifestTests(unittest.TestCase):
    def test_claude_and_cursor_manifests_name_the_same_plugin(self):
        claude, cursor = load(".claude-plugin/plugin.json"), load(".cursor-plugin/plugin.json")
        for field in ("name", "version", "description", "license", "homepage", "repository"):
            self.assertEqual(claude[field], cursor[field], field)
        self.assertEqual(claude["name"], "zils")
        self.assertEqual(claude["license"], "MIT")
        self.assertTrue((ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License"))

    def test_the_marketplace_lists_this_repository_as_the_plugin(self):
        market, plugin = load(".claude-plugin/marketplace.json"), load(".claude-plugin/plugin.json")
        (entry,) = market["plugins"]
        self.assertEqual(entry, {"name": plugin["name"], "source": "./"})
        # The directory refuses a marketplace.json with fields it does not know.
        self.assertLessEqual(set(market), {"name", "owner", "plugins", "metadata"})

    def test_no_file_starts_the_mcp_server(self):
        # The directory accepts a server that runs on the user's computer only from
        # Verified organizations, so the README tells each agent's user how to add it.
        for path in (".mcp.json", "mcp.json"):
            self.assertFalse((ROOT / path).exists(), path)
        for path in (".claude-plugin/plugin.json", ".cursor-plugin/plugin.json"):
            self.assertNotIn("mcpServers", load(path), path)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(json.dumps({"mcpServers": {"zils": RUN}}, indent=2), readme)

    def test_no_file_asks_for_or_holds_a_key(self):
        # Sign-in is `zils login`; nothing here may carry or request a credential.
        for path in (".claude-plugin/plugin.json", ".cursor-plugin/plugin.json"):
            text = (ROOT / path).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"env|userConfig|variables|zils_sk_|token", text), path)

    def test_the_logo_both_manifests_name_is_in_the_repository(self):
        claude, cursor = load(".claude-plugin/plugin.json"), load(".cursor-plugin/plugin.json")
        # The path is read from the manifests, never written here: the directory holds a
        # plugin for review when a script names a bundled image.
        self.assertEqual(claude["icon"], "./" + cursor["logo"])
        self.assertTrue((ROOT / cursor["logo"]).is_file())
        self.assertEqual((ROOT / cursor["logo"]).read_bytes()[:4], b"\x89PNG")


@unittest.skipUnless(HAVE_PLUGIN, "no zils-agent-skills checkout")
class DirectoryTests(unittest.TestCase):
    """Rules of Anthropic's plugin directory that `claude plugin validate` does not check."""

    def test_the_listing_links_to_the_policy_and_help_pages(self):
        claude = load(".claude-plugin/plugin.json")
        self.assertEqual(claude["privacyPolicyUrl"], "https://zils.ai/privacy")
        self.assertEqual(claude["termsOfServiceUrl"], "https://zils.ai/terms")
        self.assertEqual(claude["documentationUrl"], "https://docs.zils.ai")
        self.assertEqual(claude["supportUrl"], "https://zils.ai/contact")

    def test_the_readme_says_what_the_plugin_runs_and_where_data_goes(self):
        # The security scan fails behavior the README does not disclose.
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for needle in (
            f"claude mcp add zils -- npx -y @zils/cli@{CLI} mcp",
            "npm",
            "PyPI",
            "training.zils.ai",
            "zils.ai/privacy",
        ):
            self.assertIn(needle, text)
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        self.assertGreaterEqual(len(prose.split()), 40)

    def test_only_files_the_directory_accepts_are_tracked(self):
        import subprocess

        files = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
        allowed = (".json", ".md", ".png", ".gitignore", "LICENSE")
        self.assertEqual([f for f in files if not f.endswith(allowed)], [])
        self.assertEqual([f for f in files if "__pycache__" in f or ".DS_Store" in f], [])


@unittest.skipUnless(HAVE_PLUGIN, "no zils-agent-skills checkout")
class SkillTests(unittest.TestCase):
    def test_each_skill_folder_matches_its_name_and_says_when_to_use_it(self):
        folders = sorted(p.name for p in (ROOT / "skills").iterdir() if p.is_dir())
        self.assertEqual(folders, ["zils-fit", "zils-train"])
        for folder in folders:
            meta = frontmatter(f"skills/{folder}/SKILL.md")
            self.assertEqual(meta["name"], folder)
            self.assertGreater(len(meta["description"]), 60, folder)

    def test_the_training_skill_sends_the_agent_to_the_servers_own_guide(self):
        # The steps live in the MCP server (train_guide), so they cannot go stale here.
        text = (ROOT / "skills/zils-train/SKILL.md").read_text(encoding="utf-8")
        self.assertIn("train_guide", text)
        # Agents with no MCP server (skills installed by `npx skills add`) read it from the shell.
        self.assertIn(f"npx -y @zils/cli@{CLI} train guide", text)
        self.assertLess(len(text.splitlines()), 40)


if __name__ == "__main__":
    unittest.main()
