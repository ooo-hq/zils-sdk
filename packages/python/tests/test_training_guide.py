"""An agent that has never seen Zils learns the whole training flow from one tool: what to ask the user, how
long it takes, how to tell it is done, and how to use the model afterwards."""

import asyncio
import unittest


def guide():
    from zils_sdk import mcp

    server = mcp.build_server()
    _, result = asyncio.run(server.call_tool("train_guide", {}))
    return server, result["guide"]


class GuideTests(unittest.TestCase):
    def test_the_server_points_agents_to_the_guide_first(self):
        server, _ = guide()
        self.assertIn("train_guide", server.instructions)

    def test_the_guide_covers_every_step_an_agent_got_wrong(self):
        _, text = guide()
        for needed in (
            "train_prepare",  # turn the file into splits
            "give away",  # leave out columns that leak the answer
            "after the decision",  # and columns only known afterwards
            "group",  # keep one customer's rows together
            "min_accuracy",  # propose the pass bars...
            "agree",  # ...and agree them with the user
            "open",  # ask the user to choose the tier
            "private",
            "max_cost_usd",  # a cost limit for private jobs
            "train_status",  # how to tell when it is done
            "completed",
            "minutes",  # how long it takes
            "decide",  # how to use the model
            "login --api-key",  # the separate inference key
        ):
            with self.subTest(needed=needed):
                self.assertIn(needed, text)

    def test_reading_the_guide_needs_no_login_and_changes_nothing(self):
        from zils_sdk import mcp

        tools = {t.name: t for t in asyncio.run(mcp.build_server().list_tools())}
        self.assertTrue(tools["train_guide"].annotations.readOnlyHint)


if __name__ == "__main__":
    unittest.main()


def zils(*args):
    """The command line with no saved login, so the test proves the guide needs none."""
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as home:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ZILS_", "TYPESAFE_"))}
        env["ZILS_CONFIG_PATH"] = os.path.join(home, "config.json")
        return subprocess.run(
            [sys.executable, "-m", "zils_sdk.cli", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )


class CommandLineGuideTests(unittest.TestCase):
    """Agents without MCP (skills installed by skills.sh, Pi) read the same guide from the shell."""

    def test_zils_train_guide_prints_the_whole_guide_without_a_login(self):
        result = zils("train", "guide")
        self.assertEqual(result.returncode, 0, result.stderr)
        from zils_sdk import training_guide

        self.assertIn(training_guide.GUIDE, result.stdout)

    def test_it_says_which_command_stands_in_for_each_tool(self):
        text = zils("train", "guide").stdout
        head = text[: text.index("# Training a Zils model for the user")]
        for command in (
            "zils train prepare",
            "zils train submit",
            "zils train list",
            "zils train status",
            "zils train cancel",
            "zils fit scan",
            "zils fit run",
            "zils decide",
            "--max-cost",
            "--help",
        ):
            self.assertIn(command, head)

    def test_every_command_it_names_exists(self):
        import re

        from zils_sdk import training_guide

        named = set(
            re.findall(r"`(zils (?:train|fit) [a-z]+|zils decide)", training_guide.COMMANDS)
        )
        self.assertGreaterEqual(len(named), 8)
        for command in sorted(named):
            result = zils(*command.split()[1:], "--help")
            self.assertEqual(result.returncode, 0, command + ": " + result.stderr)

    def test_the_guide_is_listed_in_the_train_help(self):
        self.assertIn("guide", zils("train", "--help").stdout)
