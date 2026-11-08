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
