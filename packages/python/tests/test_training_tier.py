"""Open or private is chosen per job: open is free and trained by network miners, private is paid and stays on
Zils workers. Without a tier a job is private, as before."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from test_training_data import datasets
from training_fixture import training_server
from zils_sdk import training

POLICY = {"min_accuracy": 0.8, "min_brier_improvement": 0.01}


def created(fixture):
    return [json.loads(call[3]) for call in fixture.calls if call[:2] == ("POST", "/v1/jobs")]


class TierTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.files = datasets(self.root)

    def submit(self, fixture, **options):
        client = training.TrainingClient(session=fixture.session, root=self.root / "receipts")
        return client.submit("routing", self.files, POLICY, True, **options)

    def test_without_a_tier_the_job_is_private_as_before(self):
        with training_server() as fixture, redirect_stderr(io.StringIO()):
            result = self.submit(fixture)
            self.assertNotIn("tier", created(fixture)[0])
            self.assertEqual(result["tier"], "private")

    def test_private_can_be_asked_for(self):
        with training_server() as fixture, redirect_stderr(io.StringIO()):
            self.assertEqual(self.submit(fixture, tier="private")["tier"], "private")
            self.assertEqual(created(fixture)[0]["tier"], "private")
            self.assertNotIn("allow_open_tier_training", created(fixture)[0])

    def test_open_needs_miner_consent_before_anything_is_sent(self):
        with training_server() as fixture:
            for consent in (False, None, "yes"):
                with (
                    self.subTest(consent=consent),
                    self.assertRaisesRegex(ValueError, "network miners"),
                ):
                    self.submit(fixture, tier="open", allow_open_tier_training=consent)
            self.assertEqual(fixture.calls, [])

    def test_open_is_sent_with_its_consent_and_is_free(self):
        with training_server() as fixture, redirect_stderr(io.StringIO()) as notice:
            # Free: a cost limit far below the paid ceiling does not block an open job.
            result = self.submit(
                fixture, tier="open", allow_open_tier_training=True, max_cost_usd=0.01
            )
            body = created(fixture)[0]
            self.assertEqual((body["tier"], body["allow_open_tier_training"]), ("open", True))
            self.assertEqual(result["tier"], "open")
            self.assertIn("network miners", notice.getvalue())
            self.assertIn("free", notice.getvalue())

    def test_open_jobs_need_no_cost_limit(self):
        # Agents reasonably send 0 or nothing for a free job.
        for limit in (None, 0):
            with (
                self.subTest(limit=limit),
                training_server() as fixture,
                redirect_stderr(io.StringIO()),
            ):
                result = self.submit(
                    fixture, tier="open", allow_open_tier_training=True, max_cost_usd=limit
                )
                self.assertEqual(result["tier"], "open")

    def test_private_jobs_still_refuse_a_zero_cost_limit(self):
        with training_server() as fixture, self.assertRaisesRegex(ValueError, "cost limit"):
            self.submit(fixture, tier="private", max_cost_usd=0)
        self.assertEqual(fixture.calls, [])

    def test_unknown_tiers_are_refused_before_anything_is_sent(self):
        with training_server() as fixture:
            for tier in ("public", "", 1):
                with self.subTest(tier=tier), self.assertRaisesRegex(ValueError, "tier"):
                    self.submit(fixture, tier=tier, allow_open_tier_training=True)
            self.assertEqual(fixture.calls, [])

    def test_a_service_without_tiers_never_turns_an_open_job_into_a_paid_one(self):
        with training_server() as fixture, redirect_stderr(io.StringIO()):
            fixture.knows_tier = False
            with self.assertRaisesRegex(Exception, "open tier"):
                self.submit(fixture, tier="open", allow_open_tier_training=True)
            [job] = fixture.jobs.values()
            self.assertEqual((job["status"], job["error"]), ("failed", "Cancelled by customer."))
            self.assertFalse(
                [call for call in fixture.calls if call[0] == "PUT"], "nothing uploaded"
            )
            self.assertFalse(
                [call for call in fixture.calls if call[1].endswith("/submit")], "never submitted"
            )


class TierMcpTests(unittest.TestCase):
    def test_agents_must_choose_a_tier(self):
        import asyncio

        from zils_sdk import mcp

        tools = {t.name: t for t in asyncio.run(mcp.build_server().list_tools())}
        schema = tools["train_submit"].inputSchema
        self.assertIn("tier", schema["required"])
        self.assertIn("allow_open_tier_training", schema["properties"])
        self.assertIn("ask the user", tools["train_submit"].description.lower())

    def test_agents_name_a_cost_limit_for_private_jobs_only(self):
        import asyncio

        from mcp.server.fastmcp.exceptions import ToolError
        from zils_sdk import mcp

        server = mcp.build_server()
        tools = {t.name: t for t in asyncio.run(server.list_tools())}
        self.assertNotIn("max_cost_usd", tools["train_submit"].inputSchema["required"])
        arguments = {
            "name": "routing",
            "train": "t.jsonl",
            "calibration": "c.jsonl",
            "test": "s.jsonl",
            "min_accuracy": 0.8,
            "min_brier_improvement": 0.01,
            "allow_training_data_export": True,
            "tier": "private",
        }
        with self.assertRaisesRegex(ToolError, "cost limit"):
            asyncio.run(server.call_tool("train_submit", arguments))

    def test_agents_agree_the_pass_bars_with_the_user(self):
        import asyncio

        from zils_sdk import mcp

        tools = {t.name: t for t in asyncio.run(mcp.build_server().list_tools())}
        description = tools["train_submit"].description.lower()
        self.assertIn("pass bars", description)
        self.assertIn("agree", description)


if __name__ == "__main__":
    unittest.main()
