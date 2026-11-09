import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from sdk_helpers import QUESTIONS, api
from test_training_data import datasets
from training_fixture import training_server

TOOLS = {
    "decide",
    "list_models",
    "train_guide",
    "train_prepare",
    "train_check",
    "train_submit",
    "train_list",
    "train_status",
    "train_cancel",
    "fit_scan",
    "fit_run",
}


class McpTests(unittest.TestCase):
    def env(self, directory):
        return {
            **{k: v for k, v in os.environ.items() if not k.startswith(("ZILS_", "TYPESAFE_"))},
            "ZILS_CONFIG_PATH": str(Path(directory) / "config.json"),
        }

    def run_session(self, env, work, args=("-m", "zils_sdk.mcp")):
        self.assertIsNotNone(importlib.util.find_spec("zils_sdk.mcp"), "MCP entry point is missing")

        async def run():
            params = StdioServerParameters(command=sys.executable, args=list(args), env=env)
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                await work(session)

        asyncio.run(asyncio.wait_for(run(), timeout=25))

    def test_stdio_tools_call_real_gateway_and_reject_unknown_arguments(self):
        with tempfile.TemporaryDirectory() as directory, api() as service:
            env = self.env(directory)
            env.update(ZILS_API_KEY="valid-key", ZILS_BASE_URL=service.url)

            async def check(session):
                tools = (await session.list_tools()).tools
                self.assertEqual({t.name for t in tools}, TOOLS)
                decision = next(t for t in tools if t.name == "decide")
                self.assertFalse(decision.inputSchema["additionalProperties"])
                models = await session.call_tool("list_models", {})
                self.assertFalse(models.isError)
                result = await session.call_tool(
                    "decide", {"state": "Refund", "questions": QUESTIONS}
                )
                self.assertFalse(result.isError)
                self.assertEqual(result.structuredContent["usage"]["billable_input_tokens"], 73)
                count = len(service.calls)
                typo = await session.call_tool(
                    "decide", {"state": "Refund", "questions": QUESTIONS, "modle": "x"}
                )
                self.assertTrue(typo.isError)
                self.assertEqual(len(service.calls), count)
                service.store.limit = True
                limited = await session.call_tool(
                    "decide", {"state": "Refund", "questions": QUESTIONS}
                )
                self.assertTrue(limited.isError)
                self.assertIn("429", limited.content[0].text)
                self.assertNotIn("valid-key", limited.content[0].text)
                self.assertEqual(len(service.calls), count + 1)

            self.run_session(env, check)

    def test_zils_mcp_command_serves_the_same_tools(self):
        with tempfile.TemporaryDirectory() as directory:

            async def check(session):
                self.assertEqual({t.name for t in (await session.list_tools()).tools}, TOOLS)

            self.run_session(self.env(directory), check, ("-m", "zils_sdk.cli", "mcp"))

    def test_discovery_without_credentials_then_login_without_restart(self):
        with tempfile.TemporaryDirectory() as directory, api() as service:
            env = self.env(directory)

            async def check(session):
                self.assertEqual(len((await session.list_tools()).tools), len(TOOLS))
                missing = await session.call_tool("list_models", {})
                self.assertTrue(missing.isError)
                self.assertIn("zils login --api-key", missing.content[0].text)
                Path(env["ZILS_CONFIG_PATH"]).write_text(
                    json.dumps({"api_key": "valid-key", "base_url": service.url})
                )
                result = await session.call_tool("list_models", {})
                self.assertFalse(result.isError)

            self.run_session(env, check)

    def signin(self, env, fixture):
        session = {**fixture.session, "refresh_token": "fixture-refresh", "expires_at": 2**40}
        session["config"] = {
            **session["config"],
            "version": 1,
            "oauth": {
                "issuer": fixture.url + "/auth/v1",
                "client_id": "11111111-1111-4111-8111-111111111111",
                "redirect_uri": "http://127.0.0.1:43187/callback",
            },
        }
        Path(env["ZILS_CONFIG_PATH"]).with_suffix(".account.json").write_text(json.dumps(session))
        env["ZILS_TRAINING_URL"] = fixture.url

    def test_training_tools_check_cost_then_submit_list_status_and_cancel(self):
        with tempfile.TemporaryDirectory() as directory, training_server() as fixture:
            env = self.env(directory)
            files = {split: str(path) for split, path in datasets(Path(directory)).items()}
            submit = {
                "name": "routing",
                **files,
                "min_accuracy": 0.8,
                "min_brier_improvement": 0.01,
                "allow_training_data_export": True,
                "tier": "private",  # agents must choose; this test covers the paid path
            }

            async def check(session):
                tools = {t.name: t for t in (await session.list_tools()).tools}
                # Open jobs are free, so the limit is not in the schema; private jobs refuse without it.
                self.assertNotIn("max_cost_usd", tools["train_submit"].inputSchema["required"])
                self.assertIn("tier", tools["train_submit"].inputSchema["required"])
                self.assertTrue(tools["train_cancel"].annotations.destructiveHint)
                self.assertTrue(tools["train_check"].annotations.readOnlyHint)
                checked = await session.call_tool("train_check", files)
                self.assertFalse(checked.isError, checked.content)
                self.assertEqual(checked.structuredContent["ceiling_usd"], 2.0)
                missing = await session.call_tool("train_list", {})
                self.assertTrue(missing.isError)
                self.assertIn("zils login", missing.content[0].text)
                unbudgeted = await session.call_tool("train_submit", submit)
                self.assertTrue(unbudgeted.isError)
                self.assertIn("cost limit", unbudgeted.content[0].text)
                self.assertEqual(fixture.calls, [])
                self.signin(env, fixture)
                over = await session.call_tool("train_submit", {**submit, "max_cost_usd": 1})
                self.assertTrue(over.isError)
                self.assertIn("$2.00", over.content[0].text)
                self.assertEqual(fixture.calls, [])
                submitted = await session.call_tool("train_submit", {**submit, "max_cost_usd": 2})
                self.assertFalse(submitted.isError, submitted.content)
                job_id = submitted.structuredContent["id"]
                listed = await session.call_tool("train_list", {})
                self.assertIn(job_id, json.dumps(listed.structuredContent))
                status = await session.call_tool("train_status", {"job_id": job_id})
                self.assertEqual(status.structuredContent["id"], job_id)
                cancelled = await session.call_tool("train_cancel", {"job_id": job_id})
                self.assertTrue(cancelled.structuredContent["cancelled"])
                bad = await session.call_tool("train_status", {"job_id": "../private-secret"})
                self.assertTrue(bad.isError)
                text = json.dumps([c.model_dump() for c in submitted.content + status.content])
                self.assertNotIn("account-token", text)
                self.assertNotIn("private-grant", text)

            env["ZILS_TRAINING_URL"] = fixture.url
            self.run_session(env, check)


if __name__ == "__main__":
    unittest.main()
