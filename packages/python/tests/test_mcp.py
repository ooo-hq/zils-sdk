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


class McpTests(unittest.TestCase):
    def env(self, directory):
        return {
            **{k: v for k, v in os.environ.items() if not k.startswith(("ZILS_", "TYPESAFE_"))},
            "ZILS_CONFIG_PATH": str(Path(directory) / "config.json"),
        }

    def run_session(self, env, work):
        self.assertIsNotNone(importlib.util.find_spec("zils_sdk.mcp"), "MCP entry point is missing")

        async def run():
            params = StdioServerParameters(
                command=sys.executable, args=["-m", "zils_sdk.mcp"], env=env
            )
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
                self.assertEqual({t.name for t in tools}, {"decide", "list_models"})
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

    def test_discovery_without_credentials_then_login_without_restart(self):
        with tempfile.TemporaryDirectory() as directory, api() as service:
            env = self.env(directory)

            async def check(session):
                self.assertEqual(len((await session.list_tools()).tools), 2)
                missing = await session.call_tool("list_models", {})
                self.assertTrue(missing.isError)
                self.assertIn("zils login --api-key", missing.content[0].text)
                Path(env["ZILS_CONFIG_PATH"]).write_text(
                    json.dumps({"api_key": "valid-key", "base_url": service.url})
                )
                result = await session.call_tool("list_models", {})
                self.assertFalse(result.isError)

            self.run_session(env, check)


if __name__ == "__main__":
    unittest.main()
