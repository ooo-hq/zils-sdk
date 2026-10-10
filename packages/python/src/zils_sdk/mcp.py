"""Two Zils tools over stdio; HTTP and credentials use the shared Python client."""

import sys
from typing import Any

from . import AsyncZilsClient, ZilsError
from ._input import check_request, error_message
from .credentials import load_credentials


def build_server():
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.types import ToolAnnotations

    class ZilsMCP(FastMCP):
        # FastMCP otherwise ignores misspelled arguments on potentially paid calls.
        # Adapted from nace-mcp's MIT-licensed strict argument check; see notices.
        async def list_tools(self):
            tools = await super().list_tools()
            return [
                tool.model_copy(
                    update={
                        "inputSchema": {
                            **tool.inputSchema,
                            "additionalProperties": False,
                        }
                    }
                )
                for tool in tools
            ]

        async def call_tool(self, name, arguments):
            tools = {tool.name: tool for tool in await self.list_tools()}
            if name in tools and set(arguments) - set(
                tools[name].inputSchema.get("properties", {})
            ):
                raise ToolError(
                    "Unsupported tool arguments. Read the tool schema; nothing was sent."
                )
            return await super().call_tool(name, arguments)

    server = ZilsMCP(
        "Zils",
        instructions="Call list_models to discover available Zils models. Use decide for typed "
        "yes/no, choice, and score questions. Decisions may consume paid credit. No automatic retries.",
        log_level="WARNING",
    )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True))
    async def list_models() -> dict[str, Any]:
        """List models available to the configured Zils API key. No inference charge."""
        try:
            async with AsyncZilsClient(**load_credentials()) as client:
                response = await client.models.list()
                return response.model_dump(mode="json", exclude_none=True)
        except (ValueError, OSError, ZilsError) as error:
            raise ToolError(error_message(error)) from None

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False))
    async def decide(
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Answer named questions about text or structured data using a Zils model.

        questions maps IDs to objects with type, optional instructions, and criteria.
        noul: yes/no, with optional criteria {"true": description, "false": description}.
        choice: 2–255 labels in criteria, e.g. {"billing": "Payments", "other": null}.
        score: 2–10 ordered descriptions in criteria, e.g. ["Low", "High"].
        Instructions and descriptions can be text or structured JSON. Ask related
        questions together; shared state is billed once. Returns model, answers,
        and usage; billable_input_tokens is the billing meter when reported.
        Aliases can resolve to pinned model IDs. Each call can incur a charge;
        never automatically retry an uncertain result. Probabilities need task evaluation.
        """
        try:
            check_request(state, questions)
            async with AsyncZilsClient(**load_credentials()) as client:
                response = await client.system_one(state, questions, model=model)
                return response.model_dump(mode="json", exclude_none=True)
        except (ValueError, OSError, ZilsError) as error:
            raise ToolError(error_message(error)) from None

    return server


def main() -> int:
    try:
        build_server().run(transport="stdio")
    except ModuleNotFoundError as error:
        if error.name != "mcp":
            raise
        print(
            "From the zils-sdk repository root, install the MCP extra: "
            'python -m pip install "./packages/python[mcp]".',
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
