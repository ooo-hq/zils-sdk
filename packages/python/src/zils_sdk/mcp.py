"""Zils decision and training tools over stdio; HTTP and credentials use the shared Python client."""

import sys
from pathlib import Path
from typing import Any

from . import AsyncZilsClient, ZilsError, training_data
from ._input import check_request, error_message
from ._training_http import TrainingError
from .account import identifier
from .credentials import load_credentials
from .training import TrainingClient


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
        "yes/no, choice, and score questions. Configure an inference key with zils login --api-key. "
        "Training tools need the user's browser account login (zils login). Run train_check and "
        "agree a cost limit with the user before train_submit. Decisions and training may consume "
        "paid credit. No automatic retries.",
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

    async def training(work):
        import anyio

        try:
            return await anyio.to_thread.run_sync(work)
        except TrainingError as error:
            raise ToolError(str(error)) from None
        except (ValueError, OSError) as error:
            raise ToolError(error_message(error)) from None

    def splits(train, calibration, test):
        return {"train": Path(train), "calibration": Path(calibration), "test": Path(test)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True))
    async def train_check(train: str, calibration: str, test: str) -> dict[str, Any]:
        """Check three text JSONL training splits locally and price the most the job can cost.

        Each line is a case with id, family, group_id, state, question and label. Nothing is
        sent and no login is needed. ceiling_usd assumes every train prompt is at the 8,192-token
        cap; the service bills real train tokens at $0.20 per million, never below $2.
        """
        return await training(lambda: training_data.check(splits(train, calibration, test)))

    @server.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=False))
    async def train_submit(
        name: str,
        train: str,
        calibration: str,
        test: str,
        min_accuracy: float,
        min_brier_improvement: float,
        max_cost_usd: float,
        allow_training_data_export: bool,
        pool: str | None = None,
        base_channel: str | None = None,
    ) -> dict[str, Any]:
        """Submit a paid z3 adapter training job from three text JSONL splits.

        Refuses before sending anything when train_check's ceiling_usd is above max_cost_usd;
        agree that limit with the user first. min_accuracy (0–1) and min_brier_improvement
        (0–2) are the bars the trained adapter must clear on the test split.
        allow_training_data_export must be true: training examples go to approved workers.
        Never retry an uncertain submission; check train_list first.
        """
        return await training(
            lambda: TrainingClient().submit(
                name,
                splits(train, calibration, test),
                {"min_accuracy": min_accuracy, "min_brier_improvement": min_brier_improvement},
                allow_training_data_export,
                pool=pool,
                base_channel=base_channel,
                max_cost_usd=max_cost_usd,
            )
        )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True))
    async def train_list() -> dict[str, Any]:
        """List the account's recent training jobs and ready models."""
        return await training(lambda: TrainingClient().list())

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True))
    async def train_status(job_id: str) -> dict[str, Any]:
        """Show one job's status, evaluation and API model id once the adapter is live."""
        return await training(lambda: TrainingClient().get(identifier(job_id)))

    @server.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True))
    async def train_cancel(job_id: str) -> dict[str, Any]:
        """Cancel a training job. Costs already incurred may remain; confirm with the user first."""
        return await training(lambda: TrainingClient().cancel(identifier(job_id)))

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
