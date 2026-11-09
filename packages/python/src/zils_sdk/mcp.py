"""Zils decision and training tools over stdio; HTTP and credentials use the shared Python client."""

import sys
from pathlib import Path
from typing import Any, Literal

from . import AsyncZilsClient, ZilsError, training_data, training_guide, training_prepare
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
        "Before training a model, read train_guide: it lists what to agree with the user, including decisions "
        "made inside the user's own program (fit_scan finds them, fit_run measures a model on logged ones). "
        "Training tools "
        "need the user's browser account login (zils login). Decisions and training may consume "
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
    async def train_guide() -> dict[str, str]:
        """How to train a Zils model for the user, step by step. Read it before the other train_ tools. Local."""
        return {"guide": training_guide.GUIDE}

    def questions_from(value):
        """fit's questions: the object itself, or the path of the file zils fit scan wrote."""
        if value is None or isinstance(value, dict):
            return value
        from .training_cli import load_questions

        return load_questions(value)

    @server.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=False))
    async def train_prepare(
        input: str,
        decide: str | None = None,
        read: list[str] | None = None,
        group: str | None = None,
        name: str | None = None,
        question: str | None = None,
        out: str | None = None,
        questions: dict[str, dict[str, Any]] | str | None = None,
    ) -> dict[str, Any]:
        """Turn the user's examples into the three training splits. Local; no login.

        A table (CSV, TSV, JSON or JSONL rows): decide is the column holding the right answer. read lists the
        columns the model may use; leave out IDs and anything only known after the decision (for example a
        resolution time). group names the case each row belongs to, so related rows stay in one split. leaks
        lists columns that give the answer away; without read they are left out, with read they are kept.
        Decision logs from record() (a log, or a folder with one log per episode): decide picks the logged
        question when there are several, questions is fit's questions object or file (wording and options);
        each episode stays in one split, and repeats of an earlier situation are left out and counted.
        Writes train/calibration/test JSONL to out (never overwriting) and returns the counts, the answers per
        split, the cost ceiling and the submit command.
        """

        def work():
            if training_prepare.is_log(input):
                if read or group:
                    raise ValueError(
                        "--read and --group are for tables. In a log the state is what the model reads, and "
                        "each episode stays together."
                    )
                return training_prepare.prepare_log(
                    input,
                    questions=questions_from(questions),
                    decide=decide,
                    question=question,
                    out=out,
                    name=name,
                )
            if not decide:
                raise ValueError("Name the column holding the right answer in decide.")
            return training_prepare.prepare(
                input, decide, out=out, read=read, group=group, name=name, question=question
            )

        return await training(work)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True))
    async def fit_scan(path: str = ".", include_tests: bool = False) -> dict[str, Any]:
        """Find the model calls in the user's code that make typed decisions Zils could answer. Local; no login.

        Returns findings (each call: kind, name, function, location, question) and questions, drafted Zils
        questions for the decision calls ({id: {type, instructions, criteria}}). Read the code around each call
        and replace every TODO before using the questions. Programs that decide without a model call (rules, a
        scripted player) are not found: wrap that function with record() instead.
        """
        from .fit import draft_questions, scan

        def work():
            found = scan(Path(path), include_tests)
            return {"findings": found, "questions": draft_questions(found)}

        return await training(work)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False))
    async def fit_run(
        questions: dict[str, dict[str, Any]] | str,
        log: str,
        model: str = "zils-shared",
        max_rows: int = 500,
        target: float = 0.95,
        point_estimate: bool = False,
    ) -> dict[str, Any]:
        """Measure a Zils model on logged decisions: how often it agrees with the logged answers, per question.

        log is a record() log or a folder of them; questions is fit's questions object or file. model defaults to
        the untrained zils-shared; pass a trained model's API id to measure it after training. Sends one paid
        request per distinct logged input and refuses before sending anything if there are more than max_rows:
        agree the number with the user first. Returns, per question, the share handled at the target agreement
        and its status (ok, check_miss, no_threshold, too_few_rows).
        """
        from . import ZilsClient
        from .fit import measure, run_rows

        def work():
            asked = questions_from(questions)
            rows = run_rows(asked, log, target=target, max_rows=max_rows)
            with ZilsClient(**load_credentials()) as client:

                def predict(state, chosen):
                    result = client.system_one(state=state, questions=chosen, model=model)
                    return result.model_dump(mode="json")["answers"]

                report = measure(
                    rows, asked, predict, target=target, conservative=not point_estimate
                )
            report.pop("answers")
            return {**report, "model": model}

        try:
            return await training(work)
        except ZilsError as error:
            raise ToolError(error_message(error)) from None

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
        allow_training_data_export: bool,
        tier: Literal["open", "private"],
        max_cost_usd: float | None = None,
        allow_open_tier_training: bool = False,
        pool: str | None = None,
        base_channel: str | None = None,
    ) -> dict[str, Any]:
        """Submit a z3 adapter training job from three text JSONL splits.

        tier: ask the user before choosing. "private" is paid and trained only on Zils workers.
        "open" is free and trained by Zils network miners, who receive the training split and must
        delete it within 7 days; it also needs allow_open_tier_training true, the user's permission.
        Private jobs need max_cost_usd, agreed with the user; the tool refuses before sending anything when
        train_check's ceiling_usd is above it. Open jobs need no cost limit. min_accuracy (0–1) and
        min_brier_improvement (0–2) are the pass bars the trained adapter must clear on the test split:
        propose them and agree them with the user before submitting. allow_training_data_export must be
        true. Never retry an uncertain submission; check train_list first.
        """
        if tier == "private" and max_cost_usd is None:
            raise ToolError(
                "Private training is paid: agree a cost limit (max_cost_usd) with the user. Nothing was sent."
            )
        return await training(
            lambda: TrainingClient().submit(
                name,
                splits(train, calibration, test),
                {"min_accuracy": min_accuracy, "min_brier_improvement": min_brier_improvement},
                allow_training_data_export,
                pool=pool,
                base_channel=base_channel,
                max_cost_usd=max_cost_usd,
                tier=tier,
                allow_open_tier_training=allow_open_tier_training,
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
            "The MCP library is not installed for this Python. With the npm CLI, point "
            "ZILS_PYTHON at another Python 3.11+ and retry. From the zils-sdk repository root: "
            'python -m pip install "./packages/python[mcp]".',
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
