"""Thin clients retain TypeSafe's transport, typed questions, and error handling."""

from collections.abc import Mapping
from typing import TypedDict, Unpack

from pydantic import BaseModel, ConfigDict, Field
from typesafe_sdk import (
    AsyncTypeSafeClient,
    JSONContent,
    JSONValue,
    Question,
    RetryPolicy,
    SystemOneResponse as BaseResponse,
    TypeSafeClient,
    Usage as BaseUsage,
)

from ._config import client_config
from ._ranking import RankedItem, prepare_rank, rank_answers


class Usage(BaseUsage):
    """Resource tokens plus Zils's separate logical billing count, when reported."""

    billable_input_tokens: int | None = Field(default=None, ge=0)


class SystemOneResponse(BaseResponse):
    usage: Usage


class RankResult(BaseModel):
    """A local ranking fit and the original decision API's model and token usage."""

    model_config = ConfigDict(frozen=True)
    model: str
    ranking: list[RankedItem]
    comparisons: int
    usage: Usage


class RequestOptions(TypedDict, total=False):
    model: str | None
    retry: RetryPolicy | None
    timeout: float | None
    extra_headers: Mapping[str, str] | None


def _rank_result(
    result: SystemOneResponse, count: int, group_size: int | None = None
) -> RankResult:
    if not result.model.strip() or any(
        type(value) is not int or value < 0
        for value in (result.usage.input_tokens, result.usage.output_tokens)
    ):
        raise ValueError("Ranking response contains invalid model or usage metadata.")
    return RankResult(
        model=result.model,
        ranking=rank_answers(result.answers, count, group_size),
        comparisons=count if group_size is not None else count * (count - 1),
        usage=result.usage,
    )


class ZilsClient(TypeSafeClient):
    """Synchronous Zils client. SDK credentials come from options or ZILS_* env."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60,
        retry: RetryPolicy | None = None,
    ) -> None:
        super().__init__(
            **client_config(api_key, base_url, model),
            timeout=timeout,
            retry=retry if retry is not None else RetryPolicy(max_retries=0),
        )

    def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        **options: Unpack[RequestOptions],
    ) -> SystemOneResponse:
        return super().system_one(state, questions, response_model=SystemOneResponse, **options)

    def rank(
        self,
        items: list[JSONValue],
        instructions: str,
        *,
        group_size: int | None = None,
        **options: Unpack[RequestOptions],
    ) -> RankResult:
        """Rank 2–10 JSON items in one paid request, pairwise by default.

        Set group_size (2–10, capped to the list length) for overlapping Choice groups.
        Scores are relative strengths, not probabilities or confidence.
        All comparisons must succeed; the API's normal context and timeout limits apply.
        """
        request = prepare_rank(items, instructions, group_size)
        count = len(items)
        result = self.system_one(**request, **options)
        return _rank_result(result, count, group_size)


class AsyncZilsClient(AsyncTypeSafeClient):
    """Asynchronous Zils client with the same defaults and response types."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60,
        retry: RetryPolicy | None = None,
    ) -> None:
        super().__init__(
            **client_config(api_key, base_url, model),
            timeout=timeout,
            retry=retry if retry is not None else RetryPolicy(max_retries=0),
        )

    async def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        **options: Unpack[RequestOptions],
    ) -> SystemOneResponse:
        return await super().system_one(
            state, questions, response_model=SystemOneResponse, **options
        )

    async def rank(
        self,
        items: list[JSONValue],
        instructions: str,
        *,
        group_size: int | None = None,
        **options: Unpack[RequestOptions],
    ) -> RankResult:
        """Async equivalent of ZilsClient.rank; preserves timeout and retry options."""
        request = prepare_rank(items, instructions, group_size)
        count = len(items)
        result = await self.system_one(**request, **options)
        return _rank_result(result, count, group_size)
