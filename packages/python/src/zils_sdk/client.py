"""Thin clients retain TypeSafe's transport, typed questions, and error handling."""

from collections.abc import Mapping
from typing import TypedDict, Unpack

from pydantic import Field
from typesafe_sdk import (
    AsyncTypeSafeClient,
    JSONContent,
    Question,
    RetryPolicy,
    SystemOneResponse as BaseResponse,
    TypeSafeClient,
    Usage as BaseUsage,
)

from ._config import client_config


class Usage(BaseUsage):
    """Resource tokens plus Zils's separate logical billing count, when reported."""

    billable_input_tokens: int | None = Field(default=None, ge=0)


class SystemOneResponse(BaseResponse):
    usage: Usage


class RequestOptions(TypedDict, total=False):
    model: str | None
    retry: RetryPolicy | None
    timeout: float | None
    extra_headers: Mapping[str, str] | None


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
