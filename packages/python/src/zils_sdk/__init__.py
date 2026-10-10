"""Zils decision clients and compatible typed question helpers."""

from typesafe_sdk import (
    Answer,
    Choice,
    ChoiceAnswer,
    JSONContent,
    JSONValue,
    ListModelsResponse,
    ModelMetadata,
    Noul,
    NoulAnswer,
    Question,
    Questions,
    RetryPolicy,
    Score,
    ScoreAnswer,
    TypeSafeAPIError as APIError,
    TypeSafeError as ZilsError,
)

from .client import AsyncZilsClient, SystemOneResponse, Usage, ZilsClient

__version__ = "0.1.0"
__all__ = [
    "APIError",
    "Answer",
    "AsyncZilsClient",
    "Choice",
    "ChoiceAnswer",
    "JSONContent",
    "JSONValue",
    "ListModelsResponse",
    "ModelMetadata",
    "Noul",
    "NoulAnswer",
    "Question",
    "Questions",
    "RetryPolicy",
    "Score",
    "ScoreAnswer",
    "SystemOneResponse",
    "Usage",
    "ZilsClient",
    "ZilsError",
]
