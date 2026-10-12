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

from ._ranking import RankedItem
from .client import AsyncZilsClient, RankResult, SystemOneResponse, Usage, ZilsClient

__version__ = "0.2.0"
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
    "RankedItem",
    "RankResult",
    "RetryPolicy",
    "Score",
    "ScoreAnswer",
    "SystemOneResponse",
    "Usage",
    "ZilsClient",
    "ZilsError",
]
