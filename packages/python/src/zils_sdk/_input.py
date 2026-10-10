"""Shared request checks for CLI and MCP; the API enforces its full contract."""

import json


def parse_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key.")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("Non-finite JSON number.")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError):
        raise ValueError("Input must be valid JSON with unique keys and finite numbers.") from None


def check_request(state, questions) -> None:
    if not isinstance(state, (str, dict, list)):
        raise ValueError("State must be text, a JSON object, or an array.")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Questions must be a nonempty object keyed by question name.")
    for name, question in questions.items():
        if not isinstance(name, str) or not name or not isinstance(question, dict):
            raise ValueError("Each named question must be an object.")
        if set(question) - {"type", "instructions", "criteria"}:
            raise ValueError(
                "A question has unsupported fields; use type, instructions, and criteria."
            )
        if question.get("type") not in ("noul", "choice", "score"):
            raise ValueError("Question type must be noul, choice, or score.")


def error_message(error: Exception) -> str:
    from typesafe_sdk import TypeSafeAPIError, TypeSafeError

    if isinstance(error, TypeSafeAPIError):
        guidance = {
            401: "Check your API key with zils login.",
            402: "Check your Zils credit balance.",
            403: "This account cannot access the requested resource.",
            404: "Check the model name with zils models.",
            422: "Check the state, questions, and model against the Zils API contract.",
            429: "Wait before trying again.",
        }.get(error.status, "Check the service before trying again.")
        return f"Zils API returned HTTP {error.status}. {guidance} No automatic retry was made."
    if isinstance(error, TypeSafeError):
        return "The Zils request could not complete. Check the connection and request; a retry may incur another charge."
    if isinstance(error, OSError):
        return "Could not read or write the requested file. Check its path and permissions."
    return str(error)
