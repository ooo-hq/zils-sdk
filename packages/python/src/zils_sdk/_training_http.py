"""Bounded training responses, no redirects/retries, and no remote error text."""

import time

import httpx

from ._input import parse_json

MAX_JSON = 1024 * 1024
BILLING_URL = "https://zils.ai/billing"


class TrainingError(Exception):
    def __init__(self, status: int | None, message: str, uncertain: bool = False):
        super().__init__(message)
        self.status, self.uncertain = status, uncertain


def read_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    body: dict | None = None,
    form: dict | None = None,
    timeout: float = 30,
) -> dict:
    headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = "Bearer " + token
    deadline = time.monotonic() + timeout
    try:
        with httpx.Client(follow_redirects=False, timeout=timeout) as client:
            with client.stream(method, url, headers=headers, json=body, data=form) as response:
                if not 200 <= response.status_code < 300:
                    guidance = {
                        401: "Run zils login.",
                        402: "There is not enough prepaid credit on this account, or a payment"
                        f" needs attention. Add credit at {BILLING_URL}.",
                    }.get(response.status_code, "Check the job and account before trying again.")
                    raise TrainingError(
                        response.status_code,
                        f"Training service returned HTTP {response.status_code}. {guidance} No automatic retry was made.",
                        method != "GET" and response.status_code >= 500,
                    )
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > MAX_JSON or time.monotonic() > deadline:
                        raise ValueError("Response limit")
                result = parse_json(raw.decode("utf-8"))
                if not isinstance(result, dict):
                    raise ValueError("Response object required")
                return result
    except (httpx.HTTPError, ValueError, UnicodeError):
        raise TrainingError(
            None,
            "The training request could not be verified. No automatic retry was made.",
            method != "GET",
        ) from None
