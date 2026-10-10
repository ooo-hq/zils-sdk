"""Zils configuration never falls through to another provider's credentials."""

import ipaddress
import os
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "https://training.zils.ai/decision"
DEFAULT_MODEL = "zils-shared"


def setting(value: str | None, variable: str, default: str = "") -> str:
    return value if value is not None else os.environ.get(variable, "").strip() or default


def validate_key(value: str) -> str:
    if not value or any(not 33 <= ord(c) <= 126 for c in value):
        raise ValueError("Set ZILS_API_KEY to a nonempty API key without whitespace.")
    return value


def validate_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        loopback = host == "localhost"
        if host and not loopback:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                pass
        valid = (
            host
            and parsed.port != 0
            and (parsed.scheme == "https" or (parsed.scheme == "http" and loopback))
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() or ord(c) < 32 for c in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "Use an HTTPS base URL without credentials, query, or fragment; HTTP is allowed only on loopback."
        )
    return value.rstrip("/")


def client_config(api_key: str | None, base_url: str | None, model: str | None) -> dict[str, str]:
    key = validate_key(setting(api_key, "ZILS_API_KEY"))
    url = validate_url(setting(base_url, "ZILS_BASE_URL", DEFAULT_BASE_URL))
    name = setting(model, "ZILS_DEFAULT_MODEL", DEFAULT_MODEL)
    if not name.strip() or len(name) > 128:
        raise ValueError("Model must contain 1–128 characters.")
    return {"api_key": key, "base_url": url, "model": name}
