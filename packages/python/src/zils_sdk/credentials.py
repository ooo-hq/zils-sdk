"""Private, atomic credentials shared by the terminal and MCP entry points."""

import json
import os
from pathlib import Path

from ._config import DEFAULT_BASE_URL, client_config, setting, validate_key, validate_url
from ._files import atomic_json


def config_path() -> Path:
    configured = os.environ.get("ZILS_CONFIG_PATH", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".zils" / "config.json"


def load_credentials() -> dict[str, str]:
    if os.environ.get("ZILS_API_KEY", "").strip():
        return client_config(None, None, None)
    path = config_path()
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("Run zils login --api-key or set ZILS_API_KEY.") from None
    except (OSError, ValueError):
        raise ValueError(
            "Cannot read Zils credentials; run zils login --api-key to replace them."
        ) from None
    if not isinstance(saved, dict) or not all(
        isinstance(saved.get(k), str) for k in ("api_key", "base_url")
    ):
        raise ValueError("Invalid Zils credentials; run zils login --api-key to replace them.")
    return client_config(saved["api_key"], setting(None, "ZILS_BASE_URL", saved["base_url"]), None)


def save_credentials(api_key: str, base_url: str = DEFAULT_BASE_URL) -> None:
    value = {"api_key": validate_key(api_key), "base_url": validate_url(base_url)}
    atomic_json(config_path(), value)
