"""The zils terminal interface. Results use stdout; status and errors use stderr."""

import argparse
import getpass
import json
import sys
from pathlib import Path

from . import ZilsClient, ZilsError, __version__
from ._config import DEFAULT_BASE_URL, setting, validate_key, validate_url
from ._input import check_request, error_message, parse_json
from .credentials import config_path, load_credentials, save_credentials

MAX_INPUT = 1024 * 1024


def read_input(value: str) -> str:
    if value == "-":
        text = sys.stdin.buffer.read(MAX_INPUT + 1).decode("utf-8-sig")
    elif value.startswith("@"):
        with Path(value[1:]).open("rb") as stream:
            text = stream.read(MAX_INPUT + 1).decode("utf-8-sig")
    else:
        text = value
    if len(text.encode("utf-8")) > MAX_INPUT:
        raise ValueError("Input exceeds 1 MiB.")
    return text


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="zils", description="Call Zils decision models.")
    root.add_argument("--version", action="version", version=f"zils {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login", help="Validate and privately save a Zils API key")
    login.add_argument(
        "--api-key", choices=["-"], help="Read the key from stdin; otherwise prompt privately"
    )
    login.add_argument("--base-url", help="Zils API base URL (HTTPS, or HTTP loopback)")
    commands.add_parser(
        "logout", help="Remove locally saved credentials; does not revoke the API key"
    )
    models = commands.add_parser("models", help="List models available to your key")
    decide = commands.add_parser("decide", help="Answer typed questions about one state")
    for command in (models, decide):
        command.add_argument("--json", action="store_true", help="Write JSON to stdout")
        command.add_argument(
            "--timeout", type=float, default=60, help="Seconds per request (default 60)"
        )
    source = decide.add_mutually_exclusive_group(required=True)
    source.add_argument("--body", help="JSON request: inline, @file, or - for stdin")
    source.add_argument("--state", help="Text or JSON state: inline, @file, or - for stdin")
    decide.add_argument("--questions", help="JSON questions: inline, @file, or - for stdin")
    decide.add_argument("--model", help="Model name; defaults to ZILS_DEFAULT_MODEL or zils-shared")
    return root


def request(args):
    if args.body is not None:
        if args.questions is not None:
            raise ValueError("Use either --body or --state with --questions.")
        body = parse_json(read_input(args.body))
        if not isinstance(body, dict) or set(body) - {"state", "questions", "model"}:
            raise ValueError("A request body accepts only state, questions, and model.")
        if "state" not in body or "questions" not in body:
            raise ValueError("A request body requires state and questions.")
    else:
        if args.questions is None:
            raise ValueError("Supply --questions with --state.")
        if args.state == args.questions == "-":
            raise ValueError("Only one argument may read stdin.")
        state = read_input(args.state)
        body = {
            "state": parse_json(state) if state.lstrip().startswith(("{", "[")) else state,
            "questions": parse_json(read_input(args.questions)),
        }
    check_request(body["state"], body["questions"])
    if args.model is not None:
        body["model"] = args.model
    if "model" in body and (
        not isinstance(body["model"], str) or not 1 <= len(body["model"]) <= 128
    ):
        raise ValueError("Model must contain 1–128 characters.")
    return body


def run(args) -> None:
    if args.command == "logout":
        config_path().unlink(missing_ok=True)
        print("Saved credentials removed. Environment credentials are unchanged.", file=sys.stderr)
        return
    if args.command == "login":
        if args.api_key == "-":
            key = sys.stdin.readline(8192).strip()
        elif sys.stdin.isatty():
            key = getpass.getpass("Zils API key: ").strip()
        else:
            raise ValueError(
                "Without a terminal, use zils login --api-key - to read the key from stdin."
            )
        url = validate_url(setting(args.base_url, "ZILS_BASE_URL", DEFAULT_BASE_URL))
        with ZilsClient(api_key=validate_key(key), base_url=url) as client:
            client.models.list()
        save_credentials(key, url)
        print("Zils key verified and saved privately.", file=sys.stderr)
        return
    body = request(args) if args.command == "decide" else None
    with ZilsClient(**load_credentials(), timeout=args.timeout) as client:
        result = client.models.list() if body is None else client.system_one(**body)
    if args.json:
        print(json.dumps(result.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
    elif body is None:
        for model in result.models:
            print(f"{model.name}\t{model.description}")
    else:
        for name, answer in result.answers.items():
            print(f"{name}\t{json.dumps(answer.model_dump(mode='json'), ensure_ascii=False)}")
        print(
            f"Model: {result.model}; input tokens: {result.usage.input_tokens}; "
            f"billable input tokens: {result.usage.billable_input_tokens}",
            file=sys.stderr,
        )


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
        return 0
    except ZilsError as error:
        print(error_message(error), file=sys.stderr)
        return 1
    except (ValueError, OSError) as error:
        print(
            error_message(error)
            if not isinstance(error, UnicodeError)
            else "Input must use UTF-8.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
