"""The zils terminal interface. Results use stdout; status and errors use stderr."""

import argparse
import getpass
import json
import os
import socket
import sys
from pathlib import Path

from . import ZilsClient, ZilsError, __version__, account, fit_cli, training_cli
from ._config import DEFAULT_BASE_URL, setting, validate_key, validate_url
from ._input import check_request, error_message, parse_json
from ._training_http import TrainingError
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


def api_key_source(value):
    if value != "-":
        raise argparse.ArgumentTypeError(
            "Use --api-key alone or --api-key -; never pass a key value."
        )
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="zils", description="Train and call Zils decision models.")
    root.add_argument("--version", action="version", version=f"zils {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    login = commands.add_parser(
        "login", help="Sign in through your browser (default), or save an inference API key"
    )
    login.add_argument(
        "--api-key",
        nargs="?",
        const=True,
        type=api_key_source,
        metavar="-",
        help="Prompt privately for an inference key; add - to read from stdin",
    )
    login.add_argument("--base-url", help="Zils API base URL (HTTPS, or HTTP loopback)")
    commands.add_parser(
        "logout", help="Remove saved account and key credentials; does not revoke remote sessions"
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
    training_cli.add_parser(commands)
    fit_cli.add_parser(commands)
    return root


def offer_key() -> None:
    """After browser sign-in, offer an inference key so decide and fit work without a second step."""
    if os.environ.get("ZILS_API_KEY", "").strip() or config_path().exists():
        return
    if not sys.stdin.isatty():
        print("To call models from this machine, run zils login --api-key.", file=sys.stderr)
        return
    sys.stderr.write("Create an API key for this machine, for zils decide and zils fit? [Y/n] ")
    sys.stderr.flush()
    if sys.stdin.readline().strip().lower() not in ("", "y", "yes"):
        print("No key created. Run zils login --api-key when you need one.", file=sys.stderr)
        return
    name = f"zils CLI on {socket.gethostname()}"[:80]
    key, url = account.create_api_key(name)
    save_credentials(key, url)
    print(
        f'Created API key "{name}" and saved it privately; revoke it any time at zils.ai/keys.',
        file=sys.stderr,
    )


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
    if args.command == "train":
        training_cli.run(args)
        return
    if args.command == "fit":
        fit_cli.run(args)
        return
    if args.command == "logout":
        account.logout()
        config_path().unlink(missing_ok=True)
        print("Saved credentials removed. Environment credentials are unchanged.", file=sys.stderr)
        return
    if args.command == "login":
        if args.api_key is None:
            if args.base_url is not None:
                raise ValueError(
                    "--base-url requires --api-key. Use ZILS_TRAINING_URL for account sign-in."
                )
            print(json.dumps(account.login()))
            offer_key()
            return
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
    from .fit_guide import main_args

    args = parser().parse_args(main_args(sys.argv[1:]))
    try:
        run(args)
        return 0
    except TrainingError as error:
        print(str(error), file=sys.stderr)
        return 1
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
