"""Terminal-only customer training commands."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from .account import identifier
from .training import TrainingClient


def positive_seconds(value):
    try:
        seconds = float(value)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError
        return seconds
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Timeout must be a finite positive number of seconds."
        ) from None


def add_parser(subparsers):
    train = subparsers.add_parser(
        "train", help="Submit and manage z3 text training with your Zils account"
    )
    commands = train.add_subparsers(dest="training_command", required=True)
    for name, help_text in (
        ("submit", "Validate and submit three text JSONL splits"),
        ("resume", "Resume missing uploads for an existing job using its original files"),
        ("list", "List recent jobs and ready models"),
        ("status", "Show training results and API activation separately"),
        ("cancel", "Request cancellation; already incurred costs may remain"),
        ("download", "Verify and save an accepted model to a new directory"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--json", action="store_true", help="Write public result JSON to stdout"
        )
        command.add_argument(
            "--timeout",
            type=positive_seconds,
            default=30,
            help="Seconds per request/transfer (default 30)",
        )
        if name not in ("submit", "list"):
            command.add_argument("job_id")
        if name in ("submit", "resume"):
            for split in ("train", "calibration", "test"):
                command.add_argument(
                    "--" + split, type=Path, required=True, help=f"Original {split} JSONL file"
                )
        if name == "submit":
            command.add_argument("--name", required=True)
            command.add_argument(
                "--pool", help="Service pool key for the z3 line (default d3-mini)"
            )
            command.add_argument("--base-channel", choices=("stable", "latest"))
            command.add_argument("--model", help="Exact profile ID advertised by the service")
            command.add_argument("--min-accuracy", type=float, required=True)
            command.add_argument("--min-brier-improvement", type=float, required=True)
            command.add_argument(
                "--max-cost",
                type=float,
                help="Refuse before sending if the job could cost more than this many US dollars",
            )
            command.add_argument(
                "--allow-training-data-export",
                action="store_true",
                help="Permit training examples to be copied to approved workers",
            )
        if name == "status":
            command.add_argument(
                "--watch",
                action="store_true",
                help="Poll every 5 seconds until training finishes; Ctrl+C does not cancel",
            )
        if name == "cancel":
            command.add_argument(
                "--yes", action="store_true", help="Confirm cancellation without a terminal prompt"
            )
        if name == "download":
            command.add_argument(
                "--out",
                type=Path,
                required=True,
                help="New output directory; existing paths are never overwritten",
            )


def render(result):
    if "jobs" in result:
        rows = result["jobs"]
        rows += [job for job in result["models"] if job["id"] not in {row["id"] for row in rows}]
        return "\n\n".join(render(job) for job in rows) or "No training jobs found."
    if "directory" in result:
        return f"Verified model saved to {result['directory']}\nSHA-256: {result['sha256']}"
    lines = [f"{result['id']}  {result['name']}  {result['status']}"]
    if "delivery" in result:
        lines.append("Result: " + result["delivery"].replace("_", " "))
    if "workflow" in result:
        lines.append("Workflow: " + result["workflow"].replace("_", " "))
    if "evaluation" in result:
        lines.append("Evaluation: " + json.dumps(result["evaluation"], allow_nan=False))
    if "model_id" in result:
        lines.append("API model: " + result["model_id"])
    elif result.get("download_available"):
        lines.append("Accepted model available to download. API readiness has not been confirmed.")
    if "error" in result:
        lines.append("Reason: " + result["error"])
    if result.get("cancelled"):
        lines.append(
            "Cancellation recorded. Running workers may finish; incurred costs may remain."
        )
    return "\n".join(lines)


def run(args):
    action = args.training_command
    if hasattr(args, "job_id"):
        identifier(args.job_id)
    if action == "cancel" and not args.yes:
        if not sys.stdin.isatty():
            raise ValueError("Cancellation requires --yes without an interactive terminal.")
        print(
            f"Cancel job {args.job_id}? Costs already incurred may remain. [y/N] ",
            file=sys.stderr,
            end="",
            flush=True,
        )
        if sys.stdin.readline().strip().lower() not in ("y", "yes"):
            print("Cancellation declined.", file=sys.stderr)
            return
    client = TrainingClient(timeout=args.timeout)
    if action in ("submit", "resume"):
        files = {split: getattr(args, split) for split in ("train", "calibration", "test")}
        result = (
            client.submit(
                args.name,
                files,
                {
                    "min_accuracy": args.min_accuracy,
                    "min_brier_improvement": args.min_brier_improvement,
                },
                args.allow_training_data_export,
                pool=args.pool,
                base_channel=args.base_channel,
                model=args.model,
                max_cost_usd=args.max_cost,
            )
            if action == "submit"
            else client.resume(args.job_id, files)
        )
    elif action == "status":
        result = client.get(args.job_id)
        while args.watch and result["status"] not in ("completed", "failed"):
            print(render(result), file=sys.stderr, flush=True)
            time.sleep(5)
            result = client.get(args.job_id)
    elif action == "download":
        result = client.download(args.job_id, args.out)
    else:
        result = client.list() if action == "list" else client.cancel(args.job_id)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False) if args.json else render(result))
