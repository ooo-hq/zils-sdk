"""`zils fit`: find decision calls in a codebase, then measure Zils on logged traffic."""

import json
import sys
from pathlib import Path

from .credentials import load_credentials
from .fit import draft_questions, measure, read_log, scan


def add_parser(subparsers):
    fit = subparsers.add_parser(
        "fit", help="Find where Zils fits in your code and measure it on your logged calls"
    )
    commands = fit.add_subparsers(dest="fit_command", required=True)
    find = commands.add_parser("scan", help="List model calls that make typed decisions")
    find.add_argument("path", nargs="?", default=".", type=Path)
    find.add_argument("--questions", type=Path, help="Write draft Zils questions (JSON) here")
    find.add_argument("--include-tests", action="store_true", help="Also scan test files")
    find.add_argument("--json", action="store_true", help="Write findings as JSON to stdout")
    run = commands.add_parser(
        "run", help="Replay a fit log on Zils (one paid request per logged input)"
    )
    run.add_argument("--log", type=Path, default=Path("fit-log.jsonl"))
    run.add_argument("--questions", type=Path, required=True, help="Questions JSON from scan")
    run.add_argument("--model", default="zils-shared")
    run.add_argument("--target", type=float, default=0.95, help="Required agreement (0.95)")
    run.add_argument("--max-rows", type=int, default=500, help="Spend cap in requests (500)")
    run.add_argument(
        "--point-estimate",
        action="store_true",
        help="Pick thresholds on the measured rate, not its 95%% lower bound",
    )
    run.add_argument("--timeout", type=float, default=60)
    run.add_argument("--json", action="store_true", help="Write the report as JSON to stdout")


def _scan(args):
    found = scan(args.path, args.include_tests)
    if args.json:
        print(json.dumps(found, indent=2))
    else:
        labels = {
            "decision": None,
            "drop-in": "drop-in: Jev API shape, change base URL and model",
            "text": "skip: free text",
        }
        for item in found:
            kind = labels[item["kind"]] or item["question"]["type"]
            print(f"{item['location']}\t{item['name']}\t{kind}")
    decisions = sum(item["kind"] != "text" for item in found)
    print(f"{decisions} of {len(found)} model calls make typed decisions.", file=sys.stderr)
    if args.questions:
        questions = draft_questions(found)
        if not questions:
            raise ValueError("No decision calls to draft questions for.")
        args.questions.write_text(json.dumps(questions, indent=2) + "\n", encoding="utf-8")
        print(
            f"Wrote {len(questions)} draft questions to {args.questions}; "
            "check each TODO before running.",
            file=sys.stderr,
        )


def _run(args):
    from . import ZilsClient

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Questions must be a JSON object of Zils questions.")
    if any("TODO" in json.dumps(q) for q in questions.values()):
        raise ValueError(f"Replace every TODO in {args.questions} first.")
    if not 0 < args.target < 1 or args.max_rows < 1:
        raise ValueError("Use a target between 0 and 1 and a positive --max-rows.")
    rows = [r for r in read_log(args.log) if set(r["labels"]) & set(questions)]
    if not rows:
        raise ValueError(f"No logged answers in {args.log} match the questions.")
    if len(rows) > args.max_rows:
        raise ValueError(f"{len(rows)} logged inputs exceed --max-rows {args.max_rows}.")
    print(f"Sending {len(rows)} requests to {args.model}.", file=sys.stderr)
    with ZilsClient(**load_credentials(), timeout=args.timeout) as client:

        def predict(state, asked):
            result = client.system_one(state=state, questions=asked, model=args.model)
            return result.model_dump(mode="json")["answers"]

        report = measure(
            rows, questions, predict, target=args.target, conservative=not args.point_estimate
        )
    report["model"] = args.model
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("question\tstatus\tthreshold\thandled\tagreement when handled\tagreement overall")
        for qid, q in report["questions"].items():
            fmt = lambda v, pct=True: "-" if v is None else f"{v:.1%}" if pct else f"{v:.3f}"  # noqa: E731
            print(
                f"{qid}\t{q['status']}\t{fmt(q['threshold'], False)}\t{fmt(q['handled'])}\t"
                f"{fmt(q['accuracy_handled'])}\t{fmt(q['agreement_all'])}"
            )
        print(
            f"Inputs Zils handles completely: {report['rows_fully_handled']:.1%} "
            "(check half; the rest go to your current model)."
        )
    short = [q for q, v in report["questions"].items() if v["status"] == "too_few_rows"]
    if short:
        need = max(report["questions"][q]["inputs_needed"] for q in short)
        print(
            f"Too few logged inputs for {', '.join(short)}: log at least {need} at this "
            "target, or lower --target.",
            file=sys.stderr,
        )
    for error in report["errors"][:3]:
        print(f"Request failed for {error['id']}: {error['error']}", file=sys.stderr)
    if report["errors"]:
        print(
            f"{len(report['errors'])} of {len(rows)} requests failed and were left out.",
            file=sys.stderr,
        )
    if not report["rows"]:
        raise ValueError("Every request failed.")


def run(args):
    _scan(args) if args.fit_command == "scan" else _run(args)
