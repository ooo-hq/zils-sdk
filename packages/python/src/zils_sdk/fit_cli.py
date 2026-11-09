"""`zils fit`: find decision calls in a codebase, then measure Zils on logged traffic."""

import json
import os
import sys
from pathlib import Path

from .credentials import load_credentials
from .fit import draft_questions, measure, read_log, scan

KIND = {"noul": "yes/no", "choice": "choice", "score": "score"}
STATUS = {
    "ok": ("✓ ok", "32"),
    "check_miss": ("! missed on check half", "33"),
    "no_threshold": ("✗ below target", "31"),
    "too_few_rows": ("… too few inputs", "33"),
    "not_logged": ("– not logged", "2"),
}


def add_parser(subparsers):
    fit = subparsers.add_parser(
        "fit", help="Find where Zils fits in your code and measure it on your logged calls"
    )
    commands = fit.add_subparsers(dest="fit_command", required=True)
    start = commands.add_parser(
        "start", help="Guided tour (the default): find the questions Zils can answer, try one"
    )
    start.add_argument("path", nargs="?", default=".", type=Path)
    start.add_argument("--include-tests", action="store_true", help="Also read test files")
    agent = commands.add_parser(
        "agent", help="Install the zils-fit skill so your coding agent can do the evaluation"
    )
    agent.add_argument("path", nargs="?", default=".", type=Path, help="Project folder (.)")
    train = commands.add_parser(
        "train", help="Train a model on your logged decisions (prepares the splits, then submits)"
    )
    train.add_argument(
        "--log",
        type=Path,
        default=Path("fit-log.jsonl"),
        help="A record() log, or a folder of them",
    )
    train.add_argument(
        "--questions", type=Path, help="Questions JSON from scan (wording and options)"
    )
    train.add_argument(
        "--decide", help="Which logged question to train (needed when the log holds several)"
    )
    train.add_argument("--question", help="The decision in words, if not in the questions file")
    train.add_argument("--name", help="Training name (default from the question)")
    train.add_argument(
        "--out", type=Path, help="Where to write the splits (default zils-training/<name>)"
    )
    train.add_argument("--min-accuracy", type=float, required=True)
    train.add_argument("--min-brier-improvement", type=float, required=True)
    train.add_argument(
        "--max-cost", type=float, help="Refuse before sending if it could cost more (US dollars)"
    )
    train.add_argument(
        "--allow-training-data-export",
        action="store_true",
        help="Permit copying examples to approved workers",
    )
    train.add_argument(
        "--tier",
        choices=("open", "private"),
        help="open: free, network miners; private: paid, Zils workers",
    )
    train.add_argument(
        "--allow-open-tier-training",
        action="store_true",
        help="With --tier open: permit sending the training split to miners",
    )
    train.add_argument("--timeout", type=float, default=30)
    train.add_argument("--json", action="store_true", help="Write the result JSON to stdout")
    find = commands.add_parser("scan", help="List model calls that make typed decisions")
    find.add_argument("path", nargs="?", default=".", type=Path)
    find.add_argument("--questions", type=Path, help="Write draft Zils questions (JSON) here")
    find.add_argument("--include-tests", action="store_true", help="Also scan test files")
    find.add_argument("--all", action="store_true", help="Also list the free-text calls")
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
    run.add_argument(
        "--answers",
        type=Path,
        help="Also save every answer (JSONL: input, logged answer, Zils's full answer)",
    )
    run.add_argument("--timeout", type=float, default=60)
    run.add_argument("--json", action="store_true", help="Write the report as JSON to stdout")


def painter(stream):
    """ANSI styling for a terminal; plain text for pipes, files, NO_COLOR and TERM=dumb."""
    on = stream.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
    return lambda text, code: f"\033[{code}m{text}\033[0m" if on else text


def _table(rows, paint):
    """Rows of (text, style) cells, padded to the widest cell in each column."""
    widths = [max(len(row[i][0]) for row in rows) for i in range(len(rows[0]))]
    widths[-1] = 0  # never pad the last column
    for row in rows:
        cells = [
            paint(text.ljust(w), style) if style else text.ljust(w)
            for (text, style), w in zip(row, widths, strict=True)
        ]
        print("   " + "  ".join(cells))


def _scan(args):
    found = scan(args.path, args.include_tests)
    if args.json:
        print(json.dumps(found, indent=2))
    else:
        paint = painter(sys.stdout)
        root = args.path.resolve()
        base = root if root.is_dir() else root.parent

        def where(item):
            path, line = item["location"].rsplit(":", 1)
            return f"{Path(path).resolve().relative_to(base)}:{line}"

        decisions = [f for f in found if f["kind"] == "decision"]
        drop_ins = [f for f in found if f["kind"] == "drop-in"]
        text = [f for f in found if f["kind"] == "text"]
        print()
        print(
            f" {paint('zils fit scan', '1')}  {paint('·', '2')}  {root.name}  "
            f"{paint('·', '2')}  {len(found)} model call{'' if len(found) == 1 else 's'}"
        )
        if decisions:
            print(
                f"\n {paint('●', '32')} {paint('Decisions Zils can answer', '1')}  "
                f"{paint(str(len(decisions)), '32')}"
            )
            _table(
                [
                    [
                        (KIND[f["question"]["type"]], "36"),
                        (f["name"], "1"),
                        ("" if f["function"] == f["name"] else f["function"], ""),
                        (where(f), "2"),
                    ]
                    for f in decisions
                ],
                paint,
            )
        if drop_ins:
            print(
                f"\n {paint('◆', '35')} {paint('Already Jev-compatible', '1')}  "
                f"{paint(str(len(drop_ins)), '35')}  "
                f"{paint('point it at Zils: change the base URL and model', '2')}"
            )
            _table([[(f["function"], "1"), (where(f), "2")] for f in drop_ins], paint)
        if text:
            hint = "" if args.all else "  (--all to list)"
            print(
                f"\n {paint('○', '2')} {paint('Free-text generation, skipped', '2')}  "
                f"{paint(str(len(text)), '2')}{paint(hint, '2')}"
            )
            if args.all:
                _table([[(f["function"], "2"), (where(f), "2")] for f in text], paint)
        if not found:
            print(f"\n {paint('No model calls found.', '2')}")
        if decisions and not args.questions:
            print(
                f"\n {paint('Next', '1')}  zils fit scan "
                f"{'' if args.path == Path('.') else str(args.path) + ' '}"
                "--questions questions.json"
            )
        print()
    if args.questions:
        questions = draft_questions(found)
        if not questions:
            raise ValueError("No decision calls to draft questions for.")
        args.questions.write_text(json.dumps(questions, indent=2) + "\n", encoding="utf-8")
        todo = sum("TODO" in json.dumps(q) for q in questions.values())
        print(
            f"Wrote {len(questions)} draft questions to {args.questions}"
            + (
                f"; fill in {todo} with TODO, then wrap each decision with record()."
                if todo
                else "; wrap each decision with record() next."
            ),
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
    err = painter(sys.stderr)
    live = sys.stderr.isatty()
    done = [0]
    with ZilsClient(**load_credentials(), timeout=args.timeout) as client:

        def predict(state, asked):
            try:
                result = client.system_one(state=state, questions=asked, model=args.model)
                return result.model_dump(mode="json")["answers"]
            finally:
                done[0] += 1
                if live:
                    print(
                        f"\r {err('Asking', '2')} {args.model}  {done[0]}/{len(rows)}",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

        if not live:
            print(f"Sending {len(rows)} requests to {args.model}.", file=sys.stderr)
        try:
            report = measure(
                rows, questions, predict, target=args.target, conservative=not args.point_estimate
            )
        finally:
            if live:
                print("\r\033[K", end="", file=sys.stderr)
    answers = report.pop("answers")
    if args.answers:
        args.answers.write_text(
            "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers), encoding="utf-8"
        )
        print(f"Saved {len(answers)} answers to {args.answers}.", file=sys.stderr)
    report["model"] = args.model
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        paint = painter(sys.stdout)

        def pct(value):
            return "–" if value is None else f"{value:.1%}"

        print()
        print(
            f" {paint('zils fit run', '1')}  {paint('·', '2')}  {args.model}  "
            f"{paint('·', '2')}  {report['rows']} inputs  {paint('·', '2')}  "
            f"target {args.target:.0%}\n"
        )
        header = [(h, "2") for h in ("question", "status", "handled", "agreement", "threshold")]
        body = []
        for qid, q in report["questions"].items():
            label, style = STATUS[q["status"]]
            threshold = "–" if q["threshold"] is None else f"{q['threshold']:.2f}"
            body.append(
                [
                    (qid, "1"),
                    (label, style),
                    (pct(q["handled"]), ""),
                    (pct(q["accuracy_handled"]), ""),
                    (threshold, "2"),
                ]
            )
        _table([header, *body], paint)
        share = report["rows_fully_handled"]
        print(
            f"\n Zils answers {paint(f'{share:.1%}', '1;32' if share else '1')} of inputs on its "
            "own; the rest stay on your current model."
        )
        print(f" {paint('Agreement is with your logged answers, not proven accuracy.', '2')}\n")
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


def install_skill(project):
    """Copy the packaged skill where Claude Code (.claude) and other agents (.agents) look."""
    from importlib.resources import files

    text = files("zils_sdk").joinpath("skills/zils-fit/SKILL.md").read_text(encoding="utf-8")
    written = []
    for folder in (".claude", ".agents"):
        target = Path(project) / folder / "skills" / "zils-fit" / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written.append(target)
    return written


def _agent(args):
    if not args.path.is_dir():
        raise ValueError(f"{args.path} is not a folder.")
    written = install_skill(args.path)
    for target in written:
        print(f"Installed {target}")
    print()
    print(f"Coding agent: read {written[0]} now and follow it to answer the user.")
    print('Person: ask your coding agent "See where Zils fits in this repo."')


def _train(args):
    from .training import TrainingClient
    from .training_cli import prepare_log, render, render_prepared_log

    prepared = prepare_log(args)
    print(render_prepared_log(prepared), file=sys.stderr)
    job = TrainingClient(timeout=args.timeout).submit(
        prepared["name"],
        prepared["files"],
        {"min_accuracy": args.min_accuracy, "min_brier_improvement": args.min_brier_improvement},
        args.allow_training_data_export,
        max_cost_usd=args.max_cost,
        tier=args.tier,
        allow_open_tier_training=args.allow_open_tier_training,
    )
    if args.json:
        print(json.dumps({"prepared": prepared, "job": job}, ensure_ascii=False))
        return
    print(render(job))
    questions = f" --questions {args.questions}" if args.questions else ""
    print(
        f"When it is ready (zils train status {job['id']}), measure it on the same log: "
        f"zils fit run --log {args.log}{questions} --model <its API model id>"
    )


def run(args):
    if args.fit_command == "train":
        _train(args)
        return
    if args.fit_command == "agent":
        _agent(args)
        return
    if args.fit_command == "start":
        from .fit_guide import guide

        guide(args.path, args.include_tests)
    else:
        _scan(args) if args.fit_command == "scan" else _run(args)
