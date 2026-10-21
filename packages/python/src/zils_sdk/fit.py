"""Find where Zils fits in an existing codebase and measure it on logged traffic.

`scan` lists model calls that make typed decisions. `record` shadow-logs a decision
function's inputs and answers. `measure` replays a log on Zils and picks, per
question, the confidence above which Zils agrees with the logged answers at the
target rate: picked on one half of the rows, checked on the other.
"""

import functools
import hashlib
import inspect
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

LOG_ENV = "ZILS_FIT_LOG"
DEFAULT_LOG = "fit-log.jsonl"
_lock = threading.Lock()

# ---------------------------------------------------------------- record


def _state(args, kwargs):
    # One argument is the usual case (the ticket, the email); keep it as-is.
    if len(args) == 1 and not kwargs:
        return args[0]
    return {"args": list(args), **kwargs}


def _write(question, state, answer):
    try:
        line = json.dumps(
            {"question": question, "state": state, "answer": answer, "ts": time.time()}
        )
        path = os.environ.get(LOG_ENV) or DEFAULT_LOG
        with _lock:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
    except Exception as error:  # logging must never break the caller's request
        print(f"zils fit: call not recorded ({type(error).__name__})", file=sys.stderr)


def record(question):
    """Decorate a decision function; each call appends its input and answer to the log.

    The log holds real inputs, so it is created owner-only. Failed calls are not logged.
    """

    def wrap(fn):
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_call(*args, **kwargs):
                answer = await fn(*args, **kwargs)
                _write(question, _state(args, kwargs), answer)
                return answer

            return async_call

        @functools.wraps(fn)
        def call(*args, **kwargs):
            answer = fn(*args, **kwargs)
            _write(question, _state(args, kwargs), answer)
            return answer

        return call

    return wrap


def read_log(path):
    """Merge logged calls on the same input into rows: {id, state, labels}."""
    rows = {}
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                question, state, answer = entry["question"], entry["state"], entry["answer"]
            except (ValueError, KeyError, TypeError):
                raise ValueError(f"{path}:{number} is not a fit log entry.") from None
            key = json.dumps(state, sort_keys=True)
            row = rows.setdefault(
                key,
                {"id": hashlib.sha256(key.encode()).hexdigest()[:12], "state": state, "labels": {}},
            )
            row["labels"][question] = answer  # the latest answer wins
    return list(rows.values())


# ---------------------------------------------------------------- scan

EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "dist", "build", "__pycache__"}
CALL = re.compile(
    r"chat\.completions\.create\(|messages\.create\(|responses\.create\(|"
    r"\bgenerate(?:Object|Text)\(|\bllm\.invoke\(|\bcompletion\("
)
# Calls already in the Jev decision-API shape only need a new base URL and model.
JEV = re.compile(r"/v1/systemone|/decisions[\"'/]|\.decide\(|typesafe/jev|[\"']jev-[\d.]+")
WINDOW = 20  # ponytail: fixed line window; use a parser if call sites get long
DEF = re.compile(r"(?:def|function)\s+(\w+)\s*\((?:.*?\)\s*(?:->\s*(\w+)|:\s*(\w+))?)?")
NOUL = re.compile(
    r"startswith\(\s*[\"'](?:y|yes|true)[\"']|==\s*[\"'](?:yes|true)[\"']|z\.boolean\(\)", re.I
)
ENUM = re.compile(r"(\w+)\s*:\s*z\.enum\(\s*\[([^\]]+)\]|Literal\[([^\]]+)\]")
SCORE = re.compile(r"\bint\(|parseInt\(|\bscale\b", re.I)
ASKED = re.compile(r"[\"'`]([^\"'`]*?\?)")
LEVEL = re.compile(r"\b(\d)\s+([a-z][^,.;\"'\n]*)")


def _is_test(path):
    return (
        bool({"test", "tests", "__tests__"} & set(path.parts))
        or path.name.startswith("test_")
        or ".test." in path.name
        or ".spec." in path.name
    )


def _classify(text, name, returns):
    asked = ASKED.search(text)
    question = {"instructions": asked.group(1).strip()} if asked else {}
    if m := ENUM.search(text):
        options = re.findall(r"[\"']([^\"']+)[\"']", m.group(2) or m.group(3))
        return m.group(1) or name, {
            "type": "choice",
            **question,
            "criteria": dict.fromkeys(options),
        }
    if NOUL.search(text) or returns in ("bool", "boolean"):
        return name, {"type": "noul", **question}
    if SCORE.search(text) or returns == "int":
        levels = dict(LEVEL.findall(text))
        criteria = [levels[str(i)].strip() for i in range(len(levels)) if str(i) in levels]
        return name, {
            "type": "score",
            **question,
            "criteria": criteria if len(criteria) >= 2 else None,
        }
    return name, None


def scan(root, include_tests=False):
    """Model call sites as dicts: location, name, kind ("decision", "drop-in" or "text"), question."""
    root = Path(root)
    files = (
        [root]
        if root.is_file()
        else sorted(
            p
            for p in root.rglob("*")
            if p.suffix in EXTS
            and p.is_file()
            and not SKIP_DIRS & set(p.relative_to(root).parts)
            and (include_tests or not _is_test(p.relative_to(root)))
        )
    )
    found = []
    for path in files:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        calls = []
        for i, line in enumerate(lines):
            # ponytail: one call per 8 lines; a parser would split adjacent calls properly
            if (CALL.search(line) or JEV.search(line)) and not (calls and i - calls[-1] < 8):
                calls.append(i)
        for n, i in enumerate(calls):
            name, returns, start = "unnamed", None, max(i - 14, 0)
            for j in range(i, max(i - 15, -1), -1):
                if d := DEF.search(lines[j]):
                    name, returns, start = d.group(1), d.group(2) or d.group(3), j
                    break
            end = min(i + WINDOW, calls[n + 1] if n + 1 < len(calls) else len(lines))
            text = "\n".join(lines[start:end])
            location = f"{path}:{i + 1}"
            if JEV.search(text) and (JEV.search(lines[i]) or "questions" in text):
                found.append(
                    {"location": location, "name": name, "kind": "drop-in", "question": None}
                )
                continue
            qid, question = _classify(text, name, returns)
            kind = "decision" if question else "text"
            found.append({"location": location, "name": qid, "kind": kind, "question": question})
    return found


def draft_questions(found):
    """Zils questions for the decision calls; fill any missing instructions or criteria."""
    questions = {}
    for item in found:
        question = item["question"]
        if item["kind"] != "decision":
            continue
        draft = {"type": question["type"], "instructions": question.get("instructions") or "TODO"}
        if question["type"] == "choice":
            draft["criteria"] = question["criteria"]
        elif question["type"] == "score":
            draft["criteria"] = question["criteria"] or ["TODO lowest", "TODO highest"]
        questions[item["name"]] = draft
    return questions


# ---------------------------------------------------------------- measure


def _label(question, value):
    kind = question["type"]
    if kind == "noul":
        if isinstance(value, str):
            value = value.strip().lower() in ("true", "yes", "y", "1")
        return bool(value)
    if kind == "score":
        return int(value)
    return str(value)


def _prediction(question, answer):
    """(predicted label, confidence) from a Zils answer."""
    if question["type"] == "noul":
        p = answer["noul"]
        return p >= 0.5, max(p, 1 - p)
    probabilities = answer["probabilities"]
    best = max(probabilities, key=probabilities.get)
    return (int(best) if question["type"] == "score" else best), probabilities[best]


def _lower_bound(correct, n, z=1.96):
    """Wilson 95% lower bound on accuracy."""
    if n == 0:
        return 0.0
    p = correct / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def _check_half(row_id):
    return int(hashlib.sha256(row_id.encode()).hexdigest(), 16) % 2 == 1


def measure(rows, questions, predict, *, target=0.95, min_support=30, conservative=True):
    """Pick a confidence threshold per question on one half of the rows; report the other.

    `predict(state, questions)` returns Zils answers by question id. A row that fails
    is reported in `errors` and left out, never counted as a disagreement.
    """
    results, errors = {}, []
    for row in rows:
        asked = {q: questions[q] for q in row["labels"] if q in questions}
        if not asked:
            continue
        try:
            answers = predict(row["state"], asked)
            results[row["id"]] = {q: _prediction(asked[q], answers[q]) for q in asked}
        except Exception as error:
            errors.append({"id": row["id"], "error": f"{type(error).__name__}: {error}"})
    report = {"rows": len(results), "errors": errors, "questions": {}}
    handled_rows = {}
    for qid, question in questions.items():
        points = []
        for row in rows:
            if row["id"] in results and qid in results[row["id"]]:
                predicted, confidence = results[row["id"]][qid]
                correct = predicted == _label(question, row["labels"][qid])
                points.append((row["id"], confidence, correct))
        tune = [p for p in points if not _check_half(p[0])]
        check = [p for p in points if _check_half(p[0])]
        threshold = None
        for t in sorted({c for _, c, _ in tune}):
            kept = [ok for _, c, ok in tune if c >= t]
            if len(kept) < min_support:
                break
            score = _lower_bound(sum(kept), len(kept)) if conservative else sum(kept) / len(kept)
            if score >= target:
                threshold = t
                break
        kept = [ok for _, c, ok in check if threshold is not None and c >= threshold]
        accuracy = sum(kept) / len(kept) if kept else None
        # With the lower bound, even all-correct rows clear the target only from this many.
        needed = (
            max(min_support, math.ceil(1.96**2 * target / (1 - target)))
            if conservative
            else min_support
        )
        status = (
            "not_logged"
            if not points
            else "too_few_rows"
            if threshold is None and len(tune) < needed
            else "no_threshold"
            if threshold is None
            else "ok"
            if accuracy is not None and accuracy >= target
            else "check_miss"
        )
        report["questions"][qid] = {
            "rows": len(points),
            "threshold": threshold,
            "handled": len(kept) / len(check) if check else 0.0,
            "accuracy_handled": accuracy,
            "agreement_all": sum(ok for *_, ok in points) / len(points) if points else None,
            "status": status,
            "inputs_needed": 2 * needed,
        }
        for row_id, confidence, _ in check:
            # Only a threshold that held on the check half is one you would deploy.
            ok = status == "ok" and confidence >= threshold
            handled_rows[row_id] = handled_rows.get(row_id, True) and ok
    report["rows_fully_handled"] = (
        sum(handled_rows.values()) / len(handled_rows) if handled_rows else 0.0
    )
    return report
