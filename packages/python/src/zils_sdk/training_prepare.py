"""Turn a spreadsheet or JSON export into the three text training splits.

The rules are the website's (zils-web lib/training-csv.ts), so a file prepared here trains the same way as one
prepared at zils.ai: one choice question per row, whole case groups kept in one split, every answer present in
train, calibration and test, about 70/15/15 by rows, and the same file always giving the same splits.
"""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import training_data

SPLITS = training_data.SPLITS
MAX_ROWS = 20_000
MAX_COLUMNS = 64
MAX_OUTCOMES = 255
MIN_GROUPS_PER_ANSWER = 3  # one each for train, calibration and test
TARGET = {"train": 0.7, "calibration": 0.15, "test": 0.15}
ID_NAME = re.compile(r"(^|_)(id|uuid|key)($|_)", re.IGNORECASE)


def _text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Header names and rows, as text, from a CSV, TSV, JSON array of objects, or JSON Lines file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            records = [
                row
                for row in csv.reader(handle, delimiter="\t" if suffix == ".tsv" else ",")
                if any(cell.strip() for cell in row)
            ]
        if not records:
            raise ValueError(f"{path.name} is empty.")
        headers = [h.strip() for h in records[0]]
        rows = []
        for number, record in enumerate(records[1:], start=2):
            if len(record) != len(headers):
                raise ValueError(
                    f"{path.name} row {number}: expected {len(headers)} cells, found {len(record)}. Check commas and quotes."
                )
            rows.append(dict(zip(headers, (cell.strip() for cell in record), strict=True)))
    elif suffix in (".json", ".jsonl", ".ndjson"):
        text = path.read_text(encoding="utf-8-sig")
        try:
            items = (
                json.loads(text)
                if suffix == ".json"
                else [json.loads(line) for line in text.splitlines() if line.strip()]
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path.name} is not valid JSON: {error.msg} at line {error.lineno}."
            ) from None
        if suffix == ".json" and isinstance(items, dict):
            items = [items]
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ValueError(f"{path.name} must hold one JSON object per example.")
        headers = list(dict.fromkeys(key for item in items for key in item))
        rows = [{h: _text(item.get(h)) for h in headers} for item in items]
    else:
        raise ValueError(
            "Use a .csv, .tsv, .json or .jsonl file. Export Excel sheets as CSV first."
        )
    if len(headers) < 2 or any(not h for h in headers) or len(set(headers)) != len(headers):
        raise ValueError(
            "Give the file a header row with at least two unique, nonempty column names."
        )
    if len(headers) > MAX_COLUMNS:
        raise ValueError(
            f"Use at most {MAX_COLUMNS} columns; export only the fields this decision needs."
        )
    if not rows:
        raise ValueError(f"{path.name} has a header but no examples.")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"Use at most {MAX_ROWS:,} examples per file.")
    return headers, rows


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _compact(value) -> str:
    """The website's JSON.stringify: no spaces, keys in insertion order, text kept as written."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def split_groups(groups: list[dict], outcomes: list[str]) -> dict[str, list[dict]]:
    """Whole groups go to one split; every answer reaches all three; ratios are approximate (website splitGroups)."""
    frequency = {o: sum(1 for g in groups if o in g["labels"]) for o in outcomes}
    for outcome in outcomes:
        if frequency[outcome] < MIN_GROUPS_PER_ANSWER:
            raise ValueError(
                f"“{outcome}” needs examples from at least {MIN_GROUPS_PER_ANSWER} separate cases so it can appear in "
                f"training, calibration and testing; it has {frequency[outcome]}. Add more examples of it."
            )
    rare_first = sorted(outcomes, key=lambda o: (frequency[o], o))
    ordered = sorted(groups, key=lambda g: g["id"])
    total = sum(len(g["rows"]) for g in groups)
    for attempt in range(24):  # a few stable orderings handle groups holding more than one answer
        remaining = {g["id"]: g for g in ordered}
        assigned = {s: [] for s in SPLITS}
        coverage = {s: set() for s in SPLITS}
        possible = True
        for outcome in rare_first:
            for split in SPLITS:
                if outcome in coverage[split]:
                    continue
                choices = [g for g in ordered if g["id"] in remaining and outcome in g["labels"]]
                if not choices:
                    possible = False
                    break
                group = choices[attempt % len(choices)]
                assigned[split].append(group)
                del remaining[group["id"]]
                coverage[split] |= group["labels"]
            if not possible:
                break
        if not possible:
            continue
        sizes = {s: sum(len(g["rows"]) for g in assigned[s]) for s in SPLITS}
        for group in sorted(remaining.values(), key=lambda g: (-len(g["rows"]), g["id"])):
            split = min(SPLITS, key=lambda s: sizes[s] / (total * TARGET[s]))
            assigned[split].append(group)
            sizes[split] += len(group["rows"])
        return {
            s: sorted((row for g in assigned[s] for row in g["rows"]), key=lambda r: r["id"])
            for s in SPLITS
        }
    raise ValueError(
        "These cases could not be split while keeping every answer in all three sets. Add more separate examples."
    )


def leaks(rows: list[dict[str, str]], columns: list[str], decide: str) -> list[dict]:
    """Columns whose value almost fixes the answer, like the agent who handled a ticket: usually filled in after
    the decision, so the model would learn nothing it can use. Free text and IDs (one value per few rows) are not
    judged, and a common answer alone is not a leak."""
    base = max(Counter(r[decide] for r in rows).values()) / len(rows)
    found = []
    for column in columns:
        answers = defaultdict(Counter)
        for row in rows:
            answers[row[column]][row[decide]] += 1
        if not 2 <= len(answers) <= len(rows) // 4:
            continue
        agreement = sum(max(a.values()) for a in answers.values()) / len(rows)
        if agreement >= 0.95 and agreement - base >= 0.9 * (1 - base):
            found.append(
                {
                    "column": column,
                    "agreement": round(agreement, 3),
                    "reason": f"Its value gives away “{decide}” in {agreement:.0%} of rows. Columns like this are "
                    "usually filled in after the decision; read it only if it is known when deciding.",
                }
            )
    return found


def _slug(text: str) -> str:
    return (
        re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")[:64] or "zils-model"
    )


def prepare(
    source, decide: str, *, out=None, read=None, group=None, name=None, question=None
) -> dict:
    """Write train/calibration/test JSONL for one decision column and return what was written and what it may cost."""
    source = Path(source)
    headers, rows = read_rows(source)
    if decide not in headers:
        raise ValueError(f"“{decide}” is not a column. Columns: {', '.join(headers)}.")
    if group is not None and group not in headers:
        raise ValueError(f"“{group}” is not a column. Columns: {', '.join(headers)}.")
    if read:
        missing = [c for c in read if c not in headers]
        if missing:
            raise ValueError(f"Not columns: {', '.join(missing)}. Columns: {', '.join(headers)}.")
        if decide in read or (group and group in read):
            raise ValueError(
                "Keep the answer column (and the case column) out of what the model reads."
            )
        inputs = sorted(dict.fromkeys(read))
        skipped = [c for c in headers if c not in inputs and c not in (decide, group)]
        leaked = leaks(rows, inputs, decide)
    else:
        inputs = sorted(c for c in headers if c not in (decide, group) and not ID_NAME.search(c))
        leaked = leaks(rows, inputs, decide)
        inputs = [c for c in inputs if c not in {leak["column"] for leak in leaked}]
        skipped = [c for c in headers if c not in (decide, group) and c not in inputs]
    if not inputs:
        raise ValueError(
            "Nothing is left for the model to read. Choose at least one input column with --read."
        )

    outcomes = sorted({r[decide] for r in rows if r[decide]})
    if len(outcomes) < 2:
        raise ValueError(f"“{decide}” needs at least two different answers to learn a decision.")
    if len(outcomes) > MAX_OUTCOMES or len(outcomes) > max(2, len(rows) // 2):
        raise ValueError(
            f"“{decide}” has {len(outcomes):,} different values in {len(rows):,} rows: that looks like free text or a "
            "measurement, not a short list of answers. Choose the column that holds the decision."
        )
    if any(len(o) > 100 for o in outcomes):
        raise ValueError(f"Answers in “{decide}” must be at most 100 characters each.")

    name = name or _slug(f"{source.stem}-{decide}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise ValueError("Use a name with 1–64 lowercase letters, digits or hyphens.")
    question = (question or f"Choose the {decide.replace('_', ' ')} for this example.").strip()
    if not question or len(question) > 1000:
        raise ValueError("Describe the decision in 1–1,000 characters.")
    criteria = {o: o for o in outcomes}

    seen, groups = {}, {}
    for number, row in enumerate(rows, start=2):
        answer = row[decide]
        if not answer:
            raise ValueError(
                f"{source.name} row {number}: the answer in “{decide}” is missing. Label it or remove the row."
            )
        information = {c: row[c] for c in inputs}
        if not any(information.values()):
            raise ValueError(f"{source.name} row {number}: every column the model reads is empty.")
        fingerprint = _compact(information)
        if fingerprint in seen:
            prior, prior_answer = seen[fingerprint]
            conflict = " but conflicting answers" if prior_answer != answer else ""
            raise ValueError(
                f"{source.name} rows {prior} and {number} have identical information{conflict}. Keep one, or add a column that tells them apart."
            )
        seen[fingerprint] = (number, answer)
        raw_group = row[group] if group else fingerprint
        if not raw_group:
            raise ValueError(
                f"{source.name} row {number}: the case in “{group}” is missing; related examples must stay together."
            )
        group_id = f"group-{_digest(raw_group)}"
        case = {
            "id": f"case-{_digest(fingerprint)}",
            "group_id": group_id,
            "family": name,
            "state": {"decision": question, "information": information},
            "question": {"type": "choice", "criteria": criteria},
            "label": answer,
        }
        if len(json.dumps(case).encode()) > training_data.MAX_CASE:
            raise ValueError(
                f"{source.name} row {number}: this example is over 128 KiB. Shorten it or read fewer columns."
            )
        entry = groups.setdefault(group_id, {"id": group_id, "rows": [], "labels": set()})
        entry["rows"].append(case)
        entry["labels"].add(answer)
    splits = split_groups(list(groups.values()), outcomes)

    files, checked = _write_splits(out, name, splits)
    return {
        "name": name,
        "decide": decide,
        "read": inputs,
        "skipped": skipped,
        "leaks": leaked,
        "group": group,
        **(
            {}
            if group
            else {
                "note": "No case column given: each row is treated as a separate case. If several rows belong to one case (one customer, one thread), pass it as the group."
            }
        ),
        "question": question,
        "answers": {
            o: {s: sum(1 for c in splits[s] if c["label"] == o) for s in SPLITS} for o in outcomes
        },
        "cases": len(groups),
        "files": {s: str(p) for s, p in files.items()},
        "counts": {s: v["count"] for s, v in checked["files"].items()},
        "ceiling_usd": checked["ceiling_usd"],
        "next": (
            f"zils train submit --name {name} --train {files['train']} --calibration {files['calibration']} "
            f"--test {files['test']} --min-accuracy 0.8 --min-brier-improvement 0.01 --max-cost {checked['ceiling_usd']:g} "
            "--allow-training-data-export"
        ),
    }


def _write_splits(out, name, splits):
    """Write the three splits (never overwriting) and run the service's own dataset check on them."""
    out = Path(out) if out else Path("zils-training") / name
    files = {s: out / f"{s}.jsonl" for s in SPLITS}
    existing = [str(p) for p in files.values() if p.exists()]
    if existing:
        raise ValueError(
            f"{', '.join(existing)} already exists. Choose another --out folder; nothing was written."
        )
    out.mkdir(parents=True, exist_ok=True)
    for split, path in files.items():
        with path.open("x", encoding="utf-8") as handle:
            handle.writelines(json.dumps(case, allow_nan=False) + "\n" for case in splits[split])
    return files, training_data.check(files)


def is_log(path) -> bool:
    """A folder of logs, or a JSON Lines file whose first entry is a `record` line ({question, state, answer})."""
    path = Path(path)
    if path.is_dir():
        return True
    if path.suffix.lower() not in (".jsonl", ".ndjson") or not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    entry = json.loads(line)
                except ValueError:
                    return False
                return isinstance(entry, dict) and {"question", "state", "answer"} <= set(entry)
    return False


def _log_entries(source):
    """(episode, entry, place) for every logged decision: one file per episode in a folder, or an "episode" field."""
    source = Path(source)
    files = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
    if not files:
        raise ValueError(f"No .jsonl decision logs in {source}.")
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if (
                    not isinstance(entry["question"], str)
                    or "state" not in entry
                    or "answer" not in entry
                ):
                    raise KeyError
            except (ValueError, KeyError, TypeError):
                raise ValueError(
                    f"{path.name}:{number} is not a decision log entry: each line needs question, state and answer."
                ) from None
            episode = entry.get("episode", path.stem if source.is_dir() else None)
            yield episode, entry, f"{path.name}:{number}"


def prepare_log(source, *, questions=None, decide=None, question=None, out=None, name=None) -> dict:
    """Training splits from decision logs written by `record` (or the same shape by hand).

    Each episode (a log file in a folder, or an "episode" field) stays in one split, so the test measures new
    episodes. The program's state is kept as JSON. A situation already logged earlier is left out and counted: the
    service refuses one situation in two splits. `questions` is fit's questions file ({id: {type, instructions,
    criteria}}); without it the options are the answers seen and the question id is the wording. Cases use the
    current training layout: the wording in state.decision, the program's state in state.information."""
    entries = list(_log_entries(source))
    asked = sorted({entry["question"] for _, entry, _ in entries})
    if decide is None:
        if len(asked) != 1:
            raise ValueError(
                f"The logs answer {len(asked)} questions ({', '.join(asked)}). Train one at a time: pass --decide "
                "with one of them."
            )
        decide = asked[0]
    elif decide not in asked:
        raise ValueError(f"No logged answers for “{decide}”. Logged questions: {', '.join(asked)}.")
    spec = (questions or {}).get(decide) or {}
    kind = spec.get("type", "choice")
    if kind not in ("choice", "noul"):
        raise ValueError(
            f"“{decide}” is a {kind} question; training from logs takes choice and yes/no (noul) questions today."
        )
    criteria = spec.get("criteria")
    listed = (
        list(criteria)
        if kind == "choice" and isinstance(criteria, dict) and "TODO option" not in criteria
        else None
    )
    instructions = spec.get("instructions")
    wording = (
        question or (instructions if instructions and instructions != "TODO" else decide)
    ).strip()
    if not wording or len(wording) > 1000:
        raise ValueError("Describe the decision in 1–1,000 characters.")

    def label(answer, place):
        if kind == "noul":
            if isinstance(answer, str):
                answer = answer.strip().lower() in ("true", "yes", "y", "1")
            return "true" if answer else "false"
        text = str(answer)
        if listed is not None and text not in listed:
            raise ValueError(
                f"{place}: the answer “{text}” for “{decide}” is not one of its options ({', '.join(listed)})."
            )
        return text

    seen, repeats, conflicts, groups, outcomes = {}, 0, 0, {}, set()
    for episode, entry, place in entries:
        if entry["question"] != decide:
            continue
        answer = label(entry["answer"], place)
        key = _compact(entry["state"])
        if key in seen:
            repeats += 1
            conflicts += seen[key] != answer
            continue
        seen[key] = answer
        group_id = f"group-{_digest(str(episode) if episode is not None else key)}"
        case = {
            "id": f"case-{_digest(decide + chr(10) + key)}",
            "group_id": group_id,
            "family": "",  # set once the name is known
            "state": {"decision": wording, "information": entry["state"]},
            "question": {"type": "noul"} if kind == "noul" else None,
            "label": answer,
        }
        if len(json.dumps(case).encode()) > training_data.MAX_CASE:
            raise ValueError(f"{place}: this state is over 128 KiB. Log less of it.")
        outcomes.add(answer)
        entry_group = groups.setdefault(group_id, {"id": group_id, "rows": [], "labels": set()})
        entry_group["rows"].append(case)
        entry_group["labels"].add(answer)
    options = listed or sorted(outcomes)
    if len(outcomes) < 2:
        raise ValueError(
            f"The logs give only one answer for “{decide}”; a decision needs at least two."
        )
    if len(options) > MAX_OUTCOMES:
        raise ValueError(
            f"“{decide}” has {len(options):,} different answers; at most {MAX_OUTCOMES} train."
        )
    if not name:
        # A logged question is usually an id (is_urgent); a whole sentence gets its first words, cut at a word.
        name = _slug(decide)
        if len(name) > 40:
            name = name[:41].rsplit("-", 1)[0]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise ValueError("Use a name with 1–64 lowercase letters, digits or hyphens.")
    for group in groups.values():
        for case in group["rows"]:
            case["family"] = name
            if case["question"] is None:
                case["question"] = {"type": "choice", "criteria": {o: o for o in options}}
    splits = split_groups(list(groups.values()), sorted(outcomes))
    files, checked = _write_splits(out, name, splits)
    return {
        "name": name,
        "decide": decide,
        "question": wording,
        "type": kind,
        "answers": {
            o: {s: sum(1 for c in splits[s] if c["label"] == o) for s in SPLITS}
            for o in sorted(outcomes)
        },
        "episodes": len(groups),
        "repeats_left_out": repeats,
        "conflicting_repeats": conflicts,
        "files": {s: str(p) for s, p in files.items()},
        "counts": {s: v["count"] for s, v in checked["files"].items()},
        "ceiling_usd": checked["ceiling_usd"],
        "next": (
            f"zils train submit --name {name} --train {files['train']} --calibration {files['calibration']} "
            f"--test {files['test']} --min-accuracy 0.8 --min-brier-improvement 0.01 --max-cost {checked['ceiling_usd']:g} "
            "--allow-training-data-export"
        ),
    }
