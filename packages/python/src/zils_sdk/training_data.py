"""Text dataset admission checks and exact-byte private upload snapshots."""

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path

from ._input import parse_json
from .account import identifier, origin

SPLITS = ("train", "calibration", "test")
MAX_SPLIT = 128 * 1024 * 1024
MAX_CASE = 128 * 1024


def validate_policy(name: str, accuracy: float, improvement: float, consent: bool) -> dict:
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise ValueError("Use a name with 1–64 lowercase letters, digits or hyphens.")
    if consent is not True:
        raise ValueError(
            "Confirm permission to share training data with approved workers using --allow-training-data-export."
        )
    policy = {"min_accuracy": accuracy, "min_brier_improvement": improvement}
    for key, maximum in (("min_accuracy", 1), ("min_brier_improvement", 2)):
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"{key} must be a finite number between 0 and {maximum}.")
    return policy


def _case(case):
    if (
        not isinstance(case, dict)
        or not {"id", "family", "group_id", "state", "question", "label"} <= set(case)
        or "image" in case
    ):
        raise ValueError(
            "Each text case requires id, family, group_id, state, question and label; image inputs are unsupported."
        )
    for key in ("id", "family", "group_id", "label"):
        if not isinstance(case[key], str) or not case[key]:
            raise ValueError("Case identifiers, family and label must be nonempty strings.")
    if len(json.dumps(case, allow_nan=False).encode()) > MAX_CASE:
        raise ValueError("Case exceeds the 128 KiB size limit.")
    question = case["question"]
    if not isinstance(question, dict):
        raise ValueError("Each case requires a question object.")
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "noul" and (
        criteria is None or isinstance(criteria, dict) and not set(criteria) - {"false", "true"}
    ):
        options = ("false", "true")
    elif (
        kind == "choice"
        and isinstance(criteria, dict)
        and 2 <= len(criteria) <= 255
        and all(criteria)
    ):
        options = criteria
    elif kind == "score" and isinstance(criteria, list) and 2 <= len(criteria) <= 255:
        options = [str(index) for index in range(len(criteria))]
    else:
        raise ValueError("Use noul, choice or score questions with at most 255 outcomes.")
    if case["label"] not in options:
        raise ValueError("Every label must identify an available question outcome.")


def prepare(files: dict[str, Path], root: Path) -> dict:
    if set(files) != set(SPLITS):
        raise ValueError("Provide train, calibration and test JSONL files.")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshot = str(uuid.uuid4())
    directory = root / snapshot
    directory.mkdir(mode=0o700)
    result = {"snapshot": snapshot, "files": {}}
    ids, groups, prompts, families = set(), {}, {}, {}
    try:
        for split in SPLITS:
            target = directory / (split + ".jsonl")
            digest, size = hashlib.sha256(), 0
            with (
                Path(files[split]).open("rb") as source,
                os.fdopen(
                    os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
                ) as output,
            ):
                while chunk := source.read(64 * 1024):
                    size += len(chunk)
                    if size > MAX_SPLIT:
                        raise ValueError("Dataset exceeds the 128 MiB split size limit.")
                    digest.update(chunk)
                    output.write(chunk)
            count, families[split] = 0, set()
            with target.open("rb") as source:
                while raw := source.readline(MAX_CASE + 1):
                    if len(raw) > MAX_CASE:
                        raise ValueError("Case exceeds the 128 KiB size limit.")
                    if not raw.strip():
                        continue
                    try:
                        case = parse_json(raw.decode("utf-8"))
                        _case(case)
                        fingerprint = hashlib.sha256(
                            json.dumps(
                                [case["state"], case["question"]], sort_keys=True, allow_nan=False
                            ).encode()
                        ).hexdigest()
                    except (ValueError, RecursionError, TypeError):
                        raise ValueError(
                            "Invalid text dataset case: check UTF-8, JSON, size, required fields, outcomes and labels."
                        ) from None
                    if case["id"] in ids:
                        raise ValueError("Case IDs must be unique across all splits.")
                    ids.add(case["id"])
                    if (
                        groups.setdefault(case["group_id"], split) != split
                        or prompts.setdefault(fingerprint, split) != split
                    ):
                        raise ValueError(
                            "Related groups and duplicate prompts must not cross dataset splits."
                        )
                    families[split].add(case["family"])
                    count += 1
            if not count:
                raise ValueError("Every dataset split must contain at least one case.")
            result["files"][split] = {
                "path": str(target),
                "sha256": digest.hexdigest(),
                "size": size,
                "count": count,
            }
        if families["calibration"] != families["test"] or not families["test"] <= families["train"]:
            raise ValueError("Calibration/test families must match and be present in training.")
        return result
    except BaseException:
        shutil.rmtree(directory)
        raise


def load_receipt(root: Path, job_id: str, account_id: str, base_url: str) -> dict:
    identifier(job_id)
    try:
        with (root / (job_id + ".json")).open("rb") as stream:
            raw = stream.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ValueError("Oversized receipt")
        receipt = parse_json(raw.decode("utf-8"))
        if (
            receipt["version"] != 1
            or receipt["job_id"] != job_id
            or identifier(receipt["account_id"]) != account_id
            or origin(receipt["base_url"]) != base_url
            or set(receipt["files"]) != set(SPLITS)
        ):
            raise ValueError("Receipt binding mismatch")
        snapshot = identifier(receipt["snapshot"])
        for split, info in receipt["files"].items():
            if (
                not re.fullmatch(r"[a-f0-9]{64}", info["sha256"])
                or type(info["size"]) is not int
                or not 0 < info["size"] <= MAX_SPLIT
                or type(info["count"]) is not int
                or info["count"] < 1
                or Path(info["path"]) != root / "snapshots" / snapshot / (split + ".jsonl")
            ):
                raise ValueError("Invalid snapshot metadata")
        return receipt
    except (OSError, ValueError, TypeError, KeyError):
        raise ValueError(
            "No matching local receipt for this job, account and service. Use zils train status to inspect it."
        ) from None
