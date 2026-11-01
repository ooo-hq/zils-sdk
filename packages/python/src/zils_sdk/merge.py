"""Merge 2-5 models of the same family into a new model folder: a soup, cooked locally.

The output keeps the first model's files and safetensors layout byte for byte, except that each
weight every model shares is replaced by the merge: an even average by default (as the subnet's soup
is), custom portions with `weights`, or SLERP for exactly two. Weight names line up across the common
Hugging Face prefixes (`model.`, `model.language_model.`), so text-only fine-tunes merge into a
multimodal model; weights not every model has are kept from the first. Other models' extra output
heads (`mtp.*`, `lm_head`) and vision towers are ignored when the first model has none. Tensors
stream in chunks through memory maps, so a 4B model never sits in memory whole."""

import functools
import json
import math
import operator
import re
import shutil
import struct
from pathlib import Path

import numpy as np

from . import __version__

METHODS = ("linear", "slerp")
DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}
PREFIXES = ("model.language_model.", "language_model.model.", "language_model.", "model.")
OPTIONAL = (
    "mtp.",
    "lm_head",
    "visual.",
)  # second-model weights dropped when the first model lacks them
INDEX = "model.safetensors.index.json"
LICENSES = ("LICENSE", "LICENSE.md", "LICENSE.txt")
CHUNK = 1 << 24  # elements per chunk
SLERP_PARALLEL = 0.9995  # above this cosine, SLERP falls back to linear (as mergekit does)
REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")


def resolve(source):
    """A local model folder, or a Hugging Face repo id downloaded with its weights and config only."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    if not REPO_ID.fullmatch(source):
        raise ValueError(f"{source} is not a model folder or a Hugging Face repo id.")
    from . import _hf

    return _hf.download(source)


def canonical(name):
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def header(path):
    """(byte offset of the data, {name: info}) of one safetensors file."""
    with path.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        return 8 + size, json.loads(stream.read(size))


def layout(root):
    """{tensor name: shard file name} from the index, or the single model.safetensors."""
    if (root / INDEX).is_file():
        return json.loads((root / INDEX).read_text())["weight_map"]
    if not (root / "model.safetensors").is_file():
        raise ValueError(f"{root} has no model.safetensors or {INDEX}.")
    _, info = header(root / "model.safetensors")
    return {name: "model.safetensors" for name in info if name != "__metadata__"}


def tensors(root):
    """{canonical name: (path, data offset, info, original name)}."""
    found, headers = {}, {}
    for name, shard in layout(root).items():
        if shard not in headers:
            headers[shard] = header(root / shard)
        start, info = headers[shard]
        key = canonical(name)
        if key in found:
            raise ValueError(f"{root} has two weights with the same name after alignment: {key}.")
        found[key] = (root / shard, start, info[name], name)
    return found


def values(entry, begin, end):
    """Elements [begin, end) of a tensor as float32."""
    path, start, info, _ = entry
    width = np.dtype(DTYPES[info["dtype"]]).itemsize
    first, _ = info["data_offsets"]
    raw = np.memmap(path, np.uint8, "r", start + first + begin * width, (end - begin) * width)
    data = raw.view(DTYPES[info["dtype"]])
    if info["dtype"] == "BF16":
        return (data.astype("<u4") << 16).view("<f4")
    return data.astype("<f4")


def encode(data, dtype):
    if dtype == "BF16":  # round to nearest, ties to even
        bits = data.astype("<f4").view("<u4").astype("<u8")
        return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()
    return data.astype(DTYPES[dtype]).tobytes()


def chunks(count):
    for begin in range(0, count, CHUNK):
        yield begin, min(count, begin + CHUNK)


MAX_MODELS = 5  # a soup holds at most five bodies


def slerp(a, b, count, weight):
    """(share of a, share of b) for one tensor: SLERP's shares depend on the angle between the tensors."""
    dot = norm_a = norm_b = 0.0
    for begin, end in chunks(count):
        x, y = values(a, begin, end).astype(np.float64), values(b, begin, end).astype(np.float64)
        dot, norm_a, norm_b = dot + x @ y, norm_a + x @ x, norm_b + y @ y
    if not norm_a or not norm_b:
        return [1 - weight, weight]
    cosine = min(1.0, max(-1.0, dot / math.sqrt(norm_a * norm_b)))
    if abs(cosine) > SLERP_PARALLEL:
        return [1 - weight, weight]
    angle = math.acos(cosine)
    return [
        math.sin((1 - weight) * angle) / math.sin(angle),
        math.sin(weight * angle) / math.sin(angle),
    ]


def copy_files(models, out, shards):
    first = models[0]
    for item in first.iterdir():
        if item.name.startswith(".") or item.name in shards:
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, out / item.name)
    for number, other in enumerate(models[1:], start=2):
        for name in LICENSES:
            if (other / name).is_file():
                stem, dot, suffix = name.partition(".")
                shutil.copy2(other / name, out / f"{stem}-model-{number}{dot}{suffix}")


def portions(weights, count):
    """Each model's share, normalised to sum to 1; even shares by default."""
    if weights is None:
        return [1 / count] * count
    # A zero portion is allowed: that model then only supplies the weights the others lack. With
    # weights 0,1 the first model's layout and extra tensors (a vision tower) carry the second's values.
    if (
        len(weights) != count
        or not all(isinstance(w, int | float) and math.isfinite(w) and w >= 0 for w in weights)
        or not any(w > 0 for w in weights)
    ):
        raise ValueError(
            f"Give {count} weights, one per model: 0 or more, and at least one above 0."
        )
    total = sum(weights)
    return [w / total for w in weights]


def merge(models, out, *, method="linear", weights=None, sources=None):
    """Write the merge of `models` (2-5 model folders) to the new folder `out`, in the first model's layout.
    `weights` are each model's portion (even by default). Returns {"merged", "kept", "ignored", "out"}."""
    models, out = [Path(m) for m in models], Path(out)
    if not 2 <= len(models) <= MAX_MODELS:
        raise ValueError(f"Merge 2 to {MAX_MODELS} models.")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; use one of {', '.join(METHODS)}.")
    if method == "slerp" and len(models) != 2:
        raise ValueError("SLERP merges exactly two models; use linear for three or more.")
    shares = portions(weights, len(models))
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new output folder.")
    if len({m.resolve() for m in models}) != len(models):
        raise ValueError("Two arguments name the same model; choose different models.")
    found = [tensors(m) for m in models]
    a = found[0]
    shared = set(a).intersection(*found[1:])
    if not shared:
        raise ValueError("The models share no weights; merge models of the same family.")
    ignored = set()
    for number, other in enumerate(found[1:], start=2):
        extra = set(other) - set(a)
        ignored |= {key for key in extra if key.startswith(OPTIONAL)}
        if missing := sorted(key for key in extra if not key.startswith(OPTIONAL)):
            raise ValueError(f"Model {number}'s weight {missing[0]} is not in the first model.")
    for key in sorted(shared):
        for other in found[1:]:
            if a[key][2]["shape"] != other[key][2]["shape"]:
                raise ValueError(
                    f"Weight {key} differs in shape: {a[key][2]['shape']} vs {other[key][2]['shape']}."
                )
            for info in (a[key][2], other[key][2]):
                if info["dtype"] not in DTYPES:
                    raise ValueError(
                        f"Weight {key} has dtype {info['dtype']}; merge needs BF16, F16 or F32."
                    )
    shards = set(layout(models[0]).values())
    partial = out.with_name(f".{out.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        changed = False
        by_name = {entry[3]: (key, entry) for key, entry in a.items()}
        for shard in sorted(shards):
            start, info = header(models[0] / shard)
            with (models[0] / shard).open("rb") as source, (partial / shard).open("wb") as target:
                target.write(source.read(start))  # the first model's header, offsets unchanged
                for name, item in info.items():
                    if name == "__metadata__":
                        continue
                    key, entry = by_name[name]
                    begin, end = item["data_offsets"]
                    target.seek(start + begin)
                    if key not in shared:
                        source.seek(start + begin)
                        target.write(source.read(end - begin))
                        continue
                    count = math.prod(item["shape"])
                    entries = [entry] + [other[key] for other in found[1:]]
                    factors = slerp(*entries, count, shares[1]) if method == "slerp" else shares
                    for low, high in chunks(count):
                        parts = [values(e, low, high) for e in entries]
                        changed = changed or any(not np.array_equal(parts[0], p) for p in parts[1:])
                        # Zero-portion terms are left out and the sum starts from the first real
                        # term (not sum()'s integer 0), so a lone term keeps its exact bits, -0 included.
                        terms = [
                            f * p.astype(np.float64)
                            for f, p in zip(factors, parts, strict=True)
                            if f != 0
                        ]
                        merged = functools.reduce(operator.add, terms)
                        if not np.isfinite(merged).all():
                            raise ValueError(f"Weight {key} is not finite after merging.")
                        target.write(encode(merged.astype("<f4"), item["dtype"]))
        if not changed:
            raise ValueError("The models have identical weights; there is nothing to merge.")
        copy_files(models, partial, shards)
        recipe_models = []
        for name, path in zip(sources or [str(m) for m in models], models, strict=True):
            entry = {"source": name}
            if path.parent.name == "snapshots":
                entry["revision"] = path.name
            recipe_models.append(entry)
        recipe = {
            "tool": f"zils merge {__version__}",
            "method": method,
            "weights": shares,
            "models": recipe_models,
            "merged": len(shared),
            "kept": len(a) - len(shared),
            "ignored": len(ignored),
        }
        (partial / "merge.json").write_text(json.dumps(recipe, indent=2) + "\n")
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {
        "merged": len(shared),
        "kept": len(a) - len(shared),
        "ignored": len(ignored),
        "zero": [number for number, share in enumerate(shares, start=1) if share == 0],
        "out": str(out),
    }
