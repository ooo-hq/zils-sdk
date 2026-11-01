"""Merge two models of the same family into a new model folder.

The output keeps the first model's files and safetensors layout byte for byte, except that each
weight both models share is replaced by the merge. Weight names line up across the common Hugging
Face prefixes (`model.`, `model.language_model.`), so a text-only fine-tune merges into a multimodal
parent; weights only the first model has are kept unchanged. The second model's extra output heads
(`mtp.*`, `lm_head`) and vision tower are ignored when the first model has none, since the output keeps
the first model's architecture. Tensors stream in chunks through memory maps, so a 4B model never sits
in memory whole."""

import json
import math
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
    from huggingface_hub import snapshot_download

    ignore = ["*.gguf", "*.bin", "*.pt", "*.pth", "*.onnx", "*.h5", "*.msgpack", "*.ckpt"]
    return Path(snapshot_download(source, ignore_patterns=ignore))


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


def coefficients(a, b, count, method, weight):
    """(share of a, share of b) for this tensor; SLERP's depend on the angle between the tensors."""
    if method == "linear":
        return 1 - weight, weight
    dot = norm_a = norm_b = 0.0
    for begin, end in chunks(count):
        x, y = values(a, begin, end).astype(np.float64), values(b, begin, end).astype(np.float64)
        dot, norm_a, norm_b = dot + x @ y, norm_a + x @ x, norm_b + y @ y
    if not norm_a or not norm_b:
        return 1 - weight, weight
    cosine = min(1.0, max(-1.0, dot / math.sqrt(norm_a * norm_b)))
    if abs(cosine) > SLERP_PARALLEL:
        return 1 - weight, weight
    angle = math.acos(cosine)
    return math.sin((1 - weight) * angle) / math.sin(angle), math.sin(weight * angle) / math.sin(
        angle
    )


def copy_files(first, second, out, shards):
    for item in first.iterdir():
        if item.name.startswith(".") or item.name in shards:
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, out / item.name)
    for name in LICENSES:
        if (second / name).is_file():
            stem, dot, suffix = name.partition(".")
            shutil.copy2(second / name, out / f"{stem}-second-model{dot}{suffix}")


def merge(first, second, out, *, method="linear", weight=0.5, sources=None):
    """Write the merge of `first` and `second` (model folders) to the new folder `out`.
    `weight` is the second model's share. Returns {"merged": n, "kept": n, "ignored": n, "out": path}."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; use one of {', '.join(METHODS)}.")
    if not (isinstance(weight, int | float) and 0 < weight < 1):
        raise ValueError("Weight must be between 0 and 1 (exclusive): the second model's share.")
    first, second, out = Path(first), Path(second), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new output folder.")
    if first.resolve() == second.resolve():
        raise ValueError("Both arguments name the same model; choose two different models.")
    a, b = tensors(first), tensors(second)
    shared = set(a) & set(b)
    if not shared:
        raise ValueError("The two models share no weights; merge models of the same family.")
    ignored = sorted(key for key in set(b) - set(a) if key.startswith(OPTIONAL))
    if extra := sorted(set(b) - set(a) - set(ignored)):
        raise ValueError(f"The second model's weight {extra[0]} is not in the first model.")
    for key in sorted(shared):
        info_a, info_b = a[key][2], b[key][2]
        if info_a["shape"] != info_b["shape"]:
            raise ValueError(
                f"Weight {key} differs in shape: {info_a['shape']} vs {info_b['shape']}."
            )
        if info_a["dtype"] not in DTYPES or info_b["dtype"] not in DTYPES:
            raise ValueError(
                f"Weight {key} has dtype {info_b['dtype']}; merge needs BF16, F16 or F32."
            )
    shards = set(layout(first).values())
    partial = out.with_name(f".{out.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        changed = False
        by_name = {entry[3]: (key, entry) for key, entry in a.items()}
        for shard in sorted(shards):
            start, info = header(first / shard)
            with (first / shard).open("rb") as source, (partial / shard).open("wb") as target:
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
                    share_a, share_b = coefficients(entry, b[key], count, method, weight)
                    for low, high in chunks(count):
                        x, y = values(entry, low, high), values(b[key], low, high)
                        changed = changed or not np.array_equal(x, y)
                        merged = share_a * x.astype(np.float64) + share_b * y.astype(np.float64)
                        if not np.isfinite(merged).all():
                            raise ValueError(f"Weight {key} is not finite after merging.")
                        target.write(encode(merged.astype("<f4"), item["dtype"]))
        if not changed:
            raise ValueError("The two models have identical weights; there is nothing to merge.")
        copy_files(first, second, partial, shards)
        names = sources or (str(first), str(second))
        models = []
        for name, path in zip(names, (first, second), strict=True):
            entry = {"source": name}
            if path.parent.name == "snapshots":
                entry["revision"] = path.name
            models.append(entry)
        recipe = {
            "tool": f"zils merge {__version__}",
            "method": method,
            "weight": weight,
            "models": models,
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
        "out": str(out),
    }
