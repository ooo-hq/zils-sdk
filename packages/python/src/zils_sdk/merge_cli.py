"""`zils merge`: merge 2-5 models of the same family on this machine."""

import argparse
import subprocess
import sys
from pathlib import Path


def weight_list(text):
    try:
        return [float(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Give weights as numbers separated by commas, e.g. 1,1,2."
        ) from None


def add_parser(subparsers):
    command = subparsers.add_parser(
        "merge", help="Merge 2-5 models of the same family into a new model folder (a soup)"
    )
    command.add_argument(
        "models",
        nargs="+",
        metavar="MODEL",
        help="2-5 folders or Hugging Face repo ids; the first model's files are kept",
    )
    command.add_argument("--out", type=Path, required=True, help="New folder for the merged model")
    command.add_argument(
        "--method",
        choices=("linear", "slerp"),
        default="linear",
        help="Default linear; SLERP needs exactly two models",
    )
    portion = command.add_mutually_exclusive_group()
    portion.add_argument(
        "--weights",
        type=weight_list,
        help="Each model's portion in order, e.g. 1,1,2 (default even); 0,1 gives the second"
        " model's values in the first model's layout",
    )
    portion.add_argument(
        "--weight", type=float, help="With two models: the second model's share, 0-1"
    )
    command.add_argument(
        "--pool",
        help="Check the result as a soup body for this pool (e.g. d3-lite, d3-mini) with the"
        " Zils miner's body check, zils-miner",
    )


def body_check(out, pool):
    """Run the miner's `commit-body --check` on `out`; 0 when it passes or can't run here."""
    from .cli import MINER_INSTALL, miner_path

    later = f"zils commit-body --pool {pool} --path {out} --check"
    found = miner_path()
    if found is None:
        print(
            f"To check it as a {pool} soup body, install the Zils miner ({MINER_INSTALL})"
            f" and run:\n  {later}",
            file=sys.stderr,
        )
        return 0
    usage = subprocess.run(
        [found, "commit-body", "--help"], capture_output=True, text=True, timeout=60
    )
    if "--check" not in usage.stdout:
        print(
            "Your zils-miner is older than the soup body check. Update it with zils-miner update,"
            f" then run:\n  {later}",
            file=sys.stderr,
        )
        return 1
    code = subprocess.run(
        [found, "commit-body", "--pool", pool, "--path", str(out), "--check"]
    ).returncode
    if code:
        print(
            f"{out} holds the merge, but it is not a {pool} soup body yet; see the reason above.",
            file=sys.stderr,
        )
    return code


def run(args):
    try:
        from . import merge
    except ImportError:
        version = "%d.%d" % sys.version_info[:2]
        raise ValueError(
            f"zils merge needs numpy, which isn't installed for Python {version}."
            " From source: pip install 'zils-sdk[merge]'. With the npm CLI, set ZILS_PYTHON to"
            " another Python 3.11+ executable."
        ) from None
    weights = args.weights
    if args.weight is not None:
        if len(args.models) != 2:
            raise ValueError(
                "--weight is for two models; use --weights with one portion per model."
            )
        if not 0 < args.weight < 1:
            raise ValueError(
                "--weight must be between 0 and 1 (exclusive): the second model's share."
            )
        weights = [1 - args.weight, args.weight]
    if weights is not None and len(weights) != len(args.models):
        raise ValueError(f"Give {len(args.models)} weights, one per model.")
    if not 2 <= len(args.models) <= merge.MAX_MODELS:
        raise ValueError(f"Merge 2 to {merge.MAX_MODELS} models.")
    folders = [merge.resolve(model) for model in args.models]
    result = merge.merge(
        folders, args.out, method=args.method, weights=weights, sources=args.models
    )
    print(
        f"Merged {result['merged']} tensors from {len(folders)} models ({args.method}); "
        f"kept {result['kept']} from the first model only"
        + (f", ignored {result['ignored']} extras" if result["ignored"] else "")
        + f". Wrote {result['out']}",
        flush=True,
    )
    for number in result["zero"]:
        print(
            "Model 1 has a zero portion: it only supplies the weights the others lack"
            " (its layout and files, and tensors such as a vision tower)."
            if number == 1
            else f"Model {number} has a zero portion and adds nothing.",
            flush=True,
        )
    if args.pool and (code := body_check(result["out"], args.pool)):
        raise SystemExit(code)
