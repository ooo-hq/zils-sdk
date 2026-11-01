"""`zils merge`: merge two models of the same family on this machine."""

import sys
from pathlib import Path


def add_parser(subparsers):
    command = subparsers.add_parser(
        "merge", help="Merge two models of the same family into a new model folder"
    )
    command.add_argument("first", help="First model: a folder or Hugging Face repo id")
    command.add_argument("second", help="Second model: a folder or Hugging Face repo id")
    command.add_argument("--out", type=Path, required=True, help="New folder for the merged model")
    command.add_argument(
        "--method", choices=("linear", "slerp"), default="linear", help="Default linear"
    )
    command.add_argument(
        "--weight", type=float, default=0.5, help="The second model's share, 0–1 (default 0.5)"
    )


def run(args):
    try:
        from . import merge
    except ImportError:
        version = "%d.%d" % sys.version_info[:2]
        raise ValueError(
            f"zils merge needs numpy and huggingface_hub, which aren't installed for Python {version}."
            " From source: pip install 'zils-sdk[merge]'. With the npm CLI, set ZILS_PYTHON to a"
            " Python 3.11–3.14 executable."
        ) from None
    first, second = merge.resolve(args.first), merge.resolve(args.second)
    result = merge.merge(
        first,
        second,
        args.out,
        method=args.method,
        weight=args.weight,
        sources=(args.first, args.second),
    )
    print(
        f"Merged {result['merged']} tensors ({args.method}, weight {args.weight}); "
        f"kept {result['kept']} from the first model only"
        + (f", ignored {result['ignored']} extra in the second" if result["ignored"] else "")
        + f". Wrote {result['out']}"
    )
