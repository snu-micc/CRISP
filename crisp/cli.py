"""Command-line interface for the public CRISP release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def _compile(args: argparse.Namespace) -> None:
    from .workflow import compile_catalog

    summary = compile_catalog(args.config, args.output)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crisp",
        description="Compile chemical rules into candidate executable descriptors.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile", help="Run or resume the dataset-blind OpenAI API compilation workflow."
    )
    compile_parser.add_argument("--config", required=True, type=Path)
    compile_parser.add_argument("--output", required=True, type=Path)
    compile_parser.set_defaults(function=_compile)

    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    args.function(args)
