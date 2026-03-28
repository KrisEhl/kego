from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("ls", help="List and compare experiments")
    p.set_defaults(func=_ls)


def _ls(args: argparse.Namespace, extra_args: list[str]) -> int:
    print("ls not yet implemented")
    return 0
