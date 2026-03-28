import argparse
import sys


def main() -> None:
    from kego.cli.commands import cancel as cancel_cmd
    from kego.cli.commands import logs as logs_cmd
    from kego.cli.commands import ls as ls_cmd
    from kego.cli.commands import run as run_cmd

    parser = argparse.ArgumentParser(
        prog="kego", description="kego ML experiment engine"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    run_cmd.add_parser(subparsers)
    ls_cmd.add_parser(subparsers)
    logs_cmd.add_parser(subparsers)
    cancel_cmd.add_parser(subparsers)

    args, extra = parser.parse_known_args()
    sys.exit(args.func(args, extra) or 0)


if __name__ == "__main__":
    main()
