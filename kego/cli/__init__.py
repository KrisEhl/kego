import argparse
import os
import sys

# MLflow's default HTTP timeout is 120s. Fail fast so CLI stays responsive.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


def main() -> None:
    from kego.cli.commands import cancel as cancel_cmd
    from kego.cli.commands import cluster as cluster_cmd
    from kego.cli.commands import kernel_list as kernel_list_cmd
    from kego.cli.commands import kernel_status as kernel_status_cmd
    from kego.cli.commands import logs as logs_cmd
    from kego.cli.commands import ls as ls_cmd
    from kego.cli.commands import push as push_cmd
    from kego.cli.commands import run as run_cmd
    from kego.cli.commands import submit as submit_cmd

    parser = argparse.ArgumentParser(
        prog="kego", description="kego ML experiment engine"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    run_cmd.add_parser(subparsers)
    ls_cmd.add_parser(subparsers)
    cluster_cmd.add_parser(subparsers)
    logs_cmd.add_parser(subparsers)
    cancel_cmd.add_parser(subparsers)
    push_cmd.add_parser(subparsers)
    submit_cmd.add_parser(subparsers)
    kernel_list_cmd.add_parser(subparsers)
    kernel_status_cmd.add_parser(subparsers)

    args, extra = parser.parse_known_args()
    sys.exit(args.func(args, extra) or 0)


if __name__ == "__main__":
    main()
