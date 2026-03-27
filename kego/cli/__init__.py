import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kego", description="kego ML experiment engine"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    # Commands registered here in later tasks
    parser.parse_args(["--help"])


if __name__ == "__main__":
    main()
