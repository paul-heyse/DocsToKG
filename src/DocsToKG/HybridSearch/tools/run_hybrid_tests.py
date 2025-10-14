#!/usr/bin/env python3
"""
Helper script for running hybrid search regression suites.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hybrid search pytest suites")
    parser.add_argument(
        "mode",
        choices=("synthetic", "real", "scale", "all"),
        help="Select which hybrid search tests to run",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest",
    )
    args = parser.parse_args(argv)

    if args.mode == "synthetic":
        pytest_args = []
    elif args.mode == "real":
        pytest_args = ["--real-vectors", "-m", "real_vectors and not scale_vectors"]
    elif args.mode == "scale":
        pytest_args = ["--real-vectors", "--scale-vectors", "-m", "scale_vectors"]
    else:  # all
        pytest_args = ["--real-vectors", "--scale-vectors"]

    command = [sys.executable, "-m", "pytest", *pytest_args, *args.pytest_args]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
