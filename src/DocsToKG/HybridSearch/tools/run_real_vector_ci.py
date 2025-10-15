#!/usr/bin/env python3
"""
CI helper to execute real-vector regression tests and collect validation artifacts.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def clean_directory(path: Path) -> None:
    """Remove and recreate a directory to ensure a clean workspace.

    Args:
        path: Directory path to reset.

    Returns:
        None
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Entry point for executing real-vector CI regression suites.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Process exit code indicating success (`0`) or failure.
    """
    parser = argparse.ArgumentParser(description="Run real-vector regression suite for CI")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/real_vectors"),
        help="Directory to store validation reports",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    clean_directory(output_dir)
    os.environ["REAL_VECTOR_REPORT_DIR"] = str(output_dir)

    extra_args = list(args.pytest_args or [])

    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "--real-vectors",
            "tests/test_hybrid_search_real_vectors.py",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "--real-vectors",
            "--scale-vectors",
            "tests/test_hybrid_search_scale.py",
        ],
    ]

    for command in commands:
        result = subprocess.call([*command, *extra_args])
        if result != 0:
            print(
                f"Real-vector regression suite failed. Check artifacts in {output_dir}",
                file=sys.stderr,
            )
            return result

    print(f"Real-vector validation artifacts stored in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
