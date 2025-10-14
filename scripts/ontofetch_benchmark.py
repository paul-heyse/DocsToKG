"""Simple benchmarking harness for the ontology downloader."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from DocsToKG.OntologyDownload.cli import _handle_pull
from DocsToKG.OntologyDownload.config import ResolvedConfig


def benchmark(spec_path: Path, iterations: int) -> None:
    durations = []
    args = argparse.Namespace(
        command="pull",
        ids=[],
        spec=spec_path,
        force=False,
        resolver=None,
        target_formats=None,
        json=False,
        log_level="INFO",
    )
    base_config = ResolvedConfig.from_defaults()
    for _ in range(iterations):
        start = time.perf_counter()
        _handle_pull(args, base_config)
        durations.append(time.perf_counter() - start)
    mean = statistics.mean(durations)
    median = statistics.median(durations)
    print(f"Runs: {iterations}")
    print(f"Mean: {mean:.2f}s")
    print(f"Median: {median:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ontology downloads")
    parser.add_argument("spec", type=Path, help="Path to sources.yaml")
    parser.add_argument("--iterations", type=int, default=3, help="Number of runs to average")
    args = parser.parse_args()
    benchmark(args.spec, args.iterations)


if __name__ == "__main__":  # pragma: no cover
    main()
