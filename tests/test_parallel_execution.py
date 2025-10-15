from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("requests")

from DocsToKG.ContentDownload.resolvers import ResolverConfig, ResolverPipeline


class _NullLogger:
    def log(self, record):  # pragma: no cover - no-op sink
        pass


def test_rate_limiting_with_parallel_workers():
    config = ResolverConfig()
    config.resolver_min_interval_s = {"test": 0.1}
    pipeline = ResolverPipeline([], config, lambda *args, **kwargs: None, _NullLogger())

    # Establish initial timestamp so subsequent calls respect the interval.
    pipeline._respect_rate_limit("test")

    def call_respect_limit() -> float:
        pipeline._respect_rate_limit("test")
        return time.monotonic()

    with ThreadPoolExecutor(max_workers=4) as executor:
        timestamps = list(executor.map(lambda _: call_respect_limit(), range(4)))

    timestamps.sort()
    for first, second in zip(timestamps, timestamps[1:]):
        assert second - first >= 0.09
