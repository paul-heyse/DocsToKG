from __future__ import annotations

from typing import List

import pytest

from DocsToKG.OntologyDownload.io.network import retry_with_backoff


def test_retry_with_backoff_retries_and_records_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter(
        [
            ValueError("first"),
            ValueError("second"),
            "success",
        ]
    )

    def _fn() -> str:
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result

    sleeps: List[float] = []

    def _sleep(delay: float) -> None:
        sleeps.append(delay)

    callbacks: List[tuple[int, Exception, float]] = []

    def _callback(attempt: int, exc: Exception, delay: float) -> None:
        callbacks.append((attempt, exc, delay))

    monkeypatch.setattr(
        "DocsToKG.OntologyDownload.io.network.random.uniform",
        lambda a, b: b,
    )

    result = retry_with_backoff(
        _fn,
        retryable=lambda exc: isinstance(exc, ValueError),
        max_attempts=4,
        backoff_base=0.5,
        jitter=0.5,
        callback=_callback,
        sleep=_sleep,
    )

    assert result == "success"
    assert sleeps == [1.0, 1.5]
    assert [entry[0] for entry in callbacks] == [1, 2]
    assert all(isinstance(entry[1], ValueError) for entry in callbacks)
    assert [round(entry[2], 2) for entry in callbacks] == [1.0, 1.5]


def test_retry_with_backoff_propagates_non_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: List[float] = []

    def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        "DocsToKG.OntologyDownload.io.network.random.uniform",
        lambda a, b: b,
    )

    def _fn() -> None:
        raise RuntimeError("fatal")

    with pytest.raises(RuntimeError):
        retry_with_backoff(
            _fn,
            retryable=lambda exc: isinstance(exc, ValueError),
            max_attempts=3,
            backoff_base=0.5,
            jitter=0.5,
            sleep=_sleep,
        )

    assert sleeps == []


def test_retry_with_backoff_honours_retry_after_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter(
        [
            ValueError("first"),
            ValueError("second"),
            "done",
        ]
    )

    def _fn() -> str:
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result

    hints = iter([3.2, 1.4])

    def _retry_after(exc: Exception) -> float:
        try:
            return next(hints)
        except StopIteration:
            return 0.8

    sleeps: List[float] = []

    def _sleep(delay: float) -> None:
        sleeps.append(delay)

    callbacks: List[float] = []

    def _callback(attempt: int, exc: Exception, delay: float) -> None:
        callbacks.append(delay)

    monkeypatch.setattr(
        "DocsToKG.OntologyDownload.io.network.random.uniform",
        lambda a, b: 0.0,
    )

    result = retry_with_backoff(
        _fn,
        retryable=lambda exc: isinstance(exc, ValueError),
        max_attempts=4,
        backoff_base=0.5,
        jitter=0.25,
        retry_after=_retry_after,
        callback=_callback,
        sleep=_sleep,
    )

    assert result == "done"
    assert sleeps == [3.2, 1.4]
    assert callbacks == [3.2, 1.4]
