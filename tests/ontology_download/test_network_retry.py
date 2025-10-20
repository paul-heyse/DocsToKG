from __future__ import annotations

from typing import List

import httpx

import pytest

from DocsToKG.OntologyDownload.errors import DownloadFailure
from DocsToKG.OntologyDownload.io.network import is_retryable_error, retry_with_backoff


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


@pytest.mark.parametrize(
    "exception",
    [
        httpx.ConnectError("connect", request=httpx.Request("GET", "https://example.org/")),
        httpx.ReadTimeout("read timeout", request=httpx.Request("GET", "https://example.org/")),
        httpx.WriteTimeout("write timeout", request=httpx.Request("GET", "https://example.org/")),
        httpx.PoolTimeout("pool timeout"),
        httpx.TransportError("generic transport"),
    ],
)
def test_is_retryable_error_handles_httpx_transport_exceptions(exception):
    """Transport-level httpx exceptions should be treated as retryable."""

    assert is_retryable_error(exception) is True


def test_is_retryable_error_handles_http_status_error():
    """HTTPStatusError should be retryable only for retryable status codes."""

    request = httpx.Request("GET", "https://example.org/")
    retryable_response = httpx.Response(503, request=request)
    non_retryable_response = httpx.Response(400, request=request)

    retryable_exc = httpx.HTTPStatusError("503", request=request, response=retryable_response)
    non_retryable_exc = httpx.HTTPStatusError("400", request=request, response=non_retryable_response)

    assert is_retryable_error(retryable_exc) is True
    assert is_retryable_error(non_retryable_exc) is False


def test_is_retryable_error_preserves_download_failure_retry_flag():
    """DownloadFailure.retryable should be surfaced."""

    retryable = DownloadFailure("retryable", retryable=True)
    non_retryable = DownloadFailure("non-retryable", retryable=False)

    assert is_retryable_error(retryable) is True
    assert is_retryable_error(non_retryable) is False


def test_is_retryable_error_non_retryable_exception():
    """Unrelated exceptions should be treated as non-retryable."""

    assert is_retryable_error(RuntimeError("boom")) is False
