from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, call, patch

import pytest

try:  # pragma: no cover - dependency optional in CI
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - skip if requests missing
    requests = pytest.importorskip("requests")  # type: ignore

try:
    import hypothesis
    from hypothesis import strategies as st  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis is required for these tests", allow_module_level=True)

from DocsToKG.ContentDownload.network import (
    parse_retry_after_header,
    request_with_retries,
)

given = hypothesis.given


def _mock_response(status: int, headers: Optional[Dict[str, str]] = None) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    return response


def test_successful_request_no_retries():
    """Verify successful request completes immediately without retries."""

    session = Mock(spec=requests.Session)
    response = _mock_response(200)
    session.request.return_value = response

    result = request_with_retries(session, "GET", "https://example.org/test")

    assert result is response
    session.request.assert_called_once_with(method="GET", url="https://example.org/test")


@patch("DocsToKG.ContentDownload.network.random.random", return_value=0.0)
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_transient_503_with_exponential_backoff(mock_sleep: Mock, _: Mock) -> None:
    """Verify exponential backoff timing for transient 503 errors."""

    session = Mock(spec=requests.Session)
    response_503 = _mock_response(503, headers={})
    response_200 = _mock_response(200)
    session.request.side_effect = [response_503, response_503, response_200]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        max_retries=3,
        backoff_factor=0.5,
    )

    assert result is response_200
    assert session.request.call_count == 3
    assert mock_sleep.call_args_list == [call(0.5), call(1.0)]


def test_parse_retry_after_header_integer() -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "5"}

    assert parse_retry_after_header(response) == 5.0


def test_parse_retry_after_header_http_date() -> None:
    future = datetime.now(timezone.utc) + timedelta(seconds=30)
    header_value = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    response = requests.Response()
    response.headers = {"Retry-After": header_value}

    wait = parse_retry_after_header(response)
    assert wait is not None
    assert 0.0 <= wait <= 30.0


def test_parse_retry_after_header_invalid_date() -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Thu, 32 Foo 2024 00:00:00 GMT"}

    assert parse_retry_after_header(response) is None


@patch("DocsToKG.ContentDownload.network.random.random", return_value=0.0)
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_retry_after_header_overrides_backoff(mock_sleep: Mock, _: Mock) -> None:
    session = Mock(spec=requests.Session)
    retry_headers = {"Retry-After": "10"}
    response_retry = _mock_response(429, headers=retry_headers)
    response_success = _mock_response(200)
    session.request.side_effect = [response_retry, response_success]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        backoff_factor=0.1,
        max_retries=2,
    )

    assert result is response_success
    assert mock_sleep.call_args_list == [call(10.0)]


@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_request_exception_raises_after_retries(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    error = requests.RequestException("boom")
    session.request.side_effect = error

    with pytest.raises(requests.RequestException):
        request_with_retries(session, "GET", "https://example.org/test", max_retries=1)

    assert mock_sleep.call_count == 1
    assert session.request.call_count == 2


@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_timeout_retry_handling(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    session.request.side_effect = [requests.Timeout("slow"), _mock_response(200)]

    result = request_with_retries(session, "GET", "https://example.org/timeout", max_retries=1)

    assert result.status_code == 200
    assert mock_sleep.call_count == 1


@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_connection_error_retry_handling(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    session.request.side_effect = [requests.ConnectionError("down"), _mock_response(200)]

    result = request_with_retries(session, "GET", "https://example.org/conn", max_retries=1)

    assert result.status_code == 200
    assert mock_sleep.call_count == 1


@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_timeout_raises_after_exhaustion(mock_sleep: Mock) -> None:
    """Ensure timeout retries raise after exhausting the retry budget."""

    session = Mock(spec=requests.Session)
    session.request.side_effect = requests.Timeout("slow")

    with pytest.raises(requests.Timeout):
        request_with_retries(session, "GET", "https://example.org/timeout", max_retries=1)

    # Only the non-terminal attempt sleeps before re-raising on the final attempt.
    assert mock_sleep.call_count == 1


@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_connection_error_raises_after_exhaustion(mock_sleep: Mock) -> None:
    """Ensure connection errors propagate when retries are exhausted."""

    session = Mock(spec=requests.Session)
    session.request.side_effect = requests.ConnectionError("down")

    with pytest.raises(requests.ConnectionError):
        request_with_retries(session, "GET", "https://example.org/conn", max_retries=1)

    assert mock_sleep.call_count == 1


@given(st.text())
def test_parse_retry_after_header_property(value: str) -> None:
    response = requests.Response()
    response.headers = {"Retry-After": value}

    result = parse_retry_after_header(response)

    if result is not None:
        assert result >= 0.0 or math.isnan(result)


def test_request_with_custom_retry_statuses() -> None:
    session = Mock(spec=requests.Session)
    failing = _mock_response(404)
    success = _mock_response(200)
    session.request.side_effect = [failing, success]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        retry_statuses={404},
        max_retries=1,
    )

    assert result is success
    assert session.request.call_count == 2


def test_request_returns_after_exhausting_single_attempt() -> None:
    session = Mock(spec=requests.Session)
    retry_response = _mock_response(503)
    session.request.return_value = retry_response

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        max_retries=0,
    )

    assert result is retry_response


def test_request_with_retries_rejects_negative_retries() -> None:
    session = Mock(spec=requests.Session)

    with pytest.raises(ValueError):
        request_with_retries(session, "GET", "https://example.org/test", max_retries=-1)


def test_request_with_retries_rejects_negative_backoff() -> None:
    session = Mock(spec=requests.Session)

    with pytest.raises(ValueError):
        request_with_retries(session, "GET", "https://example.org/test", backoff_factor=-0.1)


def test_request_with_retries_requires_method_and_url() -> None:
    session = Mock(spec=requests.Session)

    with pytest.raises(ValueError):
        request_with_retries(session, "", "https://example.org/test")

    with pytest.raises(ValueError):
        request_with_retries(session, "GET", "")


def test_request_with_retries_uses_method_fallback() -> None:
    class _MinimalSession:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def get(self, url: str, **kwargs: Any):  # noqa: D401
            self.calls.append(url)
            response = Mock(spec=requests.Response)
            response.status_code = 200
            response.headers = {}
            return response

    session = _MinimalSession()

    response = request_with_retries(session, "GET", "https://example.org/fallback")

    assert response.status_code == 200
    assert session.calls == ["https://example.org/fallback"]


def test_request_with_retries_errors_when_no_callable_available() -> None:
    class _MinimalSession:
        pass

    with pytest.raises(AttributeError):
        request_with_retries(_MinimalSession(), "GET", "https://example.org/fail")


@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_retry_after_header_prefers_longer_delay(mock_sleep: Mock) -> None:
    """Verify Retry-After header longer than backoff takes precedence."""

    session = Mock(spec=requests.Session)

    retry_response = requests.Response()
    retry_response.status_code = 429
    retry_response.headers = {"Retry-After": "4"}

    success_response = requests.Response()
    success_response.status_code = 200
    success_response.headers = {}

    session.request.side_effect = [retry_response, success_response]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/with-retry-after",
        backoff_factor=0.1,
        max_retries=2,
    )

    assert result.status_code == 200
    mock_sleep.assert_called_once()
    sleep_arg = mock_sleep.call_args[0][0]
    assert pytest.approx(sleep_arg, rel=0.01) == 4.0


@patch("DocsToKG.ContentDownload.network.time.sleep")
@patch("DocsToKG.ContentDownload.network.parse_retry_after_header")
def test_respect_retry_after_false_skips_header(mock_parse: Mock, mock_sleep: Mock) -> None:
    """Ensure disabling respect_retry_after bypasses header parsing."""

    session = Mock(spec=requests.Session)
    retry_response = _mock_response(503)
    success_response = _mock_response(200)
    session.request.side_effect = [retry_response, success_response]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/no-retry-after",
        respect_retry_after=False,
        max_retries=1,
        backoff_factor=0.1,
    )

    assert result is success_response
    mock_parse.assert_not_called()
    mock_sleep.assert_called_once()


def test_parse_retry_after_header_naive_datetime() -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Mon, 01 Jan 2024 00:00:00"}

    wait = parse_retry_after_header(response)
    assert wait is not None
    assert wait >= 0.0


def test_parse_retry_after_header_handles_parse_errors(monkeypatch) -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Mon, 01 Jan 2024 00:00:00 GMT"}

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.parsedate_to_datetime",
        Mock(side_effect=TypeError("boom")),
    )

    assert parse_retry_after_header(response) is None


def test_parse_retry_after_header_returns_none_when_parser_returns_none(monkeypatch) -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Mon, 01 Jan 2024 00:00:00 GMT"}

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.parsedate_to_datetime",
        Mock(return_value=None),
    )

    assert parse_retry_after_header(response) is None
