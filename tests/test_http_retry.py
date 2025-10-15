from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from unittest.mock import Mock, call, patch

import pytest
import requests

from DocsToKG.ContentDownload.http import (
    parse_retry_after_header,
    request_with_retries,
)


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


@patch("DocsToKG.ContentDownload.http.random.random", return_value=0.0)
@patch("DocsToKG.ContentDownload.http.time.sleep")
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


@patch("DocsToKG.ContentDownload.http.random.random", return_value=0.0)
@patch("DocsToKG.ContentDownload.http.time.sleep")
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


@patch("DocsToKG.ContentDownload.http.time.sleep")
def test_request_exception_raises_after_retries(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    error = requests.RequestException("boom")
    session.request.side_effect = error

    with pytest.raises(requests.RequestException):
        request_with_retries(session, "GET", "https://example.org/test", max_retries=1)

    assert mock_sleep.call_count == 1
    assert session.request.call_count == 2
