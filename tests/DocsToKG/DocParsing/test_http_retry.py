import importlib.util
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "DocsToKG"
    / "DocParsing"
    / "core"
    / "http.py"
)

spec = importlib.util.spec_from_file_location("docparsing_http", MODULE_PATH)
assert spec is not None and spec.loader is not None
http = importlib.util.module_from_spec(spec)
spec.loader.exec_module(http)


@pytest.fixture(autouse=True)
def reset_http_singleton():
    if http._HTTP_SESSION is not None:
        http._HTTP_SESSION.close()
    http._HTTP_SESSION = None
    http._HTTP_SESSION_TIMEOUT = http.DEFAULT_HTTP_TIMEOUT
    http._HTTP_SESSION_CONFIG = None
    yield
    if http._HTTP_SESSION is not None:
        http._HTTP_SESSION.close()
    http._HTTP_SESSION = None
    http._HTTP_SESSION_TIMEOUT = http.DEFAULT_HTTP_TIMEOUT
    http._HTTP_SESSION_CONFIG = None


def test_get_http_session_reuses_global_client():
    session_one, timeout_one = http.get_http_session()
    session_two, timeout_two = http.get_http_session()

    assert session_one is session_two
    assert timeout_one == timeout_two == http.DEFAULT_HTTP_TIMEOUT


def test_get_http_session_with_headers_isolates_clone():
    session_with_header, _ = http.get_http_session(base_headers={"X-Test": "1"})
    session_default, _ = http.get_http_session()

    try:
        assert session_with_header.headers["X-Test"] == "1"
        assert "X-Test" not in session_default.headers
    finally:
        if session_with_header is not session_default:
            session_with_header.close()


def test_get_http_session_retry_override_creates_dedicated_client():
    base_session, _ = http.get_http_session()
    override = http.RetryOverrides(retry_total=1, retry_backoff=0.1, status_forcelist=(418, 503))
    override_session, _ = http.get_http_session(retry_override=override)

    try:
        assert override_session is not base_session
        assert override_session._retry_total == 1
        assert pytest.approx(override_session._retry_backoff) == 0.1
        assert override_session._status_forcelist == {418, 503}
        assert base_session._retry_total == 5
    finally:
        if override_session is not base_session:
            override_session.close()


def test_request_with_retries_respects_override_policy():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(len(attempts))
        return httpx.Response(status_code=500, request=request)

    transport = httpx.MockTransport(handler)
    session, _ = http.get_http_session()
    session._transport = transport

    response = http.request_with_retries(
        "GET",
        "https://example.com/override",
        session=session,
        retry_override=http.RetryOverrides(retry_total=1),
    )
    try:
        assert response.status_code == 500
        assert len(attempts) == 2
    finally:
        response.close()

    attempts.clear()
    response_default = http.request_with_retries(
        "GET",
        "https://example.com/default",
        session=session,
    )
    try:
        assert response_default.status_code == 500
        assert len(attempts) == session._retry_total + 1
    finally:
        response_default.close()
