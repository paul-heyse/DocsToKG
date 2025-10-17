# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_network_unit",
#   "purpose": "Pytest coverage for content download network unit scenarios",
#   "sections": [
#     {
#       "id": "dummyresponse",
#       "name": "_DummyResponse",
#       "anchor": "class-dummyresponse",
#       "kind": "class"
#     },
#     {
#       "id": "session-for-response",
#       "name": "_session_for_response",
#       "anchor": "function-session-for-response",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-precheck-accepts-pdf-content",
#       "name": "test_head_precheck_accepts_pdf_content",
#       "anchor": "function-test-head-precheck-accepts-pdf-content",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-precheck-rejects-html-payload",
#       "name": "test_head_precheck_rejects_html_payload",
#       "anchor": "function-test-head-precheck-rejects-html-payload",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-precheck-degrades-to-get",
#       "name": "test_head_precheck_degrades_to_get",
#       "anchor": "function-test-head-precheck-degrades-to-get",
#       "kind": "function"
#     },
#     {
#       "id": "test-conditional-request-helper-requires-complete-metadata",
#       "name": "test_conditional_request_helper_requires_complete_metadata",
#       "anchor": "function-test-conditional-request-helper-requires-complete-metadata",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

from __future__ import annotations

from typing import Dict
from unittest.mock import Mock, patch

import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

from DocsToKG.ContentDownload.network import (
    CircuitBreaker,
    ConditionalRequestHelper,
    ContentPolicyViolation,
    TokenBucket,
    head_precheck,
    request_with_retries,
)


class _DummyResponse:
    def __init__(self, status_code: int, headers: Dict[str, str]):
        self.status_code = status_code
        self.headers = headers
        self.closed = False

    def close(self) -> None:  # noqa: D401
        self.closed = True


# --- Helper Functions ---


def _session_for_response(response: _DummyResponse, *, method: str = "HEAD") -> Mock:
    session = Mock()
    session_request = Mock(return_value=response)
    setattr(session, "request", session_request)

    def _request_with_retries(_session, _method, url, **kwargs):
        assert _method == method
        return response

    return session, _request_with_retries


# --- Test Cases ---


def test_head_precheck_accepts_pdf_content(monkeypatch):
    response = _DummyResponse(200, {"Content-Type": "application/pdf"})
    session, helper = _session_for_response(response)
    monkeypatch.setattr("DocsToKG.ContentDownload.network.request_with_retries", helper)

    assert head_precheck(session, "https://example.org/file.pdf", timeout=10.0)
    assert response.closed


def test_head_precheck_rejects_html_payload(monkeypatch):
    response = _DummyResponse(200, {"Content-Type": "text/html"})
    session, helper = _session_for_response(response)
    monkeypatch.setattr("DocsToKG.ContentDownload.network.request_with_retries", helper)

    assert not head_precheck(session, "https://example.org/page", timeout=2.0)


def test_head_precheck_degrades_to_get(monkeypatch):
    head_response = _DummyResponse(405, {})
    get_response = _DummyResponse(200, {"Content-Type": "application/pdf"})

    def _request_with_retries(_session, method, url, **kwargs):
        if method == "HEAD":
            return head_response
        assert method == "GET"

        class _Stream:
            status_code = get_response.status_code
            headers = get_response.headers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def iter_content(self, chunk_size=1024):  # pragma: no cover - first chunk only
                yield b"%PDF"

            def close(self):
                return None

        return _Stream()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.request_with_retries", _request_with_retries
    )

    session = Mock()
    assert head_precheck(session, "https://example.org/pdf", timeout=3.0)


def test_head_precheck_blocks_disallowed_mime():
    session = Mock()
    response = _DummyResponse(200, {"Content-Type": "application/pdf"})
    session.request = Mock(return_value=response)

    policy = {"allowed_types": ("text/html",)}

    assert not head_precheck(
        session,
        "https://example.org/file.pdf",
        timeout=2.0,
        content_policy=policy,
    )
    assert response.closed


def test_conditional_request_helper_requires_complete_metadata(caplog):
    helper = ConditionalRequestHelper(
        prior_etag='"etag"',
        prior_last_modified="Wed, 01 May 2024 00:00:00 GMT",
        prior_sha256=None,
        prior_content_length=None,
        prior_path=None,
    )

    with caplog.at_level("WARNING"):
        headers = helper.build_headers()

    assert headers == {}
    assert any("resume-metadata-incomplete" in rec.message for rec in caplog.records)


def test_token_bucket_enforces_capacity():
    current = [0.0]

    def clock():
        return current[0]

    bucket = TokenBucket(rate_per_second=1.0, capacity=1.0, clock=clock)
    assert bucket.acquire() == 0.0
    wait = bucket.acquire()
    assert wait > 0.0
    current[0] += wait
    assert bucket.acquire() == 0.0


def test_circuit_breaker_open_and_cooldown(monkeypatch):
    current = [0.0]

    def fake_monotonic():
        return current[0]

    monkeypatch.setattr("DocsToKG.ContentDownload.network.time.monotonic", fake_monotonic)
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=5.0)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.allow()
    breaker.record_failure()
    assert not breaker.allow()
    remaining = breaker.cooldown_remaining()
    assert remaining == 5.0
    current[0] += 5.0
    assert breaker.allow()
    breaker.record_success()
    assert breaker.allow()


def test_request_with_retries_enforces_max_bytes_policy():
    session = Mock()
    response = _DummyResponse(200, {"Content-Length": "2048"})
    session.request = Mock(return_value=response)

    with pytest.raises(ContentPolicyViolation) as excinfo:
        request_with_retries(
            session,
            "GET",
            "https://example.org/file.pdf",
            content_policy={"max_bytes": 1024},
        )

    assert response.closed
    assert excinfo.value.violation == "max-bytes"


@st.composite
def _token_bucket_scenarios(draw):
    rate = draw(
        st.floats(
            min_value=0.5,
            max_value=5.0,
            allow_infinity=False,
            allow_nan=False,
        )
    )
    capacity = draw(
        st.floats(
            min_value=0.5,
            max_value=5.0,
            allow_infinity=False,
            allow_nan=False,
        )
    )
    steps = draw(
        st.lists(
            st.tuples(
                st.floats(
                    min_value=0.0,
                    max_value=3.0,
                    allow_infinity=False,
                    allow_nan=False,
                ),
                st.floats(
                    min_value=0.1,
                    max_value=3.0,
                    allow_infinity=False,
                    allow_nan=False,
                ),
            ),
            min_size=1,
            max_size=8,
        )
    )
    return rate, capacity, steps


@given(_token_bucket_scenarios())
def test_token_bucket_waits_match_deficits(scenario):
    rate, capacity, steps = scenario
    bucket = TokenBucket(rate_per_second=rate, capacity=capacity, clock=lambda: 0.0)
    manual_tokens = capacity
    last_time = 0.0
    for delta, tokens in steps:
        current_time = last_time + delta
        elapsed = max(current_time - last_time, 0.0)
        manual_tokens = min(capacity, manual_tokens + elapsed * rate)
        if manual_tokens >= tokens:
            expected_wait = 0.0
            manual_tokens -= tokens
        else:
            deficit = tokens - manual_tokens
            expected_wait = deficit / rate
            manual_tokens = 0.0
        wait = bucket.acquire(tokens=tokens, now=current_time)
        assert wait == pytest.approx(expected_wait, rel=1e-6, abs=1e-9)
        last_time = current_time


@given(
    st.integers(min_value=1, max_value=6),
    st.floats(
        min_value=0.05,
        max_value=20.0,
        allow_infinity=False,
        allow_nan=False,
    ),
    st.one_of(
        st.none(),
        st.floats(
            min_value=0.0,
            max_value=20.0,
            allow_infinity=False,
            allow_nan=False,
        ),
    ),
)
def test_circuit_breaker_cooldown_behaviour(failure_threshold, cooldown, retry_after):
    current = 0.0
    with patch("DocsToKG.ContentDownload.network.time.monotonic", lambda: current):
        breaker = CircuitBreaker(failure_threshold=failure_threshold, cooldown_seconds=cooldown)
        for _ in range(failure_threshold - 1):
            breaker.record_failure()
            assert breaker.allow(now=current)
        current += 1.0
        breaker.record_failure(retry_after=retry_after)
        cooldown_value = retry_after if retry_after is not None else cooldown
        addition_changes = (current + cooldown_value) != current
        should_block = cooldown_value > 0.0 and addition_changes
        if should_block:
            assert not breaker.allow(now=current)
        else:
            assert breaker.allow(now=current)
        expected_remaining = cooldown_value if should_block else 0.0
        assert breaker.cooldown_remaining(now=current) == pytest.approx(expected_remaining)
        current += expected_remaining
        assert breaker.allow(now=current)
        breaker.record_success()
        assert breaker.allow(now=current)
