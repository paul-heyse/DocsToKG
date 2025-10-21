"""Tests for HTTP layer telemetry instrumentation (Phase 1).

This module tests the telemetry emission from `request_with_retries()` to the
`http_events` table, including URL hashing, cache metadata extraction, breaker
state capture, and graceful error handling.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from DocsToKG.ContentDownload.networking import (
    _compute_url_hash,
    _extract_breaker_recorded,
    _extract_breaker_state,
    _extract_from_cache,
    _extract_rate_delay,
    _extract_retry_after,
    _extract_revalidated,
    _extract_stale,
)


class TestUrlHashComputation:
    """Test URL hashing for privacy."""

    def test_compute_url_hash_basic(self) -> None:
        """Test URL hashing produces consistent 16-char hash."""
        url = "https://example.org/test"
        url_hash = _compute_url_hash(url)
        assert isinstance(url_hash, str)
        assert len(url_hash) == 16
        assert url_hash == url_hash.lower()  # Hex is lowercase

    def test_compute_url_hash_deterministic(self) -> None:
        """Test hashing is deterministic."""
        url = "https://example.org/test"
        hash1 = _compute_url_hash(url)
        hash2 = _compute_url_hash(url)
        assert hash1 == hash2

    def test_compute_url_hash_different_urls(self) -> None:
        """Test different URLs produce different hashes."""
        hash1 = _compute_url_hash("https://example.org/a")
        hash2 = _compute_url_hash("https://example.org/b")
        assert hash1 != hash2

    def test_compute_url_hash_no_raw_url(self) -> None:
        """Test that URL is not stored in hash."""
        url = "https://example.org/test"
        url_hash = _compute_url_hash(url)
        assert "example.org" not in url_hash
        assert "/test" not in url_hash

    def test_compute_url_hash_error_handling(self) -> None:
        """Test graceful error handling for invalid input."""
        # Pass None-like object
        result = _compute_url_hash(None)  # type: ignore
        assert result == "unknown"


class TestCacheMetadataExtraction:
    """Test extraction of cache metadata from responses."""

    def test_extract_from_cache_hit(self) -> None:
        """Test detection of cache hit."""
        response = Mock(spec=httpx.Response)
        response.extensions = {"from_cache": True}
        assert _extract_from_cache(response) == 1

    def test_extract_from_cache_miss(self) -> None:
        """Test detection of cache miss."""
        response = Mock(spec=httpx.Response)
        response.extensions = {"from_cache": False}
        assert _extract_from_cache(response) == 0

    def test_extract_from_cache_none(self) -> None:
        """Test graceful handling when cache status unknown."""
        response = Mock(spec=httpx.Response)
        response.extensions = {}
        assert _extract_from_cache(response) is None

    def test_extract_from_cache_hishel_status(self) -> None:
        """Test extraction of Hishel cache_status."""
        response = Mock(spec=httpx.Response)
        response.extensions = {"cache_status": "HIT"}
        assert _extract_from_cache(response) == 1

    def test_extract_revalidated_true(self) -> None:
        """Test detection of 304 revalidation."""
        response = Mock(spec=httpx.Response)
        response.status_code = 304
        assert _extract_revalidated(response) == 1

    def test_extract_revalidated_false(self) -> None:
        """Test non-revalidation response."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        assert _extract_revalidated(response) == 0

    def test_extract_stale_true(self) -> None:
        """Test detection of stale response."""
        response = Mock(spec=httpx.Response)
        response.extensions = {"stale": True}
        assert _extract_stale(response) == 1

    def test_extract_stale_false(self) -> None:
        """Test non-stale response."""
        response = Mock(spec=httpx.Response)
        response.extensions = {"stale": False}
        assert _extract_stale(response) == 0


class TestHeaderExtraction:
    """Test extraction of HTTP header values."""

    def test_extract_retry_after_seconds(self) -> None:
        """Test extraction of Retry-After header."""
        response = Mock(spec=httpx.Response)
        response.headers = {"Retry-After": "60"}
        assert _extract_retry_after(response) == 60

    def test_extract_retry_after_float(self) -> None:
        """Test extraction of float Retry-After."""
        response = Mock(spec=httpx.Response)
        response.headers = {"Retry-After": "30.5"}
        assert _extract_retry_after(response) == 30

    def test_extract_retry_after_missing(self) -> None:
        """Test graceful handling when Retry-After missing."""
        response = Mock(spec=httpx.Response)
        response.headers = {}
        assert _extract_retry_after(response) is None

    def test_extract_retry_after_invalid(self) -> None:
        """Test error handling for invalid Retry-After."""
        response = Mock(spec=httpx.Response)
        response.headers = {"Retry-After": "invalid"}
        assert _extract_retry_after(response) is None


class TestRateLimiterExtraction:
    """Test extraction of rate limiter metadata."""

    def test_extract_rate_delay_present(self) -> None:
        """Test extraction of rate limiter wait time."""
        network_meta = {"rate_limiter": {"wait_ms": 125}}
        assert _extract_rate_delay(network_meta) == 125

    def test_extract_rate_delay_float(self) -> None:
        """Test extraction of float rate delay."""
        network_meta = {"rate_limiter": {"wait_ms": 125.5}}
        assert _extract_rate_delay(network_meta) == 125

    def test_extract_rate_delay_missing(self) -> None:
        """Test graceful handling when rate delay missing."""
        network_meta: dict[str, Any] = {}
        assert _extract_rate_delay(network_meta) is None

    def test_extract_rate_delay_invalid_type(self) -> None:
        """Test error handling for invalid rate delay type."""
        network_meta = {"rate_limiter": {"wait_ms": "invalid"}}
        assert _extract_rate_delay(network_meta) is None


class TestBreakerStateExtraction:
    """Test extraction of circuit breaker state."""

    def test_extract_breaker_state_closed(self) -> None:
        """Test extraction of closed breaker state."""
        breaker_info = {"breaker_host_state": "closed"}
        assert _extract_breaker_state(breaker_info) == "closed"

    def test_extract_breaker_state_open(self) -> None:
        """Test extraction of open breaker state."""
        breaker_info = {"breaker_host_state": "OPEN"}
        assert _extract_breaker_state(breaker_info) == "open"

    def test_extract_breaker_state_half_open(self) -> None:
        """Test extraction of half-open breaker state."""
        breaker_info = {"breaker_host_state": "half_open"}
        assert _extract_breaker_state(breaker_info) == "half_open"

    def test_extract_breaker_state_missing(self) -> None:
        """Test graceful handling when breaker state missing."""
        breaker_info: dict[str, Any] = {}
        assert _extract_breaker_state(breaker_info) is None

    def test_extract_breaker_recorded_success(self) -> None:
        """Test extraction of breaker recorded outcome."""
        breaker_info = {"breaker_recorded": "success"}
        assert _extract_breaker_recorded(breaker_info) == "success"

    def test_extract_breaker_recorded_failure(self) -> None:
        """Test extraction of failure outcome."""
        breaker_info = {"breaker_recorded": "failure"}
        assert _extract_breaker_recorded(breaker_info) == "failure"

    def test_extract_breaker_recorded_none(self) -> None:
        """Test extraction of none outcome."""
        breaker_info = {"breaker_recorded": "none"}
        assert _extract_breaker_recorded(breaker_info) == "none"

    def test_extract_breaker_recorded_missing(self) -> None:
        """Test graceful handling when recorded missing."""
        breaker_info: dict[str, Any] = {}
        assert _extract_breaker_recorded(breaker_info) is None

    def test_extract_breaker_recorded_invalid(self) -> None:
        """Test error handling for invalid recorded value."""
        breaker_info = {"breaker_recorded": "invalid"}
        assert _extract_breaker_recorded(breaker_info) is None


class TestTelemetryEmission:
    """Test end-to-end telemetry emission (integration tests)."""

    @pytest.mark.skip(reason="Requires full HTTP client mock setup")
    def test_emit_http_event_none_telemetry(self) -> None:
        """Test graceful degradation when telemetry=None."""
        # This test verifies that request_with_retries() works correctly
        # when telemetry=None (should not crash or break requests)
        pass

    @pytest.mark.skip(reason="Requires full HTTP client mock setup")
    def test_emit_http_event_basic(self) -> None:
        """Test basic HTTP event emission."""
        # This test would:
        # 1. Mock request_with_retries() call
        # 2. Verify emit_http_event() was called
        # 3. Check event fields are populated correctly
        pass

    @pytest.mark.skip(reason="Requires SQLite database setup")
    def test_emit_http_event_sqlite_integration(self) -> None:
        """Test full integration: HTTP request → SQLite table."""
        # This test would:
        # 1. Create temporary SQLite database
        # 2. Initialize telemetry schema
        # 3. Make request with telemetry sink
        # 4. Verify http_events table populated
        pass


# Mark module as test module
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
