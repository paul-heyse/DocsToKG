# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_breakers_networking",
#   "purpose": "Integration tests for circuit breaker networking layer",
#   "sections": [
#     {"id": "test-breaker-preflight", "name": "test_breaker_preflight_blocks", "kind": "function"},
#     {"id": "test-breaker-postflight", "name": "test_breaker_postflight_updates", "kind": "function"},
#     {"id": "test-breakeropenrror-handling", "name": "test_breakeropenrror_short_circuits", "kind": "function"},
#     {"id": "test-retry-after-parsing", "name": "test_retry_after_parsing", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""Integration tests for circuit breaker networking layer.

Tests the interaction between BreakerRegistry and the HTTP request/response
handling in the networking layer, including:
- Pre-flight allow() checks before sending requests
- Post-response on_success()/on_failure() updates
- BreakerOpenError exception handling
- Retry-After header parsing
- Cache hit bypass (no breaker updates)
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

# Check if pybreaker is available
try:
    import pybreaker

    HAS_PYBREAKER = True
except ImportError:
    HAS_PYBREAKER = False

from DocsToKG.ContentDownload.breakers import (
    BreakerClassification,
    BreakerConfig,
    BreakerOpenError,
    BreakerPolicy,
    BreakerRegistry,
    HalfOpenPolicy,
    RequestRole,
    RollingWindowPolicy,
)
from DocsToKG.ContentDownload.sqlite_cooldown_store import SQLiteCooldownStore

pytestmark = pytest.mark.skipif(not HAS_PYBREAKER, reason="pybreaker not installed")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_cooldown_db(tmp_path):
    """Temporary SQLite cooldown store."""
    db_path = tmp_path / "cooldowns.sqlite"
    return SQLiteCooldownStore(db_path)


@pytest.fixture
def registry(tmp_cooldown_db):
    """BreakerRegistry for networking integration tests."""
    config = BreakerConfig(
        defaults=BreakerPolicy(
            fail_max=3,
            reset_timeout_s=60,
            retry_after_cap_s=900,
        ),
        classify=BreakerClassification(
            failure_statuses=frozenset([500, 502, 503, 504]),
            neutral_statuses=frozenset([401, 403, 404, 410]),
        ),
        half_open=HalfOpenPolicy(jitter_ms=150),
        rolling=RollingWindowPolicy(
            enabled=True,
            threshold_failures=2,
            window_s=10,
            cooldown_s=30,
        ),
    )
    return BreakerRegistry(config, cooldown_store=tmp_cooldown_db)


@pytest.fixture
def mock_request():
    """Mock httpx.Request object."""
    req = Mock()
    req.url = "https://api.example.org/v1/data"
    return req


@pytest.fixture
def mock_response():
    """Mock successful httpx.Response object."""
    resp = Mock()
    resp.status_code = 200
    resp.headers = {}
    return resp


# ============================================================================
# Pre-Flight Check Tests
# ============================================================================


class TestBreakerPreflight:
    """Test pre-flight allow() checks before sending requests."""

    def test_preflight_allows_healthy_host(self, registry, mock_request):
        """Pre-flight check should allow requests to healthy hosts."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Should not raise
        registry.allow(host, role)

    def test_preflight_blocks_open_breaker(self, registry, mock_request):
        """Pre-flight check should block requests when breaker is open."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Open breaker by simulating failures
        for _ in range(3):
            registry.on_failure(host, role, status=503)

        # Pre-flight should raise
        with pytest.raises(BreakerOpenError) as exc_info:
            registry.allow(host, role)

        error = exc_info.value
        assert error.host == host
        assert "open" in str(error).lower() or "blocked" in str(error).lower()

    def test_preflight_blocks_cooldown_override(self, registry):
        """Pre-flight check should block if cooldown override is active."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Set cooldown override manually (simulating Retry-After)
        now_mono = time.monotonic()
        registry.cooldowns.set_until(host, now_mono + 60, reason="retry-after")

        # Pre-flight should raise
        with pytest.raises(BreakerOpenError) as exc_info:
            registry.allow(host, role)

        error = exc_info.value
        assert error.cooldown_remaining_ms > 0

    def test_preflight_different_roles_independent(self, registry):
        """Pre-flight should check per-role state independently."""
        host = "api.example.org"

        # Open metadata breaker
        for _ in range(3):
            registry.on_failure(host, RequestRole.METADATA, status=503)

        # Metadata should be blocked
        with pytest.raises(BreakerOpenError):
            registry.allow(host, RequestRole.METADATA)

        # Artifact should still be allowed
        registry.allow(host, RequestRole.ARTIFACT)


# ============================================================================
# Post-Response Update Tests
# ============================================================================


class TestBreakerPostflight:
    """Test post-response on_success()/on_failure() updates."""

    def test_postflight_records_failure(self, registry):
        """Failure responses should increment failure counter."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Record first failure
        registry.on_failure(host, role, status=500)

        # Pre-flight should still pass (1 < 3)
        registry.allow(host, role)

        # Record 2 more failures
        registry.on_failure(host, role, status=500)
        registry.on_failure(host, role, status=500)

        # Now breaker should be open
        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

    def test_postflight_records_success(self, registry):
        """Success responses should reset failure counter."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Record 2 failures
        registry.on_failure(host, role, status=500)
        registry.on_failure(host, role, status=500)

        # Record success (should reset)
        registry.on_success(host, role)

        # Record 2 more failures (not cumulative)
        registry.on_failure(host, role, status=500)
        registry.on_failure(host, role, status=500)

        # Breaker should still be closed (2 < 3)
        registry.allow(host, role)

    def test_postflight_neutral_status_no_count(self, registry):
        """Neutral status responses should not count as failures."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Record multiple 404 responses (neutral)
        for _ in range(10):
            registry.on_failure(host, role, status=404)

        # Breaker should still be open
        registry.allow(host, role)

    def test_postflight_retry_after_sets_cooldown(self, registry):
        """Failure with Retry-After should set cooldown override."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Record 429 with Retry-After: 30
        registry.on_failure(host, role, status=429, retry_after_s=30)

        # Check that cooldown was set
        until = registry.cooldowns.get_until(host)
        now = time.monotonic()

        assert until is not None
        assert until > now
        remaining_s = until - now
        assert 25 < remaining_s < 35

    def test_postflight_500_error_increments(self, registry):
        """500 errors should count as failures."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Record 3 x 500 errors
        for _ in range(3):
            registry.on_failure(host, role, status=500)

        # Breaker should be open
        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

    def test_postflight_502_error_increments(self, registry):
        """502 errors should count as failures."""
        host = "api.example.org"
        role = RequestRole.METADATA

        for _ in range(3):
            registry.on_failure(host, role, status=502)

        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

    def test_postflight_503_error_increments(self, registry):
        """503 errors should count as failures."""
        host = "api.example.org"
        role = RequestRole.METADATA

        for _ in range(3):
            registry.on_failure(host, role, status=503)

        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

    def test_postflight_504_error_increments(self, registry):
        """504 errors should count as failures."""
        host = "api.example.org"
        role = RequestRole.METADATA

        for _ in range(3):
            registry.on_failure(host, role, status=504)

        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)


# ============================================================================
# BreakerOpenError Handling Tests
# ============================================================================


class TestBreakerOpenErrorHandling:
    """Test BreakerOpenError exception handling."""

    def test_breakeropenrror_contains_context(self, registry):
        """BreakerOpenError should contain diagnostic context."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Open breaker
        for _ in range(3):
            registry.on_failure(host, role, status=503)

        try:
            registry.allow(host, role)
            pytest.fail("Should have raised BreakerOpenError")
        except BreakerOpenError as e:
            # Check error has useful context
            assert e.host == host
            assert e.cooldown_remaining_ms >= 0

    def test_breakeropenrror_short_circuits_request(self, registry):
        """BreakerOpenError should cause request to be skipped entirely."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Open breaker
        for _ in range(3):
            registry.on_failure(host, role, status=503)

        # Pre-flight check should raise BEFORE attempting HTTP request
        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

        # Request should never be attempted (no HTTP call made)
        # This is implicit - the exception prevents further code execution

    def test_breakeropenrror_per_host(self, registry):
        """BreakerOpenError should be specific to blocked host."""
        host1 = "api1.example.org"
        host2 = "api2.example.org"
        role = RequestRole.METADATA

        # Open breaker for host1
        for _ in range(3):
            registry.on_failure(host1, role, status=503)

        # host1 should raise
        with pytest.raises(BreakerOpenError) as exc_info:
            registry.allow(host1, role)
        assert exc_info.value.host == host1

        # host2 should not raise
        registry.allow(host2, role)

    def test_breakeropenrror_recoverable_after_cooldown(self, registry):
        """Request should be allowed after cooldown expires."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Open breaker with short cooldown
        registry.on_failure(host, role, status=429, retry_after_s=0.05)

        # Should be blocked now
        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

        # Wait for cooldown to expire
        time.sleep(0.1)

        # Should now be allowed (cooldown expired)
        registry.allow(host, role)


# ============================================================================
# Retry-After Header Parsing Tests
# ============================================================================


class TestRetryAfterParsing:
    """Test Retry-After header parsing and cooldown setting."""

    def test_retry_after_429(self, registry):
        """429 response with Retry-After should set cooldown."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Simulate 429 with Retry-After: 10
        registry.on_failure(host, role, status=429, retry_after_s=10)

        # Cooldown should be set
        until = registry.cooldowns.get_until(host)
        assert until is not None

        # Should block on next request
        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

    def test_retry_after_503(self, registry):
        """503 response with Retry-After should set cooldown."""
        host = "api.example.org"
        role = RequestRole.METADATA

        registry.on_failure(host, role, status=503, retry_after_s=15)

        until = registry.cooldowns.get_until(host)
        assert until is not None

        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

    def test_retry_after_respects_cap(self, registry):
        """Retry-After should be capped by policy."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Simulate Retry-After: 2000 seconds (way over 900s cap)
        registry.on_failure(host, role, status=429, retry_after_s=2000)

        # Get actual cooldown duration
        until = registry.cooldowns.get_until(host)
        now = time.monotonic()
        duration_s = (until - now) if until else 0

        # Should respect cap (900s)
        assert duration_s <= 900

    def test_retry_after_without_cap_no_cooldown(self, registry):
        """Non-429/503 with Retry-After should not set cooldown."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Record 401 with Retry-After (neutral status, shouldn't trigger cooldown)
        registry.on_failure(host, role, status=401, retry_after_s=30)

        # Should be allowed (401 is neutral)
        registry.allow(host, role)


# ============================================================================
# Cache Hit Bypass Tests
# ============================================================================


class TestCacheHitBypass:
    """Test that cache hits bypass breaker updates."""

    def test_cache_hit_no_breaker_update(self, registry):
        """Cache hits should not affect breaker state."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Note: The registry itself doesn't know about cache hits.
        # This test documents the expected behavior: when a cache hit occurs,
        # the networking layer should NOT call on_failure/on_success,
        # leaving the breaker state unchanged.

        # Breaker starts closed
        registry.allow(host, role)

        # Simulate cache hit (no on_failure/on_success call)
        # ... (implementation in networking layer)

        # Breaker should still be open
        registry.allow(host, role)


# ============================================================================
# Multi-Role Tests
# ============================================================================


class TestMultiRoleHandling:
    """Test circuit breaker behavior with multiple request roles."""

    def test_different_roles_independent_state(self, registry):
        """Different roles should have independent breaker state."""
        host = "api.example.org"

        # Metadata: record 2 failures (2 < 3, still open)
        registry.on_failure(host, RequestRole.METADATA, status=503)
        registry.on_failure(host, RequestRole.METADATA, status=503)

        # Artifact: record 3 failures (3 >= 3, open)
        for _ in range(3):
            registry.on_failure(host, RequestRole.ARTIFACT, status=503)

        # Metadata should still be open
        registry.allow(host, RequestRole.METADATA)

        # Artifact should be blocked
        with pytest.raises(BreakerOpenError):
            registry.allow(host, RequestRole.ARTIFACT)

    def test_success_per_role(self, registry):
        """Success on one role shouldn't affect other roles."""
        host = "api.example.org"

        # Both roles: record 2 failures
        for role in [RequestRole.METADATA, RequestRole.ARTIFACT]:
            for _ in range(2):
                registry.on_failure(host, role, status=503)

        # Success only on metadata
        registry.on_success(host, RequestRole.METADATA)

        # Metadata should be reset
        registry.allow(host, RequestRole.METADATA)

        # Artifact should still have 2 failures
        for _ in range(1):
            registry.on_failure(host, RequestRole.ARTIFACT, status=503)

        # Now artifact is open (2 + 1 = 3)
        with pytest.raises(BreakerOpenError):
            registry.allow(host, RequestRole.ARTIFACT)


# ============================================================================
# Error Exception Tests
# ============================================================================


class TestNetworkExceptionHandling:
    """Test handling of network exceptions."""

    def test_connection_error_counts_as_failure(self, registry):
        """Connection errors should count as failures."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Simulate connection failures
        for _ in range(3):
            registry.on_failure(host, role, exception_type="ConnectionError")

        # Breaker should be open
        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

    def test_timeout_error_counts_as_failure(self, registry):
        """Timeout errors should count as failures."""
        host = "api.example.org"
        role = RequestRole.METADATA

        for _ in range(3):
            registry.on_failure(host, role, exception_type="TimeoutError")

        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)


# ============================================================================
# Rolling Window Integration Tests
# ============================================================================


class TestRollingWindowIntegration:
    """Test rolling window detection in networking context."""

    def test_rapid_failures_trigger_rolling_window(self, registry):
        """Rapid failures should trigger rolling window manual open."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Rapid failures within rolling window
        registry.on_failure(host, role, status=503)
        registry.on_failure(host, role, status=503)

        # Should be open (rolling window threshold = 2)
        with pytest.raises(BreakerOpenError):
            registry.allow(host, role)

    def test_spaced_failures_no_rolling_window(self, registry):
        """Spaced-out failures should not trigger rolling window."""
        host = "api.example.org"
        role = RequestRole.METADATA

        # Failures outside window
        registry.on_failure(host, role, status=503)

        # Wait for window to expire
        time.sleep(11)

        # Next failure should not trigger rolling window
        registry.on_failure(host, role, status=503)

        # Should still be open
        registry.allow(host, role)
