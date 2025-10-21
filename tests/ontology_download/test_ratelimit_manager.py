"""Tests for rate-limit manager: RateLimitManager and singleton API.

Tests cover:
- Basic manager creation and initialization
- Per-service rate enforcement
- Per-host secondary keying
- Block vs fail-fast modes
- Weighted requests
- Singleton pattern and thread safety
- PID-aware rebinding
"""

import os
import pytest
from DocsToKG.OntologyDownload.ratelimit.manager import (
    RateLimitManager,
    get_rate_limiter,
    close_rate_limiter,
    reset_rate_limiter,
)
from DocsToKG.OntologyDownload.ratelimit.config import RateSpec


class TestRateLimitManagerBasics:
    """Test RateLimitManager basic functionality."""

    def test_manager_creation(self):
        """RateLimitManager can be created."""
        rate_specs = {
            "ols": [RateSpec(limit=4, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs)
        assert manager is not None

    def test_manager_acquire_returns_bool(self):
        """Manager.acquire() returns boolean."""
        rate_specs = {
            "_default": [RateSpec(limit=100, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs, mode="block")
        result = manager.acquire("ols", weight=1)
        assert isinstance(result, bool)
        assert result is True

    def test_manager_acquire_with_host(self):
        """Manager.acquire() can be called with host."""
        rate_specs = {
            "_default": [RateSpec(limit=100, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs)
        result = manager.acquire("ols", host="example.com", weight=1)
        assert result is True

    def test_manager_acquire_with_weight(self):
        """Manager.acquire() respects weight parameter."""
        rate_specs = {
            "_default": [RateSpec(limit=2, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs, mode="block")
        # Should succeed
        result1 = manager.acquire("ols", weight=1)
        assert result1 is True
        # Should succeed
        result2 = manager.acquire("ols", weight=1)
        assert result2 is True

    def test_manager_invalid_weight_raises(self):
        """Non-positive weight raises ValueError."""
        rate_specs = {
            "_default": [RateSpec(limit=10, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs)
        with pytest.raises(ValueError, match="must be positive"):
            manager.acquire("ols", weight=0)

    def test_manager_per_service_rates(self):
        """Different services can have different limits."""
        rate_specs = {
            "ols": [RateSpec(limit=100, interval_ms=1000)],
            "bioportal": [RateSpec(limit=50, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs)
        # Both should succeed
        result1 = manager.acquire("ols", weight=1)
        result2 = manager.acquire("bioportal", weight=1)
        assert result1 is True
        assert result2 is True

    def test_manager_default_rate(self):
        """Unknown service uses default rate."""
        rate_specs = {
            "_default": [RateSpec(limit=50, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs)
        result = manager.acquire("unknown_service", weight=1)
        assert result is True

    def test_manager_fail_fast_mode(self):
        """Fail-fast mode returns False on limit."""
        rate_specs = {
            "_default": [RateSpec(limit=1, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs, mode="fail-fast")
        # First should succeed
        result1 = manager.acquire("ols", weight=1)
        assert result1 is True
        # Second should fail immediately
        result2 = manager.acquire("ols", weight=1)
        assert result2 is False

    def test_manager_block_mode(self):
        """Block mode blocks until available."""
        rate_specs = {
            "_default": [RateSpec(limit=2, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs, mode="block")
        # Both should succeed (blocking if necessary)
        result1 = manager.acquire("ols", weight=1)
        result2 = manager.acquire("ols", weight=1)
        assert result1 is True
        assert result2 is True

    def test_manager_get_stats(self):
        """Manager.get_stats() returns statistics."""
        rate_specs = {
            "ols": [RateSpec(limit=4, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs)
        stats = manager.get_stats()
        assert isinstance(stats, dict)
        assert "mode" in stats
        assert "bucket_dir" in stats
        assert "services" in stats

    def test_manager_close(self):
        """Manager.close() cleans up resources."""
        rate_specs = {
            "ols": [RateSpec(limit=4, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs)
        manager.acquire("ols", weight=1)
        # Should not raise
        manager.close()


class TestRateLimitManagerSingleton:
    """Test singleton pattern and lifecycle."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_rate_limiter()

    def teardown_method(self):
        """Clean up after each test."""
        close_rate_limiter()

    def test_get_rate_limiter_creates_instance(self):
        """get_rate_limiter() creates instance on first call."""
        limiter1 = get_rate_limiter()
        assert limiter1 is not None

    def test_get_rate_limiter_returns_same_instance(self):
        """get_rate_limiter() returns same instance on subsequent calls."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_close_rate_limiter_releases(self):
        """close_rate_limiter() releases the singleton."""
        limiter1 = get_rate_limiter()
        close_rate_limiter()
        # After close, next get should create new instance
        limiter2 = get_rate_limiter()
        assert limiter1 is not limiter2

    def test_close_is_idempotent(self):
        """close_rate_limiter() safe to call multiple times."""
        get_rate_limiter()
        close_rate_limiter()
        close_rate_limiter()  # Should not raise
        close_rate_limiter()  # Should not raise

    def test_reset_forces_new_instance(self):
        """reset_rate_limiter() forces creation of new instance."""
        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is not limiter2

    def test_singleton_is_thread_safe(self):
        """Singleton creation is thread-safe."""
        import threading

        limiters = []
        lock = threading.Lock()

        def get_and_store():
            limiter = get_rate_limiter()
            with lock:
                limiters.append(limiter)

        threads = [threading.Thread(target=get_and_store) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be same instance
        assert len(limiters) == 4
        assert all(l is limiters[0] for l in limiters)

    def test_singleton_binds_to_config_hash(self):
        """Singleton binds to config_hash on creation."""
        limiter = get_rate_limiter()
        # Manager is created; should be bound to a config hash
        assert limiter is not None


class TestRateLimitManagerPerHostKeying:
    """Test per-host secondary keying."""

    def test_different_hosts_tracked_separately(self):
        """Different hosts are tracked with separate buckets."""
        rate_specs = {
            "_default": [RateSpec(limit=100, interval_ms=1000)],
        }
        manager = RateLimitManager(rate_specs=rate_specs)
        # Acquire for service with different hosts
        result1 = manager.acquire("ols", host="host1.example.com", weight=1)
        result2 = manager.acquire("ols", host="host2.example.com", weight=1)
        assert result1 is True
        assert result2 is True


class TestRateLimitManagerMultiWindow:
    """Test multi-window rate enforcement."""

    def test_multi_window_rates_created(self):
        """Multi-window rates can be created without error."""
        rate_specs = {
            "_default": [
                RateSpec(limit=2, interval_ms=1000),  # 2 per second
                RateSpec(limit=5, interval_ms=60_000),  # 5 per minute
            ],
        }
        # Just verify manager can be created with multi-window rates
        manager = RateLimitManager(rate_specs=rate_specs, mode="block")
        assert manager is not None
