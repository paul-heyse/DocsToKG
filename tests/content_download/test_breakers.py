"""Tests for circuit breaker functionality."""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

try:
    import pybreaker
except ImportError:
    pybreaker = None
    pytest.skip("pybreaker not available", allow_module_level=True)

from DocsToKG.ContentDownload.breakers import (
    BreakerConfig,
    BreakerPolicy,
    BreakerRegistry,
    BreakerOpenError,
    RequestRole,
    InMemoryCooldownStore,
    is_failure_for_breaker,
)
from DocsToKG.ContentDownload.networking_breaker_listener import (
    NetworkBreakerListener,
    BreakerListenerConfig,
)


class TestBreakerRegistry:
    """Test BreakerRegistry functionality."""

    def test_allow_when_closed(self):
        """Test that allow() passes when breaker is closed."""
        config = BreakerConfig()
        registry = BreakerRegistry(config)

        # Should not raise
        registry.allow("example.com", role=RequestRole.METADATA)

    def test_allow_when_open(self):
        """Test that allow() raises when breaker is open."""
        config = BreakerConfig(defaults=BreakerPolicy(fail_max=1, reset_timeout_s=60))
        registry = BreakerRegistry(config)

        # Trigger failure to open breaker
        registry.on_failure("example.com", role=RequestRole.METADATA, status=503)

        # Should raise BreakerOpenError
        with pytest.raises(BreakerOpenError):
            registry.allow("example.com", role=RequestRole.METADATA)

    def test_success_resets_breaker(self):
        """Test that success resets breaker state."""
        config = BreakerConfig(defaults=BreakerPolicy(fail_max=1, reset_timeout_s=60))
        registry = BreakerRegistry(config)

        # Trigger failure to open breaker
        registry.on_failure("example.com", role=RequestRole.METADATA, status=503)

        # Verify breaker is open
        with pytest.raises(BreakerOpenError):
            registry.allow("example.com", role=RequestRole.METADATA)

        # Success should reset breaker
        registry.on_success("example.com", role=RequestRole.METADATA)

        # Should now be allowed
        registry.allow("example.com", role=RequestRole.METADATA)

    def test_retry_after_cooldown(self):
        """Test that Retry-After headers set cooldown overrides."""
        config = BreakerConfig(defaults=BreakerPolicy(fail_max=1, reset_timeout_s=60))
        registry = BreakerRegistry(config)

        # Trigger failure with Retry-After
        registry.on_failure("example.com", role=RequestRole.METADATA, status=429, retry_after_s=5.0)

        # Should be blocked by cooldown override
        with pytest.raises(BreakerOpenError, match="cooldown_remaining_ms"):
            registry.allow("example.com", role=RequestRole.METADATA)

    def test_neutral_statuses_dont_fail(self):
        """Test that neutral statuses don't count as failures."""
        config = BreakerConfig(defaults=BreakerPolicy(fail_max=1, reset_timeout_s=60))
        registry = BreakerRegistry(config)

        # 404 should not count as failure
        registry.on_failure("example.com", role=RequestRole.METADATA, status=404)

        # Should still be allowed
        registry.allow("example.com", role=RequestRole.METADATA)

    def test_resolver_breaker(self):
        """Test per-resolver breaker functionality."""
        config = BreakerConfig(
            defaults=BreakerPolicy(fail_max=5, reset_timeout_s=60),
            resolvers={"test_resolver": BreakerPolicy(fail_max=1, reset_timeout_s=60)},
        )
        registry = BreakerRegistry(config)

        # Trigger resolver failure
        registry.on_failure(
            "example.com", role=RequestRole.METADATA, resolver="test_resolver", status=503
        )

        # Should be blocked by resolver breaker
        with pytest.raises(BreakerOpenError, match="resolver=test_resolver"):
            registry.allow("example.com", role=RequestRole.METADATA, resolver="test_resolver")

    def test_half_open_probe_limit(self):
        """Test half-open probe limits per role."""
        config = BreakerConfig(
            defaults=BreakerPolicy(
                fail_max=1,
                reset_timeout_s=1,  # Short timeout for testing
                roles={RequestRole.ARTIFACT: BreakerRolePolicy(trial_calls=2)},
            )
        )
        registry = BreakerRegistry(config)

        # Trigger failure to open breaker
        registry.on_failure("example.com", role=RequestRole.ARTIFACT, status=503)

        # Wait for breaker to become half-open
        time.sleep(1.1)

        # Should allow 2 trial calls for artifact role
        registry.allow("example.com", role=RequestRole.ARTIFACT)
        registry.allow("example.com", role=RequestRole.ARTIFACT)

        # Third call should be blocked
        with pytest.raises(BreakerOpenError, match="half-open probes exhausted"):
            registry.allow("example.com", role=RequestRole.ARTIFACT)

    def test_rolling_window_manual_open(self):
        """Test rolling window manual open functionality."""
        config = BreakerConfig(
            defaults=BreakerPolicy(fail_max=10, reset_timeout_s=60),
            rolling=RollingWindowPolicy(
                enabled=True, window_s=10, threshold_failures=3, cooldown_s=5
            ),
        )
        registry = BreakerRegistry(config)

        # Trigger 3 failures within window
        now = time.monotonic()
        with patch.object(registry, "_now", return_value=now):
            registry.on_failure("example.com", role=RequestRole.METADATA, status=503)
            registry.on_failure("example.com", role=RequestRole.METADATA, status=503)
            registry.on_failure("example.com", role=RequestRole.METADATA, status=503)

        # Should be blocked by rolling window cooldown
        with pytest.raises(BreakerOpenError, match="cooldown_remaining_ms"):
            registry.allow("example.com", role=RequestRole.METADATA)

    def test_current_state(self):
        """Test current_state() method."""
        config = BreakerConfig()
        registry = BreakerRegistry(config)

        # Initially closed
        assert "closed" in registry.current_state("example.com")

        # After failure, should be open
        registry.on_failure("example.com", role=RequestRole.METADATA, status=503)
        assert "open" in registry.current_state("example.com")


class TestBreakerListener:
    """Test NetworkBreakerListener functionality."""

    def test_listener_emits_events(self):
        """Test that listener emits telemetry events."""
        sink = Mock()
        config = BreakerListenerConfig(run_id="test-run", host="example.com")
        listener = NetworkBreakerListener(sink, config)

        # Mock breaker
        breaker = Mock()
        breaker.current_state = "closed"
        breaker.reset_timeout = 60

        # Test state change
        listener.state_change(breaker, "closed", "open")

        # Verify event was emitted
        sink.emit.assert_called()
        call_args = sink.emit.call_args[0][0]
        assert call_args["event_type"] == "breaker_state_change"
        assert call_args["host"] == "example.com"
        assert call_args["run_id"] == "test-run"
        assert call_args["old"] == "closed"
        assert call_args["new"] == "open"


class TestCooldownStore:
    """Test cooldown store functionality."""

    def test_in_memory_store(self):
        """Test InMemoryCooldownStore functionality."""
        store = InMemoryCooldownStore()

        # Initially no cooldown
        assert store.get_until("example.com") is None

        # Set cooldown
        store.set_until("example.com", 100.0, "test")
        assert store.get_until("example.com") == 100.0

        # Clear cooldown
        store.clear("example.com")
        assert store.get_until("example.com") is None


class TestBreakerClassification:
    """Test breaker classification logic."""

    def test_is_failure_for_breaker(self):
        """Test failure classification logic."""
        from DocsToKG.ContentDownload.breakers import BreakerClassification

        classify = BreakerClassification()

        # Failure statuses should count as failures
        assert is_failure_for_breaker(classify, status=503, exception=None)
        assert is_failure_for_breaker(classify, status=429, exception=None)

        # Neutral statuses should not count as failures
        assert not is_failure_for_breaker(classify, status=404, exception=None)
        assert not is_failure_for_breaker(classify, status=403, exception=None)

        # Success statuses should not count as failures
        assert not is_failure_for_breaker(classify, status=200, exception=None)

        # Exceptions should count as failures if configured
        classify_with_exceptions = BreakerClassification(failure_exceptions=(ValueError,))
        assert is_failure_for_breaker(
            classify_with_exceptions, status=None, exception=ValueError("test")
        )
        assert not is_failure_for_breaker(
            classify_with_exceptions, status=None, exception=TypeError("test")
        )


class TestBreakerConfig:
    """Test breaker configuration functionality."""

    def test_default_config(self):
        """Test default breaker configuration."""
        config = BreakerConfig()

        assert config.defaults.fail_max == 5
        assert config.defaults.reset_timeout_s == 60
        assert config.defaults.retry_after_cap_s == 900

        # Check default failure statuses
        assert 503 in config.classify.failure_statuses
        assert 429 in config.classify.failure_statuses

        # Check default neutral statuses
        assert 404 in config.classify.neutral_statuses
        assert 403 in config.classify.neutral_statuses

    def test_host_specific_config(self):
        """Test host-specific breaker configuration."""
        config = BreakerConfig(
            hosts={"api.crossref.org": BreakerPolicy(fail_max=3, reset_timeout_s=120)}
        )

        registry = BreakerRegistry(config)

        # Host-specific policy should be used
        assert registry._policy_for_host("api.crossref.org").fail_max == 3
        assert registry._policy_for_host("api.crossref.org").reset_timeout_s == 120

        # Unknown host should use defaults
        assert registry._policy_for_host("unknown.com").fail_max == 5
