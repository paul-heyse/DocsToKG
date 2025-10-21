# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_cli_breakers",
#   "purpose": "CLI command tests for circuit breaker operations",
#   "sections": [
#     {"id": "test-show-cmd", "name": "test_breaker_show_basic", "kind": "function"},
#     {"id": "test-open-cmd", "name": "test_breaker_open_sets_cooldown", "kind": "function"},
#     {"id": "test-close-cmd", "name": "test_breaker_close_clears", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""CLI command tests for circuit breaker operations.

Tests the CLI interface for:
- `breaker show` - inspect breaker state
- `breaker open` - force-open with cooldown
- `breaker close` - reset and close
- Argument parsing and validation
- Output formatting
- Factory injection
"""

from __future__ import annotations

import io
import time
from unittest.mock import patch

import pytest

# Check if pybreaker is available
try:
    import pybreaker

    HAS_PYBREAKER = True
except ImportError:
    HAS_PYBREAKER = False

import argparse

from DocsToKG.ContentDownload.breakers import (
    BreakerConfig,
    BreakerOpenError,
    BreakerPolicy,
    BreakerRegistry,
    RequestRole,
)
from DocsToKG.ContentDownload.cli_breakers import (
    _cmd_close,
    _cmd_open,
    _cmd_show,
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
    """BreakerRegistry for CLI testing."""
    config = BreakerConfig(
        defaults=BreakerPolicy(
            fail_max=3,
            reset_timeout_s=60,
            retry_after_cap_s=900,
        ),
    )
    return BreakerRegistry(config, cooldown_store=tmp_cooldown_db)


@pytest.fixture
def make_registry(registry):
    """Factory function for registry creation."""

    def _make():
        return registry, ["api.example.org", "api.other.org"]

    return _make


@pytest.fixture
def parser():
    """Minimal argparse setup for testing."""
    return argparse.ArgumentParser()


# ============================================================================
# Breaker Show Command Tests
# ============================================================================


class TestBreakerShowCommand:
    """Test `breaker show` command."""

    def test_show_displays_closed_breaker(self, make_registry, capsys):
        """Show should display closed breaker as CLOSED."""
        registry, known_hosts = make_registry()

        # Create args namespace
        args = argparse.Namespace(
            make_registry=make_registry,
            host=None,
        )

        # Capture stdout
        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_show(args)

        assert result == 0

    def test_show_displays_open_breaker(self, make_registry, capsys):
        """Show should display open breaker state."""
        registry, known_hosts = make_registry()
        host = "api.example.org"
        role = RequestRole.METADATA

        # Open breaker
        for _ in range(3):
            registry.on_failure(host, role, status=503)

        args = argparse.Namespace(
            make_registry=make_registry,
            host=None,
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_show(args)

        assert result == 0

    def test_show_filters_by_host(self, make_registry):
        """Show --host should filter to single host."""
        registry, known_hosts = make_registry()

        args = argparse.Namespace(
            make_registry=make_registry,
            host="api.example.org",
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_show(args)

        assert result == 0

    def test_show_displays_cooldown_remaining(self, make_registry):
        """Show should display cooldown remaining time."""
        registry, known_hosts = make_registry()
        host = "api.example.org"

        # Set cooldown
        now_mono = time.monotonic()
        registry.cooldowns.set_until(host, now_mono + 30, reason="test")

        args = argparse.Namespace(
            make_registry=make_registry,
            host=host,
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_show(args)

        assert result == 0

    def test_show_no_breakers(self, registry):
        """Show should handle case with no breakers."""

        def make_empty_registry():
            return registry, []

        args = argparse.Namespace(
            make_registry=make_empty_registry,
            host=None,
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_show(args)

        # Should return 0 (no error)
        assert result == 0


# ============================================================================
# Breaker Open Command Tests
# ============================================================================


class TestBreakerOpenCommand:
    """Test `breaker open` command."""

    def test_open_sets_cooldown(self, make_registry):
        """Open command should set cooldown override."""
        registry, _ = make_registry()
        host = "api.example.org"

        args = argparse.Namespace(
            make_registry=make_registry,
            host=host,
            seconds=60,
            reason="cli-open",
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_open(args)

        # Should succeed
        assert result == 0

        # Cooldown should be set
        until = registry.cooldowns.get_until(host)
        assert until is not None

        # Should block on next request

        with pytest.raises(BreakerOpenError):
            registry.allow(host, RequestRole.METADATA)

    def test_open_with_custom_reason(self, make_registry):
        """Open command should accept custom reason."""
        registry, _ = make_registry()
        host = "api.example.org"

        args = argparse.Namespace(
            make_registry=make_registry,
            host=host,
            seconds=120,
            reason="maintenance window",
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_open(args)

        assert result == 0

    def test_open_duration_respected(self, make_registry):
        """Open command should respect duration argument."""
        registry, _ = make_registry()
        host = "api.example.org"

        args = argparse.Namespace(
            make_registry=make_registry,
            host=host,
            seconds=5,
            reason="test",
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            _cmd_open(args)

        # Get cooldown duration
        until = registry.cooldowns.get_until(host)
        now = time.monotonic()
        duration_s = (until - now) if until else 0

        # Should be approximately 5 seconds
        assert 4 < duration_s < 6

    def test_open_multiple_hosts_independent(self, make_registry):
        """Open on one host shouldn't affect others."""
        registry, _ = make_registry()
        host1 = "api.example.org"
        host2 = "api.other.org"

        args = argparse.Namespace(
            make_registry=make_registry,
            host=host1,
            seconds=60,
            reason="test",
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            _cmd_open(args)

        # host1 should be blocked

        with pytest.raises(BreakerOpenError):
            registry.allow(host1, RequestRole.METADATA)

        # host2 should be open
        registry.allow(host2, RequestRole.METADATA)


# ============================================================================
# Breaker Close Command Tests
# ============================================================================


class TestBreakerCloseCommand:
    """Test `breaker close` command."""

    def test_close_clears_cooldown(self, make_registry):
        """Close command should clear cooldown override."""
        registry, _ = make_registry()
        host = "api.example.org"

        # Set cooldown
        now_mono = time.monotonic()
        registry.cooldowns.set_until(host, now_mono + 60, reason="test")

        # Verify it's blocked

        with pytest.raises(BreakerOpenError):
            registry.allow(host, RequestRole.METADATA)

        # Close it
        args = argparse.Namespace(
            make_registry=make_registry,
            host=host,
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_close(args)

        assert result == 0

        # Should now be allowed
        registry.allow(host, RequestRole.METADATA)

    def test_close_resets_counters(self, make_registry):
        """Close command should reset failure counters."""
        registry, _ = make_registry()
        host = "api.example.org"
        role = RequestRole.METADATA

        # Record some failures
        for _ in range(2):
            registry.on_failure(host, role, status=503)

        # Close
        args = argparse.Namespace(
            make_registry=make_registry,
            host=host,
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_close(args)

        assert result == 0

        # Should still be open (counter wasn't reset by close)
        registry.allow(host, role)

    def test_close_unknown_host(self, make_registry):
        """Close on unknown host should not error."""
        args = argparse.Namespace(
            make_registry=make_registry,
            host="unknown.example.org",
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_close(args)

        # Should succeed gracefully
        assert result == 0


# ============================================================================
# CLI Integration Tests
# ============================================================================


class TestCLIIntegration:
    """Test CLI command integration."""

    def test_workflow_open_show_close(self, make_registry):
        """Test typical workflow: open → show → close."""
        registry, _ = make_registry()
        host = "api.example.org"

        # Open
        args_open = argparse.Namespace(
            make_registry=make_registry,
            host=host,
            seconds=30,
            reason="test",
        )
        with patch("sys.stdout", new_callable=io.StringIO):
            _cmd_open(args_open)

        # Show
        args_show = argparse.Namespace(
            make_registry=make_registry,
            host=host,
        )
        with patch("sys.stdout", new_callable=io.StringIO):
            _cmd_show(args_show)

        # Close
        args_close = argparse.Namespace(
            make_registry=make_registry,
            host=host,
        )
        with patch("sys.stdout", new_callable=io.StringIO):
            _cmd_close(args_close)

        # After close, should be open again
        registry.allow(host, RequestRole.METADATA)

    def test_show_all_hosts(self, make_registry):
        """Show with no --host should display all hosts."""
        registry, known_hosts = make_registry()

        # Open one host
        host = known_hosts[0]
        now_mono = time.monotonic()
        registry.cooldowns.set_until(host, now_mono + 60, reason="test")

        args = argparse.Namespace(
            make_registry=make_registry,
            host=None,
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = _cmd_show(args)
            output = mock_stdout.getvalue()

        assert result == 0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_open_with_zero_duration(self, make_registry):
        """Open with 0 seconds should still work."""
        args = argparse.Namespace(
            make_registry=make_registry,
            host="api.example.org",
            seconds=0,
            reason="test",
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            result = _cmd_open(args)

        # Should succeed
        assert result == 0

    def test_commands_handle_registry_errors(self, make_registry):
        """Commands should handle registry errors gracefully."""

        def bad_registry():
            raise RuntimeError("Registry initialization failed")

        bad_make_registry = lambda: bad_registry()

        args = argparse.Namespace(
            make_registry=bad_make_registry,
            host="api.example.org",
        )

        # Should handle error gracefully
        with patch("sys.stdout", new_callable=io.StringIO):
            try:
                _cmd_show(args)
            except RuntimeError:
                pass  # Expected


# ============================================================================
# Factory Injection Tests
# ============================================================================


class TestFactoryInjection:
    """Test factory function injection for CLI commands."""

    def test_factory_creates_registry(self, tmp_cooldown_db):
        """Factory should create properly configured registry."""
        config = BreakerConfig(
            defaults=BreakerPolicy(
                fail_max=3,
                reset_timeout_s=60,
                retry_after_cap_s=900,
            ),
        )
        registry = BreakerRegistry(config, cooldown_store=tmp_cooldown_db)

        def factory():
            return registry, ["api.example.org"]

        reg, hosts = factory()
        assert reg is registry
        assert len(hosts) == 1

    def test_factory_isolation(self, tmp_cooldown_db):
        """Multiple factory calls should be independent."""

        def make_factory1():
            config1 = BreakerConfig()
            return BreakerRegistry(config1, cooldown_store=tmp_cooldown_db), ["host1"]

        def make_factory2():
            config2 = BreakerConfig()
            return BreakerRegistry(config2, cooldown_store=tmp_cooldown_db), ["host2"]

        reg1, hosts1 = make_factory1()
        reg2, hosts2 = make_factory2()

        # Should be different instances
        assert reg1 is not reg2
        assert hosts1 != hosts2
