"""Tests for URL normalization instrumentation in the networking hub.

Validates:
- Metrics tracking (normalized_total, changed_total, unique_hosts, roles_used)
- Strict mode validation and enforcement
- Role-based header injection (metadata/landing/artifact)
- Logging of URL changes (once per host)
"""

from __future__ import annotations

import os
import pytest

from DocsToKG.ContentDownload.urls_networking import (
    get_strict_mode,
    set_strict_mode,
    record_url_normalization,
    log_url_change_once,
    apply_role_headers,
    get_url_normalization_stats,
    reset_url_normalization_stats_for_tests,
    ROLE_HEADERS,
)


class TestMetricsTracking:
    """Test URL normalization metrics collection."""

    def teardown_method(self):
        """Reset metrics after each test."""
        reset_url_normalization_stats_for_tests()
        set_strict_mode(False)

    def test_track_normalized_total(self):
        """Record counter increments on each normalization."""
        assert get_url_normalization_stats()["normalized_total"] == 0
        
        record_url_normalization("http://example.com", "http://example.com/", "metadata")
        assert get_url_normalization_stats()["normalized_total"] == 1
        
        record_url_normalization("https://other.org/path", "https://other.org/path", "landing")
        assert get_url_normalization_stats()["normalized_total"] == 2

    def test_track_changed_count(self):
        """Track only URLs that were modified."""
        # Unchanged URL
        record_url_normalization("https://example.com/", "https://example.com/", "metadata")
        assert get_url_normalization_stats()["changed_total"] == 0
        
        # Changed URL (e.g., port dropped or params filtered)
        record_url_normalization("https://example.com:443/", "https://example.com/", "metadata")
        assert get_url_normalization_stats()["changed_total"] == 1
        
        # Another unchanged
        record_url_normalization("https://other.org/x", "https://other.org/x", "artifact")
        assert get_url_normalization_stats()["changed_total"] == 1  # Still 1

    def test_track_unique_hosts(self):
        """Accumulate unique hosts from canonical URLs."""
        record_url_normalization("http://example.com", "https://example.com/", "metadata")
        record_url_normalization("http://other.org/path", "https://other.org/path", "landing")
        record_url_normalization("https://example.com/file", "https://example.com/file", "artifact")
        
        stats = get_url_normalization_stats()
        assert stats["unique_hosts"] == 2
        assert "example.com" in stats["hosts_seen"]
        assert "other.org" in stats["hosts_seen"]

    def test_track_roles_used(self):
        """Count requests by role."""
        record_url_normalization("http://api.example.com", "https://api.example.com/", "metadata")
        record_url_normalization("http://api.example.com", "https://api.example.com/", "metadata")
        record_url_normalization("http://example.com/page", "https://example.com/page", "landing")
        record_url_normalization("http://cdn.example.com/file.pdf", "https://cdn.example.com/file.pdf", "artifact")
        
        stats = get_url_normalization_stats()
        assert stats["roles_used"]["metadata"] == 2
        assert stats["roles_used"]["landing"] == 1
        assert stats["roles_used"]["artifact"] == 1

    def test_stats_snapshot(self):
        """get_url_normalization_stats returns a clean snapshot."""
        record_url_normalization("http://example.com", "https://example.com/", "metadata")
        record_url_normalization("http://other.org", "https://other.org/", "landing")
        
        stats = get_url_normalization_stats()
        
        # Verify expected keys
        assert "normalized_total" in stats
        assert "changed_total" in stats
        assert "unique_hosts" in stats
        assert "hosts_seen" in stats
        assert "roles_used" in stats
        assert "strict_mode" in stats
        
        # Verify values
        assert stats["normalized_total"] == 2
        assert stats["unique_hosts"] == 2
        assert isinstance(stats["hosts_seen"], list)
        assert isinstance(stats["roles_used"], dict)


class TestStrictMode:
    """Test strict URL validation mode."""

    def teardown_method(self):
        """Reset state after each test."""
        reset_url_normalization_stats_for_tests()
        set_strict_mode(False)

    def test_strict_mode_disabled_by_default(self):
        """Strict mode should be off by default."""
        assert get_strict_mode() is False

    def test_set_strict_mode_enabled(self):
        """Can toggle strict mode on."""
        set_strict_mode(True)
        assert get_strict_mode() is True

    def test_strict_mode_disabled_allows_changes(self):
        """With strict mode off, URL changes are allowed."""
        set_strict_mode(False)
        # Should not raise
        record_url_normalization("http://example.com:443/", "https://example.com/", "metadata")

    def test_strict_mode_enabled_rejects_changes(self):
        """With strict mode on, non-canonical URLs raise ValueError."""
        set_strict_mode(True)
        
        with pytest.raises(ValueError, match="Non-canonical URL in DOCSTOKG_URL_STRICT=1 mode"):
            record_url_normalization("HTTP://EXAMPLE.COM", "https://example.com/", "metadata")

    def test_strict_mode_allows_canonical_urls(self):
        """Strict mode allows URLs that don't change."""
        set_strict_mode(True)
        # Same URL - should not raise
        record_url_normalization("https://example.com/", "https://example.com/", "metadata")

    def test_strict_mode_stats_reflect_setting(self):
        """Stats snapshot includes strict_mode value."""
        set_strict_mode(True)
        stats = get_url_normalization_stats()
        assert stats["strict_mode"] is True
        
        set_strict_mode(False)
        stats = get_url_normalization_stats()
        assert stats["strict_mode"] is False


class TestRoleBasedHeaders:
    """Test role-based header injection."""

    def test_metadata_role_headers(self):
        """Metadata role gets JSON Accept header."""
        headers = apply_role_headers(None, "metadata")
        assert headers["Accept"] == "application/json, text/javascript;q=0.9, */*;q=0.1"

    def test_landing_role_headers(self):
        """Landing role gets HTML Accept header."""
        headers = apply_role_headers(None, "landing")
        assert headers["Accept"] == "text/html, application/xhtml+xml;q=0.9, */*;q=0.8"

    def test_artifact_role_headers(self):
        """Artifact role gets PDF Accept header."""
        headers = apply_role_headers(None, "artifact")
        assert headers["Accept"] == "application/pdf, */*;q=0.1"

    def test_preserve_existing_headers(self):
        """Existing headers are preserved."""
        input_headers = {"Authorization": "Bearer token", "X-Custom": "value"}
        result = apply_role_headers(input_headers, "metadata")
        
        assert result["Authorization"] == "Bearer token"
        assert result["X-Custom"] == "value"
        assert result["Accept"] == "application/json, text/javascript;q=0.9, */*;q=0.1"

    def test_caller_accept_header_overrides_role_default(self):
        """If caller provides Accept header, don't override."""
        input_headers = {"Accept": "text/plain"}
        result = apply_role_headers(input_headers, "metadata")
        
        # Should keep caller's Accept, not use metadata default
        assert result["Accept"] == "text/plain"

    def test_none_headers_creates_new_dict(self):
        """Passing None creates a new dict."""
        result = apply_role_headers(None, "metadata")
        assert isinstance(result, dict)
        assert "Accept" in result

    def test_does_not_mutate_input(self):
        """apply_role_headers doesn't mutate the input dict."""
        input_headers = {"X-Original": "value"}
        apply_role_headers(input_headers, "metadata")
        
        # Input should be unchanged
        assert "Accept" not in input_headers
        assert input_headers == {"X-Original": "value"}


class TestLoggingBehavior:
    """Test URL change logging (once per host)."""

    def teardown_method(self):
        """Reset after each test."""
        reset_url_normalization_stats_for_tests()

    def test_log_url_change_once_no_change(self, caplog):
        """No logging if URL didn't change."""
        log_url_change_once("https://example.com/", "https://example.com/")
        assert len(caplog.records) == 0

    def test_log_url_change_once_with_change(self, caplog):
        """Logs when URL changes."""
        log_url_change_once("http://example.com:443/", "https://example.com/")
        assert len(caplog.records) >= 1
        assert "URL normalized" in caplog.text

    def test_log_once_per_host(self, caplog):
        """Only logs once per host, not for every URL."""
        caplog.clear()
        log_url_change_once("http://example.com/page1", "https://example.com/page1")
        first_count = len(caplog.records)
        
        caplog.clear()
        log_url_change_once("http://example.com/page2", "https://example.com/page2")
        second_count = len(caplog.records)
        
        # Second call should not log (already logged for this host)
        assert first_count >= 1
        assert second_count == 0

    def test_log_multiple_hosts(self, caplog):
        """Logs for different hosts."""
        log_url_change_once("http://example.com:443/", "https://example.com/")
        log_url_change_once("http://other.org:443/", "https://other.org/")
        
        # Should log for both hosts
        assert "example.com" in caplog.text
        assert "other.org" in caplog.text


class TestRoleHeadersConstant:
    """Test ROLE_HEADERS constant."""

    def test_role_headers_has_all_roles(self):
        """All three roles are defined."""
        assert "metadata" in ROLE_HEADERS
        assert "landing" in ROLE_HEADERS
        assert "artifact" in ROLE_HEADERS

    def test_role_headers_have_accept(self):
        """Each role has an Accept header."""
        for role, headers in ROLE_HEADERS.items():
            assert "Accept" in headers
            assert isinstance(headers["Accept"], str)
            assert len(headers["Accept"]) > 0

    def test_role_headers_distinct(self):
        """Each role has a different Accept header."""
        values = {headers["Accept"] for headers in ROLE_HEADERS.values()}
        assert len(values) == 3  # All different


class TestResetForTests:
    """Test test fixture reset behavior."""

    def test_reset_clears_metrics(self):
        """Reset clears all metrics."""
        record_url_normalization("http://example.com", "https://example.com/", "metadata")
        assert get_url_normalization_stats()["normalized_total"] == 1
        
        reset_url_normalization_stats_for_tests()
        assert get_url_normalization_stats()["normalized_total"] == 0
        assert get_url_normalization_stats()["changed_total"] == 0
        assert get_url_normalization_stats()["unique_hosts"] == 0

    def test_reset_clears_logged_urls(self, caplog):
        """Reset clears the logged URL changes set."""
        log_url_change_once("http://example.com:443/", "https://example.com/")
        initial_count = len(caplog.records)
        
        # Clear logs
        caplog.clear()
        reset_url_normalization_stats_for_tests()
        
        # After reset, same host should log again
        log_url_change_once("http://example.com:443/", "https://example.com/")
        assert len(caplog.records) >= 1  # Should log again
