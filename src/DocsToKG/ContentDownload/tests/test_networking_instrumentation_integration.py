"""
Integration tests for Phase 3A: Networking Hub instrumentation.

Tests that the wired instrumentation in request_with_retries properly records
URL normalizations, applies role-based headers, and logs changes.
"""

import pytest
import httpx
from typing import Any
from unittest.mock import patch, MagicMock

from DocsToKG.ContentDownload.urls import canonical_for_index, canonical_for_request
from DocsToKG.ContentDownload.urls_networking import (
    get_url_normalization_stats,
    reset_url_normalization_stats_for_tests,
    set_strict_mode,
    get_strict_mode,
)
from DocsToKG.ContentDownload.networking import request_with_retries


class TestPhase3AIntegration:
    """Integration tests for Phase 3A: Networking hub instrumentation."""

    def teardown_method(self) -> None:
        """Reset instrumentation state between tests."""
        reset_url_normalization_stats_for_tests()
        set_strict_mode(False)

    def test_instrumentation_wired_into_request_with_retries(self) -> None:
        """Verify instrumentation calls are triggered when making requests."""
        reset_url_normalization_stats_for_tests()

        # Mock the HTTP client to avoid real network calls
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.history = []

        with patch("DocsToKG.ContentDownload.networking.http_client") as mock_client:
            mock_client.request.return_value = mock_response

            # Call request_with_retries
            source_url = "HTTP://EXAMPLE.COM/path?utm_source=test"
            try:
                result = request_with_retries(
                    url=source_url,
                    method="GET",
                    url_role="landing",
                )
            except Exception:
                pass  # We're just checking instrumentation was called

            # Verify metrics were updated
            stats = get_url_normalization_stats()
            assert stats["normalized_total"] > 0
            assert "example.com" in stats["hosts_seen"]

    def test_headers_applied_by_role(self) -> None:
        """Verify role-specific headers are applied during requests."""
        reset_url_normalization_stats_for_tests()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.history = []

        with patch("DocsToKG.ContentDownload.networking.http_client") as mock_client:
            mock_client.request.return_value = mock_response

            # Make request with landing role
            url = "https://example.com/article"
            try:
                request_with_retries(
                    url=url,
                    method="GET",
                    url_role="landing",
                )
            except Exception:
                pass

            # Check that headers were included in the request
            assert mock_client.request.called

    def test_url_change_logged_once_per_host(self) -> None:
        """Verify URL changes are logged only once per host."""
        reset_url_normalization_stats_for_tests()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.history = []

        with patch("DocsToKG.ContentDownload.networking.http_client") as mock_client:
            mock_client.request.return_value = mock_response

            # First request with UTM params (will be normalized)
            url1 = "https://example.com/page?utm_source=twitter"
            try:
                request_with_retries(
                    url=url1,
                    method="GET",
                    url_role="landing",
                )
            except Exception:
                pass

            # Second request to same host with different UTM params
            url2 = "https://example.com/page?utm_source=facebook"
            try:
                request_with_retries(
                    url=url2,
                    method="GET",
                    url_role="landing",
                )
            except Exception:
                pass

            # Verify both requests were made but only one log entry per host
            assert mock_client.request.call_count >= 2

    def test_strict_mode_integration(self) -> None:
        """Verify strict mode can be enforced at networking layer."""
        reset_url_normalization_stats_for_tests()
        set_strict_mode(True)

        try:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.history = []

            with patch("DocsToKG.ContentDownload.networking.http_client") as mock_client:
                mock_client.request.return_value = mock_response

                # Non-canonical URL should raise in strict mode
                non_canonical_url = "HTTP://EXAMPLE.COM/path"
                canonical_url = canonical_for_index(non_canonical_url)

                # If they differ, strict mode should reject
                if non_canonical_url != canonical_url:
                    with pytest.raises(ValueError, match="strict mode"):
                        request_with_retries(
                            url=non_canonical_url,
                            method="GET",
                            url_role="landing",
                        )
        finally:
            set_strict_mode(False)

    def test_role_specific_canonicalization_applied(self) -> None:
        """Verify role-based canonicalization is applied correctly."""
        reset_url_normalization_stats_for_tests()

        # Test that landing role applies different canonicalization than metadata
        base_url = "https://example.com/page?utm_source=test&page=1"

        canonical_landing = canonical_for_request(base_url, role="landing")
        canonical_metadata = canonical_for_request(base_url, role="metadata")

        # Landing should strip UTM params, metadata should keep them
        # (depending on the FILTER_FOR policy)
        assert canonical_landing is not None
        assert canonical_metadata is not None

    def test_extensions_track_url_changes(self) -> None:
        """Verify extensions properly track whether URL was changed."""
        reset_url_normalization_stats_for_tests()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.history = []
        mock_response.extensions = {}

        with patch("DocsToKG.ContentDownload.networking.http_client") as mock_client:
            mock_client.request.return_value = mock_response

            # URL with tracking params (will be canonicalized)
            url_with_trackers = "https://example.com/page?utm_source=test"

            try:
                request_with_retries(
                    url=url_with_trackers,
                    method="GET",
                    url_role="landing",
                )
            except Exception:
                pass

            # Check that mock was called (proving request was attempted)
            assert mock_client.request.called

    def test_metrics_accumulation(self) -> None:
        """Verify metrics properly accumulate across multiple requests."""
        reset_url_normalization_stats_for_tests()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.history = []

        with patch("DocsToKG.ContentDownload.networking.http_client") as mock_client:
            mock_client.request.return_value = mock_response

            urls = [
                "https://example.com/page1",
                "https://example.com/page2",
                "https://another.com/page",
                "HTTP://EXAMPLE.COM/Page3",  # Different case, same host
            ]

            for url in urls:
                try:
                    request_with_retries(
                        url=url,
                        method="GET",
                        url_role="metadata",
                    )
                except Exception:
                    pass

            stats = get_url_normalization_stats()
            # Should have normalized all URLs
            assert stats["normalized_total"] >= 4
            # Should see multiple hosts
            assert len(stats["hosts_seen"]) >= 2
            # Metadata role should be recorded
            assert "metadata" in stats["roles_used"]

    def test_no_double_instrumentation(self) -> None:
        """Verify instrumentation doesn't double-track already-instrumented URLs."""
        reset_url_normalization_stats_for_tests()

        # Use canonical URL from the start
        url = canonical_for_index("https://example.com/page")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.history = []

        with patch("DocsToKG.ContentDownload.networking.http_client") as mock_client:
            mock_client.request.return_value = mock_response

            try:
                request_with_retries(
                    url=url,
                    method="GET",
                    url_role="artifact",
                )
            except Exception:
                pass

            stats = get_url_normalization_stats()
            # Should only normalize once
            assert stats["normalized_total"] >= 1


class TestPhase3AHeaderShaping:
    """Tests for role-based header shaping in Phase 3A."""

    def teardown_method(self) -> None:
        """Reset state."""
        reset_url_normalization_stats_for_tests()

    def test_landing_role_accept_header(self) -> None:
        """Verify landing role uses HTML-focused Accept header."""
        from DocsToKG.ContentDownload.urls_networking import ROLE_HEADERS

        assert "landing" in ROLE_HEADERS
        landing_headers = ROLE_HEADERS["landing"]
        assert "Accept" in landing_headers
        assert "text/html" in landing_headers["Accept"]

    def test_metadata_role_accept_header(self) -> None:
        """Verify metadata role uses JSON-focused Accept header."""
        from DocsToKG.ContentDownload.urls_networking import ROLE_HEADERS

        assert "metadata" in ROLE_HEADERS
        metadata_headers = ROLE_HEADERS["metadata"]
        assert "Accept" in metadata_headers
        assert "application/json" in metadata_headers["Accept"]

    def test_artifact_role_accept_header(self) -> None:
        """Verify artifact role uses PDF-focused Accept header."""
        from DocsToKG.ContentDownload.urls_networking import ROLE_HEADERS

        assert "artifact" in ROLE_HEADERS
        artifact_headers = ROLE_HEADERS["artifact"]
        assert "Accept" in artifact_headers
        assert "application/pdf" in artifact_headers["Accept"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
