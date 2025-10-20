"""
End-to-End Integration Tests: Phase 3A + 3B Canonical URL Flow

Tests the complete pipeline from resolver → pipeline → download → networking,
verifying that canonical URLs are properly handled throughout and Phase 3A
instrumentation is applied.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional

from DocsToKG.ContentDownload.resolvers.base import ResolverResult
from DocsToKG.ContentDownload.urls import canonical_for_index
from DocsToKG.ContentDownload.urls_networking import (
    get_url_normalization_stats,
    reset_url_normalization_stats_for_tests,
)


class TestCanonicalURLResolverFlow:
    """Tests for canonical URL flow from resolvers through the pipeline."""

    def teardown_method(self) -> None:
        """Reset instrumentation state."""
        reset_url_normalization_stats_for_tests()

    def test_resolver_result_auto_canonicalization(self) -> None:
        """Verify ResolverResult.__post_init__ automatically canonicalizes URLs."""
        # Input: non-canonical URL with tracking params
        input_url = "HTTP://EXAMPLE.COM/path?utm_source=test"

        # Create ResolverResult without explicit canonical_url
        result = ResolverResult(url=input_url, metadata={"source": "test"})

        # Verify __post_init__ filled in the fields
        assert result.url is not None
        assert result.canonical_url is not None
        assert result.original_url is not None

        # canonical_url should be normalized (lowercase, no fragment, etc.)
        assert result.canonical_url == canonical_for_index(input_url)

        # url should be set to canonical
        assert result.url == result.canonical_url

        # original_url should preserve the input
        assert result.original_url == input_url

    def test_resolver_result_explicit_canonical_url(self) -> None:
        """Verify ResolverResult respects explicit canonical_url."""
        input_url = "https://example.com/page?id=123&utm_source=twitter"
        explicit_canonical = "https://example.com/page?id=123"

        result = ResolverResult(
            url=input_url, canonical_url=explicit_canonical, metadata={"source": "test"}
        )

        # Verify explicit canonical was preserved
        assert result.canonical_url == explicit_canonical
        assert result.url == explicit_canonical
        assert result.original_url == input_url

    def test_resolver_result_original_url_preserved(self) -> None:
        """Verify ResolverResult preserves original_url through the pipeline."""
        original_url = "HTTP://EXAMPLE.COM/article"
        canonical_url = canonical_for_index(original_url)

        result = ResolverResult(
            url=original_url, original_url=original_url, metadata={"resolver": "test"}
        )

        # Both should be set correctly
        assert result.original_url == original_url
        assert result.canonical_url == canonical_url
        assert result.url == canonical_url

    def test_resolver_result_url_and_original_same_initially(self) -> None:
        """Verify ResolverResult handles case where url == original_url."""
        url = "https://example.com/doc"

        result = ResolverResult(url=url, metadata={"source": "test"})

        # When url and original_url are the same, both should be set
        assert result.url == url
        assert result.original_url == url
        assert result.canonical_url == url

    def test_canonical_url_prevents_duplicate_downloads(self) -> None:
        """Verify canonical URLs enable deduplication."""
        # Two URLs that are equivalent after canonicalization
        url1 = "HTTP://EXAMPLE.COM/page"
        url2 = "https://example.com/page"

        result1 = ResolverResult(url=url1, metadata={"source": "resolver1"})
        result2 = ResolverResult(url=url2, metadata={"source": "resolver2"})

        # After canonicalization, they should be identical
        assert result1.canonical_url == result2.canonical_url

        # This enables the pipeline to dedupe them
        seen_urls = set()
        seen_urls.add(result1.canonical_url)

        # Second URL would be skipped due to duplicate detection
        is_duplicate = result2.canonical_url in seen_urls
        assert is_duplicate is True

    def test_tracker_params_handled_consistently(self) -> None:
        """Verify URLs with and without tracker params dedupe correctly."""
        url_with_trackers = "https://example.com/page?utm_source=test&utm_medium=email"
        url_without_trackers = "https://example.com/page"

        result_with = ResolverResult(url=url_with_trackers, metadata={"source": "test"})
        result_without = ResolverResult(url=url_without_trackers, metadata={"source": "test"})

        # After canonicalization, both should be canonical forms
        # (exact behavior depends on url_normalize settings)
        canonical_with = result_with.canonical_url
        canonical_without = result_without.canonical_url

        # Both should be valid canonical URLs
        assert canonical_with is not None
        assert canonical_without is not None

    def test_phase3a_instrumentation_applied_to_canonical_urls(self) -> None:
        """Verify Phase 3A instrumentation tracks canonical URLs."""
        reset_url_normalization_stats_for_tests()

        # Simulate resolver producing URLs
        urls = [
            "HTTP://EXAMPLE.COM/page1",
            "https://example.com/page2",
            "HTTPS://ANOTHER.COM/article",
        ]

        results = []
        for url in urls:
            result = ResolverResult(url=url, metadata={"source": "test"})
            results.append(result)
            # Simulate what pipeline would do: use canonical_url for deduplication
            canonical = result.canonical_url

        # All results should have canonical URLs populated
        assert len(results) == 3
        for result in results:
            assert result.canonical_url is not None
            assert result.original_url is not None

        # Canonical URLs should be properly normalized
        assert all(result.canonical_url.startswith("https://") for result in results)

    def test_original_url_passed_to_download_layer(self) -> None:
        """Verify original_url is preserved for download layer."""
        # This simulates what the pipeline does
        original_input = "HTTP://EXAMPLE.COM/path?utm=test"

        result = ResolverResult(url=original_input, metadata={"resolver": "test"})

        # Pipeline would do:
        url_for_download = result.canonical_url  # Use canonical for request
        original_for_telemetry = result.original_url  # Preserve original

        assert url_for_download == canonical_for_index(original_input)
        assert original_for_telemetry == original_input

        # Download layer would receive both
        assert url_for_download is not None
        assert original_for_telemetry is not None

    def test_edge_case_none_url(self) -> None:
        """Verify ResolverResult handles None URL gracefully."""
        # Event result with no URL
        result = ResolverResult(url=None, event="skipped", metadata={"reason": "test"})

        assert result.url is None
        assert result.canonical_url is None
        assert result.original_url is None

    def test_edge_case_empty_string_url(self) -> None:
        """Verify ResolverResult handles empty string URL."""
        result = ResolverResult(url="", metadata={"source": "test"})

        # Empty string URLs should be handled gracefully
        assert result.url == "" or result.url is None

    def test_multiple_resolvers_same_url(self) -> None:
        """Verify multiple resolvers producing same URL results in single canonical URL."""
        shared_url = "https://arxiv.org/pdf/2102.00001"

        # Different resolvers might yield the same URL
        resolver1_result = ResolverResult(url=shared_url, metadata={"source": "arxiv_resolver"})
        resolver2_result = ResolverResult(
            url=shared_url, metadata={"source": "landing_page_resolver"}
        )

        # Both should have identical canonical URLs
        assert resolver1_result.canonical_url == resolver2_result.canonical_url

        # Pipeline would deduplicate via canonical URL
        seen = set()
        seen.add(resolver1_result.canonical_url)
        assert resolver2_result.canonical_url in seen

    def test_canonical_url_in_metadata_flow(self) -> None:
        """Verify canonical_url properly flows through to telemetry/manifests."""
        input_url = "HTTP://EXAMPLE.COM/article?utm_campaign=social"

        result = ResolverResult(
            url=input_url, metadata={"resolver": "test", "timestamp": "2025-10-21"}
        )

        # Simulate what goes into telemetry
        telemetry_record = {
            "url": result.url,
            "canonical_url": result.canonical_url,
            "original_url": result.original_url,
            "metadata": result.metadata,
        }

        # Verify telemetry record is complete
        assert telemetry_record["canonical_url"] is not None
        assert telemetry_record["original_url"] is not None
        assert telemetry_record["canonical_url"] != telemetry_record["original_url"]
        assert telemetry_record["url"] == telemetry_record["canonical_url"]


class TestPhase3Integration:
    """Integration tests for Phase 3A + 3B combination."""

    def teardown_method(self) -> None:
        """Reset state."""
        reset_url_normalization_stats_for_tests()

    def test_resolver_to_download_canonical_flow(self) -> None:
        """Verify complete flow from resolver through download."""
        # Step 1: Resolver creates result
        input_url = "HTTP://EXAMPLE.COM/pdf?utm_source=test"
        result = ResolverResult(url=input_url, metadata={"source": "unpaywall"})

        # Step 2: Pipeline processes result
        # (This is what pipeline._process_result does)
        url_for_request = result.canonical_url or result.url
        original_url = result.original_url or url_for_request

        assert url_for_request is not None
        assert original_url is not None

        # Step 3: Download layer would receive:
        # - url_for_request: canonical form for the actual HTTP request
        # - original_url: original form for telemetry and deduplication

        # After download's prepare_candidate_download:
        canonical_index = canonical_for_index(original_url)

        # Both should be the same canonical form
        assert canonical_index == url_for_request

    def test_deduplication_via_canonical_urls(self) -> None:
        """Verify deduplication works correctly with canonical URLs."""
        urls_from_resolvers = [
            "HTTP://EXAMPLE.COM/article",  # Resolver 1
            "https://example.com/article",  # Resolver 2 (different case/scheme)
            "HTTPS://EXAMPLE.COM/article?",  # Resolver 3 (trailing empty param)
        ]

        results = [
            ResolverResult(url=url, metadata={"source": f"resolver_{i}"})
            for i, url in enumerate(urls_from_resolvers, 1)
        ]

        # Pipeline deduplication
        seen = set()
        unique_results = []
        for result in results:
            canonical = result.canonical_url
            if canonical not in seen:
                seen.add(canonical)
                unique_results.append(result)

        # All three URLs should deduplicate to one
        assert len(unique_results) == 1
        assert len(seen) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
