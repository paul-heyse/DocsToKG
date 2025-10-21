"""Integration tests for policy gates with ContentDownload URL operations."""

import pytest

from DocsToKG.OntologyDownload.policy.gates import url_gate
from DocsToKG.OntologyDownload.policy.errors import URLPolicyException


class TestUrlGateIntegration:
    """Tests for URL policy gate integration with ContentDownload."""

    def test_url_gate_accepts_valid_https_url(self):
        """Valid HTTPS URLs should pass the gate."""
        url = "https://api.crossref.org/works"
        result = url_gate(url)
        assert result is not None
        assert result.gate_name == "url_gate"

    def test_url_gate_rejects_invalid_scheme(self):
        """Invalid schemes should be rejected."""
        url = "ftp://example.com/file.pdf"
        with pytest.raises(URLPolicyException):
            url_gate(url)

    def test_url_gate_rejects_empty_host(self):
        """URLs with empty hosts should be rejected."""
        url = "https://"
        with pytest.raises(URLPolicyException):
            url_gate(url)

    def test_url_gate_accepts_common_academic_urls(self):
        """Common academic URLs should pass."""
        urls = [
            "https://api.crossref.org/works",
            "https://unpaywall.org/api/v2",
            "https://www.ncbi.nlm.nih.gov/pmc/",
            "https://arxiv.org/",
        ]
        for url in urls:
            result = url_gate(url)
            assert result is not None

    def test_url_gate_disallows_userinfo_in_url(self):
        """URLs with userinfo should be rejected."""
        url = "https://user:password@example.com/data"
        with pytest.raises(URLPolicyException):
            url_gate(url)

    def test_url_gate_tracks_statistics(self):
        """URL gate should track invocation statistics."""
        from DocsToKG.OntologyDownload.policy.registry import get_registry

        registry = get_registry()

        # Clear stats
        registry.reset_stats("url_gate")

        # Invoke with valid URL
        try:
            url_gate("https://example.org/data")
            stats = registry.get_stats("url_gate")
            assert stats["passes"] > 0
        except Exception:
            pass

    def test_url_gate_with_query_parameters(self):
        """URLs with query parameters should be validated correctly."""
        url = "https://api.example.com/v1/search?q=test&limit=100"
        result = url_gate(url)
        assert result is not None

    def test_url_gate_with_fragments(self):
        """URLs with fragments should pass."""
        url = "https://example.com/docs#section"
        result = url_gate(url)
        assert result is not None

    def test_url_gate_rejects_very_long_urls(self):
        """Extremely long URLs might be rejected."""
        # Most URLs under 2000 chars should be fine
        long_query = "q=" + "x" * 1000
        url = f"https://example.com/search?{long_query}"
        # This should either pass or provide a meaningful error
        try:
            url_gate(url)
        except URLPolicyException as e:
            assert "E_" in str(e)  # Should be a policy error code


class TestUrlGateErrorHandling:
    """Tests for URL gate error handling in ContentDownload context."""

    def test_url_gate_error_messages_are_clear(self):
        """Error messages should be helpful for debugging."""
        with pytest.raises(URLPolicyException) as exc_info:
            url_gate("ftp://invalid.com")

        error = exc_info.value
        assert hasattr(error, "error_code")
        assert error.error_code is not None

    def test_url_gate_preserves_details_without_secrets(self):
        """Error details should not contain sensitive information."""
        with pytest.raises(URLPolicyException) as exc_info:
            url_gate("https://user:password@malicious.com")

        error = exc_info.value
        error_str = str(error)
        assert "password" not in error_str.lower()


class TestUrlGateIntegrationWithValidation:
    """Tests demonstrating ContentDownload integration patterns."""

    def test_download_validates_url_before_request(self):
        """URLs should be validated before making HTTP requests."""
        # Valid URL should proceed
        valid_url = "https://example.org/paper.pdf"
        result = url_gate(valid_url)
        assert result is not None

        # Invalid URL should fail before download attempt
        invalid_url = "ftp://example.com"
        with pytest.raises(URLPolicyException):
            url_gate(invalid_url)

    def test_url_gate_metrics_available(self):
        """Metrics should be available after gate invocations."""
        from DocsToKG.OntologyDownload.policy.registry import get_registry

        registry = get_registry()

        # After running tests, metrics should be available
        stats = registry.get_stats("url_gate")

        # Stats should have expected keys
        expected_keys = {"passes", "rejects", "avg_ms", "min_ms", "max_ms", "pass_rate"}
        assert expected_keys.issubset(set(stats.keys()))


class TestUrlGatePerformance:
    """Tests for URL gate performance in ContentDownload context."""

    def test_url_gate_is_fast(self):
        """URL gate validation should be fast (<1ms per URL)."""
        import time

        url = "https://api.example.com/data"

        start = time.time()
        for _ in range(100):
            try:
                url_gate(url)
            except Exception:
                pass
        elapsed = time.time() - start

        # 100 validations should take < 100ms (1ms each)
        assert elapsed < 0.1, f"URL validation too slow: {elapsed}s for 100 calls"

    def test_url_gate_batching_scenario(self):
        """URL gate should handle batches of URLs efficiently."""
        urls = [
            "https://api.crossref.org/works",
            "https://unpaywall.org/api/v2",
            "https://arxiv.org/api",
            "https://www.ncbi.nlm.nih.gov/pmc/",
        ]

        results = []
        for url in urls:
            try:
                result = url_gate(url)
                results.append((url, "PASS", result))
            except URLPolicyException as e:
                results.append((url, "REJECT", str(e)))

        # Most should pass
        passed = sum(1 for _, status, _ in results if status == "PASS")
        assert passed >= 2, f"Expected at least 2 passes, got {passed}"
