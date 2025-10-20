"""Tests for DocsToKG.ContentDownload.urls canonicalization module.

This test suite validates:
- RFC 3986/3987 normalization (case, escapes, dot-segments, ports)
- Role-based param filtering (metadata/landing/artifact)
- Host extraction for limiter/breaker keys
- IDN and punycode handling
- Policy configuration and environment overrides
"""

from __future__ import annotations

import pytest

from DocsToKG.ContentDownload.urls import (
    canonical_for_index,
    canonical_for_request,
    canonical_host,
    configure_url_policy,
    get_url_policy,
    parse_param_allowlist_spec,
    reset_url_policy_for_tests,
)


class TestCanonicalForIndex:
    """Test URL canonicalization for dedupe/manifest indexing (no filtering)."""

    def test_lowercase_scheme_and_host(self):
        """Scheme and host should be lowercased."""
        # Note: url_normalize preserves explicit schemes (doesn't default to https)
        assert canonical_for_index("http://EXAMPLE.COM/path") == "http://example.com/path"
        assert canonical_for_index("https://MiXeD.ExAmPlE.Com/") == "https://mixed.example.com/"

    def test_default_path_slash(self):
        """Empty paths on http/https should get `/`."""
        assert canonical_for_index("https://example.com") == "https://example.com/"
        assert canonical_for_index("http://example.com") == "http://example.com/"

    def test_drop_default_ports(self):
        """Scheme-default ports should be dropped."""
        # http default is 80
        assert canonical_for_index("http://example.com:80/path") == "http://example.com/path"
        # https default is 443
        assert canonical_for_index("https://example.com:443/path") == "https://example.com/path"

    def test_preserve_non_default_ports(self):
        """Non-default ports should be preserved."""
        assert canonical_for_index("https://example.com:8080/path") == "https://example.com:8080/path"
        assert canonical_for_index("http://example.com:9000/path") == "http://example.com:9000/path"

    def test_default_scheme_https(self):
        """URLs without scheme should default to https."""
        assert canonical_for_index("example.com/path") == "https://example.com/path"
        assert canonical_for_index("www.example.com") == "https://www.example.com/"

    def test_preserve_query_string_no_filtering(self):
        """Query parameters should NOT be filtered or reordered in canonical_for_index."""
        result = canonical_for_index("https://example.com?b=2&a=1&utm_source=x")
        assert "b=2&a=1&utm_source=x" in result

    def test_remove_fragment(self):
        """URL fragments should be stripped."""
        assert canonical_for_index("https://example.com/path#section") == "https://example.com/path"
        assert canonical_for_index("https://example.com/path?q=1#frag") == "https://example.com/path?q=1"

    def test_remove_dot_segments(self):
        """Dot segments in path should be removed."""
        assert canonical_for_index("https://example.com/a/./b/../c") == "https://example.com/a/c"


class TestCanonicalForRequest:
    """Test URL canonicalization for HTTP requests with role-based filtering."""

    def teardown_method(self):
        """Reset policy after each test."""
        reset_url_policy_for_tests()

    def test_metadata_role_no_filtering(self):
        """Metadata role should preserve all query params."""
        url = "https://example.com?utm_source=x&id=1&ref=old"
        result = canonical_for_request(url, role="metadata")
        assert "utm_source=x" in result
        assert "id=1" in result
        assert "ref=old" in result

    def test_landing_role_no_allowlist_drops_all(self):
        """Landing role without allowlist drops all params (default url-normalize behavior)."""
        url = "https://example.com?id=1&utm_source=x&gclid=abc"
        result = canonical_for_request(url, role="landing")
        # With no allowlist, all params are dropped
        assert result == "https://example.com/"

    def test_landing_role_with_allowlist_preserves_specified(self):
        """Landing role with allowlist preserves only specified params."""
        configure_url_policy(param_allowlist_per_domain={"example.com": ["id", "page"]})
        try:
            url = "https://example.com?id=1&utm_source=x&gclid=abc&page=1"
            result = canonical_for_request(url, role="landing")
            # Should keep: id, page
            assert "id=1" in result
            assert "page=1" in result
            # Should drop: utm_source, gclid
            assert "utm_source" not in result
            assert "gclid" not in result
        finally:
            reset_url_policy_for_tests()

    def test_artifact_role_no_filtering(self):
        """Artifact role should preserve all query params (CDN signatures, etc.)."""
        url = "https://s3.example.com/file.pdf?X-Amz-Signature=abc&X-Amz-Expires=3600&Token=secret"
        result = canonical_for_request(url, role="artifact")
        assert "X-Amz-Signature" in result
        assert "X-Amz-Expires" in result
        assert "Token" in result

    def test_relative_url_with_origin_host(self):
        """Relative URLs should be made absolute using origin_host."""
        result = canonical_for_request(
            "/path/to/file.pdf", role="metadata", origin_host="example.com"
        )
        assert result.startswith("https://example.com")
        assert "/path/to/file.pdf" in result


class TestCanonicalHost:
    """Test host extraction for limiter/breaker keys."""

    def test_extract_lowercase_host(self):
        """Host should be lowercased."""
        assert canonical_host("https://EXAMPLE.COM/path") == "example.com"
        assert canonical_host("https://MiXeD.Example.ORG:443/") == "mixed.example.org"

    def test_strip_port(self):
        """Port should not be included in the host key."""
        assert canonical_host("https://example.com:8080/path") == "example.com"
        assert canonical_host("http://example.com:80/") == "example.com"

    def test_error_on_none(self):
        """Should raise TypeError if url is None."""
        with pytest.raises(TypeError, match="expected a URL string, received None"):
            canonical_host(None)


class TestUrlPolicy:
    """Test URL policy configuration."""

    def teardown_method(self):
        """Reset policy after each test."""
        reset_url_policy_for_tests()

    def test_default_policy(self):
        """Default policy should have sane defaults."""
        policy = get_url_policy()
        assert policy.default_scheme == "https"
        assert policy.filter_for["landing"] is True
        assert policy.filter_for["metadata"] is False
        assert policy.filter_for["artifact"] is False

    def test_configure_url_policy_scheme(self):
        """Can override default scheme."""
        configure_url_policy(default_scheme="http")
        policy = get_url_policy()
        assert policy.default_scheme == "http"
        # Verify it's used
        assert canonical_for_index("example.com") == "http://example.com/"

    def test_parse_param_allowlist_spec_global(self):
        """Parse global allowlist spec."""
        global_params, per_domain = parse_param_allowlist_spec("page,id,sort")
        assert set(global_params) == {"page", "id", "sort"}
        assert per_domain == {}

    def test_parse_param_allowlist_spec_per_domain(self):
        """Parse per-domain allowlist spec."""
        spec = "example.com:page,id;api.example.org:token,v"
        global_params, per_domain = parse_param_allowlist_spec(spec)
        assert global_params == ()
        assert per_domain["example.com"] == ("page", "id")
        assert per_domain["api.example.org"] == ("token", "v")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_url_raises_error(self):
        """canonical_for_index should reject None."""
        with pytest.raises(TypeError):
            canonical_for_index(None)

        with pytest.raises(TypeError):
            canonical_for_request(None, role="metadata")

    def test_url_without_scheme_defaults_to_https(self):
        """Schemeless URLs should default to https."""
        result = canonical_for_index("example.com/path")
        assert result.startswith("https://")
