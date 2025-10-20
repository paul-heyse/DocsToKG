"""Tests for cache_control.py - RFC 9111 directive parsing and interpretation."""

from unittest import TestCase

from DocsToKG.ContentDownload.cache_control import (
    CacheControlDirective,
    can_serve_stale,
    is_fresh,
    parse_cache_control,
    should_cache,
)


class TestCacheControlDirectiveDataclass(TestCase):
    """Tests for CacheControlDirective dataclass."""

    def test_empty_directive_defaults(self) -> None:
        """Empty directive has all conservative defaults."""
        directive = CacheControlDirective()
        self.assertFalse(directive.no_cache)
        self.assertFalse(directive.no_store)
        self.assertIsNone(directive.max_age)
        self.assertEqual(directive.stale_while_revalidate, 0)

    def test_directive_is_frozen(self) -> None:
        """CacheControlDirective instances are immutable."""
        directive = CacheControlDirective(max_age=3600)
        with self.assertRaises(Exception):
            directive.max_age = 7200  # type: ignore[misc]

    def test_directive_with_all_fields(self) -> None:
        """Directive can be created with all fields."""
        directive = CacheControlDirective(
            no_cache=True,
            max_age=3600,
            stale_while_revalidate=60,
            public=True,
        )
        self.assertTrue(directive.no_cache)
        self.assertEqual(directive.max_age, 3600)
        self.assertEqual(directive.stale_while_revalidate, 60)
        self.assertTrue(directive.public)


class TestParseCacheControl(TestCase):
    """Tests for parse_cache_control function."""

    def test_parse_max_age(self) -> None:
        """Parse max-age directive."""
        headers = {"cache-control": "max-age=3600"}
        directive = parse_cache_control(headers)
        self.assertEqual(directive.max_age, 3600)

    def test_parse_no_cache(self) -> None:
        """Parse no-cache directive."""
        headers = {"cache-control": "no-cache"}
        directive = parse_cache_control(headers)
        self.assertTrue(directive.no_cache)

    def test_parse_no_store(self) -> None:
        """Parse no-store directive."""
        headers = {"cache-control": "no-store"}
        directive = parse_cache_control(headers)
        self.assertTrue(directive.no_store)

    def test_parse_public(self) -> None:
        """Parse public directive."""
        headers = {"cache-control": "public"}
        directive = parse_cache_control(headers)
        self.assertTrue(directive.public)

    def test_parse_private(self) -> None:
        """Parse private directive."""
        headers = {"cache-control": "private"}
        directive = parse_cache_control(headers)
        self.assertTrue(directive.private)

    def test_parse_multiple_directives(self) -> None:
        """Parse multiple directives separated by comma."""
        headers = {"cache-control": "max-age=3600, public, must-revalidate"}
        directive = parse_cache_control(headers)
        self.assertEqual(directive.max_age, 3600)
        self.assertTrue(directive.public)
        self.assertTrue(directive.must_revalidate)

    def test_parse_stale_while_revalidate(self) -> None:
        """Parse stale-while-revalidate directive."""
        headers = {"cache-control": "max-age=3600, stale-while-revalidate=60"}
        directive = parse_cache_control(headers)
        self.assertEqual(directive.stale_while_revalidate, 60)

    def test_parse_stale_if_error(self) -> None:
        """Parse stale-if-error directive."""
        headers = {"cache-control": "max-age=3600, stale-if-error=120"}
        directive = parse_cache_control(headers)
        self.assertEqual(directive.stale_if_error, 120)

    def test_parse_case_insensitive_header(self) -> None:
        """Header key lookup is case-insensitive."""
        headers = {"Cache-Control": "max-age=3600"}
        directive = parse_cache_control(headers)
        self.assertEqual(directive.max_age, 3600)

    def test_parse_missing_header(self) -> None:
        """Missing Cache-Control header returns default."""
        headers = {}
        directive = parse_cache_control(headers)
        self.assertFalse(directive.no_cache)
        self.assertIsNone(directive.max_age)

    def test_parse_empty_header(self) -> None:
        """Empty Cache-Control header returns default."""
        headers = {"cache-control": ""}
        directive = parse_cache_control(headers)
        self.assertFalse(directive.no_cache)

    def test_parse_s_maxage(self) -> None:
        """Parse s-maxage directive."""
        headers = {"cache-control": "max-age=3600, s-maxage=7200"}
        directive = parse_cache_control(headers)
        self.assertEqual(directive.s_maxage, 7200)

    def test_parse_must_revalidate(self) -> None:
        """Parse must-revalidate directive."""
        headers = {"cache-control": "must-revalidate"}
        directive = parse_cache_control(headers)
        self.assertTrue(directive.must_revalidate)

    def test_parse_proxy_revalidate(self) -> None:
        """Parse proxy-revalidate directive."""
        headers = {"cache-control": "proxy-revalidate"}
        directive = parse_cache_control(headers)
        self.assertTrue(directive.proxy_revalidate)

    def test_parse_immutable(self) -> None:
        """Parse immutable directive."""
        headers = {"cache-control": "max-age=31536000, immutable"}
        directive = parse_cache_control(headers)
        self.assertTrue(directive.immutable)

    def test_parse_invalid_max_age_value(self) -> None:
        """Invalid max-age value is ignored gracefully."""
        headers = {"cache-control": "max-age=invalid"}
        directive = parse_cache_control(headers)
        # Should not crash, but value might be None or 0 depending on implementation
        self.assertIsNotNone(directive)

    def test_parse_whitespace_handling(self) -> None:
        """Directives with extra whitespace are parsed correctly."""
        headers = {"cache-control": "  max-age=3600  ,  public  "}
        directive = parse_cache_control(headers)
        self.assertEqual(directive.max_age, 3600)
        self.assertTrue(directive.public)


class TestIsFresh(TestCase):
    """Tests for is_fresh function."""

    def test_fresh_response_young_age(self) -> None:
        """Young response is fresh."""
        directive = CacheControlDirective(max_age=3600)
        self.assertTrue(is_fresh(directive, 1800.0))

    def test_stale_response_past_max_age(self) -> None:
        """Response past max-age is stale."""
        directive = CacheControlDirective(max_age=3600)
        self.assertFalse(is_fresh(directive, 3600.1))

    def test_max_age_zero_is_stale(self) -> None:
        """Response with max-age=0 is immediately stale."""
        directive = CacheControlDirective(max_age=0)
        self.assertFalse(is_fresh(directive, 0.0))

    def test_no_cache_is_never_fresh(self) -> None:
        """no-cache responses are never fresh."""
        directive = CacheControlDirective(no_cache=True, max_age=3600)
        self.assertFalse(is_fresh(directive, 0.0))

    def test_no_store_is_never_fresh(self) -> None:
        """no-store responses are never fresh."""
        directive = CacheControlDirective(no_store=True, max_age=3600)
        self.assertFalse(is_fresh(directive, 0.0))

    def test_s_maxage_overrides_max_age(self) -> None:
        """s-maxage takes precedence over max-age."""
        directive = CacheControlDirective(max_age=3600, s_maxage=7200)
        # At 5400 seconds: fresh with s-maxage, stale with max-age
        self.assertTrue(is_fresh(directive, 5400.0))

    def test_no_max_age_is_stale(self) -> None:
        """Response without max-age is considered stale (conservative)."""
        directive = CacheControlDirective()
        self.assertFalse(is_fresh(directive, 0.0))

    def test_exact_max_age_boundary(self) -> None:
        """Response at exact max-age boundary is stale."""
        directive = CacheControlDirective(max_age=3600)
        self.assertFalse(is_fresh(directive, 3600.0))

    def test_just_before_stale(self) -> None:
        """Response just before stale is fresh."""
        directive = CacheControlDirective(max_age=3600)
        self.assertTrue(is_fresh(directive, 3599.999))


class TestCanServeStale(TestCase):
    """Tests for can_serve_stale function."""

    def test_serve_stale_within_swrv(self) -> None:
        """Stale response can be served within stale-while-revalidate window."""
        directive = CacheControlDirective(max_age=3600, stale_while_revalidate=60)
        # 10 seconds into the SWrV window
        self.assertTrue(can_serve_stale(directive, 3610.0))

    def test_serve_stale_past_swrv(self) -> None:
        """Stale response past SWrV window cannot be served."""
        directive = CacheControlDirective(max_age=3600, stale_while_revalidate=60)
        # 70 seconds past max-age, beyond SWrV
        self.assertFalse(can_serve_stale(directive, 3670.0))

    def test_must_revalidate_forbids_stale(self) -> None:
        """must-revalidate forbids serving stale."""
        directive = CacheControlDirective(
            max_age=3600,
            must_revalidate=True,
            stale_while_revalidate=60,
        )
        self.assertFalse(can_serve_stale(directive, 3610.0))

    def test_stale_if_error_on_error(self) -> None:
        """Stale-if-error allows serving stale on revalidation error."""
        directive = CacheControlDirective(max_age=3600, stale_if_error=120)
        # 10 seconds past max-age, within SIE window
        self.assertTrue(can_serve_stale(directive, 3610.0, is_revalidation_error=True))

    def test_stale_if_error_past_window(self) -> None:
        """Stale-if-error does not allow past its window."""
        directive = CacheControlDirective(max_age=3600, stale_if_error=120)
        # 130 seconds past max-age, beyond SIE window
        self.assertFalse(can_serve_stale(directive, 3730.0, is_revalidation_error=True))

    def test_no_stale_extensions_returns_false(self) -> None:
        """Without SWrV or SIE, cannot serve stale."""
        directive = CacheControlDirective(max_age=3600)
        self.assertFalse(can_serve_stale(directive, 3610.0))

    def test_zero_stale_while_revalidate(self) -> None:
        """zero stale-while-revalidate means no grace period."""
        directive = CacheControlDirective(max_age=3600, stale_while_revalidate=0)
        self.assertFalse(can_serve_stale(directive, 3600.1))

    def test_no_max_age_no_stale(self) -> None:
        """Without max-age, cannot serve stale."""
        directive = CacheControlDirective(stale_while_revalidate=60)
        self.assertFalse(can_serve_stale(directive, 60.0))


class TestShouldCache(TestCase):
    """Tests for should_cache function."""

    def test_no_store_forbids_cache(self) -> None:
        """no-store forbids caching."""
        directive = CacheControlDirective(no_store=True)
        self.assertFalse(should_cache(directive))

    def test_no_cache_allows_cache(self) -> None:
        """no-cache allows caching (with revalidation)."""
        directive = CacheControlDirective(no_cache=True)
        self.assertTrue(should_cache(directive))

    def test_default_allows_cache(self) -> None:
        """Default directive allows caching."""
        directive = CacheControlDirective()
        self.assertTrue(should_cache(directive))

    def test_with_max_age_allows_cache(self) -> None:
        """Directive with max-age allows caching."""
        directive = CacheControlDirective(max_age=3600)
        self.assertTrue(should_cache(directive))

    def test_public_allows_cache(self) -> None:
        """public directive allows caching."""
        directive = CacheControlDirective(public=True)
        self.assertTrue(should_cache(directive))

    def test_private_allows_cache(self) -> None:
        """private directive allows caching."""
        directive = CacheControlDirective(private=True)
        self.assertTrue(should_cache(directive))
