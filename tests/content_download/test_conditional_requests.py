"""Tests for conditional_requests.py - RFC 7232 conditional request handling."""

from datetime import datetime
from unittest import TestCase

from DocsToKG.ContentDownload.conditional_requests import (
    EntityValidator,
    build_conditional_headers,
    is_validator_available,
    merge_validators,
    parse_entity_validator,
    should_revalidate,
)


class TestEntityValidator(TestCase):
    """Tests for EntityValidator dataclass."""

    def test_empty_validator(self) -> None:
        """Empty validator has all None/False defaults."""
        v = EntityValidator()
        self.assertIsNone(v.etag)
        self.assertFalse(v.etag_strong)
        self.assertIsNone(v.last_modified)
        self.assertIsNone(v.last_modified_dt)

    def test_validator_is_frozen(self) -> None:
        """EntityValidator instances are immutable."""
        v = EntityValidator(etag='"abc"')
        with self.assertRaises(Exception):
            v.etag = '"def"'  # type: ignore[misc]

    def test_validator_with_all_fields(self) -> None:
        """Validator can be created with all fields."""
        dt = datetime(2025, 10, 21, 7, 28, 0)
        v = EntityValidator(
            etag='"abc123"',
            etag_strong=True,
            last_modified="Wed, 21 Oct 2025 07:28:00 GMT",
            last_modified_dt=dt,
        )
        self.assertEqual(v.etag, '"abc123"')
        self.assertTrue(v.etag_strong)
        self.assertEqual(v.last_modified, "Wed, 21 Oct 2025 07:28:00 GMT")
        self.assertEqual(v.last_modified_dt, dt)


class TestParseEntityValidator(TestCase):
    """Tests for parse_entity_validator function."""

    def test_parse_strong_etag(self) -> None:
        """Parse strong ETag (without W/ prefix)."""
        headers = {"etag": '"abc123"'}
        v = parse_entity_validator(headers)
        self.assertEqual(v.etag, '"abc123"')
        self.assertTrue(v.etag_strong)

    def test_parse_weak_etag(self) -> None:
        """Parse weak ETag (with W/ prefix)."""
        headers = {"etag": 'W/"abc123"'}
        v = parse_entity_validator(headers)
        self.assertEqual(v.etag, 'W/"abc123"')
        self.assertFalse(v.etag_strong)

    def test_parse_last_modified(self) -> None:
        """Parse Last-Modified header."""
        headers = {"last-modified": "Wed, 21 Oct 2025 07:28:00 GMT"}
        v = parse_entity_validator(headers)
        self.assertEqual(v.last_modified, "Wed, 21 Oct 2025 07:28:00 GMT")
        self.assertIsNotNone(v.last_modified_dt)

    def test_parse_case_insensitive_headers(self) -> None:
        """Header parsing is case-insensitive."""
        headers = {
            "ETag": '"abc123"',
            "Last-Modified": "Wed, 21 Oct 2025 07:28:00 GMT",
        }
        v = parse_entity_validator(headers)
        self.assertEqual(v.etag, '"abc123"')
        self.assertIsNotNone(v.last_modified)

    def test_parse_both_validators(self) -> None:
        """Parse both ETag and Last-Modified."""
        headers = {
            "etag": '"abc123"',
            "last-modified": "Wed, 21 Oct 2025 07:28:00 GMT",
        }
        v = parse_entity_validator(headers)
        self.assertEqual(v.etag, '"abc123"')
        self.assertEqual(v.last_modified, "Wed, 21 Oct 2025 07:28:00 GMT")

    def test_parse_missing_validators(self) -> None:
        """Missing validators return defaults."""
        headers = {"content-type": "application/json"}
        v = parse_entity_validator(headers)
        self.assertIsNone(v.etag)
        self.assertIsNone(v.last_modified)

    def test_parse_empty_headers(self) -> None:
        """Empty headers return defaults."""
        v = parse_entity_validator({})
        self.assertIsNone(v.etag)
        self.assertIsNone(v.last_modified)

    def test_parse_invalid_last_modified(self) -> None:
        """Invalid Last-Modified is handled gracefully."""
        headers = {"last-modified": "not a valid date"}
        v = parse_entity_validator(headers)
        self.assertEqual(v.last_modified, "not a valid date")
        self.assertIsNone(v.last_modified_dt)

    def test_parse_etag_with_whitespace(self) -> None:
        """ETag with surrounding whitespace is trimmed."""
        headers = {"etag": '  "abc123"  '}
        v = parse_entity_validator(headers)
        self.assertEqual(v.etag, '"abc123"')


class TestBuildConditionalHeaders(TestCase):
    """Tests for build_conditional_headers function."""

    def test_build_with_etag_only(self) -> None:
        """Build headers with ETag only."""
        v = EntityValidator(etag='"abc123"')
        headers = build_conditional_headers(v)
        self.assertEqual(headers, {"If-None-Match": '"abc123"'})

    def test_build_with_last_modified_only(self) -> None:
        """Build headers with Last-Modified only."""
        v = EntityValidator(last_modified="Wed, 21 Oct 2025 07:28:00 GMT")
        headers = build_conditional_headers(v)
        self.assertEqual(headers, {"If-Modified-Since": "Wed, 21 Oct 2025 07:28:00 GMT"})

    def test_build_with_both_validators(self) -> None:
        """Build headers with both ETag and Last-Modified."""
        v = EntityValidator(
            etag='"abc123"',
            last_modified="Wed, 21 Oct 2025 07:28:00 GMT",
        )
        headers = build_conditional_headers(v)
        self.assertEqual(
            headers,
            {
                "If-None-Match": '"abc123"',
                "If-Modified-Since": "Wed, 21 Oct 2025 07:28:00 GMT",
            },
        )

    def test_build_with_no_validators(self) -> None:
        """Build headers with no validators returns empty dict."""
        v = EntityValidator()
        headers = build_conditional_headers(v)
        self.assertEqual(headers, {})

    def test_build_preserves_weak_etag(self) -> None:
        """Weak ETag is preserved in conditional headers."""
        v = EntityValidator(etag='W/"abc123"')
        headers = build_conditional_headers(v)
        self.assertEqual(headers, {"If-None-Match": 'W/"abc123"'})


class TestShouldRevalidate(TestCase):
    """Tests for should_revalidate function."""

    def test_revalidate_etag_match(self) -> None:
        """ETag match indicates successful revalidation."""
        original = EntityValidator(etag='"abc123"')
        response_headers = {"etag": '"abc123"'}
        self.assertTrue(should_revalidate(original, response_headers))

    def test_revalidate_etag_mismatch(self) -> None:
        """ETag mismatch indicates revalidation failed."""
        original = EntityValidator(etag='"abc123"')
        response_headers = {"etag": '"def456"'}
        self.assertFalse(should_revalidate(original, response_headers))

    def test_revalidate_weak_etag_match(self) -> None:
        """Weak ETag match (weak comparison)."""
        original = EntityValidator(etag='W/"abc123"')
        response_headers = {"etag": 'W/"abc123"'}
        self.assertTrue(should_revalidate(original, response_headers))

    def test_revalidate_weak_vs_strong_etag(self) -> None:
        """Weak and strong ETags can match via weak comparison."""
        original = EntityValidator(etag='"abc123"')
        response_headers = {"etag": 'W/"abc123"'}
        # Weak comparison: both normalize to "abc123"
        self.assertTrue(should_revalidate(original, response_headers))

    def test_revalidate_last_modified_match(self) -> None:
        """Last-Modified match indicates successful revalidation."""
        original = EntityValidator(last_modified="Wed, 21 Oct 2025 07:28:00 GMT")
        response_headers = {"last-modified": "Wed, 21 Oct 2025 07:28:00 GMT"}
        self.assertTrue(should_revalidate(original, response_headers))

    def test_revalidate_last_modified_mismatch(self) -> None:
        """Last-Modified mismatch indicates revalidation failed."""
        original = EntityValidator(last_modified="Wed, 21 Oct 2025 07:28:00 GMT")
        response_headers = {"last-modified": "Wed, 21 Oct 2025 07:29:00 GMT"}
        self.assertFalse(should_revalidate(original, response_headers))

    def test_revalidate_both_validators_etag_match(self) -> None:
        """With both validators, ETag match is successful."""
        original = EntityValidator(
            etag='"abc123"',
            last_modified="Wed, 21 Oct 2025 07:28:00 GMT",
        )
        response_headers = {
            "etag": '"abc123"',
            "last-modified": "Wed, 21 Oct 2025 07:29:00 GMT",
        }
        self.assertTrue(should_revalidate(original, response_headers))

    def test_revalidate_both_validators_last_modified_match(self) -> None:
        """With both validators, Last-Modified match is successful."""
        original = EntityValidator(
            etag='"abc123"',
            last_modified="Wed, 21 Oct 2025 07:28:00 GMT",
        )
        response_headers = {
            "etag": '"def456"',
            "last-modified": "Wed, 21 Oct 2025 07:28:00 GMT",
        }
        self.assertTrue(should_revalidate(original, response_headers))

    def test_revalidate_no_validators(self) -> None:
        """No validators to check returns False."""
        original = EntityValidator()
        response_headers = {}
        self.assertFalse(should_revalidate(original, response_headers))

    def test_revalidate_no_matching_validators(self) -> None:
        """No matching validators returns False."""
        original = EntityValidator(etag='"abc123"')
        response_headers = {"last-modified": "Wed, 21 Oct 2025 07:28:00 GMT"}
        self.assertFalse(should_revalidate(original, response_headers))


class TestMergeValidators(TestCase):
    """Tests for merge_validators function."""

    def test_merge_update_etag(self) -> None:
        """Updated ETag replaces original."""
        original = EntityValidator(etag='"old"')
        updated = EntityValidator(etag='"new"')
        merged = merge_validators(original, updated)
        self.assertEqual(merged.etag, '"new"')

    def test_merge_keep_etag_if_not_updated(self) -> None:
        """Original ETag kept if not in updated."""
        original = EntityValidator(etag='"abc"')
        updated = EntityValidator()
        merged = merge_validators(original, updated)
        self.assertEqual(merged.etag, '"abc"')

    def test_merge_update_last_modified(self) -> None:
        """Updated Last-Modified replaces original."""
        original = EntityValidator(last_modified="Wed, 21 Oct 2025 07:28:00 GMT")
        updated = EntityValidator(last_modified="Wed, 21 Oct 2025 07:29:00 GMT")
        merged = merge_validators(original, updated)
        self.assertEqual(merged.last_modified, "Wed, 21 Oct 2025 07:29:00 GMT")

    def test_merge_keep_last_modified_if_not_updated(self) -> None:
        """Original Last-Modified kept if not in updated."""
        original = EntityValidator(last_modified="Wed, 21 Oct 2025 07:28:00 GMT")
        updated = EntityValidator()
        merged = merge_validators(original, updated)
        self.assertEqual(merged.last_modified, "Wed, 21 Oct 2025 07:28:00 GMT")

    def test_merge_both_validators(self) -> None:
        """Merge both ETag and Last-Modified."""
        original = EntityValidator(
            etag='"old"',
            last_modified="Wed, 21 Oct 2025 07:28:00 GMT",
        )
        updated = EntityValidator(
            etag='"new"',
            last_modified="Wed, 21 Oct 2025 07:29:00 GMT",
        )
        merged = merge_validators(original, updated)
        self.assertEqual(merged.etag, '"new"')
        self.assertEqual(merged.last_modified, "Wed, 21 Oct 2025 07:29:00 GMT")

    def test_merge_empty_to_empty(self) -> None:
        """Merge two empty validators."""
        original = EntityValidator()
        updated = EntityValidator()
        merged = merge_validators(original, updated)
        self.assertIsNone(merged.etag)
        self.assertIsNone(merged.last_modified)


class TestIsValidatorAvailable(TestCase):
    """Tests for is_validator_available function."""

    def test_available_with_etag(self) -> None:
        """Available when ETag is set."""
        v = EntityValidator(etag='"abc123"')
        self.assertTrue(is_validator_available(v))

    def test_available_with_last_modified(self) -> None:
        """Available when Last-Modified is set."""
        v = EntityValidator(last_modified="Wed, 21 Oct 2025 07:28:00 GMT")
        self.assertTrue(is_validator_available(v))

    def test_available_with_both(self) -> None:
        """Available when both validators are set."""
        v = EntityValidator(
            etag='"abc123"',
            last_modified="Wed, 21 Oct 2025 07:28:00 GMT",
        )
        self.assertTrue(is_validator_available(v))

    def test_not_available_empty(self) -> None:
        """Not available when no validators set."""
        v = EntityValidator()
        self.assertFalse(is_validator_available(v))

    def test_available_with_only_metadata(self) -> None:
        """etag_strong metadata alone doesn't make validator available."""
        v = EntityValidator(etag_strong=True)
        self.assertFalse(is_validator_available(v))
