"""Tests for rate-limit configuration: RateSpec parsing and validation.

Tests cover:
- Basic rate string parsing
- Duration normalization and aliases
- Multi-window validation and ordering
- Per-service rate configuration
- Error handling and validation
- Human-readable representations
"""

import pytest
from DocsToKG.OntologyDownload.ratelimit.config import (
    RateSpec,
    parse_rate_string,
    validate_rate_list,
    normalize_rate_list,
    normalize_per_service_rates,
    get_schema_summary,
)


class TestRateSpecBasics:
    """Test RateSpec data model."""

    def test_rate_spec_creation(self):
        """RateSpec can be created with limit and interval."""
        spec = RateSpec(limit=5, interval_ms=1000)
        assert spec.limit == 5
        assert spec.interval_ms == 1000

    def test_rate_spec_frozen(self):
        """RateSpec is immutable (frozen dataclass)."""
        spec = RateSpec(limit=5, interval_ms=1000)
        with pytest.raises(AttributeError):
            spec.limit = 10

    def test_rate_spec_rps_property(self):
        """RateSpec.rps computes requests per second."""
        spec = RateSpec(limit=5, interval_ms=1000)
        assert spec.rps == 5.0

        spec2 = RateSpec(limit=300, interval_ms=60_000)
        assert spec2.rps == 5.0

    def test_rate_spec_rpm_property(self):
        """RateSpec.rpm computes requests per minute."""
        spec = RateSpec(limit=5, interval_ms=1000)
        assert spec.rpm == 300.0

        spec2 = RateSpec(limit=300, interval_ms=60_000)
        assert spec2.rpm == 300.0

    def test_rate_spec_string_representation(self):
        """RateSpec.__str__ returns human-readable format."""
        spec = RateSpec(limit=5, interval_ms=1000)
        assert str(spec) == "5/second"

        spec2 = RateSpec(limit=300, interval_ms=60_000)
        assert str(spec2) == "300/minute"

        spec3 = RateSpec(limit=1000, interval_ms=3_600_000)
        assert str(spec3) == "1000/hour"

    def test_rate_spec_repr(self):
        """RateSpec.__repr__ returns full representation."""
        spec = RateSpec(limit=5, interval_ms=1000)
        assert repr(spec) == "RateSpec(limit=5, interval_ms=1000)"


class TestRateStringParsing:
    """Test parse_rate_string function."""

    def test_parse_basic_rate_string(self):
        """Parse simple rate strings."""
        spec = parse_rate_string("5/second")
        assert spec.limit == 5
        assert spec.interval_ms == 1000

    def test_parse_minute_rate(self):
        """Parse minute-based rates."""
        spec = parse_rate_string("300/minute")
        assert spec.limit == 300
        assert spec.interval_ms == 60_000

    def test_parse_hour_rate(self):
        """Parse hour-based rates."""
        spec = parse_rate_string("10/hour")
        assert spec.limit == 10
        assert spec.interval_ms == 3_600_000

    def test_parse_day_rate(self):
        """Parse day-based rates."""
        spec = parse_rate_string("100/day")
        assert spec.limit == 100
        assert spec.interval_ms == 24 * 3_600_000

    def test_parse_with_duration_aliases(self):
        """Parse rates with duration aliases."""
        spec_sec = parse_rate_string("5/sec")
        assert spec_sec.limit == 5
        assert spec_sec.interval_ms == 1000

        spec_min = parse_rate_string("300/min")
        assert spec_min.limit == 300
        assert spec_min.interval_ms == 60_000

        spec_hr = parse_rate_string("10/hr")
        assert spec_hr.limit == 10
        assert spec_hr.interval_ms == 3_600_000

    def test_parse_with_whitespace(self):
        """Parse rates with extra whitespace."""
        spec = parse_rate_string("  5  /  second  ")
        assert spec.limit == 5
        assert spec.interval_ms == 1000

    def test_parse_invalid_format_raises(self):
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid rate spec"):
            parse_rate_string("invalid")

    def test_parse_unknown_duration_raises(self):
        """Unknown duration raises ValueError."""
        with pytest.raises(ValueError, match="Unknown duration"):
            parse_rate_string("5/futureminute")

    def test_parse_non_positive_limit_raises(self):
        """Non-positive limit raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            parse_rate_string("0/second")

        with pytest.raises(ValueError, match="Invalid rate spec"):
            parse_rate_string("-5/second")


class TestRateListValidation:
    """Test validate_rate_list function."""

    def test_single_rate_is_valid(self):
        """Single rate always validates."""
        spec = RateSpec(limit=5, interval_ms=1000)
        assert validate_rate_list([spec]) is True

    def test_empty_list_is_valid(self):
        """Empty list always validates."""
        assert validate_rate_list([]) is True

    def test_correctly_ordered_rates_validate(self):
        """Correctly ordered multi-window rates validate."""
        rates = [
            RateSpec(limit=5, interval_ms=1000),
            RateSpec(limit=300, interval_ms=60_000),
        ]
        assert validate_rate_list(rates) is True

    def test_incorrectly_ordered_raises(self):
        """Incorrectly ordered rates raise ValueError."""
        rates = [
            RateSpec(limit=300, interval_ms=60_000),
            RateSpec(limit=5, interval_ms=1000),
        ]
        with pytest.raises(ValueError, match="not ordered by interval"):
            validate_rate_list(rates)


class TestNormalizeRateList:
    """Test normalize_rate_list function."""

    def test_normalize_single_rate(self):
        """Normalize single rate string."""
        result = normalize_rate_list(["5/second"])
        assert len(result) == 1
        assert result[0].limit == 5
        assert result[0].interval_ms == 1000

    def test_normalize_multiple_rates(self):
        """Normalize multiple rate strings."""
        result = normalize_rate_list(["5/second", "300/minute"])
        assert len(result) == 2
        assert result[0].limit == 5
        assert result[1].limit == 300

    def test_normalize_sorts_by_interval(self):
        """Normalize sorts rates by interval (ascending)."""
        # Input in wrong order
        result = normalize_rate_list(["300/minute", "5/second"])
        # Should be sorted by interval
        assert result[0].interval_ms == 1000
        assert result[1].interval_ms == 60_000

    def test_normalize_invalid_spec_raises(self):
        """Invalid spec in list raises ValueError."""
        with pytest.raises(ValueError):
            normalize_rate_list(["5/second", "invalid"])


class TestPerServiceRates:
    """Test normalize_per_service_rates function."""

    def test_normalize_service_dict(self):
        """Normalize dictionary of service rates."""
        config = normalize_per_service_rates(
            {
                "ols": "4/second",
                "bioportal": "2/second",
            }
        )
        assert "ols" in config
        assert "bioportal" in config
        assert config["ols"][0].limit == 4
        assert config["bioportal"][0].limit == 2

    def test_normalize_with_default(self):
        """Normalize with default rate."""
        config = normalize_per_service_rates(
            {"ols": "4/second"},
            default_rate="8/second",
        )
        assert "_default" in config
        assert config["_default"][0].limit == 8

    def test_normalize_invalid_service_rate_raises(self):
        """Invalid rate in service dict raises ValueError."""
        with pytest.raises(ValueError, match="Invalid rate for service"):
            normalize_per_service_rates({"ols": "invalid"})

    def test_normalize_invalid_default_raises(self):
        """Invalid default rate raises ValueError."""
        with pytest.raises(ValueError, match="Invalid default rate"):
            normalize_per_service_rates(
                {"ols": "4/second"},
                default_rate="invalid",
            )


class TestSchemaSummary:
    """Test get_schema_summary function."""

    def test_schema_summary_returns_dict(self):
        """get_schema_summary returns a dictionary."""
        summary = get_schema_summary()
        assert isinstance(summary, dict)

    def test_schema_summary_has_expected_keys(self):
        """Schema summary has expected documentation keys."""
        summary = get_schema_summary()
        assert "format" in summary
        assert "duration_options" in summary
        assert "examples" in summary

    def test_schema_summary_examples_are_valid(self):
        """Examples in schema summary are parseable."""
        summary = get_schema_summary()
        for example in summary["examples"]:
            spec = parse_rate_string(example)
            assert spec is not None
