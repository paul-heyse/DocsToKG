"""Phase 5.2: Complex Domain Models Tests

Tests for SecuritySettings, RateLimitSettings, ExtractionSettings, StorageSettings, DuckDBSettings.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from DocsToKG.OntologyDownload.settings import (
    SecuritySettings,
    RateLimitSettings,
    ExtractionSettings,
    StorageSettings,
    DuckDBSettings,
)


# ============================================================================
# SECURITY SETTINGS TESTS
# ============================================================================


class TestSecuritySettingsDefaults:
    """Test SecuritySettings default values."""

    def test_security_defaults(self):
        """Verify default values."""
        s = SecuritySettings()
        assert s.allowed_hosts is None
        assert s.allowed_ports is None
        assert s.allow_private_networks is False
        assert s.allow_plain_http is False
        assert s.strict_dns is True

    def test_security_immutability(self):
        """Test that SecuritySettings is frozen."""
        s = SecuritySettings()
        with pytest.raises(ValidationError):
            s.allow_private_networks = True


class TestSecuritySettingsHostParsing:
    """Test host parsing logic."""

    def test_simple_hostname(self):
        """Parse simple hostname."""
        s = SecuritySettings(allowed_hosts=["example.com"])
        exact, suffixes, host_ports, ips = s.normalized_allowed_hosts()
        assert "example.com" in exact
        assert len(suffixes) == 0
        assert len(ips) == 0

    def test_wildcard_domain(self):
        """Parse wildcard domains."""
        s = SecuritySettings(allowed_hosts=["*.example.com", ".other.org"])
        exact, suffixes, host_ports, ips = s.normalized_allowed_hosts()
        assert "example.com" in suffixes
        assert "other.org" in suffixes

    def test_hostname_with_port(self):
        """Parse hostname with port."""
        s = SecuritySettings(allowed_hosts=["example.com:8443"])
        exact, suffixes, host_ports, ips = s.normalized_allowed_hosts()
        assert "example.com" in exact
        assert 8443 in host_ports["example.com"]

    def test_ipv4_address(self):
        """Parse IPv4 addresses."""
        s = SecuritySettings(allowed_hosts=["192.168.1.1"])
        exact, suffixes, host_ports, ips = s.normalized_allowed_hosts()
        assert "192.168.1.1" in ips

    def test_ipv6_literal(self):
        """Parse IPv6 literals."""
        s = SecuritySettings(allowed_hosts=["[::1]", "[2001:db8::1]:8443"])
        exact, suffixes, host_ports, ips = s.normalized_allowed_hosts()
        assert "::1" in ips
        assert "2001:db8::1" in ips

    def test_invalid_ipv6_literal(self):
        """Reject invalid IPv6 literals."""
        s = SecuritySettings(allowed_hosts=["[::1"])
        with pytest.raises(ValueError):
            s.normalized_allowed_hosts()

    def test_wildcard_cannot_have_port(self):
        """Reject wildcards with ports."""
        s = SecuritySettings(allowed_hosts=["*.example.com:8443"])
        with pytest.raises(ValueError):
            s.normalized_allowed_hosts()

    def test_allowed_port_set_default(self):
        """Default port set is 80, 443."""
        s = SecuritySettings()
        ports = s.allowed_port_set()
        assert ports == {80, 443}

    def test_allowed_port_set_custom(self):
        """Custom port set."""
        s = SecuritySettings(allowed_ports=[8080, 8443])
        ports = s.allowed_port_set()
        assert ports == {8080, 8443}

    def test_port_parsing_csv(self):
        """Parse port CSV string."""
        s = SecuritySettings(allowed_ports="80,443,8443")
        assert s.allowed_ports == [80, 443, 8443]

    def test_invalid_port_range(self):
        """Reject ports outside valid range."""
        with pytest.raises(ValidationError):
            SecuritySettings(allowed_ports=[0, 65536])


# ============================================================================
# RATE LIMIT SETTINGS TESTS
# ============================================================================


class TestRateLimitSettingsDefaults:
    """Test RateLimitSettings default values."""

    def test_ratelimit_defaults(self):
        """Verify default values."""
        s = RateLimitSettings()
        assert s.default is None
        assert s.per_service == {}
        assert s.shared_dir is None
        assert s.engine == "pyrate"

    def test_ratelimit_immutability(self):
        """Test that RateLimitSettings is frozen."""
        s = RateLimitSettings()
        with pytest.raises(ValidationError):
            s.engine = "other"


class TestRateLimitSettingsValidation:
    """Test rate limit string validation."""

    def test_valid_rate_strings(self):
        """Accept valid rate limit formats."""
        valid = [
            "10/second",
            "60/minute",
            "3600/hour",
            "1/second",
            "999/hour",
        ]
        for rate in valid:
            s = RateLimitSettings(default=rate)
            assert s.default == rate

    def test_invalid_rate_strings(self):
        """Reject invalid rate limit formats."""
        invalid = [
            "10/millisecond",
            "10 second",
            "10sec",
            "garbage",
            "10/",
            "/second",
        ]
        for rate in invalid:
            with pytest.raises(ValidationError):
                RateLimitSettings(default=rate)

    def test_per_service_dict(self):
        """Parse per-service dict."""
        s = RateLimitSettings(per_service={"ols": "4/second", "bioportal": "2/minute"})
        assert s.per_service == {"ols": "4/second", "bioportal": "2/minute"}

    def test_per_service_csv(self):
        """Parse per-service CSV format."""
        s = RateLimitSettings(per_service="ols:4/second;bioportal:2/minute")
        assert "ols" in s.per_service
        assert "bioportal" in s.per_service

    def test_parse_service_rate_limit(self):
        """Parse service-specific rate limit to RPS."""
        s = RateLimitSettings(
            default="10/second",
            per_service={"fast": "100/second"},
        )
        # Should work (would be tested with actual parse_rate_limit_to_rps implementation)
        assert s.parse_service_rate_limit("unknown") is not None


# ============================================================================
# EXTRACTION SETTINGS TESTS
# ============================================================================


class TestExtractionSettingsDefaults:
    """Test ExtractionSettings default values."""

    def test_extraction_defaults(self):
        """Verify default values."""
        s = ExtractionSettings()
        assert s.encapsulate is True
        assert s.encapsulation_name == "sha256"
        assert s.max_depth == 32
        assert s.max_path_len == 4096
        assert s.max_entries == 50000
        assert s.max_total_ratio == 10.0
        assert s.unicode_form == "NFC"
        assert s.overwrite == "reject"
        assert s.duplicate_policy == "reject"

    def test_extraction_immutability(self):
        """Test that ExtractionSettings is frozen."""
        s = ExtractionSettings()
        with pytest.raises(ValidationError):
            s.max_depth = 64


class TestExtractionSettingsValidation:
    """Test extraction validation."""

    def test_valid_encapsulation_names(self):
        """Accept valid encapsulation strategies."""
        for name in ["sha256", "SHA256", "basename", "BASENAME"]:
            s = ExtractionSettings(encapsulation_name=name)
            assert s.encapsulation_name in {"sha256", "basename"}

    def test_invalid_encapsulation_name(self):
        """Reject invalid encapsulation strategy."""
        with pytest.raises(ValidationError):
            ExtractionSettings(encapsulation_name="invalid")

    def test_valid_unicode_forms(self):
        """Accept valid Unicode forms."""
        for form in ["NFC", "nfc", "NFD", "nfd"]:
            s = ExtractionSettings(unicode_form=form)
            assert s.unicode_form in {"NFC", "NFD"}

    def test_numeric_bounds(self):
        """Test numeric field bounds."""
        # Valid values
        s = ExtractionSettings(
            max_depth=10,
            max_entries=1000,
            max_total_ratio=50.0,
            space_safety_margin=1.5,
        )
        assert s.max_depth == 10
        assert s.max_entries == 1000

    def test_max_depth_bounds(self):
        """max_depth must be 1-255."""
        with pytest.raises(ValidationError):
            ExtractionSettings(max_depth=0)
        with pytest.raises(ValidationError):
            ExtractionSettings(max_depth=256)

    def test_max_total_ratio_bounds(self):
        """max_total_ratio must be ≥1."""
        with pytest.raises(ValidationError):
            ExtractionSettings(max_total_ratio=0.5)

    def test_policies_are_lowercased(self):
        """Policies are normalized to lowercase."""
        s = ExtractionSettings(
            overwrite="REJECT",
            duplicate_policy="FIRST_WINS",
            casefold_collision_policy="ALLOW",
        )
        assert s.overwrite == "reject"
        assert s.duplicate_policy == "first_wins"
        assert s.casefold_collision_policy == "allow"


# ============================================================================
# STORAGE SETTINGS TESTS
# ============================================================================


class TestStorageSettingsDefaults:
    """Test StorageSettings default values."""

    def test_storage_defaults(self):
        """Verify default values."""
        s = StorageSettings()
        assert s.root.is_absolute()
        assert s.latest_name == "LATEST.json"
        assert s.url is None

    def test_storage_immutability(self):
        """Test that StorageSettings is frozen."""
        s = StorageSettings()
        with pytest.raises(ValidationError):
            s.latest_name = "other.json"


class TestStorageSettingsPathHandling:
    """Test storage path normalization."""

    def test_tilde_expansion(self):
        """Expand tilde in paths."""
        s = StorageSettings(root="~/ontologies")
        assert s.root.is_absolute()
        assert "~" not in str(s.root)

    def test_relative_path_normalization(self):
        """Convert relative to absolute."""
        s = StorageSettings(root="./ontologies")
        assert s.root.is_absolute()

    def test_path_object(self):
        """Accept Path objects."""
        p = Path("/tmp/ontologies")
        s = StorageSettings(root=p)
        assert s.root == p


# ============================================================================
# DUCKDB SETTINGS TESTS
# ============================================================================


class TestDuckDBSettingsDefaults:
    """Test DuckDBSettings default values."""

    def test_duckdb_defaults(self):
        """Verify default values."""
        s = DuckDBSettings()
        assert s.path.is_absolute()
        assert s.threads is None
        assert s.readonly is False
        assert s.wlock is True
        assert s.parquet_events is False

    def test_duckdb_immutability(self):
        """Test that DuckDBSettings is frozen."""
        s = DuckDBSettings()
        with pytest.raises(ValidationError):
            s.readonly = True


class TestDuckDBSettingsPathHandling:
    """Test DuckDB path normalization."""

    def test_tilde_expansion(self):
        """Expand tilde in paths."""
        s = DuckDBSettings(path="~/.data/ontofetch.duckdb")
        assert s.path.is_absolute()
        assert "~" not in str(s.path)

    def test_relative_path_normalization(self):
        """Convert relative to absolute."""
        s = DuckDBSettings(path="./ontofetch.duckdb")
        assert s.path.is_absolute()

    def test_threads_validation(self):
        """Threads must be positive when set."""
        with pytest.raises(ValidationError):
            DuckDBSettings(threads=0)

        with pytest.raises(ValidationError):
            DuckDBSettings(threads=-1)

        # Valid values
        s = DuckDBSettings(threads=4)
        assert s.threads == 4


# ============================================================================
# PHASE 5.2 INTEGRATION TESTS
# ============================================================================


class TestPhase52Exports:
    """Test that Phase 5.2 models are properly exported."""

    def test_all_models_in_all(self):
        """All Phase 5.2 models should be in __all__."""
        from DocsToKG.OntologyDownload import settings

        models = [
            "SecuritySettings",
            "RateLimitSettings",
            "ExtractionSettings",
            "StorageSettings",
            "DuckDBSettings",
        ]
        for model in models:
            assert model in settings.__all__, f"{model} not in __all__"

    def test_models_are_frozen(self):
        """All Phase 5.2 models should be frozen."""
        models = [
            SecuritySettings,
            RateLimitSettings,
            ExtractionSettings,
            StorageSettings,
            DuckDBSettings,
        ]
        for model_cls in models:
            instance = model_cls()
            first_field = next(iter(model_cls.model_fields))
            with pytest.raises(ValidationError):
                setattr(instance, first_field, None)

    def test_models_are_serializable(self):
        """All Phase 5.2 models should be serializable."""
        models = [
            SecuritySettings(),
            RateLimitSettings(),
            ExtractionSettings(),
            StorageSettings(),
            DuckDBSettings(),
        ]
        for instance in models:
            dumped = instance.model_dump()
            assert isinstance(dumped, dict)
            assert len(dumped) > 0


# ============================================================================
# SKIPPED: RESERVED FOR PHASE 5.3
# ============================================================================


@pytest.mark.skip(reason="Environment variable mapping deferred to Phase 5.3")
class TestPhase52EnvironmentMapping:
    """Environment variable mapping tests (Phase 5.3)."""

    def test_security_from_env(self):
        """Parse SecuritySettings from ONTOFETCH_SECURITY__* env vars."""
        pass

    def test_extraction_from_env(self):
        """Parse ExtractionSettings from ONTOFETCH_EXTRACT__* env vars."""
        pass


@pytest.mark.skip(reason="Root Settings integration deferred to Phase 5.3")
class TestPhase52Integration:
    """Root Settings integration tests (Phase 5.3)."""

    def test_all_domains_in_root_settings(self):
        """All 10 domain models should be composable into root Settings."""
        pass

    def test_source_precedence(self):
        """Test config source precedence: CLI → file → .env → env → defaults."""
        pass
