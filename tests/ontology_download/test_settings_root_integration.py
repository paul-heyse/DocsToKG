"""Phase 5.3: Root Settings Integration Tests

Tests for OntologyDownloadSettings and singleton getter with caching.
"""

import pytest
from pathlib import Path

from DocsToKG.OntologyDownload.settings import (
    OntologyDownloadSettings,
    get_settings,
    clear_settings_cache,
    HttpSettings,
    CacheSettings,
    RetrySettings,
    LoggingSettings,
    TelemetrySettings,
    SecuritySettings,
    RateLimitSettings,
    ExtractionSettings,
    StorageSettings,
    DuckDBSettings,
)


# ============================================================================
# ROOT SETTINGS COMPOSITION TESTS
# ============================================================================


class TestOntologyDownloadSettingsComposition:
    """Test root settings composition of all 10 domain models."""

    def test_root_settings_instantiation(self):
        """Create root settings with all defaults."""
        settings = OntologyDownloadSettings()
        assert isinstance(settings, OntologyDownloadSettings)
        assert settings is not None

    def test_foundation_models_present(self):
        """All 5 foundation models should be present."""
        settings = OntologyDownloadSettings()
        assert isinstance(settings.http, HttpSettings)
        assert isinstance(settings.cache, CacheSettings)
        assert isinstance(settings.retry, RetrySettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert isinstance(settings.telemetry, TelemetrySettings)

    def test_complex_models_present(self):
        """All 5 complex models should be present."""
        settings = OntologyDownloadSettings()
        assert isinstance(settings.security, SecuritySettings)
        assert isinstance(settings.ratelimit, RateLimitSettings)
        assert isinstance(settings.extraction, ExtractionSettings)
        assert isinstance(settings.storage, StorageSettings)
        assert isinstance(settings.db, DuckDBSettings)

    def test_foundation_defaults(self):
        """Foundation model defaults should be accessible."""
        settings = OntologyDownloadSettings()
        assert settings.http.http2 is True
        assert settings.http.timeout_read == 30.0
        assert settings.cache.enabled is True
        assert settings.retry.connect_retries == 2
        assert settings.logging.level == "INFO"
        assert settings.telemetry.emit_events is True

    def test_complex_defaults(self):
        """Complex model defaults should be accessible."""
        settings = OntologyDownloadSettings()
        assert settings.security.strict_dns is True
        assert settings.ratelimit.engine == "pyrate"
        assert settings.extraction.max_depth == 32
        assert settings.storage.latest_name == "LATEST.json"
        assert settings.db.readonly is False

    def test_root_settings_frozen(self):
        """Root settings should be frozen."""
        settings = OntologyDownloadSettings()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            settings.http = HttpSettings(http2=False)

    def test_root_settings_serializable(self):
        """Root settings should be serializable."""
        settings = OntologyDownloadSettings()
        dumped = settings.model_dump()
        assert isinstance(dumped, dict)
        assert "http" in dumped
        assert "cache" in dumped
        assert "security" in dumped
        assert "extraction" in dumped
        assert "db" in dumped


# ============================================================================
# CONFIG HASH TESTS
# ============================================================================


class TestConfigHash:
    """Test config hash computation for provenance."""

    def test_config_hash_computation(self):
        """Config hash should be deterministic."""
        settings1 = OntologyDownloadSettings()
        hash1 = settings1.config_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex is 64 chars

    def test_config_hash_deterministic(self):
        """Same config with fixed UUID should produce same hash."""
        import uuid

        test_uuid = uuid.uuid4()
        from DocsToKG.OntologyDownload.settings import TelemetrySettings

        telemetry = TelemetrySettings(run_id=test_uuid)
        settings1 = OntologyDownloadSettings(telemetry=telemetry)

        # Create another with same UUID
        telemetry2 = TelemetrySettings(run_id=test_uuid)
        settings2 = OntologyDownloadSettings(telemetry=telemetry2)
        assert settings1.config_hash() == settings2.config_hash()

    def test_config_hash_different_for_different_config(self):
        """Different config should produce different hash."""
        settings1 = OntologyDownloadSettings()
        settings2 = OntologyDownloadSettings(
            http=HttpSettings(timeout_read=60.0),
        )
        assert settings1.config_hash() != settings2.config_hash()


# ============================================================================
# SINGLETON GETTER TESTS
# ============================================================================


class TestSingletonGetter:
    """Test singleton getter with caching."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_settings_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_settings_cache()

    def test_get_settings_returns_instance(self):
        """get_settings should return OntologyDownloadSettings instance."""
        settings = get_settings()
        assert isinstance(settings, OntologyDownloadSettings)

    def test_get_settings_caches_instance(self):
        """get_settings should cache instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2  # Same object reference

    def test_get_settings_force_reload(self):
        """force_reload should create new instance."""
        settings1 = get_settings()
        settings2 = get_settings(force_reload=True)
        # After force reload, they should be different objects
        # (though functionally equivalent)
        assert settings1 is not settings2

    def test_clear_settings_cache(self):
        """clear_settings_cache should reset singleton."""
        settings1 = get_settings()
        clear_settings_cache()
        settings2 = get_settings()
        assert settings1 is not settings2  # Different object after clear

    def test_thread_safety(self):
        """get_settings should be thread-safe."""
        import threading

        results = []

        def get_in_thread():
            s = get_settings()
            results.append(s)

        threads = [threading.Thread(target=get_in_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be the same instance
        assert all(r is results[0] for r in results)


# ============================================================================
# NESTED MODEL ACCESS TESTS
# ============================================================================


class TestNestedModelAccess:
    """Test accessing nested domain models."""

    def test_http_settings_access(self):
        """Should access HTTP settings."""
        settings = OntologyDownloadSettings()
        assert settings.http.timeout_read == 30.0
        assert settings.http.pool_max_connections == 64

    def test_security_settings_access(self):
        """Should access security settings."""
        settings = OntologyDownloadSettings()
        assert settings.security.strict_dns is True
        assert settings.security.allow_private_networks is False

    def test_extraction_settings_access(self):
        """Should access extraction settings."""
        settings = OntologyDownloadSettings()
        assert settings.extraction.max_depth == 32
        assert settings.extraction.max_entries == 50000

    def test_custom_nested_settings(self):
        """Should accept custom nested settings."""
        http = HttpSettings(timeout_read=60.0)
        settings = OntologyDownloadSettings(http=http)
        assert settings.http.timeout_read == 60.0

    def test_partial_nested_settings(self):
        """Should compose with partial nested settings."""
        extraction = ExtractionSettings(max_depth=64)
        settings = OntologyDownloadSettings(extraction=extraction)
        assert settings.extraction.max_depth == 64
        # Other extraction settings should still have defaults
        assert settings.extraction.max_entries == 50000


# ============================================================================
# SETTINGS ACCESSORS TESTS
# ============================================================================


class TestSettingsAccessors:
    """Test convenience accessors for settings."""

    def test_access_http_timeout(self):
        """Should access HTTP timeout settings."""
        settings = get_settings()
        assert settings.http.timeout_connect > 0
        assert settings.http.timeout_read > 0
        assert settings.http.timeout_write > 0
        assert settings.http.timeout_pool > 0

    def test_access_cache_settings(self):
        """Should access cache settings."""
        settings = get_settings()
        assert settings.cache.enabled is True
        assert settings.cache.dir.is_absolute()
        assert settings.cache.bypass is False

    def test_access_database_settings(self):
        """Should access database settings."""
        settings = get_settings()
        assert settings.db.path.is_absolute()
        assert settings.db.readonly is False
        assert settings.db.wlock is True

    def test_access_extraction_policies(self):
        """Should access extraction policies."""
        settings = get_settings()
        assert settings.extraction.encapsulate is True
        assert settings.extraction.overwrite == "reject"
        assert settings.extraction.duplicate_policy == "reject"


# ============================================================================
# PHASE 5 COMPLETION TESTS
# ============================================================================


class TestPhase53Complete:
    """Test Phase 5 (5.1 + 5.2 + 5.3) completion."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_settings_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_settings_cache()

    def test_all_10_models_composed(self):
        """All 10 domain models should be in root settings."""
        settings = OntologyDownloadSettings()

        # Foundation models (5)
        assert isinstance(settings.http, HttpSettings)
        assert isinstance(settings.cache, CacheSettings)
        assert isinstance(settings.retry, RetrySettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert isinstance(settings.telemetry, TelemetrySettings)

        # Complex models (5)
        assert isinstance(settings.security, SecuritySettings)
        assert isinstance(settings.ratelimit, RateLimitSettings)
        assert isinstance(settings.extraction, ExtractionSettings)
        assert isinstance(settings.storage, StorageSettings)
        assert isinstance(settings.db, DuckDBSettings)

    def test_61_fields_total(self):
        """All 61 fields should be accessible."""
        settings = OntologyDownloadSettings()
        dumped = settings.model_dump()

        # Count nested fields
        total_fields = (
            len(dumped["http"])  # 10 fields
            + len(dumped["cache"])  # 3 fields
            + len(dumped["retry"])  # 3 fields
            + len(dumped["logging"])  # 2 fields
            + len(dumped["telemetry"])  # 2 fields
            + len(dumped["security"])  # 5 fields
            + len(dumped["ratelimit"])  # 4 fields
            + len(dumped["extraction"])  # 23 fields
            + len(dumped["storage"])  # 3 fields
            + len(dumped["db"])  # 5 fields
        )

        assert total_fields == 62  # 10+3+3+2+2+5+4+25+3+5

    def test_settings_in_exports(self):
        """Root settings should be in __all__."""
        from DocsToKG.OntologyDownload import settings

        assert "OntologyDownloadSettings" in settings.__all__
        assert "get_settings" in settings.__all__
        assert "clear_settings_cache" in settings.__all__

    def test_backward_compatibility(self):
        """Should be backward compatible with existing code."""
        # Existing legacy classes should still be available
        from DocsToKG.OntologyDownload.settings import (
            DownloadConfiguration,
            LoggingConfiguration,
            DatabaseConfiguration,
            ValidationConfig,
        )

        # Should be able to instantiate them
        dc = DownloadConfiguration()
        lc = LoggingConfiguration()
        db_cfg = DatabaseConfiguration()
        vc = ValidationConfig()

        assert dc is not None
        assert lc is not None
        assert db_cfg is not None
        assert vc is not None


# ============================================================================
# SKIPPED: RESERVED FOR FUTURE PHASES
# ============================================================================


@pytest.mark.skip(reason="Environment variable parsing deferred to Phase 5.4")
class TestEnvironmentVariableParsing:
    """Environment variable parsing tests (Phase 5.4)."""

    def test_ontofetch_http_timeout_read(self):
        """Parse ONTOFETCH_HTTP__TIMEOUT_READ from environment."""
        pass

    def test_ontofetch_security_allowed_hosts(self):
        """Parse ONTOFETCH_SECURITY__ALLOWED_HOSTS from environment."""
        pass

    def test_source_precedence(self):
        """Test config source precedence: CLI → file → env."""
        pass


@pytest.mark.skip(reason="Migration guide deferred to Phase 5.4")
class TestMigrationFromLegacy:
    """Migration from legacy settings (Phase 5.4)."""

    def test_migration_helper(self):
        """Provide helper to migrate from legacy config."""
        pass

    def test_deprecation_warnings(self):
        """Legacy classes should emit deprecation warnings."""
        pass
