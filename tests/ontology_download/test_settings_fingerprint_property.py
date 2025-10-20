"""Tests for source_fingerprint property on OntologyDownloadSettings.

Tests verify:
- Property existence and accessibility
- Correct return type and values
- Defensive copying
- Integration with source tracing
"""

import pytest
from pydantic_core import ValidationError as PydanticValidationError

from DocsToKG.OntologyDownload.settings import OntologyDownloadSettings
from DocsToKG.OntologyDownload.settings_sources import (
    set_source_fingerprint,
    clear_source_context,
)


class TestSourceFingerprintProperty:
    """Tests for source_fingerprint property on settings."""

    def setup_method(self):
        """Clear context before each test."""
        clear_source_context()

    def test_source_fingerprint_property_exists(self):
        """OntologyDownloadSettings should have source_fingerprint property."""
        settings = OntologyDownloadSettings()
        assert hasattr(settings, "source_fingerprint")

    def test_source_fingerprint_is_readable(self):
        """source_fingerprint property should be readable."""
        settings = OntologyDownloadSettings()
        fp = settings.source_fingerprint
        assert isinstance(fp, dict)

    def test_source_fingerprint_returns_dict(self):
        """source_fingerprint should return a dict."""
        settings = OntologyDownloadSettings()
        fp = settings.source_fingerprint
        assert isinstance(fp, dict)

    def test_source_fingerprint_empty_by_default(self):
        """source_fingerprint should be empty dict by default."""
        clear_source_context()
        settings = OntologyDownloadSettings()
        fp = settings.source_fingerprint
        assert fp == {}

    def test_source_fingerprint_with_traced_sources(self):
        """source_fingerprint should contain traced sources."""
        test_sources = {
            "http__timeout_read": "cli",
            "security__allowed_hosts": "env",
            "db__path": "default",
        }
        set_source_fingerprint(test_sources)

        settings = OntologyDownloadSettings()
        fp = settings.source_fingerprint

        assert fp == test_sources
        assert fp["http__timeout_read"] == "cli"
        assert fp["security__allowed_hosts"] == "env"
        assert fp["db__path"] == "default"

    def test_source_fingerprint_returns_defensive_copy(self):
        """source_fingerprint should return a defensive copy."""
        test_sources = {"field": "cli"}
        set_source_fingerprint(test_sources)

        settings = OntologyDownloadSettings()
        fp1 = settings.source_fingerprint
        fp1["field"] = "modified"

        fp2 = settings.source_fingerprint
        assert fp2["field"] == "cli"

    def test_source_fingerprint_with_config_file_source(self):
        """source_fingerprint should track config file sources."""
        test_sources = {"security__allowed_hosts": "config:/etc/settings.yaml"}
        set_source_fingerprint(test_sources)

        settings = OntologyDownloadSettings()
        fp = settings.source_fingerprint

        assert fp["security__allowed_hosts"] == "config:/etc/settings.yaml"

    def test_source_fingerprint_with_dot_env_source(self):
        """source_fingerprint should track .env sources."""
        test_sources = {"http__timeout_read": ".env.ontofetch", "db__path": ".env"}
        set_source_fingerprint(test_sources)

        settings = OntologyDownloadSettings()
        fp = settings.source_fingerprint

        assert fp["http__timeout_read"] == ".env.ontofetch"
        assert fp["db__path"] == ".env"

    def test_source_fingerprint_integration_with_config_hash(self):
        """source_fingerprint should work alongside config_hash method."""
        set_source_fingerprint({"field1": "cli", "field2": "env"})

        settings = OntologyDownloadSettings()

        # Both should be accessible
        fp = settings.source_fingerprint
        assert isinstance(fp, dict)

        # config_hash should also work
        config_hash = settings.config_hash()
        assert isinstance(config_hash, str)

    def test_source_fingerprint_is_readonly(self):
        """source_fingerprint property should not be settable."""
        settings = OntologyDownloadSettings()

        with pytest.raises(PydanticValidationError):
            settings.source_fingerprint = {"field": "value"}

    def test_source_fingerprint_value_types(self):
        """source_fingerprint values should be strings."""
        test_sources = {
            "field1": "cli",
            "field2": "env",
            "field3": "config:/path",
        }
        set_source_fingerprint(test_sources)

        settings = OntologyDownloadSettings()
        fp = settings.source_fingerprint

        for key, value in fp.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_source_fingerprint_key_names(self):
        """source_fingerprint keys should be field names."""
        test_sources = {
            "http__timeout_connect": "cli",
            "security__allowed_hosts": "env",
        }
        set_source_fingerprint(test_sources)

        settings = OntologyDownloadSettings()
        fp = settings.source_fingerprint

        for key in fp.keys():
            assert isinstance(key, str)
            assert len(key) > 0

    def test_source_fingerprint_docstring_exists(self):
        """source_fingerprint property should have docstring."""
        assert OntologyDownloadSettings.source_fingerprint.__doc__ is not None

    def test_source_fingerprint_multiple_calls_consistent(self):
        """Multiple calls to source_fingerprint should return same data."""
        test_sources = {"field": "cli"}
        set_source_fingerprint(test_sources)

        settings = OntologyDownloadSettings()

        fp1 = settings.source_fingerprint
        fp2 = settings.source_fingerprint
        fp3 = settings.source_fingerprint

        assert fp1 == fp2 == fp3
