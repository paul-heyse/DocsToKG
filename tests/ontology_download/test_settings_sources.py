"""Tests for settings_sources: TracingSettingsSource and source attribution.

Tests verify:
- Source tracking for each field
- Thread-safe context management
- Integration with Pydantic settings sources
- Proper handling of defaults vs provided values
"""

from unittest.mock import Mock
import pytest

from DocsToKG.OntologyDownload.settings_sources import (
    TracingSettingsSource,
    get_source_fingerprint,
    set_source_fingerprint,
    clear_source_context,
    init_source_context,
)


class MockSettingsSource:
    """Mock Pydantic settings source for testing."""

    def __init__(self, values=None):
        self.values = values or {}

    def get_field_value(self, field, field_name):
        """Mock get_field_value that returns value if present."""
        if field_name in self.values:
            return self.values[field_name], field_name, False
        return None, field_name, True  # using_default=True


class TestTracingSettingsSourceBasic:
    """Tests for basic TracingSettingsSource functionality."""

    def setup_method(self):
        """Clear context before each test."""
        clear_source_context()

    def test_records_cli_source_when_value_provided(self):
        """TracingSettingsSource should record source for CLI values."""
        mock_source = MockSettingsSource({"http__timeout": 5})
        traced = TracingSettingsSource(mock_source, "cli")
        mock_field = Mock()

        traced.get_field_value(mock_field, "http__timeout")

        fp = get_source_fingerprint()
        assert fp["http__timeout"] == "cli"

    def test_records_env_source_when_value_provided(self):
        """TracingSettingsSource should record source for environment values."""
        mock_source = MockSettingsSource({"security__allowed_hosts": ["localhost"]})
        traced = TracingSettingsSource(mock_source, "env")
        mock_field = Mock()

        traced.get_field_value(mock_field, "security__allowed_hosts")

        fp = get_source_fingerprint()
        assert fp["security__allowed_hosts"] == "env"

    def test_does_not_record_default_values(self):
        """TracingSettingsSource should not record defaults in fingerprint."""
        mock_source = MockSettingsSource({})  # No values
        traced = TracingSettingsSource(mock_source, "cli")
        mock_field = Mock()

        traced.get_field_value(mock_field, "http__timeout")

        fp = get_source_fingerprint()
        assert "http__timeout" not in fp

    def test_multiple_fields_accumulated(self):
        """Multiple get_field_value calls should accumulate sources."""
        mock_source = MockSettingsSource({"field1": "val1", "field2": "val2"})
        traced = TracingSettingsSource(mock_source, "cli")
        mock_field = Mock()

        traced.get_field_value(mock_field, "field1")
        traced.get_field_value(mock_field, "field2")

        fp = get_source_fingerprint()
        assert fp["field1"] == "cli"
        assert fp["field2"] == "cli"


class TestTracingSettingsSourcePrecedence:
    """Tests for source precedence and override behavior."""

    def setup_method(self):
        """Clear context before each test."""
        clear_source_context()

    def test_later_source_overrides_earlier_source(self):
        """Later sources should override earlier ones for same field."""
        mock_cli = MockSettingsSource({"field": "cli_val"})
        mock_env = MockSettingsSource({"field": "env_val"})

        traced_cli = TracingSettingsSource(mock_cli, "cli")
        traced_env = TracingSettingsSource(mock_env, "env")
        mock_field = Mock()

        traced_cli.get_field_value(mock_field, "field")
        traced_env.get_field_value(mock_field, "field")

        fp = get_source_fingerprint()
        assert fp["field"] == "env"  # Later source wins

    def test_source_names_are_preserved(self):
        """Source names should match what was provided."""
        mock_source = MockSettingsSource({"field": "value"})
        traced = TracingSettingsSource(mock_source, "config:/path/to/file.yaml")
        mock_field = Mock()

        traced.get_field_value(mock_field, "field")

        fp = get_source_fingerprint()
        assert fp["field"] == "config:/path/to/file.yaml"


class TestTracingSettingsSourceCallable:
    """Tests for __call__ interface."""

    def setup_method(self):
        """Clear context before each test."""
        clear_source_context()

    def test_call_interface_records_all_fields(self):
        """__call__ should record all returned fields."""
        class CallableSource:
            def __call__(self):
                return {"field1": "val1", "field2": "val2", "field3": "val3"}

        mock_source = CallableSource()
        traced = TracingSettingsSource(mock_source, "config")

        traced()

        fp = get_source_fingerprint()
        assert fp["field1"] == "config"
        assert fp["field2"] == "config"
        assert fp["field3"] == "config"


class TestTracingSettingsSourceContext:
    """Tests for context variable management."""

    def test_get_source_fingerprint_returns_copy(self):
        """get_source_fingerprint should return a defensive copy."""
        set_source_fingerprint({"field": "cli"})

        fp1 = get_source_fingerprint()
        fp1["field"] = "modified"

        fp2 = get_source_fingerprint()
        assert fp2["field"] == "cli"  # Should not be modified

    def test_clear_source_context_resets_fingerprint(self):
        """clear_source_context should reset the fingerprint."""
        set_source_fingerprint({"field": "cli"})
        clear_source_context()

        fp = get_source_fingerprint()
        assert fp == {}

    def test_init_source_context_resets_fingerprint(self):
        """init_source_context should reset the fingerprint."""
        set_source_fingerprint({"field": "env"})
        init_source_context()

        fp = get_source_fingerprint()
        assert fp == {}

    def test_set_source_fingerprint_stores_copy(self):
        """set_source_fingerprint should store a copy."""
        original = {"field": "source"}
        set_source_fingerprint(original)
        original["field"] = "modified"

        fp = get_source_fingerprint()
        assert fp["field"] == "source"  # Should not be modified


class TestTracingSettingsSourceErrorHandling:
    """Tests for error handling and edge cases."""

    def setup_method(self):
        """Clear context before each test."""
        clear_source_context()

    def test_propagates_exceptions_from_source(self):
        """TracingSettingsSource should propagate source exceptions."""
        class FailingSource:
            def get_field_value(self, field, field_name):
                raise ValueError("Source error")

        mock_source = FailingSource()
        traced = TracingSettingsSource(mock_source, "cli")
        mock_field = Mock()

        with pytest.raises(ValueError, match="Source error"):
            traced.get_field_value(mock_field, "field")

    def test_handles_none_values(self):
        """TracingSettingsSource should handle None values."""
        mock_source = MockSettingsSource({"field": None})
        traced = TracingSettingsSource(mock_source, "cli")
        mock_field = Mock()

        traced.get_field_value(mock_field, "field")

        fp = get_source_fingerprint()
        assert fp["field"] == "cli"  # None is still a provided value

    def test_handles_empty_dict_in_callable(self):
        """__call__ should handle empty dict gracefully."""
        class EmptyCallableSource:
            def __call__(self):
                return {}

        mock_source = EmptyCallableSource()
        traced = TracingSettingsSource(mock_source, "cli")

        result = traced()

        assert result == {}
        fp = get_source_fingerprint()
        assert fp == {}


class TestTracingSettingsSourceRepr:
    """Tests for string representation."""

    def test_repr_shows_source_class_and_name(self):
        """__repr__ should show source class and name."""
        mock_source = MockSettingsSource()
        traced = TracingSettingsSource(mock_source, "cli")

        repr_str = repr(traced)

        assert "TracingSettingsSource" in repr_str
        assert "cli" in repr_str


class TestTracingSettingsSourceIntegration:
    """Integration tests for complete workflows."""

    def setup_method(self):
        """Clear context before each test."""
        clear_source_context()

    def test_multiple_sources_accumulate(self):
        """Multiple sources should accumulate in fingerprint."""
        cli_source = MockSettingsSource({"http__timeout": 5})
        env_source = MockSettingsSource({"security__allowed": ["localhost"]})
        config_source = MockSettingsSource({"db__path": "/data/db"})

        traced_cli = TracingSettingsSource(cli_source, "cli")
        traced_env = TracingSettingsSource(env_source, "env")
        traced_config = TracingSettingsSource(config_source, "config:/etc/config.yaml")

        mock_field = Mock()

        traced_cli.get_field_value(mock_field, "http__timeout")
        traced_env.get_field_value(mock_field, "security__allowed")
        traced_config.get_field_value(mock_field, "db__path")

        fp = get_source_fingerprint()

        assert fp["http__timeout"] == "cli"
        assert fp["security__allowed"] == "env"
        assert fp["db__path"] == "config:/etc/config.yaml"

    def test_precedence_workflow(self):
        """Test typical precedence workflow: CLI > Config > Env > Default."""
        # CLI layer provides http__timeout and security__allowed
        cli_source = MockSettingsSource({"http__timeout": 10, "security__allowed": "cli_hosts"})
        # Config layer provides security__allowed and db__path
        config_source = MockSettingsSource({"security__allowed": "config_hosts", "db__path": "/data"})
        # Env layer provides only db__path
        env_source = MockSettingsSource({"db__path": "/env/data"})

        traced_cli = TracingSettingsSource(cli_source, "cli")
        traced_config = TracingSettingsSource(config_source, "config")
        traced_env = TracingSettingsSource(env_source, "env")

        mock_field = Mock()

        # Apply in precedence order
        traced_cli.get_field_value(mock_field, "http__timeout")
        traced_cli.get_field_value(mock_field, "security__allowed")

        traced_config.get_field_value(mock_field, "security__allowed")  # Overrides CLI
        traced_config.get_field_value(mock_field, "db__path")

        traced_env.get_field_value(mock_field, "db__path")  # Overrides config

        fp = get_source_fingerprint()

        assert fp["http__timeout"] == "cli"
        assert fp["security__allowed"] == "config"  # Last source wins (config called after cli)
        assert fp["db__path"] == "env"  # Env is last

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
