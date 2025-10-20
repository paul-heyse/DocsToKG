"""Tests for settings_schema: JSON Schema generation and validation.

Tests verify:
- Deterministic schema generation (sorted keys)
- Top-level and submodel schema generation
- Disk writing and file generation
- Configuration file validation
- Schema statistics and summaries
"""

import json
import tempfile
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.settings_schema import (
    CanonicalJsonSchema,
    generate_settings_schema,
    generate_submodel_schemas,
    get_schema_summary,
    validate_config_file,
    write_schemas_to_disk,
)


class TestCanonicalJsonSchema:
    """Tests for CanonicalJsonSchema deterministic generation."""

    def test_canonical_schema_has_sorted_keys(self):
        """Schema generation should sort keys alphabetically."""
        schema = generate_settings_schema()

        # Convert to JSON and back to ensure no random ordering
        json_str = json.dumps(schema, sort_keys=True)
        schema_from_json = json.loads(json_str)

        # Keys should be sorted at top level
        keys = list(schema.keys())
        assert keys == sorted(keys)

    def test_canonical_schema_is_deterministic(self):
        """Multiple generations should produce identical output."""
        schema1_json = json.dumps(generate_settings_schema(), sort_keys=True)
        schema2_json = json.dumps(generate_settings_schema(), sort_keys=True)
        schema3_json = json.dumps(generate_settings_schema(), sort_keys=True)

        assert schema1_json == schema2_json == schema3_json

    def test_sort_dict_recursive(self):
        """CanonicalJsonSchema._sort_dict should recursively sort nested dicts."""
        nested = {
            "z": 1,
            "a": {"z": 2, "a": 3},
            "m": [{"z": 4, "a": 5}],
        }

        sorted_dict = CanonicalJsonSchema._sort_dict(nested)

        # Top level sorted
        top_keys = list(sorted_dict.keys())
        assert top_keys == ["a", "m", "z"]

        # Nested dict sorted
        nested_keys = list(sorted_dict["a"].keys())
        assert nested_keys == ["a", "z"]

        # Dict in list sorted
        list_dict_keys = list(sorted_dict["m"][0].keys())
        assert list_dict_keys == ["a", "z"]


class TestSettingsSchemaGeneration:
    """Tests for top-level and submodel schema generation."""

    def test_generate_settings_schema_returns_dict(self):
        """generate_settings_schema should return valid JSON Schema."""
        schema = generate_settings_schema()

        assert isinstance(schema, dict)
        assert "$schema" in schema or "type" in schema

    def test_settings_schema_has_title(self):
        """Top-level schema should have a title."""
        schema = generate_settings_schema()

        assert "title" in schema
        assert schema["title"] == "OntologyDownloadSettings"

    def test_settings_schema_has_10_properties(self):
        """Top-level schema should have 10 domain model properties."""
        schema = generate_settings_schema()

        assert "properties" in schema
        assert len(schema["properties"]) == 10

    def test_generate_submodel_schemas_returns_dict(self):
        """generate_submodel_schemas should return dict of schemas."""
        schemas = generate_submodel_schemas()

        assert isinstance(schemas, dict)
        assert len(schemas) == 10

    def test_each_submodel_is_valid_schema(self):
        """Each submodel schema should be valid JSON Schema."""
        schemas = generate_submodel_schemas()

        for name, schema in schemas.items():
            assert isinstance(schema, dict), f"Schema for {name} should be dict"
            assert "type" in schema or "$ref" in schema, f"Schema for {name} missing type"


class TestSchemaWriting:
    """Tests for writing schemas to disk."""

    def test_write_schemas_to_disk_returns_path_and_count(self):
        """write_schemas_to_disk should return (path, file_count)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            path, count = write_schemas_to_disk(tmppath)

            assert path == tmppath
            assert isinstance(count, int)
            assert count > 0

    def test_write_schemas_to_disk_creates_11_files(self):
        """write_schemas_to_disk should create 11 files (1 top + 10 sub)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _, count = write_schemas_to_disk(tmppath)

            files = list(tmppath.glob("*.json"))
            assert len(files) == 11
            assert count == 11

    def test_write_schemas_creates_top_level_schema(self):
        """write_schemas_to_disk should create settings.schema.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            write_schemas_to_disk(tmppath)

            schema_file = tmppath / "settings.schema.json"
            assert schema_file.exists()

            schema = json.loads(schema_file.read_text())
            assert schema["title"] == "OntologyDownloadSettings"

    def test_schema_files_are_valid_json(self):
        """All written schema files should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            write_schemas_to_disk(tmppath)

            for schema_file in tmppath.glob("*.json"):
                content = schema_file.read_text()
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    pytest.fail(f"File {schema_file.name} is not valid JSON")


class TestConfigValidation:
    """Tests for configuration file validation."""

    def test_validate_config_file_with_valid_yaml(self):
        """validate_config_file should accept valid YAML config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("http:\n  timeout_connect: 5\n  timeout_read: 30\n")

            valid, errors = validate_config_file(config_file)
            assert valid is True
            assert errors == []

    def test_validate_config_file_with_valid_json(self):
        """validate_config_file should accept valid JSON config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text('{"http": {"timeout_connect": 5}}')

            valid, errors = validate_config_file(config_file)
            assert valid is True
            assert errors == []

    def test_validate_config_file_missing_file(self):
        """validate_config_file should handle missing files."""
        missing_file = Path("/nonexistent/config.yaml")

        valid, errors = validate_config_file(missing_file)
        assert valid is False
        assert len(errors) > 0


class TestSchemaSummary:
    """Tests for schema summary statistics."""

    def test_get_schema_summary_returns_dict(self):
        """get_schema_summary should return dict with statistics."""
        summary = get_schema_summary()

        assert isinstance(summary, dict)
        assert "total_models" in summary
        assert "total_properties" in summary
        assert "required_properties" in summary
        assert "optional_properties" in summary
        assert "models" in summary

    def test_schema_summary_has_10_models(self):
        """Schema summary should show 10 domain models."""
        summary = get_schema_summary()

        assert summary["total_models"] == 10
        assert len(summary["models"]) == 10

    def test_schema_summary_property_counts(self):
        """Schema summary should have reasonable property counts."""
        summary = get_schema_summary()

        # Should have more than just the 10 models
        assert summary["total_properties"] > 10

        # Required + optional should equal total
        assert (
            summary["required_properties"] + summary["optional_properties"]
            == summary["total_properties"]
        )


class TestSchemaIntegration:
    """Integration tests for complete schema workflow."""

    def test_generated_schema_validates_default_settings(self):
        """Generated schema should validate default settings when serialized."""
        from DocsToKG.OntologyDownload.settings import OntologyDownloadSettings

        settings = OntologyDownloadSettings()
        # Serialize to JSON-compatible format (converts UUID to string, etc)
        settings_json = json.dumps(settings.model_dump(mode="json"), default=str)
        settings_dict = json.loads(settings_json)

        schema = generate_settings_schema()

        try:
            import jsonschema

            jsonschema.validate(settings_dict, schema)
        except ImportError:
            pytest.skip("jsonschema not available")

    def test_schema_generation_performance(self):
        """Schema generation should be fast."""
        import time

        start = time.perf_counter()
        _ = generate_settings_schema()
        elapsed = time.perf_counter() - start

        # Should complete in <100ms
        assert elapsed < 0.1, f"Schema generation took {elapsed:.3f}s"

    def test_subschema_generation_performance(self):
        """Subschema generation should be fast."""
        import time

        start = time.perf_counter()
        _ = generate_submodel_schemas()
        elapsed = time.perf_counter() - start

        # Should complete in <200ms for all 10
        assert elapsed < 0.2, f"Subschema generation took {elapsed:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
