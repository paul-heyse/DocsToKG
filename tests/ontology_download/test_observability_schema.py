"""Tests for observability event schema generation and validation.

Covers:
- Schema generation (canonical, submodels)
- Schema validation
- Schema persistence (disk I/O)
- Schema comparison
- Schema summary
"""

import json
import tempfile
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.observability.events import (
    Event,
    EventContext,
    EventIds,
)
from DocsToKG.OntologyDownload.observability.schema import (
    EVENT_JSON_SCHEMA,
    EVENT_JSON_SCHEMA_VERSION,
    compare_schemas,
    generate_settings_schema,
    generate_submodel_schemas,
    get_schema_summary,
    load_schema_from_file,
    validate_event,
    write_schemas_to_disk,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_event():
    """Create a sample event for testing."""
    ctx = EventContext(
        app_version="1.0.0",
        os_name="Linux",
        python_version="3.10.0",
    )
    return Event(
        ts="2025-10-20T01:23:45Z",
        type="test.event",
        level="INFO",
        run_id="run-123",
        config_hash="hash-abc",
        service="test",
        context=ctx,
        ids=EventIds(),
        payload={"test": "data"},
    )


# ============================================================================
# Canonical Schema Tests
# ============================================================================


class TestCanonicalSchema:
    """Test the canonical event schema definition."""

    def test_schema_exists(self):
        """Canonical schema is defined."""
        assert EVENT_JSON_SCHEMA is not None
        assert isinstance(EVENT_JSON_SCHEMA, dict)

    def test_schema_version(self):
        """Schema has version metadata."""
        assert EVENT_JSON_SCHEMA_VERSION == "1.0"
        assert EVENT_JSON_SCHEMA.get("version") == "1.0"

    def test_schema_has_required_fields(self):
        """Schema specifies required fields."""
        required = EVENT_JSON_SCHEMA.get("required", [])
        assert "ts" in required
        assert "type" in required
        assert "level" in required
        assert "run_id" in required
        assert "config_hash" in required
        assert "service" in required
        assert "context" in required
        assert "ids" in required
        assert "payload" in required

    def test_schema_has_properties(self):
        """Schema defines all property types."""
        props = EVENT_JSON_SCHEMA.get("properties", {})
        assert "ts" in props
        assert props["ts"]["type"] == "string"
        assert "level" in props
        assert "enum" in props["level"]


# ============================================================================
# Schema Generation Tests
# ============================================================================


class TestSchemaGeneration:
    """Test schema generation functions."""

    def test_generate_settings_schema(self):
        """generate_settings_schema() returns canonical schema."""
        schema = generate_settings_schema()
        assert schema == EVENT_JSON_SCHEMA
        # Verify it's a copy
        assert schema is not EVENT_JSON_SCHEMA

    def test_generate_submodel_schemas(self):
        """generate_submodel_schemas() returns all submodel schemas."""
        schemas = generate_submodel_schemas()
        assert "Event" in schemas
        assert "EventContext" in schemas
        assert "EventIds" in schemas

    def test_submodel_schema_event(self):
        """Event submodel schema matches canonical schema."""
        schemas = generate_submodel_schemas()
        assert schemas["Event"] == EVENT_JSON_SCHEMA

    def test_submodel_schema_context(self):
        """EventContext submodel schema is valid."""
        schemas = generate_submodel_schemas()
        ctx_schema = schemas["EventContext"]
        assert ctx_schema["title"] == "EventContext"
        assert "app_version" in ctx_schema.get("properties", {})

    def test_submodel_schema_ids(self):
        """EventIds submodel schema is valid."""
        schemas = generate_submodel_schemas()
        ids_schema = schemas["EventIds"]
        assert ids_schema["title"] == "EventIds"
        assert "version_id" in ids_schema.get("properties", {})


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestSchemaValidation:
    """Test event validation against schema."""

    def test_validate_valid_event(self, sample_event):
        """Valid event passes validation."""
        is_valid, errors = validate_event(sample_event.to_dict())
        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_required_field(self, sample_event):
        """Missing required field fails validation."""
        event_dict = sample_event.to_dict()
        del event_dict["ts"]
        is_valid, errors = validate_event(event_dict)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_invalid_type(self, sample_event):
        """Invalid field type fails validation."""
        event_dict = sample_event.to_dict()
        event_dict["ts"] = 12345  # Should be string
        is_valid, errors = validate_event(event_dict)
        assert not is_valid

    def test_validate_invalid_level(self, sample_event):
        """Invalid level value fails validation."""
        event_dict = sample_event.to_dict()
        event_dict["level"] = "DEBUG"  # Should be INFO|WARN|ERROR
        is_valid, errors = validate_event(event_dict)
        assert not is_valid

    def test_validate_invalid_event_type_format(self, sample_event):
        """Invalid event type format fails validation."""
        event_dict = sample_event.to_dict()
        event_dict["type"] = "INVALID TYPE"  # Should be lowercase with dots
        is_valid, errors = validate_event(event_dict)
        assert not is_valid

    def test_validate_returns_errors(self, sample_event):
        """Validation errors are descriptive."""
        event_dict = sample_event.to_dict()
        event_dict["level"] = "INVALID"
        is_valid, errors = validate_event(event_dict)
        assert not is_valid
        assert len(errors) == 1
        assert isinstance(errors[0], str)


# ============================================================================
# Schema Persistence Tests
# ============================================================================


class TestSchemaPersistence:
    """Test schema disk I/O."""

    def test_write_schemas_to_disk(self):
        """write_schemas_to_disk() writes schema files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            written = write_schemas_to_disk(output_dir)

            assert "event-schema.json" in written
            assert "EventContext-schema.json" in written
            assert "EventIds-schema.json" in written
            assert "Event-schema.json" in written

    def test_written_schemas_are_valid_json(self):
        """Written schemas are valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            written = write_schemas_to_disk(output_dir)

            for filename, filepath in written.items():
                with open(filepath) as f:
                    # Should not raise
                    json.load(f)

    def test_schema_files_are_reproducible(self):
        """Schema files have deterministic output."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                write_schemas_to_disk(Path(tmpdir1))
                write_schemas_to_disk(Path(tmpdir2))

                # Compare file contents
                schema1 = (Path(tmpdir1) / "event-schema.json").read_text()
                schema2 = (Path(tmpdir2) / "event-schema.json").read_text()
                assert schema1 == schema2

    def test_load_schema_from_file(self):
        """load_schema_from_file() reads schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            written = write_schemas_to_disk(output_dir)

            schema = load_schema_from_file(written["event-schema.json"])
            assert schema is not None
            assert schema["title"] == "OntologyDownload Event Schema"

    def test_load_nonexistent_file(self):
        """load_schema_from_file() handles missing files gracefully."""
        schema = load_schema_from_file(Path("/nonexistent/path"))
        assert schema is None


# ============================================================================
# Schema Comparison Tests
# ============================================================================


class TestSchemaComparison:
    """Test schema comparison for CI drift detection."""

    def test_compare_identical_schemas(self):
        """Identical schemas compare as equal."""
        schema1 = generate_settings_schema()
        schema2 = generate_settings_schema()

        result = compare_schemas(schema1, schema2)
        assert result["identical"] is True
        assert len(result["differences"]) == 0

    def test_compare_different_schemas(self):
        """Different schemas detect differences."""
        schema1 = generate_settings_schema()
        schema2 = generate_settings_schema()
        schema2["version"] = "2.0"

        result = compare_schemas(schema1, schema2)
        assert result["identical"] is False
        assert len(result["differences"]) > 0

    def test_compare_detects_added_required_fields(self):
        """Comparison detects added required fields."""
        schema1 = generate_settings_schema()
        # Make a proper copy and modify it
        import copy

        schema2 = copy.deepcopy(schema1)
        schema2["required"] = list(schema2["required"]) + ["new_field"]

        result = compare_schemas(schema1, schema2)
        assert result["identical"] is False
        # Should mention the addition
        assert any("new_field" in diff for diff in result["differences"])

    def test_compare_detects_removed_properties(self):
        """Comparison detects removed properties."""
        schema1 = generate_settings_schema()
        # Make a proper copy and modify it
        import copy

        schema2 = copy.deepcopy(schema1)
        del schema2["properties"]["payload"]

        result = compare_schemas(schema1, schema2)
        assert result["identical"] is False
        assert any("payload" in diff for diff in result["differences"])


# ============================================================================
# Schema Summary Tests
# ============================================================================


class TestSchemaSummary:
    """Test schema summary generation."""

    def test_get_schema_summary(self):
        """get_schema_summary() returns summary dict."""
        summary = get_schema_summary()
        assert isinstance(summary, dict)

    def test_summary_has_version(self):
        """Summary includes schema version."""
        summary = get_schema_summary()
        assert "schema_version" in summary
        assert summary["schema_version"] == "1.0"

    def test_summary_has_property_counts(self):
        """Summary includes property counts."""
        summary = get_schema_summary()
        assert "total_properties" in summary
        assert "required_properties" in summary
        assert summary["total_properties"] > 0

    def test_summary_has_required_fields(self):
        """Summary lists required fields."""
        summary = get_schema_summary()
        assert "required_fields" in summary
        assert "ts" in summary["required_fields"]

    def test_summary_has_property_overview(self):
        """Summary includes property overview."""
        summary = get_schema_summary()
        assert "properties_overview" in summary
        overview = summary["properties_overview"]
        assert "ts" in overview
        assert "type" in overview["ts"]
        assert "required" in overview["ts"]

    def test_summary_has_created_at(self):
        """Summary includes creation timestamp."""
        summary = get_schema_summary()
        assert "created_at" in summary
        assert "T" in summary["created_at"]  # ISO format
