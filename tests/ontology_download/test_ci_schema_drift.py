"""Tests for CI schema drift detection.

Tests verify:
- Schema comparison logic
- Drift detection
- Report generation
- CLI functionality
"""

import json
import tempfile
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.ci_schema_drift import (
    check_schema_drift,
    _normalize_schema,
)
from DocsToKG.OntologyDownload.settings_schema import (
    generate_settings_schema,
    generate_submodel_schemas,
)


class TestSchemaNormalization:
    """Tests for schema normalization."""

    def test_normalize_schema_produces_deterministic_output(self):
        """Normalized schemas should be deterministic."""
        schema = {"z": 1, "a": {"z": 2, "a": 3}, "m": 4}
        
        norm1 = _normalize_schema(schema)
        norm2 = _normalize_schema(schema)
        
        assert norm1 == norm2

    def test_normalize_schema_sorts_keys(self):
        """Normalized schema should have sorted keys."""
        schema = {"z": 1, "a": 2, "m": 3}
        normalized = _normalize_schema(schema)
        
        # Parse back and check key order
        parsed = json.loads(normalized)
        keys = list(parsed.keys())
        assert keys == ["a", "m", "z"]

    def test_normalize_schema_deep_sorts(self):
        """Normalized schema should deeply sort nested keys."""
        schema = {"z": {"z": 1, "a": 2}, "a": 3}
        normalized = _normalize_schema(schema)
        parsed = json.loads(normalized)
        
        # Top level sorted
        assert list(parsed.keys()) == ["a", "z"]
        # Nested sorted
        assert list(parsed["z"].keys()) == ["a", "z"]


class TestSchemaDriftDetection:
    """Tests for schema drift detection."""

    def test_check_schema_drift_no_drift(self):
        """check_schema_drift should return 0 when no drift detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            
            # Write current schemas
            top_schema = generate_settings_schema()
            top_schema_path = schema_dir / "settings.schema.json"
            top_schema_path.write_text(
                json.dumps(top_schema, indent=2, sort_keys=True)
            )
            
            sub_schemas = generate_submodel_schemas()
            for domain, schema in sub_schemas.items():
                schema_path = schema_dir / f"settings.{domain}.subschema.json"
                schema_path.write_text(
                    json.dumps(schema, indent=2, sort_keys=True)
                )
            
            # Check drift
            exit_code = check_schema_drift(
                expected_dir=schema_dir,
                fail_on_drift=True,
                verbose=False,
            )
            
            assert exit_code == 0

    def test_check_schema_drift_with_drift_fails(self):
        """check_schema_drift should return 1 when drift detected and fail_on_drift=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            
            # Write modified schema (with extra field)
            top_schema = generate_settings_schema()
            top_schema["extra_field"] = "extra_value"
            top_schema_path = schema_dir / "settings.schema.json"
            top_schema_path.write_text(
                json.dumps(top_schema, indent=2, sort_keys=True)
            )
            
            sub_schemas = generate_submodel_schemas()
            for domain, schema in sub_schemas.items():
                schema_path = schema_dir / f"settings.{domain}.subschema.json"
                schema_path.write_text(
                    json.dumps(schema, indent=2, sort_keys=True)
                )
            
            # Check drift - should detect mismatch
            exit_code = check_schema_drift(
                expected_dir=schema_dir,
                fail_on_drift=True,
                verbose=False,
            )
            
            # Since we added a field, schemas won't match
            # (current won't have the extra_field)
            assert exit_code == 1

    def test_check_schema_drift_missing_expected_dir(self):
        """check_schema_drift should handle missing expected dir gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir) / "nonexistent"
            
            # Should still run but report missing schemas
            exit_code = check_schema_drift(
                expected_dir=schema_dir,
                fail_on_drift=False,  # Don't fail, just report
                verbose=False,
            )
            
            # Should return 0 since fail_on_drift=False
            assert exit_code == 0

    def test_check_schema_drift_no_fail_on_drift(self):
        """check_schema_drift with fail_on_drift=False should return 0 even with drift."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            
            # Write modified schema
            top_schema = generate_settings_schema()
            top_schema["extra"] = "value"
            top_schema_path = schema_dir / "settings.schema.json"
            top_schema_path.write_text(
                json.dumps(top_schema, indent=2, sort_keys=True)
            )
            
            sub_schemas = generate_submodel_schemas()
            for domain, schema in sub_schemas.items():
                schema_path = schema_dir / f"settings.{domain}.subschema.json"
                schema_path.write_text(
                    json.dumps(schema, indent=2, sort_keys=True)
                )
            
            # Check drift with fail_on_drift=False
            exit_code = check_schema_drift(
                expected_dir=schema_dir,
                fail_on_drift=False,
                verbose=False,
            )
            
            # Should return 0 even though drift exists
            assert exit_code == 0


class TestDriftDetectionIntegration:
    """Integration tests for drift detection."""

    def test_full_workflow_no_drift(self):
        """Full workflow should detect when schemas are identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            
            # Generate and save schemas
            top_schema = generate_settings_schema()
            sub_schemas = generate_submodel_schemas()
            
            # Write top-level
            (schema_dir / "settings.schema.json").write_text(
                json.dumps(top_schema, indent=2, sort_keys=True)
            )
            
            # Write submodels
            for domain, schema in sub_schemas.items():
                (schema_dir / f"settings.{domain}.subschema.json").write_text(
                    json.dumps(schema, indent=2, sort_keys=True)
                )
            
            # Now check drift (should match since we just wrote them)
            exit_code = check_schema_drift(
                expected_dir=schema_dir,
                fail_on_drift=True,
            )
            
            assert exit_code == 0

    def test_drift_detection_with_multiple_files(self):
        """Drift detection should handle multiple schema files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            
            # Write schemas
            top_schema = generate_settings_schema()
            sub_schemas = generate_submodel_schemas()
            
            (schema_dir / "settings.schema.json").write_text(
                json.dumps(top_schema, indent=2, sort_keys=True)
            )
            
            for domain, schema in sub_schemas.items():
                (schema_dir / f"settings.{domain}.subschema.json").write_text(
                    json.dumps(schema, indent=2, sort_keys=True)
                )
            
            # Verify all files exist
            schema_files = list(schema_dir.glob("*.json"))
            assert len(schema_files) == 11  # 1 top + 10 sub


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
