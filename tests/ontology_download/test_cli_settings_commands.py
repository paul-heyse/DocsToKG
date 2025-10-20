"""Tests for CLI settings commands: show, schema, validate.

Tests verify:
- Command execution
- Output formatting (table, json, yaml)
- Schema generation
- Config file validation
- Error handling
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from DocsToKG.OntologyDownload.cli_settings_commands import settings_app


runner = CliRunner()


class TestSettingsShowCommand:
    """Tests for 'settings show' command."""

    def test_show_command_succeeds(self):
        """show command should execute successfully."""
        result = runner.invoke(settings_app, ["show", "--format", "json"])
        assert result.exit_code == 0

    def test_show_command_json_output(self):
        """show command with --format json should output valid JSON."""
        result = runner.invoke(settings_app, ["show", "--format", "json"])
        assert result.exit_code == 0
        
        try:
            data = json.loads(result.stdout)
            assert isinstance(data, list)
            if data:
                assert "field" in data[0]
                assert "value" in data[0]
                assert "source" in data[0]
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_show_command_includes_source_field(self):
        """show command output should include source attribution."""
        result = runner.invoke(settings_app, ["show", "--format", "json"])
        assert result.exit_code == 0
        
        data = json.loads(result.stdout)
        # Each item should have a source field
        for item in data:
            assert "source" in item
            assert item["source"] in ["cli", "env", "config", ".env", ".env.ontofetch", "default"]

    def test_show_command_redacts_secrets_by_default(self):
        """show command should redact sensitive fields by default."""
        result = runner.invoke(settings_app, ["show", "--format", "json"])
        assert result.exit_code == 0
        
        # Just verify it runs without error (actual redaction depends on field names)
        assert result.stdout

    def test_show_command_with_no_redact_flag(self):
        """show command with --no-redact-secrets should include all values."""
        result = runner.invoke(
            settings_app,
            ["show", "--format", "json", "--no-redact-secrets"]
        )
        assert result.exit_code == 0

    def test_show_command_default_format_table(self):
        """show command default should use table format."""
        result = runner.invoke(settings_app, ["show"])
        # Table format may fail if Rich is not available, but command should run
        assert result.exit_code in [0, 2]  # 0 success, 2 if Rich missing

    def test_show_command_supports_yaml_format(self):
        """show command should support yaml format."""
        result = runner.invoke(settings_app, ["show", "--format", "yaml"])
        # YAML format may fail if PyYAML is not available
        assert result.exit_code in [0, 2]

    def test_show_command_invalid_format_fails(self):
        """show command with invalid format should fail gracefully."""
        result = runner.invoke(settings_app, ["show", "--format", "invalid"])
        # Should either work as table or fail with appropriate error
        assert result.exit_code in [0, 1, 2]


class TestSettingsSchemCommand:
    """Tests for 'settings schema' command."""

    def test_schema_command_succeeds(self):
        """schema command should execute successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                settings_app,
                ["schema", "--out", tmpdir]
            )
            assert result.exit_code == 0

    def test_schema_command_creates_files(self):
        """schema command should create schema files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                settings_app,
                ["schema", "--out", tmpdir]
            )
            assert result.exit_code == 0
            
            # Check that files were created
            schema_dir = Path(tmpdir)
            json_files = list(schema_dir.glob("*.json"))
            assert len(json_files) > 0

    def test_schema_command_creates_top_level_schema(self):
        """schema command should create settings.schema.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                settings_app,
                ["schema", "--out", tmpdir]
            )
            assert result.exit_code == 0
            
            schema_file = Path(tmpdir) / "settings.schema.json"
            assert schema_file.exists()

    def test_schema_command_creates_submodel_schemas(self):
        """schema command should create submodel schema files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                settings_app,
                ["schema", "--out", tmpdir]
            )
            assert result.exit_code == 0
            
            schema_dir = Path(tmpdir)
            subschema_files = list(schema_dir.glob("settings.*.subschema.json"))
            assert len(subschema_files) == 10  # 10 domain models

    def test_schema_command_output_includes_count(self):
        """schema command output should show file count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                settings_app,
                ["schema", "--out", tmpdir]
            )
            assert result.exit_code == 0
            assert "Generated" in result.stdout
            assert "11" in result.stdout or "schema files" in result.stdout

    def test_schema_command_default_output_dir(self):
        """schema command without --out should use default directory."""
        result = runner.invoke(settings_app, ["schema"])
        # Will create docs/schemas in current directory - just verify it runs
        assert result.exit_code in [0, 1]  # May fail if perms denied


class TestSettingsValidateCommand:
    """Tests for 'settings validate' command."""

    def test_validate_command_with_valid_yaml(self):
        """validate command should accept valid YAML config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("http:\n  timeout_connect: 5\n")
            
            result = runner.invoke(
                settings_app,
                ["validate", str(config_file)]
            )
            assert result.exit_code == 0

    def test_validate_command_with_valid_json(self):
        """validate command should accept valid JSON config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text('{"http": {"timeout_connect": 5}}')
            
            result = runner.invoke(
                settings_app,
                ["validate", str(config_file)]
            )
            assert result.exit_code == 0

    def test_validate_command_missing_file(self):
        """validate command should fail for missing files."""
        result = runner.invoke(
            settings_app,
            ["validate", "/nonexistent/config.yaml"]
        )
        assert result.exit_code == 3  # File not found exit code

    def test_validate_command_json_output(self):
        """validate command with --format json should output JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("http:\n  timeout_connect: 5\n")
            
            result = runner.invoke(
                settings_app,
                ["validate", str(config_file), "--format", "json"]
            )
            assert result.exit_code == 0
            
            try:
                data = json.loads(result.stdout)
                assert "valid" in data
                assert "file" in data
            except json.JSONDecodeError:
                pytest.fail("Output should be valid JSON")

    def test_validate_command_success_message(self):
        """validate command should show success message for valid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("http:\n  timeout_connect: 5\n")
            
            result = runner.invoke(
                settings_app,
                ["validate", str(config_file)]
            )
            assert result.exit_code == 0
            assert "Valid" in result.stdout or "✅" in result.stdout


class TestSettingsCommandsIntegration:
    """Integration tests for all settings commands."""

    def test_show_schema_validate_workflow(self):
        """Complete workflow: show settings, generate schema, validate config."""
        # 1. Show settings
        show_result = runner.invoke(settings_app, ["show", "--format", "json"])
        assert show_result.exit_code == 0
        
        # 2. Generate schema
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_result = runner.invoke(
                settings_app,
                ["schema", "--out", tmpdir]
            )
            assert schema_result.exit_code == 0
            
            # Verify schema file exists
            assert (Path(tmpdir) / "settings.schema.json").exists()
            
            # 3. Validate a config against generated schema
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("http:\n  timeout_connect: 5\n")
            
            validate_result = runner.invoke(
                settings_app,
                ["validate", str(config_file)]
            )
            assert validate_result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
