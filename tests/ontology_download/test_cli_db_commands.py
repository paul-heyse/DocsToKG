# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli_db_commands",
#   "purpose": "CLI tests for DuckDB catalog commands (Task 1.2)",
#   "sections": [
#     {"id": "setup", "name": "Test Setup", "anchor": "SETUP", "kind": "infra"},
#     {"id": "tests", "name": "CLI Command Tests", "anchor": "TESTS", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Tests for DuckDB catalog CLI commands (Task 1.2)."""

from __future__ import annotations

import sys
from pathlib import Path

# Direct import to avoid __init__.py dependencies
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from typer.testing import CliRunner

# Import directly from db_cmd module
from DocsToKG.OntologyDownload.cli.db_cmd import app


@pytest.fixture
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()


class TestAllCommands:
    """Test all CLI commands."""
    
    def test_migrate_command_exists(self, cli_runner):
        """Test migrate command."""
        result = cli_runner.invoke(app, ["migrate", "--help"])
        assert result.exit_code == 0
        assert "migrations" in result.output.lower()
    
    def test_latest_command_exists(self, cli_runner):
        """Test latest command."""
        result = cli_runner.invoke(app, ["latest", "--help"])
        assert result.exit_code == 0
    
    def test_versions_command_exists(self, cli_runner):
        """Test versions command."""
        result = cli_runner.invoke(app, ["versions", "--help"])
        assert result.exit_code == 0
    
    def test_files_command_exists(self, cli_runner):
        """Test files command."""
        result = cli_runner.invoke(app, ["files", "--help"])
        assert result.exit_code == 0
    
    def test_stats_command_exists(self, cli_runner):
        """Test stats command."""
        result = cli_runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0
    
    def test_delta_command_exists(self, cli_runner):
        """Test delta command."""
        result = cli_runner.invoke(app, ["delta", "--help"])
        assert result.exit_code == 0
    
    def test_doctor_command_exists(self, cli_runner):
        """Test doctor command."""
        result = cli_runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
    
    def test_prune_command_exists(self, cli_runner):
        """Test prune command."""
        result = cli_runner.invoke(app, ["prune", "--help"])
        assert result.exit_code == 0
    
    def test_backup_command_exists(self, cli_runner):
        """Test backup command."""
        result = cli_runner.invoke(app, ["backup", "--help"])
        assert result.exit_code == 0
    
    def test_main_help(self, cli_runner):
        """Test main help."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "DuckDB catalog" in result.output.lower()


class TestCommandBehavior:
    """Test command behavior."""
    
    def test_migrate_dry_run(self, cli_runner):
        """Test migrate dry run."""
        result = cli_runner.invoke(app, ["migrate", "--dry-run"])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
    
    def test_latest_get(self, cli_runner):
        """Test latest get."""
        result = cli_runner.invoke(app, ["latest", "get"])
        assert result.exit_code == 0
    
    def test_latest_set_requires_version(self, cli_runner):
        """Test latest set requires version."""
        result = cli_runner.invoke(app, ["latest", "set"])
        assert result.exit_code == 1
        assert "version" in result.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

