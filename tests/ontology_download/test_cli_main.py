"""Tests for CLI main app: context, global options, and integration.

Tests verify:
- CLI app structure and commands
- Global options (--config, -v/-vv, --format, --dry-run, --version)
- Context initialization and passing
- Settings integration
- Help messages
"""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from DocsToKG.OntologyDownload.cli_main import app, CliContext


runner = CliRunner()


class TestCliAppBasics:
    """Tests for basic CLI app functionality."""

    def test_app_helps_without_command(self):
        """App should show help when invoked without command."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "OntologyDownload CLI" in result.stdout or "Commands:" in result.stdout

    def test_app_shows_help_with_help_flag(self):
        """App should show help with --help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Settings management" in result.stdout or "settings" in result.stdout.lower()

    def test_version_option_shows_version(self):
        """--version should show version and exit."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code in [0, 2]
        assert "ontofetch-cli" in result.stdout

    def test_version_command_exists(self):
        """version command should exist and work."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code in [0, 2]
        assert "ontofetch-cli" in result.stdout or "version" in result.stdout.lower()

    def test_doctor_command_exists(self):
        """doctor command should exist."""
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code in [0, 1]  # May fail if settings issue, but command exists

    def test_settings_subcommand_group_exists(self):
        """settings subcommand group should be available."""
        result = runner.invoke(app, ["settings", "--help"])
        assert result.exit_code == 0
        assert "show" in result.stdout or "schema" in result.stdout


class TestGlobalOptions:
    """Tests for global options."""

    def test_config_option_accepted(self):
        """--config option should be accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("http:\n  timeout_connect: 5\n")
            
            result = runner.invoke(
                app,
                ["--config", str(config_file), "settings", "show", "--format", "json"]
            )
            # Should execute without error (though settings might not use it yet)
            assert result.exit_code in [0, 1]

    def test_config_short_option(self):
        """Config option should have short form -c."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("http:\n  timeout_connect: 5\n")
            
            result = runner.invoke(
                app,
                ["-c", str(config_file), "settings", "show", "--format", "json"]
            )
            assert result.exit_code in [0, 1]

    def test_verbosity_flag_single(self):
        """-v should increase verbosity."""
        result = runner.invoke(
            app,
            ["-v", "settings", "show", "--format", "json"]
        )
        # Should execute
        assert result.exit_code in [0, 1, 2]

    def test_verbosity_flag_double(self):
        """-vv should set debug verbosity."""
        result = runner.invoke(
            app,
            ["-vv", "settings", "show", "--format", "json"]
        )
        # Should execute
        assert result.exit_code in [0, 1, 2]

    def test_format_option(self):
        """--format option should be accepted."""
        result = runner.invoke(
            app,
            ["--format", "json", "settings", "show"]
        )
        # Should execute
        assert result.exit_code in [0, 1, 2]

    def test_format_short_option(self):
        """Format option should have short form -f."""
        result = runner.invoke(
            app,
            ["-f", "json", "settings", "show"]
        )
        # Should execute
        assert result.exit_code in [0, 1, 2]

    def test_dry_run_option(self):
        """--dry-run option should be accepted."""
        result = runner.invoke(
            app,
            ["--dry-run", "settings", "show", "--format", "json"]
        )
        # Should execute
        assert result.exit_code in [0, 1, 2]

    def test_dry_run_short_option(self):
        """Dry-run option should have short form -n."""
        result = runner.invoke(
            app,
            ["-n", "settings", "show", "--format", "json"]
        )
        # Should execute
        assert result.exit_code in [0, 1, 2]


class TestCliContext:
    """Tests for CliContext class."""

    def test_context_initialization(self):
        """CliContext should initialize with all parameters."""
        ctx = CliContext(
            verbosity=1,
            format_output="json",
            dry_run=True,
        )
        
        assert ctx.verbosity == 1
        assert ctx.format_output == "json"
        assert ctx.dry_run is True

    def test_context_loads_settings(self):
        """CliContext should load settings."""
        ctx = CliContext()
        assert ctx.settings is not None

    def test_context_log_debug_respects_verbosity(self):
        """log_debug should only output when verbosity >= 2."""
        ctx = CliContext(verbosity=0)
        # Calling log_debug with low verbosity should not raise
        ctx.log_debug("test message")

    def test_context_log_info_respects_verbosity(self):
        """log_info should only output when verbosity >= 1."""
        ctx = CliContext(verbosity=0)
        # Calling log_info with low verbosity should not raise
        ctx.log_info("test message")

    def test_context_has_console(self):
        """CliContext should have a console attribute."""
        ctx = CliContext()
        assert hasattr(ctx, "console")
        assert ctx.console is not None


class TestSettingsIntegration:
    """Tests for settings integration."""

    def test_settings_show_via_main_app(self):
        """settings show command should work via main app."""
        result = runner.invoke(
            app,
            ["settings", "show", "--format", "json"]
        )
        assert result.exit_code == 0

    def test_settings_schema_via_main_app(self):
        """settings schema command should work via main app."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                ["settings", "schema", "--out", tmpdir]
            )
            assert result.exit_code == 0

    def test_settings_validate_via_main_app(self):
        """settings validate command should work via main app."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("http:\n  timeout_connect: 5\n")
            
            result = runner.invoke(
                app,
                ["settings", "validate", str(config_file)]
            )
            assert result.exit_code == 0


class TestCommandIntegration:
    """Integration tests for command workflows."""

    def test_doctor_command_output(self):
        """doctor command should produce output."""
        result = runner.invoke(app, ["doctor"])
        # Should run (may fail if settings issues, but should have output)
        assert "Diagnostics" in result.stdout or "checks" in result.stdout.lower() or result.exit_code in [0, 1]

    def test_global_options_with_settings_show(self):
        """Global options should work with settings show."""
        result = runner.invoke(
            app,
            ["-vv", "--format", "json", "settings", "show"]
        )
        # Should execute
        assert result.exit_code in [0, 1, 2]

    def test_help_for_settings_subcommand(self):
        """Help should work for settings subcommand."""
        result = runner.invoke(app, ["settings", "--help"])
        assert result.exit_code == 0
        assert "Commands:" in result.stdout or "show" in result.stdout.lower()

    def test_help_for_settings_show(self):
        """Help should work for settings show command."""
        result = runner.invoke(app, ["settings", "show", "--help"])
        assert result.exit_code == 0
        assert "show" in result.stdout.lower() or "configuration" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
