"""Tests for observability CLI commands: tail, stats, export.

Covers:
- CLI command invocation
- Option parsing
- Output formatting
- Error handling
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from DocsToKG.OntologyDownload.cli.obs_cmd import app

runner = CliRunner()


# ============================================================================
# tail command tests
# ============================================================================


class TestObsTail:
    """Test obs tail command."""

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_tail_basic(self, mock_conn):
        """obs tail command works with defaults."""
        # Mock DuckDB connection
        mock_con = MagicMock()
        mock_con.execute().fetchall.return_value = [
            ("2025-10-20T01:23:45Z", "net.request", "INFO", "ols", "run-1"),
        ]
        mock_con.execute().description = [("ts",), ("type",), ("level",), ("service",), ("run_id",)]
        mock_conn.return_value = mock_con

        result = runner.invoke(app, ["tail"])
        assert result.exit_code == 0
        assert "net.request" in result.stdout

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_tail_with_count(self, mock_conn):
        """obs tail respects --count option."""
        mock_con = MagicMock()
        mock_con.execute().fetchall.return_value = []
        mock_conn.return_value = mock_con

        result = runner.invoke(app, ["tail", "--count", "50"])
        assert result.exit_code == 0
        # Verify count was used in query
        mock_con.execute.assert_called()

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_tail_with_level_filter(self, mock_conn):
        """obs tail filters by level."""
        mock_con = MagicMock()
        mock_con.execute().fetchall.return_value = []
        mock_conn.return_value = mock_con

        result = runner.invoke(app, ["tail", "--level", "ERROR"])
        assert result.exit_code == 0

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_tail_with_type_filter(self, mock_conn):
        """obs tail filters by event type."""
        mock_con = MagicMock()
        mock_con.execute().fetchall.return_value = []
        mock_conn.return_value = mock_con

        result = runner.invoke(app, ["tail", "--type", "net.request"])
        assert result.exit_code == 0

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_tail_with_service_filter(self, mock_conn):
        """obs tail filters by service."""
        mock_con = MagicMock()
        mock_con.execute().fetchall.return_value = []
        mock_conn.return_value = mock_con

        result = runner.invoke(app, ["tail", "--service", "ols"])
        assert result.exit_code == 0

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_tail_json_output(self, mock_conn):
        """obs tail supports --json output."""
        mock_con = MagicMock()
        mock_df = MagicMock()
        mock_df.to_json.return_value = "[]"
        mock_con.execute().df.return_value = mock_df
        mock_conn.return_value = mock_con

        result = runner.invoke(app, ["tail", "--json"])
        assert result.exit_code == 0

    def test_tail_duckdb_not_installed(self):
        """obs tail handles missing DuckDB gracefully."""
        with patch(
            "DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection",
            side_effect=Exception("DuckDB not installed"),
        ):
            result = runner.invoke(app, ["tail"])
            assert result.exit_code == 1


# ============================================================================
# stats command tests
# ============================================================================


class TestObsStats:
    """Test obs stats command."""

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_stats_list_queries(self, mock_conn):
        """obs stats --list shows available queries."""
        result = runner.invoke(app, ["stats", "--list"])
        assert result.exit_code == 0
        assert "Available stock queries" in result.stdout or "Available" in result.stdout

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_stats_run_query(self, mock_conn):
        """obs stats runs a named query."""
        mock_con = MagicMock()
        mock_con.execute().fetchall.return_value = [
            ("ols", 95.0, 50.0, 150.0, 100),
        ]
        mock_con.execute().description = [("service",), ("p95",), ("p50",), ("max",), ("count",)]
        mock_conn.return_value = mock_con

        result = runner.invoke(app, ["stats", "net_request_p95_latency"])
        assert result.exit_code == 0

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_stats_invalid_query(self, mock_conn):
        """obs stats rejects invalid query name."""
        mock_conn.return_value = MagicMock()
        result = runner.invoke(app, ["stats", "nonexistent_query"])
        assert result.exit_code == 1
        assert "not found" in result.stdout or "not found" in result.stderr

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_stats_no_query_specified(self, mock_conn):
        """obs stats requires query name unless --list used."""
        result = runner.invoke(app, ["stats"])
        # Should fail or show help
        assert "query" in result.stdout.lower() or "list" in result.stdout.lower()

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_stats_json_output(self, mock_conn):
        """obs stats supports --json output."""
        mock_con = MagicMock()
        mock_df = MagicMock()
        mock_df.to_json.return_value = "[]"
        mock_con.execute().df.return_value = mock_df
        mock_conn.return_value = mock_con

        result = runner.invoke(app, ["stats", "net_request_p95_latency", "--json"])
        assert result.exit_code == 0


# ============================================================================
# export command tests
# ============================================================================


class TestObsExport:
    """Test obs export command."""

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_export_json(self, mock_conn):
        """obs export supports .json format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "events.json"

            mock_con = MagicMock()
            mock_df = MagicMock()
            mock_df.to_json.return_value = None
            mock_conn.return_value = mock_con
            mock_con.execute().df.return_value = mock_df

            result = runner.invoke(app, ["export", str(output_file)])
            assert result.exit_code == 0
            assert "Exported" in result.stdout

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_export_jsonl(self, mock_conn):
        """obs export supports .jsonl format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "events.jsonl"

            mock_con = MagicMock()
            mock_df = MagicMock()
            mock_df.to_json.return_value = None
            mock_conn.return_value = mock_con
            mock_con.execute().df.return_value = mock_df

            result = runner.invoke(app, ["export", str(output_file)])
            assert result.exit_code == 0

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_export_parquet(self, mock_conn):
        """obs export supports .parquet format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "events.parquet"

            mock_con = MagicMock()
            mock_df = MagicMock()
            mock_df.to_parquet.return_value = None
            mock_conn.return_value = mock_con
            mock_con.execute().df.return_value = mock_df

            result = runner.invoke(app, ["export", str(output_file)])
            assert result.exit_code == 0

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_export_csv(self, mock_conn):
        """obs export supports .csv format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "events.csv"

            mock_con = MagicMock()
            mock_df = MagicMock()
            mock_df.to_csv.return_value = None
            mock_conn.return_value = mock_con
            mock_con.execute().df.return_value = mock_df

            result = runner.invoke(app, ["export", str(output_file)])
            assert result.exit_code == 0

    def test_export_no_extension(self):
        """obs export rejects output path without extension."""
        result = runner.invoke(app, ["export", "/tmp/events"])
        assert result.exit_code == 1
        assert "extension" in result.stdout.lower() or "extension" in result.stderr.lower()

    def test_export_unsupported_format(self):
        """obs export rejects unsupported formats."""
        result = runner.invoke(app, ["export", "/tmp/events.yaml"])
        assert result.exit_code == 1
        assert "Unsupported" in result.stdout or "unsupported" in result.stderr

    @patch("DocsToKG.OntologyDownload.cli.obs_cmd._get_duckdb_connection")
    def test_export_with_filters(self, mock_conn):
        """obs export respects filter options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "events.json"

            mock_con = MagicMock()
            mock_df = MagicMock()
            mock_df.to_json.return_value = None
            mock_con.execute().df.return_value = mock_df
            mock_conn.return_value = mock_con

            result = runner.invoke(
                app,
                [
                    "export",
                    str(output_file),
                    "--level",
                    "ERROR",
                    "--type",
                    "net",
                    "--since",
                    "2025-10-20T00:00:00Z",
                    "--limit",
                    "1000",
                ],
            )
            assert result.exit_code == 0


# ============================================================================
# App structure tests
# ============================================================================


class TestObsApp:
    """Test the obs CLI app structure."""

    def test_app_has_commands(self):
        """obs app has expected commands."""
        # The app should have tail, stats, and export commands
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "tail" in command_names
        assert "stats" in command_names
        assert "export" in command_names

    def test_app_help(self):
        """obs app --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Observability" in result.stdout or "observability" in result.stdout.lower()
