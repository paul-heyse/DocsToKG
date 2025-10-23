"""Regression tests for CLI re-export expectations."""

from __future__ import annotations
import sys
from pathlib import Path

from typer import Typer
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runner = CliRunner()


def test_contentdownload_cli_help() -> None:
    """Import the legacy CLI module and ensure the Typer app loads."""

    from DocsToKG.ContentDownload import cli

    assert isinstance(cli.app, Typer)
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "DocsToKG ContentDownload" in result.output
    assert "queue" in result.output


def test_cli_orchestrator_exports_and_help() -> None:
    """Ensure orchestrator CLI re-exports config and queue helpers."""

    from DocsToKG.ContentDownload import cli_orchestrator

    assert hasattr(cli_orchestrator, "OrchestratorConfig")
    assert hasattr(cli_orchestrator, "WorkQueue")

    result = runner.invoke(cli_orchestrator.app, ["queue-stats", "--help"])
    assert result.exit_code == 0
    assert "Display work queue statistics." in result.output


def test_bootstrap_download_and_resolver_reexports() -> None:
    """Verify bootstrap, download, and resolver modules expose legacy symbols."""

    from DocsToKG.ContentDownload.bootstrap import OrchestratorConfig as BootstrapOrchestratorConfig
    from DocsToKG.ContentDownload.download import RateLimitError as DownloadRateLimitError
    from DocsToKG.ContentDownload.errors import RateLimitError as CanonicalRateLimitError
    from DocsToKG.ContentDownload import resolvers

    assert "config.models" in BootstrapOrchestratorConfig.__module__
    assert DownloadRateLimitError is CanonicalRateLimitError
    assert callable(resolvers.build_resolvers)
    assert callable(resolvers.get_registry)
    assert hasattr(resolvers, "register_v2")
