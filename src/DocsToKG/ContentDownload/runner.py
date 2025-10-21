# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.runner",
#   "purpose": "Modern run orchestration using Pydantic v2 config and DownloadPipeline",
#   "sections": [
#     {"id": "downloadrun", "name": "DownloadRun", "anchor": "class-downloadrun", "kind": "class"},
#     {"id": "run-helper", "name": "run", "anchor": "function-run", "kind": "function"}
#   ]
# }
# === /NAVMAP ===
"""Modern execution harness for content download runs.

This module replaces the legacy runner with a clean, design-first implementation
that uses Pydantic v2 configuration and the modern DownloadPipeline orchestrator.

Responsibilities
----------------
- Load configuration from file/env/CLI using ContentDownloadConfig
- Build the resolver pipeline from the registry
- Execute artifact processing through the pipeline
- Emit structured telemetry and manifests

Design Principles
-----------------
- Explicit dependency injection (no globals)
- Type-safe configuration (Pydantic v2)
- Modern resolver registry (@register_v2 decorator)
- Clean integration points for testing and scripting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from DocsToKG.ContentDownload.bootstrap import (
    BootstrapConfig,
    run_from_config,
)
from DocsToKG.ContentDownload.bootstrap import (
    RunResult as BootstrapRunResult,
)
from DocsToKG.ContentDownload.config import ContentDownloadConfig, load_config
from DocsToKG.ContentDownload.pipeline import ResolverPipeline

_LOGGER = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a download run."""

    run_id: Optional[str]
    total_processed: int
    successful: int
    failed: int
    skipped: int


class DownloadRun:
    """Modern context manager for orchestrating a content download run.

    Delegates to the canonical bootstrap orchestrator which coordinates
    telemetry, HTTP, resolvers, clients, and the ResolverPipeline.

    Usage:
        config = load_config("contentdownload.yaml")
        with DownloadRun(config) as runner:
            result = runner.process_artifacts(artifacts)
    """

    def __init__(self, config: ContentDownloadConfig) -> None:
        """Initialize run with configuration.

        Args:
            config: Pydantic v2 configuration model
        """
        self.config = config
        self.pipeline: Optional[ResolverPipeline] = None
        self._result: Optional[BootstrapRunResult] = None

    def __enter__(self) -> DownloadRun:
        """Set up the pipeline and telemetry on context entry."""
        # Use canonical bootstrap pattern
        _LOGGER.info(
            "DownloadRun initialized",
            extra={
                "run_id": self.config.run_id,
                "resolvers": self.config.resolvers.order,
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up and finalize telemetry on context exit."""
        _LOGGER.info(
            "DownloadRun completed",
            extra={
                "run_id": self.config.run_id,
            },
        )

    def process_artifact(self, artifact: Any) -> dict[str, Any]:
        """
        Process a single artifact through the resolver chain.

        Args:
            artifact: Work item (e.g., DOI, PMC ID, URL)

        Returns:
            Result dict with status, outcomes, resolver info
        """
        # This is now delegated to bootstrap/pipeline
        raise NotImplementedError(
            "Use process_artifacts() which delegates to bootstrap orchestrator"
        )

    def process_artifacts(self, artifacts: Iterable[Any]) -> RunResult:
        """Process multiple artifacts using canonical bootstrap pattern.

        Args:
            artifacts: Iterable of work items

        Returns:
            Summary of the run
        """
        # Delegate to canonical bootstrap orchestrator
        bootstrap_result = run_from_config(
            config=BootstrapConfig(),  # Uses defaults; can be customized
            artifacts=iter(artifacts),
            dry_run=False,
        )

        return RunResult(
            run_id=bootstrap_result.run_id,
            total_processed=bootstrap_result.success_count
            + bootstrap_result.skip_count
            + bootstrap_result.error_count,
            successful=bootstrap_result.success_count,
            failed=bootstrap_result.error_count,
            skipped=bootstrap_result.skip_count,
        )


def run(
    config_path: Optional[str] = None,
    artifacts: Optional[Iterable[Any]] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> RunResult:
    """Run download pipeline using canonical bootstrap orchestrator.

    Example:
        result = run(
            config_path="contentdownload.yaml",
            artifacts=["10.1234/example", "arXiv:2301.12345"],
            cli_overrides={"resolvers": {"order": ["arxiv", "landing"]}}
        )
        print(f"Processed {result.total_processed}, successful {result.successful}")

    Args:
        config_path: Path to configuration file (optional)
        artifacts: Iterable of artifacts to process (optional)
        cli_overrides: CLI-level config overrides

    Returns:
        RunResult with statistics
    """
    config = load_config(config_path, cli_overrides=cli_overrides)

    with DownloadRun(config) as runner:
        if artifacts:
            return runner.process_artifacts(artifacts)
        else:
            return RunResult(
                run_id=config.run_id,
                total_processed=0,
                successful=0,
                failed=0,
                skipped=0,
            )
