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
from pathlib import Path
from typing import Any, Iterable, Optional

from DocsToKG.ContentDownload.config import ContentDownloadConfig, load_config
from DocsToKG.ContentDownload.download_pipeline import DownloadPipeline, build_pipeline

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
        self.pipeline: Optional[DownloadPipeline] = None
        self._stats = {"processed": 0, "successful": 0, "failed": 0, "skipped": 0}

    def __enter__(self) -> DownloadRun:
        """Set up the pipeline and telemetry on context entry."""
        self.pipeline = build_pipeline(self.config)
        _LOGGER.info(
            f"DownloadRun initialized",
            extra={
                "run_id": self.config.run_id,
                "resolvers": self.config.resolvers.order,
                "resolver_count": len(self.pipeline.resolvers),
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up and finalize telemetry on context exit."""
        if self.pipeline:
            _LOGGER.info(
                "DownloadRun completed",
                extra={
                    "run_id": self.config.run_id,
                    "stats": self._stats,
                },
            )

    def process_artifact(self, artifact: Any) -> dict[str, Any]:
        """Process a single artifact through the resolver pipeline.
        
        Args:
            artifact: Work item (e.g., DOI, PMC ID, URL)
            
        Returns:
            Result dict with status, outcomes, resolver info
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized (use as context manager)")

        self._stats["processed"] += 1
        outcome = self.pipeline.process_artifact(artifact)

        if outcome.get("status") == "resolved":
            self._stats["successful"] += 1
        elif outcome.get("status") == "unresolved":
            self._stats["skipped"] += 1
        else:
            self._stats["failed"] += 1

        return outcome

    def process_artifacts(self, artifacts: Iterable[Any]) -> RunResult:
        """Process multiple artifacts.
        
        Args:
            artifacts: Iterable of work items
            
        Returns:
            Summary of the run
        """
        for artifact in artifacts:
            self.process_artifact(artifact)

        return RunResult(
            run_id=self.config.run_id,
            total_processed=self._stats["processed"],
            successful=self._stats["successful"],
            failed=self._stats["failed"],
            skipped=self._stats["skipped"],
        )


def run(
    config_path: Optional[str] = None,
    artifacts: Optional[Iterable[Any]] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> RunResult:
    """Convenience helper for running a download job.
    
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
