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

from DocsToKG.ContentDownload.bootstrap import (
    BootstrapConfig,
    run_from_config,
)
from DocsToKG.ContentDownload.bootstrap import (
    RunResult as BootstrapRunResult,
)
from DocsToKG.ContentDownload.config import ContentDownloadConfig, load_config
from DocsToKG.ContentDownload.http_session import HttpConfig
from DocsToKG.ContentDownload.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.resolver_http_client import RetryConfig
from DocsToKG.ContentDownload.resolvers.registry_v2 import build_resolvers

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
        self._bootstrap_config: Optional[BootstrapConfig] = None
        self._bootstrap_config_signature: Optional[str] = None

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
        self._refresh_bootstrap_config()
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
        self._refresh_bootstrap_config()

        bootstrap_result = run_from_config(
            config=self._bootstrap_config,
            artifacts=iter(artifacts),
            dry_run=False,
        )
        self._result = bootstrap_result

        return RunResult(
            run_id=bootstrap_result.run_id,
            total_processed=bootstrap_result.success_count
            + bootstrap_result.skip_count
            + bootstrap_result.error_count,
            successful=bootstrap_result.success_count,
            failed=bootstrap_result.error_count,
            skipped=bootstrap_result.skip_count,
        )

    def _refresh_bootstrap_config(self) -> None:
        """Ensure cached bootstrap config matches the current run configuration."""

        signature = self._compute_config_signature()

        if (
            self._bootstrap_config is None
            or self._bootstrap_config_signature != signature
        ):
            self._bootstrap_config = self._build_bootstrap_config()
            self._bootstrap_config_signature = signature

    def _build_bootstrap_config(self) -> BootstrapConfig:
        """Translate ContentDownloadConfig into BootstrapConfig."""

        http_cfg = self.config.http
        http = HttpConfig(
            user_agent=http_cfg.user_agent,
            mailto=http_cfg.mailto,
            timeout_connect_s=http_cfg.timeout_connect_s,
            timeout_read_s=http_cfg.timeout_read_s,
            pool_connections=http_cfg.max_keepalive_connections,
            pool_maxsize=http_cfg.max_connections,
            verify_tls=http_cfg.verify_tls,
            proxies=http_cfg.proxies or None,
        )

        telemetry_paths = self._build_telemetry_paths()
        resolver_registry, retry_configs = self._build_resolvers()
        policy_knobs = self._build_policy_knobs()

        return BootstrapConfig(
            http=http,
            telemetry_paths=telemetry_paths,
            resolver_registry=resolver_registry,
            resolver_retry_configs=retry_configs,
            policy_knobs=policy_knobs,
            run_id=self.config.run_id,
        )

    def _compute_config_signature(self) -> str:
        """Return a deterministic signature for the current configuration."""

        config = self.config

        if hasattr(config, "config_hash") and callable(config.config_hash):
            return config.config_hash()

        return repr(config)

    def _build_telemetry_paths(self) -> dict[str, Path]:
        telemetry_cfg = self.config.telemetry
        telemetry_paths: dict[str, Path] = {}

        if "csv" in telemetry_cfg.sinks:
            csv_path = Path(telemetry_cfg.csv_path)
            telemetry_paths["csv"] = csv_path
            telemetry_paths["last_attempt"] = csv_path.with_name("last.csv")

        if "jsonl" in telemetry_cfg.sinks:
            manifest_path = Path(telemetry_cfg.manifest_path)
            telemetry_paths["manifest_index"] = manifest_path.with_name("index.json")
            telemetry_paths["summary"] = manifest_path.with_name("summary.json")
            telemetry_paths["sqlite"] = manifest_path.with_suffix(".sqlite")

        return telemetry_paths

    def _build_resolvers(self) -> tuple[dict[str, Any], dict[str, RetryConfig]]:
        resolver_registry: dict[str, Any] = {}
        retry_configs: dict[str, RetryConfig] = {}

        for resolver in build_resolvers(self.config):
            resolver_name = getattr(
                resolver,
                "_registry_name",
                getattr(resolver, "name", resolver.__class__.__name__.lower()),
            )
            resolver_registry[resolver_name] = resolver

            resolver_cfg = getattr(self.config.resolvers, resolver_name, None)
            if resolver_cfg is None:
                continue

            retry_policy = resolver_cfg.retry
            rate_policy = resolver_cfg.rate_limit

            retry_configs[resolver_name] = RetryConfig(
                max_attempts=retry_policy.max_attempts,
                retry_statuses=tuple(retry_policy.retry_statuses),
                base_delay_ms=retry_policy.base_delay_ms,
                max_delay_ms=retry_policy.max_delay_ms,
                jitter_ms=retry_policy.jitter_ms,
                rate_capacity=rate_policy.capacity,
                rate_refill_per_sec=rate_policy.refill_per_sec,
                rate_burst=rate_policy.burst,
                timeout_read_s=resolver_cfg.timeout_read_s,
            )

        return resolver_registry, retry_configs

    def _build_policy_knobs(self) -> dict[str, Any]:
        download_cfg = self.config.download
        http_cfg = self.config.http

        policy_knobs: dict[str, Any] = {}

        if download_cfg.max_bytes is not None:
            policy_knobs["max_bytes"] = download_cfg.max_bytes

        policy_knobs["chunk_size_bytes"] = download_cfg.chunk_size_bytes
        policy_knobs["atomic_write"] = download_cfg.atomic_write
        policy_knobs["verify_content_length"] = download_cfg.verify_content_length
        policy_knobs["timeout_s"] = http_cfg.timeout_read_s

        return policy_knobs


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
