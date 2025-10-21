# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.bootstrap",
#   "purpose": "Bootstrap orchestrator that coordinates all layers (telemetry, HTTP, resolvers, pipeline)",
#   "sections": [
#     {"id": "run-from-config", "name": "run_from_config", "anchor": "function-run-from-config", "kind": "function"},
#     {"id": "bootstrap-context", "name": "BootstrapContext", "anchor": "class-bootstrapcontext", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""Bootstrap orchestrator for ContentDownload telemetry system.

**Purpose**
-----------
Coordinates the full pipeline:
1. Build telemetry sinks from config + run_id
2. Build shared HTTPX session with polite headers
3. Materialize resolvers from registry in configured order
4. For each resolver: create per-resolver HTTP client (rate limit + retry)
5. Create ResolverPipeline with client map + policies
6. Process artifact iterator through pipeline
7. Record manifest + attempt logs

**Design**
----------
This is the **single entrypoint** for running the download system. It keeps:
- CLI separate from runtime
- Bootstrap logic testable
- Resolver/client injection explicit
- Telemetry wiring centralized

**Contract**
-----------
run_from_config(cfg, artifacts=None, ...)
  - Builds all layers
  - If artifacts provided: iterates and downloads
  - If artifacts=None: validates wiring only
  - Returns RunResult with metrics
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from DocsToKG.ContentDownload.api import DownloadOutcome, ResolverResult
from DocsToKG.ContentDownload.http_session import HttpConfig, get_http_session
from DocsToKG.ContentDownload.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.resolver_http_client import (
    PerResolverHttpClient,
    RetryConfig,
)
from DocsToKG.ContentDownload.telemetry import (
    JsonlSink,
    ManifestEntry,
    MultiSink,
    RunTelemetry,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapConfig:
    """Complete configuration for bootstrap."""

    http: HttpConfig
    """HTTP session configuration (timeouts, headers, pooling)."""

    telemetry_paths: Optional[Mapping[str, Path]] = None
    """Telemetry sink paths (jsonl, csv, sqlite, etc.)."""

    run_id: Optional[str] = None
    """Unique run identifier (auto-generated if None)."""

    resolver_registry: Optional[Mapping[str, Any]] = None
    """Resolver instances (resolver_name → resolver object)."""

    resolver_retry_configs: Optional[Mapping[str, RetryConfig]] = None
    """Per-resolver retry/rate-limit configs."""

    policy_knobs: Optional[Mapping[str, Any]] = None
    """Policy configuration (robots, download, etc.)."""


@dataclass(frozen=True)
class RunResult:
    """Result of a download run."""

    run_id: str
    """Run identifier."""

    success_count: int = 0
    """Artifacts successfully downloaded."""

    skip_count: int = 0
    """Artifacts skipped (policy, duplicate, etc.)."""

    error_count: int = 0
    """Artifacts failed."""

    total_attempts: int = 0
    """Total resolver attempts across all artifacts."""


def run_from_config(
    config: BootstrapConfig,
    artifacts: Optional[Iterator[Any]] = None,
    *,
    dry_run: bool = False,
) -> RunResult:
    """
    Run the download pipeline end-to-end.

    **Usage**

        config = BootstrapConfig(
            http=HttpConfig(user_agent="MyBot", timeout_read_s=60),
            telemetry_paths={"jsonl": Path("logs/manifest.jsonl")},
            resolver_registry={"unpaywall": resolver_instance},
            policy_knobs={"robots": {"enabled": True}},
        )

        result = run_from_config(config, artifacts=artifact_iter)
        print(f"Downloaded {result.success_count} / {result.total_attempts}")

    **Parameters**

        config : BootstrapConfig
            Bootstrap configuration with HTTP, telemetry, resolvers, policies.

        artifacts : Iterator, optional
            Iterator of artifacts to download. If None, validates wiring only.

        dry_run : bool
            If True, log attempts but don't write final files.

    **Returns**

        RunResult
            Summary of the run (counts, run_id).

    **Flow**

        1. Generate run_id if not provided
        2. Initialize telemetry sinks + RunTelemetry façade
        3. Get shared HTTP session with polite headers
        4. Materialize resolvers (ensure ordered)
        5. Create per-resolver HTTP clients (rate limit + retry + telemetry)
        6. Create ResolverPipeline with client map + policies
        7. If artifacts provided: iterate through pipeline
        8. Record manifests and metrics
        9. Return RunResult with counts
    """

    # Step 1: Generate or use provided run_id
    run_id = config.run_id or f"{uuid.uuid4().hex[:8]}-{__import__('time').time():.0f}"
    LOGGER.info(f"Starting download run: {run_id}")

    # Step 2: Build telemetry sinks
    telemetry = _build_telemetry(config.telemetry_paths, run_id)

    try:
        # Step 3: Get shared HTTP session
        http_session = get_http_session(config.http)
        LOGGER.debug("Shared HTTP session acquired")

        # Step 4: Materialize resolvers in order
        resolver_registry = config.resolver_registry or {}
        if not resolver_registry:
            LOGGER.warning("No resolvers configured")

        # Step 5: Create per-resolver HTTP clients
        client_map = _build_client_map(
            http_session,
            resolver_registry,
            config.resolver_retry_configs or {},
            telemetry,
        )

        # Step 6: Create pipeline
        policy_knobs = config.policy_knobs or {}
        pipeline = ResolverPipeline(
            resolvers=list(resolver_registry.values()),
            client_map=client_map,
            telemetry=telemetry,
            run_id=run_id,
            **policy_knobs,
        )
        LOGGER.info(
            f"Pipeline ready: {len(resolver_registry)} resolvers, {len(client_map)} clients"
        )

        # Step 7: Process artifacts if provided
        result = _process_artifacts(
            pipeline=pipeline,
            artifacts=artifacts,
            telemetry=telemetry,
            run_id=run_id,
            dry_run=dry_run,
        )

        LOGGER.info(
            f"Run complete: {result.success_count} success, "
            f"{result.skip_count} skip, {result.error_count} error"
        )

        return result

    finally:
        # Cleanup telemetry
        if hasattr(telemetry, "close"):
            telemetry.close()


def _build_telemetry(paths: Optional[Mapping[str, Path]], run_id: str) -> RunTelemetry:
    """Build telemetry sinks and RunTelemetry façade."""
    sinks = []

    if paths:
        if "jsonl" in paths:
            sinks.append(JsonlSink(paths["jsonl"]))
            LOGGER.debug(f"JSONL sink: {paths['jsonl']}")

        # Add CSV, SQLite, etc. sinks as needed
        # (implementation deferred to Phase 4)

    # Create multi-sink or single sink
    if not sinks:
        LOGGER.warning("No telemetry sinks configured")
        # Create a no-op sink
        sink = _NullSink()
    elif len(sinks) == 1:
        sink = sinks[0]
    else:
        sink = MultiSink(sinks)

    return RunTelemetry(sink=sink)


def _build_client_map(
    session: Any,
    resolver_registry: Mapping[str, Any],
    retry_configs: Mapping[str, RetryConfig],
    telemetry: RunTelemetry,
) -> Mapping[str, PerResolverHttpClient]:
    """Create per-resolver HTTP clients with rate limiting."""
    client_map = {}

    for resolver_name, resolver in resolver_registry.items():
        retry_config = retry_configs.get(resolver_name, RetryConfig())

        client = PerResolverHttpClient(
            session=session,
            resolver_name=resolver_name,
            retry_config=retry_config,
            telemetry=telemetry,
        )

        client_map[resolver_name] = client
        LOGGER.debug(
            f"Client for {resolver_name}: "
            f"rate={retry_config.rate_capacity}/s, "
            f"max_retries={retry_config.max_attempts}"
        )

    return client_map


def _process_artifacts(
    pipeline: ResolverPipeline,
    artifacts: Optional[Iterator[Any]],
    telemetry: RunTelemetry,
    run_id: str,
    dry_run: bool,
) -> RunResult:
    """Process artifact iterator through pipeline."""
    success_count = 0
    skip_count = 0
    error_count = 0
    total_attempts = 0

    if not artifacts:
        LOGGER.info("No artifacts provided; validation complete")
        return RunResult(
            run_id=run_id,
            success_count=0,
            skip_count=0,
            error_count=0,
            total_attempts=0,
        )

    for artifact in artifacts:
        try:
            outcome = pipeline.run(artifact)

            # Record manifest entry
            entry = ManifestEntry(
                artifact_id=getattr(artifact, "artifact_id", None),
                url=getattr(outcome, "url", None),
                resolver=getattr(outcome, "resolver", None),
                outcome="success"
                if outcome.ok
                else ("skip" if outcome.classification == "skip" else "error"),
                ok=outcome.ok,
                reason=getattr(outcome, "reason", None),
                path=str(outcome.path) if hasattr(outcome, "path") and outcome.path else None,
                dry_run=dry_run,
            )
            telemetry.log_manifest(entry)

            # Update counts
            if outcome.ok:
                success_count += 1
            elif getattr(outcome, "classification", None) == "skip":
                skip_count += 1
            else:
                error_count += 1

            total_attempts += 1

        except Exception as e:  # pylint: disable=broad-except
            LOGGER.error(f"Error processing artifact {artifact}: {e}")
            error_count += 1
            total_attempts += 1

    return RunResult(
        run_id=run_id,
        success_count=success_count,
        skip_count=skip_count,
        error_count=error_count,
        total_attempts=total_attempts,
    )


class _NullSink:
    """No-op telemetry sink (when no sinks configured)."""

    def log_attempt(self, record: Any, **kwargs: Any) -> None:
        """No-op."""

    def log_io_attempt(self, record: Any) -> None:
        """No-op."""

    def log_manifest(self, entry: Any) -> None:
        """No-op."""

    def log_summary(self, summary: Any) -> None:
        """No-op."""

    def close(self) -> None:
        """No-op."""
