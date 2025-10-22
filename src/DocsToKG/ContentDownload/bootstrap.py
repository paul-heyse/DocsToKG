# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.bootstrap",
#   "purpose": "Bootstrap orchestrator that coordinates all layers (telemetry, HTTP, resolvers, pipeline)",
#   "sections": [
#     {"id": "run-from-config", "name": "run_from_config", "anchor": "#function-run-from-config", "kind": "function"},
#     {"id": "bootstrap-config", "name": "BootstrapConfig", "anchor": "#class-bootstrapconfig", "kind": "class"},
#     {"id": "run-result", "name": "RunResult", "anchor": "#class-runresult", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""Bootstrap orchestrator for ContentDownload telemetry system.

**Purpose**
-----------
Coordinates the full pipeline:
1. Build telemetry sinks from config + run_id
2. Build shared HTTPX session with polite headers
3. Materialize resolvers in configured order
4. Create per-resolver HTTP clients with independent policies
5. Create ResolverPipeline with client_map
6. Process artifacts through pipeline
7. Record manifests and metrics

**Design**
----------
- Wires all layers together (telemetry, HTTP, resolvers, clients, pipeline)
- Per-resolver clients have independent rate limits and retry policies
- Fallback to shared session if client not in client_map
- Artifact iteration with manifest recording
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional
from uuid import uuid4

from DocsToKG.ContentDownload.http_session import HttpConfig, get_http_session
from DocsToKG.ContentDownload.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.resolver_http_client import (
    PerResolverHttpClient,
    RetryConfig,
)
from DocsToKG.ContentDownload.telemetry import (
    AttemptSink,
    CsvSink,
    LastAttemptCsvSink,
    ManifestIndexSink,
    MultiSink,
    RunTelemetry,
    SqliteSink,
    SummarySink,
)

# Feature flags support
try:
    from DocsToKG.ContentDownload.config.feature_flags import (
        get_feature_flags,
        FeatureFlag,
    )

    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False

# New bootstrap helpers (when feature enabled)
if FEATURE_FLAGS_AVAILABLE:
    try:
        from DocsToKG.ContentDownload.config.bootstrap import (
            build_http_client,
            build_telemetry_sinks,
        )

        BOOTSTRAP_HELPERS_AVAILABLE = True
    except ImportError:
        BOOTSTRAP_HELPERS_AVAILABLE = False
else:
    BOOTSTRAP_HELPERS_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def _should_use_new_bootstrap() -> bool:
    """Check if new bootstrap helpers should be used.

    Checks the DTKG_FEATURE_UNIFIED_BOOTSTRAP environment variable.
    Defaults to False for backward compatibility.

    Returns:
        True if unified bootstrap feature is enabled
    """
    if not FEATURE_FLAGS_AVAILABLE or not BOOTSTRAP_HELPERS_AVAILABLE:
        return False

    try:
        flags = get_feature_flags()
        return flags.is_enabled(FeatureFlag.UNIFIED_BOOTSTRAP)
    except Exception as e:
        LOGGER.debug(f"Could not check bootstrap feature flag: {e}")
        return False


@dataclass
class BootstrapConfig:
    """Complete bootstrap configuration."""

    http: HttpConfig = field(default_factory=HttpConfig)
    telemetry_paths: Optional[Mapping[str, Path]] = None
    resolver_registry: Optional[dict[str, Any]] = None
    resolver_retry_configs: Optional[dict[str, RetryConfig]] = None
    policy_knobs: Optional[dict[str, Any]] = None
    run_id: Optional[str] = None


@dataclass
class RunResult:
    """Result from running the pipeline."""

    run_id: str
    success_count: int = 0
    skip_count: int = 0
    error_count: int = 0


def run_from_config(
    config: BootstrapConfig,
    artifacts: Optional[Iterator[Any]] = None,
    dry_run: bool = False,
) -> RunResult:
    """
    Run bootstrap orchestrator.

    Coordinates all layers:
    1. Telemetry sinks
    2. HTTP session (shared, pooled)
    3. Resolvers
    4. Per-resolver HTTP clients
    5. Pipeline execution

    Args:
        config: BootstrapConfig with all settings
        artifacts: Optional iterator of artifacts to download
        dry_run: If True, don't actually download (validation only)

    Returns:
        RunResult with counts and run_id
    """
    # Step 1: Generate or validate run_id
    run_id = config.run_id or str(uuid4())
    LOGGER.info(f"Starting run {run_id}")

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
            session=http_session,
            telemetry=telemetry,
            run_id=run_id,
            client_map=client_map,
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
    """Build telemetry sinks from configuration.

    Creates sinks based on provided telemetry_paths dictionary:
    - CSV sink (key: 'csv')
    - SQLite sink (key: 'sqlite')
    - Manifest index sink (key: 'manifest_index')
    - Last attempt CSV sink (key: 'last_attempt')
    - Summary sink (key: 'summary')

    At least one path must be provided. Calling without paths will raise an error
    to ensure telemetry configuration is explicit.

    Args:
        paths: Mapping of sink names to file paths (required)
        run_id: Unique run identifier for this execution

    Returns:
        RunTelemetry instance with configured sinks

    Raises:
        ValueError: If no telemetry paths are provided
    """
    # Check if unified bootstrap should be used
    if _should_use_new_bootstrap() and BOOTSTRAP_HELPERS_AVAILABLE:
        try:
            # When unified bootstrap is enabled, use Pydantic v2 config if available
            LOGGER.debug("Using unified bootstrap for telemetry (feature enabled)")
            # Try to import and use Pydantic v2 config
            try:
                from DocsToKG.ContentDownload.config.models import (
                    ContentDownloadConfig,
                )
                from DocsToKG.ContentDownload.config.loader import load_config

                # Load config and use new telemetry builder
                cfg = load_config()
                telemetry = build_telemetry_sinks(cfg.telemetry, run_id)
                return telemetry
            except (ImportError, Exception) as e:
                LOGGER.debug(f"Pydantic v2 config not available, falling back to legacy: {e}")
                # Fall through to legacy code below
        except Exception as e:
            LOGGER.debug(f"Unified bootstrap failed, falling back to legacy: {e}")

    # Legacy telemetry building code (always works)
    if not paths:
        raise ValueError(
            "telemetry_paths must be provided to _build_telemetry(). "
            "Provide at least one of: 'csv', 'sqlite', 'manifest_index', "
            "'last_attempt', 'summary'"
        )

    sinks: list[AttemptSink] = []

    # Create sinks for each provided path
    if "csv" in paths:
        sinks.append(CsvSink(paths["csv"]))
    if "sqlite" in paths:
        sinks.append(SqliteSink(paths["sqlite"]))
    if "manifest_index" in paths:
        sinks.append(ManifestIndexSink(paths["manifest_index"]))
    if "last_attempt" in paths:
        sinks.append(LastAttemptCsvSink(paths["last_attempt"]))
    if "summary" in paths:
        sinks.append(SummarySink(paths["summary"]))

    if not sinks:
        raise ValueError(
            "No recognized telemetry paths provided. "
            "Expected one of: 'csv', 'sqlite', 'manifest_index', 'last_attempt', 'summary'"
        )

    # Composite sink handles all outputs
    multi_sink = MultiSink(sinks) if len(sinks) > 1 else sinks[0]

    return RunTelemetry(sink=multi_sink)


def _build_client_map(
    http_session: Any,
    resolver_registry: dict[str, Any],
    retry_configs: dict[str, RetryConfig],
    telemetry: Any,
) -> dict[str, PerResolverHttpClient]:
    """Create per-resolver HTTP clients with independent policies."""
    client_map = {}

    for resolver_name, resolver in resolver_registry.items():
        # Get retry config for this resolver (or use default)
        retry_config = retry_configs.get(resolver_name) or RetryConfig(
            rate_capacity=5.0,
            rate_refill_per_sec=1.0,
            max_attempts=4,
        )

        # Create per-resolver client
        client = PerResolverHttpClient(
            session=http_session,
            resolver_name=resolver_name,
            retry_config=retry_config,
        )

        client_map[resolver_name] = client
        LOGGER.debug(
            f"Created client for resolver '{resolver_name}' "
            f"(capacity={retry_config.rate_capacity}, refill={retry_config.rate_refill_per_sec}/s)"
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
    result = RunResult(run_id=run_id)

    if not artifacts:
        LOGGER.info("No artifacts provided; validation complete")
        return result

    for artifact in artifacts:
        if dry_run:
            LOGGER.info(f"[DRY-RUN] Would process artifact {getattr(artifact, 'id', '?')}")
            continue

        # Run pipeline for this artifact
        outcome = pipeline.run(artifact, ctx=None)

        # Record metrics
        if outcome.ok:
            result.success_count += 1
        elif outcome.classification == "skip":
            result.skip_count += 1
        else:
            result.error_count += 1

    return result


__all__ = [
    "BootstrapConfig",
    "RunResult",
    "run_from_config",
]
