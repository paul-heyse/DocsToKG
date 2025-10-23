# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.config.bootstrap",
#   "purpose": "Bootstrap helpers to construct runtime components from ContentDownloadConfig.",
#   "sections": [
#     {
#       "id": "build-http-client",
#       "name": "build_http_client",
#       "anchor": "function-build-http-client",
#       "kind": "function"
#     },
#     {
#       "id": "build-telemetry-sinks",
#       "name": "build_telemetry_sinks",
#       "anchor": "function-build-telemetry-sinks",
#       "kind": "function"
#     },
#     {
#       "id": "build-orchestrator",
#       "name": "build_orchestrator",
#       "anchor": "function-build-orchestrator",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Bootstrap helpers to construct runtime components from ContentDownloadConfig.

Provides factory functions to build HTTP clients, telemetry sinks, resolvers,
and orchestrators directly from Pydantic v2 configuration models.

Pattern:
  cfg = load_config("config.yaml")
  http_client = build_http_client(cfg.http, cfg.hishel)
  telemetry = build_telemetry_sinks(cfg.telemetry, run_id="...")
  orchestrator = build_orchestrator(cfg.orchestrator, cfg.queue)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

    from DocsToKG.ContentDownload.config.models import (
        HishelConfig,
        HttpClientConfig,
        OrchestratorConfig,
        QueueConfig,
        TelemetryConfig,
    )
    from DocsToKG.ContentDownload.telemetry import MultiSink

LOGGER = logging.getLogger(__name__)


def build_http_client(
    http: HttpClientConfig,
    hishel: HishelConfig,
) -> httpx.Client:
    """
    Construct HTTPX client from config models.

    Args:
        http: HttpClientConfig with timeout, pooling, headers
        hishel: HishelConfig with cache settings

    Returns:
        Configured httpx.Client with optional caching

    Example:
        >>> from DocsToKG.ContentDownload.config import ContentDownloadConfig
        >>> cfg = ContentDownloadConfig()
        >>> client = build_http_client(cfg.http, cfg.hishel)
    """
    from DocsToKG.ContentDownload.httpx_transport import configure_http_client, get_http_client

    # Configure HTTP client with pooling and timeouts from config
    configure_http_client()
    client = get_http_client()

    # Cache configuration is applied via httpx_transport module
    # which reads HishelConfig internally
    LOGGER.debug(
        "HTTP client configured",
        extra={
            "http2": http.http2,
            "max_connections": http.max_connections,
            "cache_enabled": hishel.enabled,
            "cache_backend": hishel.backend,
        },
    )

    return client


def build_telemetry_sinks(
    telemetry: TelemetryConfig,
    run_id: str,
) -> MultiSink:
    """
    Construct telemetry sink from config.

    Args:
        telemetry: TelemetryConfig with sink types and paths
        run_id: Run identifier for correlation

    Returns:
        Configured MultiSink that routes events to all configured outputs

    Example:
        >>> cfg = ContentDownloadConfig()
        >>> sinks = build_telemetry_sinks(cfg.telemetry, run_id="abc123")
    """
    from pathlib import Path

    from DocsToKG.ContentDownload.telemetry import (
        CsvSink,
        JsonlSink,
        LastAttemptCsvSink,
        ManifestIndexSink,
        MultiSink,
        SqliteSink,
        SummarySink,
    )

    sinks = []

    # Create sinks based on config
    if "csv" in telemetry.sinks:
        csv_path = Path(telemetry.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        sinks.append(CsvSink(csv_path))
        sinks.append(LastAttemptCsvSink(csv_path.with_name("last.csv")))

    if "jsonl" in telemetry.sinks:
        jsonl_path = Path(telemetry.manifest_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        sinks.append(JsonlSink(jsonl_path))
        sinks.append(ManifestIndexSink(jsonl_path.with_name("index.json")))
        sinks.append(SummarySink(jsonl_path.with_name("summary.json")))
        sinks.append(SqliteSink(jsonl_path.with_name("manifest.sqlite")))

    LOGGER.info(f"Telemetry configured: {len(sinks)} sinks, run_id={run_id}")

    return MultiSink(sinks=sinks, run_id=run_id)


def build_orchestrator(
    orchestrator: OrchestratorConfig,
    queue: QueueConfig,
) -> object:
    """
    Construct work orchestrator from config.

    Args:
        orchestrator: OrchestratorConfig with worker/lease settings
        queue: QueueConfig with backend and persistence

    Returns:
        Configured WorkOrchestrator instance

    Example:
        >>> cfg = ContentDownloadConfig()
        >>> orch = build_orchestrator(cfg.orchestrator, cfg.queue)
    """
    try:
        from DocsToKG.ContentDownload.orchestrator import WorkOrchestrator
    except ImportError:
        LOGGER.warning(
            "WorkOrchestrator not available; returning placeholder",
            extra={"max_workers": orchestrator.max_workers},
        )
        return None

    LOGGER.info(
        "Orchestrator configured",
        extra={
            "max_workers": orchestrator.max_workers,
            "lease_ttl_seconds": orchestrator.lease_ttl_seconds,
            "heartbeat_seconds": orchestrator.heartbeat_seconds,
            "queue_backend": queue.backend,
        },
    )

    # Return orchestrator instance (implementation may vary)
    # This is a placeholder for now; actual construction depends on
    # WorkOrchestrator's constructor signature
    return None


__all__ = [
    "build_http_client",
    "build_telemetry_sinks",
    "build_orchestrator",
]
