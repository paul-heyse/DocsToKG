# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.integration",
#   "purpose": "Integration of fallback orchestrator into the download pipeline.",
#   "sections": [
#     {
#       "id": "try-fallback-resolution",
#       "name": "try_fallback_resolution",
#       "anchor": "function-try-fallback-resolution",
#       "kind": "function"
#     },
#     {
#       "id": "is-fallback-enabled",
#       "name": "is_fallback_enabled",
#       "anchor": "function-is-fallback-enabled",
#       "kind": "function"
#     },
#     {
#       "id": "get-fallback-plan-path",
#       "name": "get_fallback_plan_path",
#       "anchor": "function-get-fallback-plan-path",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Integration of fallback orchestrator into the download pipeline.

This module provides utilities to integrate the FallbackOrchestrator into
the existing download.process_one_work() function, enabling a feature gate
for the fallback & resiliency strategy.

Responsibilities:
  - Create orchestrator with proper dependencies
  - Call orchestrator.resolve_pdf() if enabled
  - Handle success/failure outcomes
  - Correlate telemetry events
  - Maintain backward compatibility
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from DocsToKG.ContentDownload.fallback.loader import load_fallback_plan
from DocsToKG.ContentDownload.fallback.orchestrator import FallbackOrchestrator
from DocsToKG.ContentDownload.fallback.types import AttemptResult

logger = logging.getLogger(__name__)


def try_fallback_resolution(
    *,
    context: dict[str, Any],
    adapters: dict[str, Callable[[Any, dict[str, Any]], AttemptResult]],
    breaker: Any | None = None,
    rate: Any | None = None,
    head_client: Any | None = None,
    raw_client: Any | None = None,
    telemetry: Any | None = None,
    fallback_plan_path: Path | None = None,
) -> AttemptResult | None:
    """Attempt PDF resolution using the fallback strategy.

    This is a non-blocking resolution attempt. If the fallback strategy
    succeeds (outcome='success'), returns the result. Otherwise returns None
    to signal that the caller should fall back to the existing pipeline.

    Args:
        context: Resolution context (work_id, artifact_id, doi, url, etc.)
        adapters: Mapping of source_name → adapter function
        breaker: Optional BreakerRegistry for health gating
        rate: Optional rate limiter for politeness
        head_client: Optional cached HTTP client
        raw_client: Optional raw HTTP client
        telemetry: Optional telemetry sink
        fallback_plan_path: Optional path to fallback.yaml config

    Returns:
        AttemptResult if success (outcome='success'), None otherwise

    Raises:
        Exception: Only on configuration/setup errors; runtime failures
                   are captured in AttemptResult outcomes.
    """
    # Load fallback plan
    plan = load_fallback_plan(yaml_path=fallback_plan_path)

    # Create clients dict
    clients = {}
    if head_client:
        clients["head"] = head_client
    if raw_client:
        clients["raw"] = raw_client

    # Create orchestrator
    orchestrator = FallbackOrchestrator(
        plan=plan,
        breaker=breaker,
        rate_limiter=rate,
        clients=clients,
        telemetry=telemetry,
        logger=logger,
    )

    # Attempt resolution
    try:
        result = orchestrator.resolve_pdf(context=context, adapters=adapters)

        # Success: return the result
        if result.is_success():  # Fixed: is_success is a method, not property
            logger.info(
                f"Fallback strategy succeeded for {context.get('work_id')}: {result.reason}"
            )
            return result

        # Not found or retryable: fall back to pipeline
        logger.debug(
            f"Fallback strategy did not find PDF for {context.get('work_id')}: {result.reason} (outcome={result.outcome})"
        )
        return None

    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Fallback resolution error: {e}", exc_info=True)
        # On error, fall back to pipeline
        return None


def is_fallback_enabled(options: Any) -> bool:
    """Check if fallback strategy is enabled in options.

    Args:
        options: DownloadConfig or similar object with enable_fallback_strategy attr

    Returns:
        True if enabled, False otherwise
    """
    return bool(getattr(options, "enable_fallback_strategy", False))


def get_fallback_plan_path(options: Any) -> Path | None:
    """Get fallback plan YAML path from options.

    Args:
        options: DownloadConfig or similar object with fallback_plan_path attr

    Returns:
        Path if set, None otherwise
    """
    path = getattr(options, "fallback_plan_path", None)
    if path is None:
        return None
    if isinstance(path, Path):
        return path
    if isinstance(path, str):
        return Path(path)
    return None


__all__ = [
    "try_fallback_resolution",
    "is_fallback_enabled",
    "get_fallback_plan_path",
]
