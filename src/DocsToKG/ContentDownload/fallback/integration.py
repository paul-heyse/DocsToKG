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
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from DocsToKG.ContentDownload.fallback.loader import load_fallback_plan
from DocsToKG.ContentDownload.fallback.orchestrator import FallbackOrchestrator
from DocsToKG.ContentDownload.fallback.types import AttemptResult

logger = logging.getLogger(__name__)


def try_fallback_resolution(
    *,
    context: Dict[str, Any],
    adapters: Dict[str, Callable[[Any, Dict[str, Any]], AttemptResult]],
    breaker: Optional[Any] = None,
    rate: Optional[Any] = None,
    head_client: Optional[Any] = None,
    raw_client: Optional[Any] = None,
    telemetry: Optional[Any] = None,
    fallback_plan_path: Optional[Path] = None,
) -> Optional[AttemptResult]:
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

    # Create orchestrator
    orchestrator = FallbackOrchestrator(
        plan=plan,
        breaker=breaker,
        rate=rate,
        head_client=head_client,
        raw_client=raw_client,
        telemetry=telemetry,
        logger=logger,
    )

    # Attempt resolution
    try:
        result = orchestrator.resolve_pdf(context=context, adapters=adapters)

        # Success: return the result
        if result.is_success:
            logger.info(f"Fallback strategy succeeded for {context.get('work_id')}: {result.reason}")
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


def get_fallback_plan_path(options: Any) -> Optional[Path]:
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
