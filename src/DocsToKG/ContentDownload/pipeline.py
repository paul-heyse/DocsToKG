"""Legacy resolver pipeline and configuration types.

⚠️  DEPRECATED: This module is maintained for backward compatibility only.
The new system uses:
  - download_pipeline.py for the modern DownloadPipeline orchestrator
  - config/models.py for Pydantic v2 configuration
  - registry_v2.py for the modern @register_v2 decorator pattern

This module will be removed after all dependents migrate to the modern architecture.
"""

from __future__ import annotations

# Legacy placeholder types (not in api/types.py - kept for minimal compatibility)
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Re-export data contract types from api/types for backward compatibility
from DocsToKG.ContentDownload.api.types import (
    AttemptRecord,
    AttemptStatus,
    DownloadOutcome,
    DownloadPlan,
    DownloadStreamResult,
    OutcomeClass,
    ReasonCode,
    ResolverResult,
)


@dataclass
class ResolverMetrics:
    """Legacy metrics placeholder for backward compatibility.
    
    ⚠️  DEPRECATED: Use new telemetry infrastructure instead.
    Kept only to avoid breaking download.py type hints during transition.
    """
    attempts: Dict[str, int] = field(default_factory=dict)
    successes: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, int] = field(default_factory=dict)

@dataclass(frozen=True)
class PipelineResult:
    """Legacy pipeline result placeholder for backward compatibility.
    
    ⚠️  DEPRECATED: Not used in modern pipeline.
    Kept only for TYPE_CHECKING imports in telemetry.py.
    """
    success: bool
    resolver_name: Optional[str] = None
    outcome: Optional[DownloadOutcome] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# Legacy ResolverPipeline stub (not exported - delete after telemetry cleanup)
class ResolverPipeline:
    """⚠️  DEPRECATED: Legacy pipeline class removed.
    
    Use download_pipeline.DownloadPipeline instead.
    This stub is kept only to catch import errors.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "ResolverPipeline is deprecated and no longer functional. "
            "Use DocsToKG.ContentDownload.download_pipeline.DownloadPipeline instead."
        )

@dataclass
class ResolverConfig:
    """⚠️  DEPRECATED: Legacy resolver config placeholder for backward compatibility.
    
    Kept only for legacy test imports.
    Do not use in new code.
    """
    pass

__all__ = [
    # Re-exported from api/types
    "AttemptRecord",
    "DownloadOutcome",
    "DownloadPlan",
    "DownloadStreamResult",
    "ResolverResult",
    "AttemptStatus",
    "OutcomeClass",
    "ReasonCode",
    # Legacy placeholders
    "ResolverMetrics",
    "PipelineResult",
    "ResolverPipeline",
    "ResolverConfig",
]
