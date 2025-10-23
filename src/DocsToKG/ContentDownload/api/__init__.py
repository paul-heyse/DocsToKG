# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.api.__init__",
#   "purpose": "ContentDownload API Surface.",
#   "sections": []
# }
# === /NAVMAP ===

"""
ContentDownload API Surface

Canonical types module for unified contracts between resolvers,
download execution, and pipeline orchestration.

All types exported here represent the stable public API:
- DownloadPlan: Resolver → Pipeline
- ResolverResult: Resolver return type
- DownloadStreamResult: Download execution intermediate
- DownloadOutcome: Final result for manifest
- AttemptRecord: Telemetry/diagnostics

Plus stable vocabulary types:
- OutcomeClass: "success" | "skip" | "error"
- AttemptStatus: HTTP attempt status tokens
- ReasonCode: Normalized reason codes
"""

from .types import (
    AttemptRecord,
    AttemptStatus,
    DownloadOutcome,
    DownloadPlan,
    DownloadStreamResult,
    OutcomeClass,
    ReasonCode,
    ResolverResult,
)

__all__ = [
    # Core dataclasses
    "DownloadPlan",
    "DownloadStreamResult",
    "DownloadOutcome",
    "ResolverResult",
    "AttemptRecord",
    # Vocabulary types (Literals)
    "OutcomeClass",
    "AttemptStatus",
    "ReasonCode",
]
