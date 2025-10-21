"""
ContentDownload API Surface

Exports unified types for resolver and pipeline contracts.
"""

from .types import (
    AttemptRecord,
    DownloadOutcome,
    DownloadPlan,
    DownloadStreamResult,
    ResolverResult,
)

__all__ = [
    "DownloadPlan",
    "DownloadStreamResult",
    "DownloadOutcome",
    "ResolverResult",
    "AttemptRecord",
]
