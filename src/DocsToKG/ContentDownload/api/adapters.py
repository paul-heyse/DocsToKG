"""
Adapters & Legacy Shims for Canonical Types

Thin wrapper functions to help legacy code migrate to canonical types.
Used when old call sites can't be refactored immediately.

Over time, these should be removed as code migrates.
"""

from __future__ import annotations

from typing import Any, Optional

from .exceptions import DownloadError, SkipDownload
from .types import (
    DownloadOutcome,
    DownloadPlan,
    ReasonCode,
    ResolverResult,
)

# ============================================================================
# DownloadPlan Constructors
# ============================================================================


def to_download_plan(
    url: str,
    resolver_name: str,
    *,
    referer: Optional[str] = None,
    expected_mime: Optional[str] = None,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    max_bytes_override: Optional[int] = None,
) -> DownloadPlan:
    """
    Construct a DownloadPlan from scalar arguments.

    Convenience wrapper for code that builds plans piece-by-piece.
    """
    return DownloadPlan(
        url=url,
        resolver_name=resolver_name,
        referer=referer,
        expected_mime=expected_mime,
        etag=etag,
        last_modified=last_modified,
        max_bytes_override=max_bytes_override,
    )


# ============================================================================
# DownloadOutcome Constructors
# ============================================================================


def to_outcome_success(path: str, **meta: Any) -> DownloadOutcome:
    """Create a successful DownloadOutcome."""
    return DownloadOutcome(
        ok=True,
        classification="success",
        path=path,
        reason=None,
        meta=meta,
    )


def to_outcome_skip(reason: ReasonCode, **meta: Any) -> DownloadOutcome:
    """Create a skip DownloadOutcome."""
    return DownloadOutcome(
        ok=False,
        classification="skip",
        path=None,
        reason=reason,
        meta=meta,
    )


def to_outcome_error(reason: ReasonCode, **meta: Any) -> DownloadOutcome:
    """Create an error DownloadOutcome."""
    return DownloadOutcome(
        ok=False,
        classification="error",
        path=None,
        reason=reason,
        meta=meta,
    )


# ============================================================================
# ResolverResult Constructors
# ============================================================================


def to_resolver_result(
    plans: list[DownloadPlan] | None = None,
    **notes: Any,
) -> ResolverResult:
    """Create a ResolverResult from plans and optional notes."""
    return ResolverResult(
        plans=plans or [],
        notes=notes,
    )


# ============================================================================
# Exception Factories
# ============================================================================


def make_skip(reason: ReasonCode, message: Optional[str] = None) -> SkipDownload:
    """Factory for SkipDownload exception."""
    return SkipDownload(reason, message)


def make_error(reason: ReasonCode, message: Optional[str] = None) -> DownloadError:
    """Factory for DownloadError exception."""
    return DownloadError(reason, message)
