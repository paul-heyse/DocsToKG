# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.adapters.__init__",
#   "purpose": "Fallback source adapters for PDF resolution.",
#   "sections": [
#     {
#       "id": "head-pdf",
#       "name": "head_pdf",
#       "anchor": "function-head-pdf",
#       "kind": "function"
#     },
#     {
#       "id": "canonicalize-url",
#       "name": "canonicalize_url",
#       "anchor": "function-canonicalize-url",
#       "kind": "function"
#     },
#     {
#       "id": "parse-retry-after",
#       "name": "parse_retry_after",
#       "anchor": "function-parse-retry-after",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Fallback source adapters for PDF resolution.

This package contains adapters for 7 different sources:
- Unpaywall PDF (metadata API)
- arXiv PDF (direct URL construction)
- PubMed Central (metadata + direct PDF)
- DOI Redirects (follow chains)
- Landing Page Scraping (HTML parsing)
- Europe PMC (metadata API)
- Wayback Machine (archive + CDX)

Each adapter is a callable with signature: (policy: AttemptPolicy, context: Dict) → AttemptResult

Shared utilities:
- head_pdf(): Validate PDF URLs with HEAD checks
- canonicalize_url(): Normalize URLs with role awareness
- parse_retry_after(): Extract retry-after headers
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def head_pdf(
    url: str,
    client: Any,
    timeout_s: float = 10.0,
    breaker: Optional[Any] = None,
    logger_: Optional[logging.Logger] = None,
) -> Tuple[bool, int, Optional[str], str]:
    """Validate a PDF URL via HEAD request with content-type checking.

    Args:
        url: URL to validate
        client: HTTP client (with timeout support)
        timeout_s: Timeout in seconds
        breaker: Optional BreakerRegistry for pre-flight checks
        logger_: Optional logger

    Returns:
        Tuple of (is_valid_pdf, status_code, content_type, reason)
        - is_valid_pdf: True if status 200 and content-type includes 'pdf'
        - status_code: HTTP status
        - content_type: From response header
        - reason: Short reason code
    """
    if logger_ is None:
        logger_ = logger

    # Pre-flight breaker check (if provided)
    if breaker is not None:
        try:
            from DocsToKG.ContentDownload.urls import canonical_host

            # pylint: disable=unused-variable
            _ = canonical_host(url)
            # Adapters should call breaker.allow() here if available
            # This is a placeholder - actual implementation in adapters
        except Exception:  # pylint: disable=broad-except
            pass

    try:
        # HEAD request with timeout
        resp = client.head(url, follow_redirects=True, timeout=(5, timeout_s))
        status = resp.status_code
        ct = resp.headers.get("Content-Type", "").lower()

        # Validate status and content-type
        if status == 200:
            if "pdf" in ct or "application/octet-stream" in ct:
                return True, status, ct, "valid_pdf"
            else:
                return False, status, ct, "wrong_content_type"
        else:
            return False, status, ct, f"http_{status}"
    except Exception as e:  # pylint: disable=broad-except
        logger_.debug(f"HEAD failed for {url}: {e}")
        return False, 0, None, "head_error"


def canonicalize_url(url: str, role: str = "artifact") -> str:
    """Canonicalize a URL for consistent handling.

    Args:
        url: URL to canonicalize
        role: Request role ("metadata" or "artifact")

    Returns:
        Canonicalized URL
    """
    # Placeholder - actual implementation would use canonical_for_request
    # from DocsToKG.ContentDownload.urls
    return url


def parse_retry_after(resp: Any, cap_s: int = 900) -> Optional[int]:
    """Parse Retry-After header from response.

    Args:
        resp: HTTP response object
        cap_s: Maximum retry time in seconds

    Returns:
        Retry after in seconds, capped at cap_s, or None
    """
    try:
        retry_after = resp.headers.get("Retry-After")
        if not retry_after:
            return None

        # Try to parse as integer (seconds)
        try:
            delay_s = int(retry_after)
            return min(delay_s, cap_s)
        except ValueError:
            # Try to parse as HTTP-date (not implemented for brevity)
            return None
    except Exception:  # pylint: disable=broad-except
        return None


__all__ = [
    "head_pdf",
    "canonicalize_url",
    "parse_retry_after",
]
