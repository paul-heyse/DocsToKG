"""arXiv PDF adapter for direct PDF resolution."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from DocsToKG.ContentDownload.fallback.adapters import head_pdf
from DocsToKG.ContentDownload.fallback.types import AttemptPolicy, AttemptResult


def extract_arxiv_id(doi: Optional[str]) -> Optional[str]:
    """Extract arXiv ID from DOI or context.

    arXiv DOIs are typically in the form: 10.48550/arXiv.XXXX.XXXXX
    """
    if not doi:
        return None

    # Try to extract from DOI
    match = re.search(r"arXiv[:/]?(\d{4}\.\d{4,5})", doi, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def adapter_arxiv_pdf(
    policy: AttemptPolicy,
    context: Dict[str, Any],
) -> AttemptResult:
    """Query arXiv for PDF via direct URL construction.

    arXiv PDFs can be accessed directly via https://arxiv.org/pdf/{arxiv_id}.pdf
    This adapter constructs the URL and validates it via HEAD request.

    Args:
        policy: AttemptPolicy with timeout and retry configuration
        context: Dict with doi, arxiv_id (optional), head_client, raw_client

    Returns:
        AttemptResult indicating success/failure of PDF resolution
    """
    # Extract arXiv ID from context or DOI
    arxiv_id = context.get("arxiv_id")
    if not arxiv_id:
        arxiv_id = extract_arxiv_id(context.get("doi"))

    if not arxiv_id:
        return AttemptResult(
            outcome="skipped",
            reason="no_arxiv_id",
            elapsed_ms=0,
        )

    raw_client = context.get("raw_client")

    if not raw_client:
        return AttemptResult(
            outcome="error",
            reason="missing_client",
            elapsed_ms=0,
            meta={"source": "arxiv"},
        )

    try:
        # Construct arXiv PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Validate PDF via HEAD
        ok, status, ct, reason = head_pdf(
            pdf_url,
            raw_client,
            timeout_s=policy.timeout_ms / 1000,
        )

        if ok:
            return AttemptResult(
                outcome="success",
                reason="arxiv_pdf",
                elapsed_ms=0,
                url=pdf_url,
                status=status,
                host="arxiv.org",
                meta={"source": "arxiv", "arxiv_id": arxiv_id},
            )
        else:
            outcome = "retryable" if status in (429, 503) else "nonretryable"
            return AttemptResult(
                outcome=outcome,
                reason=reason,
                elapsed_ms=0,
                status=status,
                host="arxiv.org",
                meta={"source": "arxiv", "arxiv_id": arxiv_id},
            )

    except Exception as e:  # pylint: disable=broad-except
        return AttemptResult(
            outcome="error",
            reason="exception",
            elapsed_ms=0,
            meta={"source": "arxiv", "error": str(e)},
        )
