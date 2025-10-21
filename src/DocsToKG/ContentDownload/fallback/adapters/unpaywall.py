"""Unpaywall API adapter for Open Access PDF resolution."""

from __future__ import annotations

from typing import Any, Dict

from DocsToKG.ContentDownload.fallback.adapters import head_pdf
from DocsToKG.ContentDownload.fallback.types import AttemptPolicy, AttemptResult


def adapter_unpaywall_pdf(
    policy: AttemptPolicy,
    context: Dict[str, Any],
) -> AttemptResult:
    """Query Unpaywall API for Open Access PDF.

    Unpaywall provides the best open access location for a given DOI via
    their REST API. This adapter queries the API and validates the returned
    PDF URL via HEAD request.

    Args:
        policy: AttemptPolicy with timeout and retry configuration
        context: Dict with doi, head_client, raw_client

    Returns:
        AttemptResult indicating success/failure of PDF resolution
    """
    doi = context.get("doi")
    if not doi:
        return AttemptResult(
            outcome="skipped",
            reason="no_doi",
            elapsed_ms=0,
        )

    email = context.get("email") or "unknown@example.com"
    head_client = context.get("head_client")
    raw_client = context.get("raw_client")

    if not head_client or not raw_client:
        return AttemptResult(
            outcome="error",
            reason="missing_client",
            elapsed_ms=0,
            meta={"source": "unpaywall"},
        )

    try:
        # Query Unpaywall API (metadata call, cached)
        api_url = f"https://api.unpaywall.org/v2/{doi}"
        resp = head_client.get(
            api_url,
            params={"email": email},
            timeout=(5, policy.timeout_ms / 1000),
        )

        if resp.status_code != 200:
            outcome = "retryable" if resp.status_code in (429, 503) else "nonretryable"
            return AttemptResult(
                outcome=outcome,
                reason="api_error",
                elapsed_ms=0,
                status=resp.status_code,
                host="api.unpaywall.org",
                meta={"source": "unpaywall"},
            )

        # Parse response
        data = resp.json()
        best_oa = data.get("best_oa_location")

        if not best_oa or not best_oa.get("url_for_pdf"):
            return AttemptResult(
                outcome="no_pdf",
                reason="no_oa_pdf",
                elapsed_ms=0,
                status=200,
                host="api.unpaywall.org",
                meta={"source": "unpaywall"},
            )

        pdf_url = best_oa["url_for_pdf"]

        # Validate PDF via HEAD
        ok, status, ct, reason = head_pdf(
            pdf_url,
            raw_client,
            timeout_s=policy.timeout_ms / 1000,
        )

        if ok:
            return AttemptResult(
                outcome="success",
                reason="oa_pdf",
                elapsed_ms=0,
                url=pdf_url,
                status=status,
                host="api.unpaywall.org",
                meta={"source": "unpaywall", "oa_location": best_oa.get("license")},
            )
        else:
            return AttemptResult(
                outcome="nonretryable",
                reason=reason,
                elapsed_ms=0,
                status=status,
                host="api.unpaywall.org",
                meta={"source": "unpaywall", "validation_failed": reason},
            )

    except Exception as e:  # pylint: disable=broad-except
        return AttemptResult(
            outcome="error",
            reason="exception",
            elapsed_ms=0,
            meta={"source": "unpaywall", "error": str(e)},
        )
