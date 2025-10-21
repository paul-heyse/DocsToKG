"""Europe PMC API adapter for PDF resolution."""

from __future__ import annotations

from typing import Any, Dict

from DocsToKG.ContentDownload.fallback.adapters import head_pdf
from DocsToKG.ContentDownload.fallback.types import AttemptPolicy, AttemptResult


def adapter_europe_pmc_pdf(
    policy: AttemptPolicy,
    context: Dict[str, Any],
) -> AttemptResult:
    """Query Europe PMC API for PDF.

    Europe PMC provides metadata and PDFs for biomedical/life sciences articles.
    This adapter queries the REST API and validates the returned PDF URL.

    Args:
        policy: AttemptPolicy with timeout and retry configuration
        context: Dict with doi, pmid, pmcid, head_client, raw_client

    Returns:
        AttemptResult indicating success/failure of PDF resolution
    """
    head_client = context.get("head_client")
    raw_client = context.get("raw_client")

    if not head_client or not raw_client:
        return AttemptResult(
            outcome="error",  # type: ignore[arg-type]
            reason="missing_client",
            elapsed_ms=0,
            meta={"source": "europe_pmc"},
        )

    # Try DOI first, then PMID
    doi = context.get("doi")
    pmid = context.get("pmid")
    pmcid = context.get("pmcid")

    query_id = None
    query_type = None

    if doi:
        query_id = doi
        query_type = "doi"
    elif pmid:
        query_id = pmid
        query_type = "pmid"
    elif pmcid:
        query_id = pmcid
        query_type = "pmcid"

    if not query_id:
        return AttemptResult(
            outcome="skipped",  # type: ignore[arg-type]
            reason="no_identifier",
            elapsed_ms=0,
        )

    try:
        # Query Europe PMC API
        api_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": f"{query_type}:{query_id}",
            "format": "json",
            "pageSize": 1,
        }

        resp = head_client.get(
            api_url,
            params=params,
            timeout=(5, policy.timeout_ms / 1000),
        )

        if resp.status_code != 200:
            outcome = "retryable" if resp.status_code in (429, 503) else "nonretryable"
            return AttemptResult(
                outcome=outcome,  # type: ignore[arg-type]
                reason="api_error",
                elapsed_ms=0,
                status=resp.status_code,
                host="ebi.ac.uk",
                meta={"source": "europe_pmc"},
            )

        # Parse response
        data = resp.json()
        results = data.get("resultList", {}).get("result", [])

        if not results:
            return AttemptResult(
                outcome="no_pdf",  # type: ignore[arg-type]
                reason="not_found",
                elapsed_ms=0,
                status=200,
                host="ebi.ac.uk",
                meta={"source": "europe_pmc"},
            )

        article = results[0]

        # Check for PDF URL
        pdf_url = article.get("fullTextUrl")
        if not pdf_url:
            # Try alternate field
            pdf_url = article.get("pmcPdfUrl")

        if not pdf_url:
            return AttemptResult(
                outcome="no_pdf",  # type: ignore[arg-type]
                reason="no_pdf_field",
                elapsed_ms=0,
                status=200,
                host="ebi.ac.uk",
                meta={"source": "europe_pmc"},
            )

        # Validate PDF via HEAD
        ok, status, ct, reason = head_pdf(
            pdf_url,
            raw_client,
            timeout_s=policy.timeout_ms / 1000,
        )

        if ok:
            return AttemptResult(
                outcome="success",  # type: ignore[arg-type]
                reason="europe_pmc_pdf",
                elapsed_ms=0,
                url=pdf_url,
                status=status,
                host="ebi.ac.uk",
                meta={"source": "europe_pmc", "query_type": query_type},
            )
        else:
            return AttemptResult(
                outcome="nonretryable",  # type: ignore[arg-type]
                reason=reason,
                elapsed_ms=0,
                status=status,
                host="ebi.ac.uk",
                meta={"source": "europe_pmc", "validation_failed": reason},
            )

    except Exception as e:  # pylint: disable=broad-except
        return AttemptResult(
            outcome="error",  # type: ignore[arg-type]
            reason="exception",
            elapsed_ms=0,
            meta={"source": "europe_pmc", "error": str(e)},
        )
