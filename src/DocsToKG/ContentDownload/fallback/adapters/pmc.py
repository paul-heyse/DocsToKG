# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.adapters.pmc",
#   "purpose": "PubMed Central adapter for PDF resolution.",
#   "sections": [
#     {
#       "id": "extract-pmcid",
#       "name": "extract_pmcid",
#       "anchor": "function-extract-pmcid",
#       "kind": "function"
#     },
#     {
#       "id": "adapter-pmc-pdf",
#       "name": "adapter_pmc_pdf",
#       "anchor": "function-adapter-pmc-pdf",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""PubMed Central adapter for PDF resolution."""

from __future__ import annotations

from typing import Any

from DocsToKG.ContentDownload.fallback.adapters import head_pdf
from DocsToKG.ContentDownload.fallback.types import AttemptPolicy, AttemptResult


def extract_pmcid(pmcid_str: str | None) -> str | None:
    """Extract numeric PMCID from various formats.

    Handles: "PMC1234567", "1234567", "pmc1234567"
    """
    if not pmcid_str:
        return None

    # Remove common prefixes
    pmcid = str(pmcid_str).upper().replace("PMC", "").strip()

    # Ensure it's numeric
    if pmcid.isdigit():
        return pmcid

    return None


def adapter_pmc_pdf(
    policy: AttemptPolicy,
    context: dict[str, Any],
) -> AttemptResult:
    """Query PubMed Central for PDF.

    PubMed Central (PMC) provides free full-text biomedical literature.
    This adapter constructs PMC URLs or queries the PMC API.

    Args:
        policy: AttemptPolicy with timeout and retry configuration
        context: Dict with pmcid, pmid, head_client, raw_client

    Returns:
        AttemptResult indicating success/failure of PDF resolution
    """
    raw_client = context.get("raw_client")

    if not raw_client:
        return AttemptResult(
            outcome="error",  # type: ignore[arg-type]
            reason="missing_client",
            elapsed_ms=0,
            meta={"source": "pmc"},
        )

    # Extract PMCID
    pmcid = context.get("pmcid")
    if not pmcid:
        pmcid = extract_pmcid(context.get("pmcid"))

    if not pmcid:
        return AttemptResult(
            outcome="skipped",  # type: ignore[arg-type]
            reason="no_pmcid",
            elapsed_ms=0,
        )

    try:
        # Construct PMC PDF URL
        # Format: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"

        # Validate PDF via HEAD
        ok, status, ct, reason = head_pdf(
            pdf_url,
            raw_client,
            timeout_s=policy.timeout_ms / 1000,
        )

        if ok:
            return AttemptResult(
                outcome="success",  # type: ignore[arg-type]
                reason="pmc_pdf",
                elapsed_ms=0,
                url=pdf_url,
                status=status,
                host="ncbi.nlm.nih.gov",
                meta={"source": "pmc", "pmcid": pmcid},
            )
        else:
            outcome = "retryable" if status in (429, 503) else "nonretryable"
            return AttemptResult(
                outcome=outcome,  # type: ignore[arg-type]
                reason=reason,
                elapsed_ms=0,
                status=status,
                host="ncbi.nlm.nih.gov",
                meta={"source": "pmc", "pmcid": pmcid},
            )

    except Exception as e:  # pylint: disable=broad-except
        return AttemptResult(
            outcome="error",  # type: ignore[arg-type]
            reason="exception",
            elapsed_ms=0,
            meta={"source": "pmc", "error": str(e)},
        )
