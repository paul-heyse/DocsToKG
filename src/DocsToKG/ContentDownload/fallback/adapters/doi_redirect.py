"""DOI Redirect adapter for following DOI chains to PDFs."""

from __future__ import annotations

from typing import Any, Dict

from DocsToKG.ContentDownload.fallback.adapters import head_pdf
from DocsToKG.ContentDownload.fallback.types import AttemptPolicy, AttemptResult


def adapter_doi_redirect_pdf(
    policy: AttemptPolicy,
    context: Dict[str, Any],
) -> AttemptResult:
    """Follow DOI redirects to find PDF or landing page.

    This adapter resolves a DOI via https://doi.org/{doi} which typically
    redirects to the publisher's landing page. It follows the redirect chain
    and attempts to extract or validate a PDF URL.

    Args:
        policy: AttemptPolicy with timeout and retry configuration
        context: Dict with doi, raw_client

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

    raw_client = context.get("raw_client")

    if not raw_client:
        return AttemptResult(
            outcome="error",
            reason="missing_client",
            elapsed_ms=0,
            meta={"source": "doi_redirect"},
        )

    try:
        # Follow DOI redirects
        doi_url = f"https://doi.org/{doi}"
        resp = raw_client.get(
            doi_url,
            follow_redirects=True,
            timeout=(5, policy.timeout_ms / 1000),
        )

        if resp.status_code != 200:
            outcome = "retryable" if resp.status_code in (429, 503) else "nonretryable"
            return AttemptResult(
                outcome=outcome,
                reason="doi_redirect_failed",
                elapsed_ms=0,
                status=resp.status_code,
                host="doi.org",
                meta={"source": "doi_redirect", "final_url": str(resp.url)},
            )

        final_url = str(resp.url)

        # Check if we ended up at a PDF
        if final_url.endswith(".pdf"):
            ok, status, ct, reason = head_pdf(
                final_url,
                raw_client,
                timeout_s=policy.timeout_ms / 1000,
            )

            if ok:
                return AttemptResult(
                    outcome="success",
                    reason="doi_redirect_pdf",
                    elapsed_ms=0,
                    url=final_url,
                    status=status,
                    host="doi.org",
                    meta={"source": "doi_redirect"},
                )

        # Otherwise, we're on a landing page
        # Try to extract PDF URL from HTML (basic approach)
        content_type = resp.headers.get("Content-Type", "").lower()
        if "html" in content_type:
            # Basic heuristic: look for .pdf in the HTML
            html = resp.text[:10000]  # First 10k chars

            if ".pdf" in html:
                # Try common PDF URL patterns
                pdf_patterns = [
                    r'href="([^"]*\.pdf[^"]*)"',
                    r"href='([^']*\.pdf[^']*)'",
                    r'src="([^"]*\.pdf[^"]*)"',
                ]

                import re

                for pattern in pdf_patterns:
                    match = re.search(pattern, html, re.IGNORECASE)
                    if match:
                        candidate_url = match.group(1)
                        # Resolve relative URLs
                        if not candidate_url.startswith(("http://", "https://")):
                            from urllib.parse import urljoin

                            candidate_url = urljoin(final_url, candidate_url)

                        # Validate candidate
                        ok, status, ct, reason = head_pdf(
                            candidate_url,
                            raw_client,
                            timeout_s=policy.timeout_ms / 1000,
                        )

                        if ok:
                            return AttemptResult(
                                outcome="success",
                                reason="doi_redirect_extracted_pdf",
                                elapsed_ms=0,
                                url=candidate_url,
                                status=status,
                                host="doi.org",
                                meta={"source": "doi_redirect", "extracted": True},
                            )

        # No PDF found after following redirects
        return AttemptResult(
            outcome="no_pdf",
            reason="no_pdf_after_redirect",
            elapsed_ms=0,
            status=200,
            host="doi.org",
            meta={"source": "doi_redirect", "final_url": final_url},
        )

    except Exception as e:  # pylint: disable=broad-except
        return AttemptResult(
            outcome="error",
            reason="exception",
            elapsed_ms=0,
            meta={"source": "doi_redirect", "error": str(e)},
        )
