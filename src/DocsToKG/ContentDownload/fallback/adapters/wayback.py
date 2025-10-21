"""Wayback Machine adapter for archived PDF resolution."""

from __future__ import annotations

from typing import Any, Dict, Optional

from DocsToKG.ContentDownload.fallback.adapters import head_pdf
from DocsToKG.ContentDownload.fallback.types import AttemptPolicy, AttemptResult


def extract_domain_from_url(url: str) -> Optional[str]:
    """Extract domain from URL for Wayback queries."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split("/")[0]
    except Exception:  # pylint: disable=broad-except
        return None


def adapter_wayback_pdf(
    policy: AttemptPolicy,
    context: Dict[str, Any],
) -> AttemptResult:
    """Query Wayback Machine CDX API for archived PDFs.

    The Wayback Machine (archive.org) indexes snapshots of websites.
    This adapter queries the CDX API for PDF snapshots and validates them.

    Args:
        policy: AttemptPolicy with timeout and retry configuration
        context: Dict with landing_url, url, head_client, raw_client

    Returns:
        AttemptResult indicating success/failure of PDF resolution
    """
    head_client = context.get("head_client")
    raw_client = context.get("raw_client")

    if not head_client or not raw_client:
        return AttemptResult(
            outcome="error",
            reason="missing_client",
            elapsed_ms=0,
            meta={"source": "wayback"},
        )

    # Get URL to archive-search
    landing_url = context.get("landing_url") or context.get("url")

    if not landing_url:
        return AttemptResult(
            outcome="skipped",
            reason="no_landing_url",
            elapsed_ms=0,
        )

    try:
        # Query CDX API for PDF snapshots
        # Example: https://archive.org/wayback/available?url=example.org/paper&output=json
        api_url = "https://archive.org/wayback/available"
        params = {
            "url": landing_url,
            "output": "json",
        }

        resp = head_client.get(
            api_url,
            params=params,
            timeout=(5, policy.timeout_ms / 1000),
        )

        if resp.status_code != 200:
            outcome = "retryable" if resp.status_code in (429, 503) else "nonretryable"
            return AttemptResult(
                outcome=outcome,
                reason="cdx_api_error",
                elapsed_ms=0,
                status=resp.status_code,
                host="archive.org",
                meta={"source": "wayback"},
            )

        # Parse response
        data = resp.json()
        snapshots = data.get("archived_snapshots", {})

        if not snapshots:
            return AttemptResult(
                outcome="no_pdf",
                reason="no_snapshots",
                elapsed_ms=0,
                status=200,
                host="archive.org",
                meta={"source": "wayback"},
            )

        # Try the closest snapshot
        closest = snapshots.get("closest")
        if not closest:
            return AttemptResult(
                outcome="no_pdf",
                reason="no_closest_snapshot",
                elapsed_ms=0,
                status=200,
                host="archive.org",
                meta={"source": "wayback"},
            )

        # Get the Wayback URL
        wayback_url = closest.get("url")
        if not wayback_url:
            return AttemptResult(
                outcome="no_pdf",
                reason="no_wayback_url",
                elapsed_ms=0,
                status=200,
                host="archive.org",
                meta={"source": "wayback"},
            )

        # Check if Wayback URL ends in .pdf
        if wayback_url.endswith(".pdf"):
            ok, status, ct, reason = head_pdf(
                wayback_url,
                raw_client,
                timeout_s=policy.timeout_ms / 1000,
            )

            if ok:
                return AttemptResult(
                    outcome="success",
                    reason="wayback_pdf",
                    elapsed_ms=0,
                    url=wayback_url,
                    status=status,
                    host="archive.org",
                    meta={"source": "wayback", "timestamp": closest.get("timestamp")},
                )

        # Try to fetch and parse the Wayback snapshot for PDF links
        snapshot_resp = raw_client.get(
            wayback_url,
            follow_redirects=True,
            timeout=(5, policy.timeout_ms / 1000),
        )

        if snapshot_resp.status_code == 200:
            # Check if response is PDF
            content_type = snapshot_resp.headers.get("Content-Type", "").lower()
            if "pdf" in content_type:
                return AttemptResult(
                    outcome="success",
                    reason="wayback_pdf_content",
                    elapsed_ms=0,
                    url=wayback_url,
                    status=200,
                    host="archive.org",
                    meta={"source": "wayback", "timestamp": closest.get("timestamp")},
                )

            # Check if it's HTML and look for PDF links
            if "html" in content_type:
                html = snapshot_resp.text[:10000]  # First 10k chars
                if ".pdf" in html:
                    # Basic extraction
                    import re

                    match = re.search(r'href="([^"]*\.pdf[^"]*)"', html, re.IGNORECASE)
                    if match:
                        candidate_url = match.group(1)
                        # Resolve relative URLs
                        if not candidate_url.startswith(("http://", "https://")):
                            from urllib.parse import urljoin

                            candidate_url = urljoin(wayback_url, candidate_url)

                        ok, status, ct, reason = head_pdf(
                            candidate_url,
                            raw_client,
                            timeout_s=policy.timeout_ms / 1000,
                        )

                        if ok:
                            return AttemptResult(
                                outcome="success",
                                reason="wayback_extracted_pdf",
                                elapsed_ms=0,
                                url=candidate_url,
                                status=status,
                                host="archive.org",
                                meta={
                                    "source": "wayback",
                                    "timestamp": closest.get("timestamp"),
                                    "extracted": True,
                                },
                            )

        # No PDF found in Wayback
        return AttemptResult(
            outcome="no_pdf",
            reason="no_wayback_pdf",
            elapsed_ms=0,
            status=200,
            host="archive.org",
            meta={
                "source": "wayback",
                "timestamp": closest.get("timestamp"),
            },
        )

    except Exception as e:  # pylint: disable=broad-except
        return AttemptResult(
            outcome="error",
            reason="exception",
            elapsed_ms=0,
            meta={"source": "wayback", "error": str(e)},
        )
