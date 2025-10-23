# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.adapters.landing_scrape",
#   "purpose": "Landing page scraper adapter for PDF extraction from HTML.",
#   "sections": [
#     {
#       "id": "pdflinkextractor",
#       "name": "PDFLinkExtractor",
#       "anchor": "class-pdflinkextractor",
#       "kind": "class"
#     },
#     {
#       "id": "adapter-landing-scrape-pdf",
#       "name": "adapter_landing_scrape_pdf",
#       "anchor": "function-adapter-landing-scrape-pdf",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Landing page scraper adapter for PDF extraction from HTML."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Dict, List
from urllib.parse import urljoin

from DocsToKG.ContentDownload.fallback.adapters import head_pdf
from DocsToKG.ContentDownload.fallback.types import AttemptPolicy, AttemptResult


class PDFLinkExtractor(HTMLParser):
    """Extract PDF URLs from HTML using common citation metadata patterns."""

    def __init__(self) -> None:
        """Initialize parser."""
        super().__init__()
        self.pdf_urls: List[str] = []
        self.in_script = False

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        """Handle HTML tags."""
        attrs_dict = dict(attrs)

        # Skip script tags
        if tag == "script":
            self.in_script = True
            return

        # meta name="citation_pdf_url"
        if tag == "meta" and attrs_dict.get("name") == "citation_pdf_url":
            content = attrs_dict.get("content")
            if content:
                self.pdf_urls.append(content)

        # link rel="alternate" type="application/pdf"
        elif tag == "link":
            rel = attrs_dict.get("rel", "").lower()
            type_ = attrs_dict.get("type", "").lower()
            if ("alternate" in rel or "pdf" in rel) and "pdf" in type_:
                href = attrs_dict.get("href")
                if href:
                    self.pdf_urls.append(href)

        # a href="...pdf"
        elif tag == "a":
            href = attrs_dict.get("href", "")
            if ".pdf" in href.lower():
                self.pdf_urls.append(href)

    def handle_endtag(self, tag: str) -> None:
        """Handle HTML end tags."""
        if tag == "script":
            self.in_script = False


def adapter_landing_scrape_pdf(
    policy: AttemptPolicy,
    context: Dict[str, Any],
) -> AttemptResult:
    """Scrape landing page HTML for PDF URLs.

    This adapter fetches a landing page (URL or DOI redirect), parses the HTML
    for common citation metadata patterns, and validates candidate PDFs.

    Args:
        policy: AttemptPolicy with timeout and retry configuration
        context: Dict with landing_url, url, doi, head_client, raw_client

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
            meta={"source": "landing_scrape"},
        )

    # Get landing URL
    landing_url = context.get("landing_url") or context.get("url")

    if not landing_url:
        return AttemptResult(
            outcome="skipped",  # type: ignore[arg-type]
            reason="no_landing_url",
            elapsed_ms=0,
        )

    try:
        # Fetch landing page (cached metadata call)
        resp = head_client.get(
            landing_url,
            follow_redirects=True,
            timeout=(5, policy.timeout_ms / 1000),
        )

        if resp.status_code != 200:
            outcome = "retryable" if resp.status_code in (429, 503) else "nonretryable"
            return AttemptResult(
                outcome=outcome,  # type: ignore[arg-type]
                reason="landing_fetch_failed",
                elapsed_ms=0,
                status=resp.status_code,
                meta={"source": "landing_scrape"},
            )

        # Check if response is HTML
        content_type = resp.headers.get("Content-Type", "").lower()
        if "html" not in content_type:
            return AttemptResult(
                outcome="no_pdf",  # type: ignore[arg-type]
                reason="not_html",
                elapsed_ms=0,
                status=200,
                meta={"source": "landing_scrape"},
            )

        # Parse HTML for PDF URLs
        parser = PDFLinkExtractor()
        try:
            parser.feed(resp.text)
        except Exception:  # pylint: disable=broad-except
            pass

        if not parser.pdf_urls:
            return AttemptResult(
                outcome="no_pdf",  # type: ignore[arg-type]
                reason="no_pdf_urls_in_html",
                elapsed_ms=0,
                status=200,
                meta={"source": "landing_scrape"},
            )

        # Try each candidate PDF URL
        for pdf_url in parser.pdf_urls:
            # Resolve relative URLs
            if not pdf_url.startswith(("http://", "https://")):
                pdf_url = urljoin(resp.url, pdf_url)

            # Validate PDF via HEAD
            ok, status, ct, reason = head_pdf(
                pdf_url,
                raw_client,
                timeout_s=policy.timeout_ms / 1000,
            )

            if ok:
                return AttemptResult(
                    outcome="success",  # type: ignore[arg-type]
                    reason="scraped_pdf",
                    elapsed_ms=0,
                    url=pdf_url,
                    status=status,
                    meta={"source": "landing_scrape", "landing_url": landing_url},
                )

        # No valid PDF found
        return AttemptResult(
            outcome="no_pdf",  # type: ignore[arg-type]
            reason="scraped_urls_invalid",
            elapsed_ms=0,
            status=200,
            meta={
                "source": "landing_scrape",
                "candidates": len(parser.pdf_urls),
            },
        )

    except Exception as e:  # pylint: disable=broad-except
        return AttemptResult(
            outcome="error",  # type: ignore[arg-type]
            reason="exception",
            elapsed_ms=0,
            meta={"source": "landing_scrape", "error": str(e)},
        )
