"""Landing page scraper resolver using BeautifulSoup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable
from urllib.parse import urljoin, urlparse

import requests

from DocsToKG.ContentDownload.http import request_with_retries

from ..types import ResolverConfig, ResolverResult

try:  # Optional dependency guarded at runtime
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    BeautifulSoup = None

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


def _absolute_url(base: str, href: str) -> str:
    """Resolve relative links on a landing page against the page URL."""

    parsed = urlparse(href)
    if parsed.scheme and parsed.netloc:
        return href
    return urljoin(base, href)


LOGGER = logging.getLogger(__name__)


class LandingPageResolver:
    """Attempt to scrape landing pages when explicit PDFs are unavailable.

    Attributes:
        name: Resolver identifier surfaced to the pipeline dispatcher.

    Examples:
        >>> resolver = LandingPageResolver()
        >>> resolver.name
        'landing_page'
    """

    name = "landing_page"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact exposes landing page URLs.

        Args:
            config: Resolver configuration controlling scraping behaviour.
            artifact: Work artifact containing landing page URLs.

        Returns:
            Boolean indicating whether landing pages should be scraped.
        """

        return bool(artifact.landing_urls)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs discovered by scraping landing pages.

        Args:
            session: HTTP session used to download landing pages.
            config: Resolver configuration providing headers and timeouts.
            artifact: Work artifact listing landing page URLs to inspect.

        Returns:
            Iterable of resolver results representing discovered download URLs.
        """

        if BeautifulSoup is None:
            yield ResolverResult(
                url=None,
                event="skipped",
                event_reason="no-beautifulsoup",
            )
            return
        for landing in artifact.landing_urls:
            try:
                resp = request_with_retries(
                    session,
                    "get",
                    landing,
                    headers=config.polite_headers,
                    timeout=config.get_timeout(self.name),
                )
            except requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={
                        "landing": landing,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                continue
            except requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"landing": landing, "error": str(exc)},
                )
                continue
            except requests.RequestException as exc:  # pragma: no cover - network errors
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"landing": landing, "error": str(exc)},
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected error scraping landing page")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={
                        "landing": landing,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                continue

            if resp.status_code != 200:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=resp.status_code,
                    metadata={
                        "landing": landing,
                        "error_detail": f"Landing page returned {resp.status_code}",
                    },
                )
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if meta and meta.get("content"):
                url = _absolute_url(landing, meta["content"].strip())
                yield ResolverResult(url=url, referer=landing, metadata={"pattern": "meta"})
                continue

            for link in soup.find_all("link"):
                rel = " ".join(link.get("rel") or []).lower()
                typ = (link.get("type") or "").lower()
                href = link.get("href") or ""
                if "alternate" in rel and "application/pdf" in typ and href:
                    url = _absolute_url(landing, href.strip())
                    yield ResolverResult(url=url, referer=landing, metadata={"pattern": "link"})
                    break

            for anchor in soup.find_all("a"):
                href = (anchor.get("href") or "").strip()
                if not href:
                    continue
                text = (anchor.get_text() or "").strip().lower()
                href_lower = href.lower()
                if href_lower.endswith(".pdf") or "pdf" in text:
                    candidate = _absolute_url(landing, href)
                    if candidate.lower().endswith(".pdf"):
                        yield ResolverResult(
                            url=candidate,
                            referer=landing,
                            metadata={"pattern": "anchor"},
                        )
                        break


__all__ = ["LandingPageResolver"]
