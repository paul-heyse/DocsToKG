# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.landing_page",
#   "purpose": "Landing page scraping resolver",
#   "sections": [
#     {
#       "id": "landingpageresolver",
#       "name": "LandingPageResolver",
#       "anchor": "class-landingpageresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver that scrapes landing pages for PDF links when metadata fails."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Iterable

import requests as _requests

from .base import (
    BeautifulSoup,
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    XMLParsedAsHTMLWarning,
    find_pdf_via_anchor,
    find_pdf_via_link,
    find_pdf_via_meta,
    request_with_retries,
)

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


class LandingPageResolver(RegisteredResolver):
    """Attempt to scrape landing pages when explicit PDFs are unavailable."""

    name = "landing_page"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        """Return ``True`` when landing page URLs are available to scrape.

        Args:
            config: Resolver configuration (unused for enablement).
            artifact: Work record containing landing URLs.

        Returns:
            bool: Whether the resolver should attempt scraping.
        """
        return bool(artifact.landing_urls)

    def iter_urls(
        self,
        session: _requests.Session,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Scrape landing pages for PDF links and yield matching results.

        Args:
            session: Requests session for HTTP interactions.
            config: Resolver configuration providing timeouts and headers.
            artifact: Work metadata containing landing URLs.

        Yields:
            ResolverResult: Candidate download URLs or diagnostic events.
        """
        if BeautifulSoup is None:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_BEAUTIFULSOUP,
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
            except _requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.TIMEOUT,
                    metadata={
                        "landing": landing,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                continue
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.CONNECTION_ERROR,
                    metadata={"landing": landing, "error": str(exc)},
                )
                continue
            except _requests.RequestException as exc:  # pragma: no cover
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.REQUEST_ERROR,
                    metadata={"landing": landing, "error": str(exc)},
                )
                continue
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Unexpected error scraping landing page")
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.UNEXPECTED_ERROR,
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
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.HTTP_ERROR,
                    http_status=resp.status_code,
                    metadata={
                        "landing": landing,
                        "error_detail": f"Landing page returned {resp.status_code}",
                    },
                )
                continue

            parser_name = "lxml"
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "xml" in content_type:
                parser_name = "lxml-xml"
            if XMLParsedAsHTMLWarning is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                    soup = BeautifulSoup(resp.text, parser_name)
            else:
                soup = BeautifulSoup(resp.text, parser_name)
            for pattern, finder in (
                ("meta", find_pdf_via_meta),
                ("link", find_pdf_via_link),
                ("anchor", find_pdf_via_anchor),
            ):
                candidate = finder(soup, landing)
                if candidate:
                    yield ResolverResult(
                        url=candidate,
                        referer=landing,
                        metadata={"pattern": pattern},
                    )
                    break
