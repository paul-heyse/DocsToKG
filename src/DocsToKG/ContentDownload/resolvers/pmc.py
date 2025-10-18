# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.pmc",
#   "purpose": "PMC resolver implementation",
#   "sections": [
#     {
#       "id": "pmcresolver",
#       "name": "PmcResolver",
#       "anchor": "class-pmcresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for PubMed Central content."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Iterable, List

import requests as _requests

from DocsToKG.ContentDownload.core import dedupe, normalize_doi, normalize_pmcid

from .base import (
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    _absolute_url,
    request_with_retries,
)

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


class PmcResolver(RegisteredResolver):
    """Resolve PubMed Central articles via identifiers and lookups."""

    name = "pmc"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        """Return ``True`` when PMC identifiers or DOI metadata are available.

        Args:
            config: Resolver configuration (unused for enablement checks).
            artifact: Work record containing identifiers such as DOI, PMID, or PMCID.

        Returns:
            bool: Whether the resolver should attempt the work.
        """
        return bool(artifact.pmcid or artifact.pmid or artifact.doi)

    def _lookup_pmcids(
        self, session: _requests.Session, identifiers: List[str], config: "ResolverConfig"
    ) -> List[str]:
        if not identifiers:
            return []
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={
                    "ids": ",".join(identifiers),
                    "format": "json",
                    "tool": "docs-to-kg",
                    "email": config.unpaywall_email or "",
                },
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
            )
        except _requests.Timeout as exc:
            LOGGER.debug("PMC ID lookup timed out: %s", exc)
            return []
        except _requests.ConnectionError as exc:
            LOGGER.debug("PMC ID lookup connection error: %s", exc)
            return []
        except _requests.RequestException as exc:
            LOGGER.debug("PMC ID lookup request error: %s", exc)
            return []
        except Exception:  # pragma: no cover
            LOGGER.exception("Unexpected error looking up PMC IDs")
            return []
        if resp.status_code != 200:
            LOGGER.debug("PMC ID lookup HTTP error status: %s", resp.status_code)
            return []
        try:
            data = resp.json()
        except ValueError as json_err:
            LOGGER.debug("PMC ID lookup JSON error: %s", json_err)
            return []
        results: List[str] = []
        for record in data.get("records", []) or []:
            pmcid = record.get("pmcid")
            if pmcid:
                results.append(normalize_pmcid(pmcid))
        return [pmc for pmc in results if pmc]

    def iter_urls(
        self,
        session: _requests.Session,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield PubMed Central PDF URLs matched to ``artifact``.

        Args:
            session: Requests session for HTTP requests.
            config: Resolver configuration providing timeouts and headers.
            artifact: Work metadata supplying PMC identifiers.

        Yields:
            ResolverResult: Candidate PDF URLs or diagnostic events.
        """
        pmcids: List[str] = []
        if artifact.pmcid:
            pmcids.append(normalize_pmcid(artifact.pmcid))
        identifiers: List[str] = []
        doi = normalize_doi(artifact.doi)
        if doi:
            identifiers.append(doi)
        elif artifact.pmid:
            identifiers.append(artifact.pmid)
        if not pmcids:
            pmcids.extend(self._lookup_pmcids(session, identifiers, config))

        if not pmcids:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_PMCID,
            )
            return

        for pmcid in dedupe(pmcids):
            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
            fallback_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
            try:
                resp = request_with_retries(
                    session,
                    "get",
                    oa_url,
                    timeout=config.get_timeout(self.name),
                    headers=config.polite_headers,
                )
            except _requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.TIMEOUT,
                    metadata={
                        "pmcid": pmcid,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.CONNECTION_ERROR,
                    metadata={"pmcid": pmcid, "error": str(exc)},
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except _requests.RequestException as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.REQUEST_ERROR,
                    metadata={"pmcid": pmcid, "error": str(exc)},
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Unexpected PMC OA lookup error")
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                    metadata={"pmcid": pmcid, "error": str(exc), "error_type": type(exc).__name__},
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            if resp.status_code != 200:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.HTTP_ERROR,
                    http_status=resp.status_code,
                    metadata={
                        "pmcid": pmcid,
                        "error_detail": f"OA endpoint returned {resp.status_code}",
                    },
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            pdf_links_emitted = False
            try:
                root = ET.fromstring(resp.text)
            except ET.ParseError as exc:
                LOGGER.debug("PMC OA XML parse error for %s: %s", pmcid, exc)
                for href in re.findall(r'href=["\']([^"\']+)["\']', resp.text or ""):
                    candidate = href.strip()
                    if candidate.lower().endswith(".pdf"):
                        pdf_links_emitted = True
                        yield ResolverResult(
                            url=_absolute_url(oa_url, candidate),
                            metadata={"pmcid": pmcid, "source": "oa"},
                        )
            else:
                for link in root.iter():
                    tag = link.tag.rsplit("}", 1)[-1].lower()
                    if tag not in {"link", "a"}:
                        continue
                    href = (
                        link.attrib.get("href")
                        or link.attrib.get("{http://www.w3.org/1999/xlink}href")
                        or ""
                    ).strip()
                    fmt = (link.attrib.get("format") or "").lower()
                    mime = (link.attrib.get("type") or "").lower()
                    if not href:
                        continue
                    if fmt == "pdf" or mime == "application/pdf" or href.lower().endswith(".pdf"):
                        pdf_links_emitted = True
                        yield ResolverResult(
                            url=_absolute_url(oa_url, href),
                            metadata={"pmcid": pmcid, "source": "oa"},
                        )
            if pdf_links_emitted:
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
            else:
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
