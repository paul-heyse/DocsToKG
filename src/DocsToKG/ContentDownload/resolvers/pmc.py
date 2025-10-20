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
from typing import TYPE_CHECKING, Iterable, List, Optional

import httpx

from DocsToKG.ContentDownload.core import dedupe, normalize_doi, normalize_pmcid
from DocsToKG.ContentDownload.networking import request_with_retries

from .base import (
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    _absolute_url,
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
        self, client: httpx.Client, identifiers: List[str], config: "ResolverConfig"
    ) -> List[str]:
        if not identifiers:
            return []
        resp: Optional[httpx.Response] = None
        try:
            resp = request_with_retries(
                client,
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
                retry_after_cap=config.retry_after_cap,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            LOGGER.debug("PMC ID lookup timed out: %s", exc)
            return []
        except httpx.TransportError as exc:
            LOGGER.debug("PMC ID lookup connection error: %s", exc)
            return []
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            LOGGER.debug("PMC ID lookup HTTP error status: %s", status)
            return []
        except httpx.RequestError as exc:
            LOGGER.debug("PMC ID lookup request error: %s", exc)
            return []
        except ValueError as json_err:
            LOGGER.debug("PMC ID lookup JSON error: %s", json_err)
            return []
        except Exception:  # pragma: no cover
            LOGGER.exception("Unexpected error looking up PMC IDs")
            return []
        finally:
            if resp is not None:
                resp.close()

        results: List[str] = []
        for record in data.get("records", []) or []:
            pmcid = record.get("pmcid")
            if pmcid:
                results.append(normalize_pmcid(pmcid))
        return [pmc for pmc in results if pmc]

    def iter_urls(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield PubMed Central PDF URLs matched to ``artifact``.

        Args:
            client: HTTPX client for HTTP requests.
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
            pmcids.extend(self._lookup_pmcids(client, identifiers, config))

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
            resp: Optional[httpx.Response] = None
            try:
                resp = request_with_retries(
                    client,
                    "get",
                    oa_url,
                    timeout=config.get_timeout(self.name),
                    headers=config.polite_headers,
                    retry_after_cap=config.retry_after_cap,
                )
                resp.raise_for_status()
                try:
                    root = ET.fromstring(resp.text)
                except ET.ParseError as exc:
                    LOGGER.debug("PMC OA XML parse error for %s: %s", pmcid, exc)
                    for href in re.findall(r'href=["\']([^"\']+)["\']', resp.text or ""):
                        candidate = href.strip()
                        if candidate.lower().endswith(".pdf"):
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
                        if (
                            fmt == "pdf"
                            or mime == "application/pdf"
                            or href.lower().endswith(".pdf")
                        ):
                            yield ResolverResult(
                                url=_absolute_url(oa_url, href),
                                metadata={"pmcid": pmcid, "source": "oa"},
                            )
            except httpx.TimeoutException as exc:
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
            except httpx.TransportError as exc:
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
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.HTTP_ERROR,
                    http_status=status,
                    metadata={
                        "pmcid": pmcid,
                        "error_detail": str(exc),
                    },
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except httpx.RequestError as exc:
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
            finally:
                if resp is not None:
                    resp.close()

            yield ResolverResult(
                url=fallback_url,
                metadata={"pmcid": pmcid, "source": "pdf-fallback"},
            )
