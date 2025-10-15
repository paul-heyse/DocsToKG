"""PubMed Central resolver leveraging NCBI utilities and OA endpoints."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, List
from urllib.parse import urljoin, urlparse

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import dedupe, normalize_doi, normalize_pmcid

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


def _absolute_url(base: str, href: str) -> str:
    """Resolve relative ``href`` values against ``base`` to obtain absolute URLs."""

    parsed = urlparse(href)
    if parsed.scheme and parsed.netloc:
        return href
    return urljoin(base, href)


class PmcResolver:
    """Resolve PubMed Central articles via identifiers and lookups.

    Attributes:
        name: Resolver identifier published to the pipeline scheduler.

    Examples:
        >>> resolver = PmcResolver()
        >>> resolver.name
        'pmc'
    """

    name = "pmc"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has PMC, PMID, or DOI identifiers.

        Args:
            config: Resolver configuration providing PMC connectivity hints.
            artifact: Work artifact capturing PMC/PMID/DOI metadata.

        Returns:
            Boolean indicating whether PMC resolution should be attempted.
        """

        return bool(artifact.pmcid or artifact.pmid or artifact.doi)

    def _lookup_pmcids(
        self, session: requests.Session, identifiers: List[str], config: ResolverConfig
    ) -> List[str]:
        """Return PMCIDs resolved from DOI/PMID identifiers using NCBI utilities.

        Args:
            session: Requests session reused across resolver calls.
            identifiers: DOI or PMID identifiers to convert to PMCIDs.
            config: Resolver configuration providing timeout and polite headers.

        Returns:
            List of PMCIDs corresponding to the supplied identifiers.
        """

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
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        results: List[str] = []
        for record in data.get("records", []) or []:
            pmcid = record.get("pmcid")
            if pmcid:
                results.append(normalize_pmcid(pmcid))
        return [pmc for pmc in results if pmc]

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate PMC download URLs derived from identifiers.

        Args:
            session: HTTP session used to issue PMC and utility API requests.
            config: Resolver configuration, including timeouts and headers.
            artifact: Work artifact describing the scholarly item under resolution.

        Returns:
            Iterable of resolver results pointing to PMC hosted PDFs.
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
            except requests.RequestException:
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            if resp.status_code != 200:
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            for match in re.finditer(r'href="([^"]+\.pdf)"', resp.text, flags=re.I):
                href = match.group(1)
                url = _absolute_url(oa_url, href)
                yield ResolverResult(
                    url=url,
                    metadata={"pmcid": pmcid, "source": "oa"},
                )
            yield ResolverResult(
                url=fallback_url,
                metadata={"pmcid": pmcid, "source": "pdf-fallback"},
            )


__all__ = ["PmcResolver"]
