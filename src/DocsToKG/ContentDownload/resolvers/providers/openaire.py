"""Resolver connecting to the OpenAIRE research infrastructure for discovery."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterable, List

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import dedupe, normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


def _collect_candidate_urls(node: object, results: List[str]) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _collect_candidate_urls(value, results)
    elif isinstance(node, list):
        for item in node:
            _collect_candidate_urls(item, results)
    elif isinstance(node, str):
        if node.lower().startswith("http"):
            results.append(node)


class OpenAireResolver:
    """Resolve URLs using the OpenAIRE API.

    Attributes:
        name: Resolver identifier used when interacting with the pipeline.

    Examples:
        >>> resolver = OpenAireResolver()
        >>> resolver.name
        'openaire'
    """

    name = "openaire"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has a DOI.

        Args:
            config: Resolver configuration controlling OpenAIRE behaviour.
            artifact: Work artifact containing metadata such as DOI.

        Returns:
            Boolean indicating whether an OpenAIRE lookup should be attempted.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs discovered via OpenAIRE search.

        Args:
            session: HTTP session available for outbound requests.
            config: Resolver configuration with polite headers and timeouts.
            artifact: Work artifact describing the item being resolved.

        Returns:
            Iterable of resolver results representing discovered URLs.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            return []
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://api.openaire.eu/search/publications",
                params={"doi": doi},
                headers=config.polite_headers,
                timeout=config.get_timeout(self.name),
            )
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            try:
                data = json.loads(resp.text)
            except ValueError:
                return []
        results: List[str] = []
        _collect_candidate_urls(data, results)
        for url in dedupe(results):
            if url.lower().endswith(".pdf"):
                yield ResolverResult(url=url, metadata={"source": "openaire"})


__all__ = ["OpenAireResolver"]
