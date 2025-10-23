# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.hal",
#   "purpose": "HAL resolver implementation",
#   "sections": [
#     {
#       "id": "halresolver",
#       "name": "HalResolver",
#       "anchor": "class-halresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the HAL open archive API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List

import httpx

from DocsToKG.ContentDownload.core import dedupe, normalize_doi
from DocsToKG.ContentDownload.resolvers.base import (
    ApiResolverBase,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
)

from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact


@register_v2("hal")
class HalResolver(ApiResolverBase):
    """Resolve publications from the HAL open archive."""

    name = "hal"

    def is_enabled(self, config: Any, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the work includes a DOI for HAL search.

        Args:
            config: Resolver configuration (unused for enablement).
            artifact: Work record being evaluated.

        Returns:
            bool: Whether the resolver is applicable to the work.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        client: httpx.Client,
        config: Any,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield HAL download URLs referencing ``artifact``.

        Args:
            client: HTTPX client to execute HTTP calls.
            config: Resolver configuration with request limits.
            artifact: Work metadata that supplies the DOI.

        Yields:
            ResolverResult: Candidate PDF URLs or diagnostic events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_DOI,
            )
            return
        data, error = self._request_json(
            client,
            "GET",
            "https://api.archives-ouvertes.fr/search/",
            config=config,
            params={"q": f"doiId_s:{doi}", "fl": "fileMain_s,file_s"},
        )
        if error:
            yield error
            return
        docs = (data.get("response") or {}).get("docs") or []
        urls: List[str] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            main = doc.get("fileMain_s")
            if isinstance(main, str):
                urls.append(main)
            files = doc.get("file_s")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, str):
                        urls.append(item)
        for url in dedupe(urls):
            if url.lower().endswith(".pdf"):
                yield ResolverResult(url=url, metadata={"source": "hal"})
