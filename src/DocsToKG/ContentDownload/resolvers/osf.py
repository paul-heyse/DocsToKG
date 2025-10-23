# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.osf",
#   "purpose": "OSF resolver implementation",
#   "sections": [
#     {
#       "id": "osfresolver",
#       "name": "OsfResolver",
#       "anchor": "class-osfresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the Open Science Framework API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

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


@register_v2("osf")
class OsfResolver(ApiResolverBase):
    """Resolve artefacts hosted on the Open Science Framework."""

    name = "osf"

    def is_enabled(self, config: Any, artifact: WorkArtifact) -> bool:
        """Return ``True`` when a DOI is available for OSF lookups.

        Args:
            config: Resolver configuration (unused for enablement).
            artifact: Work record describing the document.

        Returns:
            bool: Whether the resolver should attempt the work.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        client: httpx.Client,
        config: Any,
        artifact: WorkArtifact,
    ) -> Iterable[ResolverResult]:
        """Yield OSF download URLs corresponding to ``artifact``.

        Args:
            client: HTTPX client for HTTP operations.
            config: Resolver configuration managing limits.
            artifact: Work metadata providing DOI information.

        Yields:
            ResolverResult: Candidate download URLs or skip events.
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
            "https://api.osf.io/v2/preprints/",
            config=config,
            params={"filter[doi]": doi},
        )
        if error:
            yield error
            return
        urls: list[str] = []
        for item in data.get("data", []) or []:
            if not isinstance(item, dict):
                continue
            links = item.get("links") or {}
            download = links.get("download")
            if isinstance(download, str):
                urls.append(download)
            attributes = item.get("attributes") or {}
            primary = attributes.get("primary_file") or {}
            if isinstance(primary, dict):
                file_links = primary.get("links") or {}
                href = file_links.get("download")
                if isinstance(href, str):
                    urls.append(href)
        for url in dedupe(urls):
            yield ResolverResult(url=url, metadata={"source": "osf"})
