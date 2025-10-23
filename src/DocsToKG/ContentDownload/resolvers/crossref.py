# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.crossref",
#   "purpose": "Crossref resolver implementation",
#   "sections": [
#     {
#       "id": "crossrefresolver",
#       "name": "CrossrefResolver",
#       "anchor": "class-crossrefresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the Crossref metadata API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote

import httpx

from DocsToKG.ContentDownload.core import normalize_doi
from DocsToKG.ContentDownload.resolvers.base import (
    ApiResolverBase,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
)
from DocsToKG.ContentDownload.urls import canonical_for_index

from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact


@register_v2("crossref")
class CrossrefResolver(ApiResolverBase):
    """Resolve candidate URLs from the Crossref metadata API."""

    name = "crossref"
    api_display_name = "Crossref"

    def is_enabled(self, config: Any, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the work exposes a DOI Crossref can query.

        Args:
            config: Resolver configuration (unused, required for signature).
            artifact: Work record under consideration.

        Returns:
            bool: Whether the resolver should attempt this work.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        client: httpx.Client,
        config: Any,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield PDF URLs referenced by Crossref metadata for ``artifact``.

        Args:
            client: HTTPX client for outbound HTTP calls.
            config: Resolver configuration controlling behaviour.
            artifact: Work record containing DOI and other metadata.

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
        email = config.mailto or config.unpaywall_email
        params: Optional[Dict[str, str]] = {"mailto": email} if email else None
        data, error = self._request_json(
            client,
            "GET",
            f"https://api.crossref.org/works/{quote(doi)}",
            config=config,
            params=params,
        )
        if error:
            yield error
            return

        message = (data.get("message") if isinstance(data, dict) else None) or {}
        link_section = message.get("link") or []
        if not isinstance(link_section, list):
            link_section = []

        pdf_candidates: List[Tuple[str, Dict[str, Any]]] = []
        for entry in link_section:
            if not isinstance(entry, dict):
                continue
            url = entry.get("URL") or entry.get("url")
            content_type = (entry.get("content-type") or entry.get("content_type") or "").lower()
            if not url or "application/pdf" not in content_type:
                continue
            pdf_candidates.append((url, entry))

        if not pdf_candidates:
            return

        def _score(candidate: Tuple[str, Dict[str, Any]]) -> int:
            _, meta = candidate
            version = (meta.get("content-version") or meta.get("content_version") or "").lower()
            return 1 if version == "vor" else 0

        pdf_candidates.sort(key=_score, reverse=True)

        seen: Set[str] = set()
        for url, meta in pdf_candidates:
            try:
                normalized = canonical_for_index(url)
            except Exception:
                normalized = url
            if normalized in seen:
                continue
            seen.add(normalized)
            yield ResolverResult(
                url=url,
                canonical_url=normalized,
                metadata={
                    "source": "crossref",
                    "content-version": meta.get("content-version") or meta.get("content_version"),
                    "content_type": meta.get("content-type") or meta.get("content_type"),
                },
            )
