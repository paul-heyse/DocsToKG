# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.zenodo",
#   "purpose": "Zenodo resolver implementation",
#   "sections": [
#     {
#       "id": "zenodoresolver",
#       "name": "ZenodoResolver",
#       "anchor": "class-zenodoresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the Zenodo REST API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

import httpx

from DocsToKG.ContentDownload.core import normalize_doi

from .registry_v2 import register_v2

class ResolverResult:
    """Result from resolver attempt."""
    def __init__(self, url=None, referer=None, metadata=None, 
                 event=None, event_reason=None, **kwargs):
        self.url = url
        self.referer = referer
        self.metadata = metadata or {}
        self.event = event
        self.event_reason = event_reason
        for k, v in kwargs.items():
            setattr(self, k, v)



if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    

LOGGER = logging.getLogger(__name__)


@register_v2("zenodo")
class ZenodoResolver:
    """Resolve Zenodo records into downloadable open-access PDF URLs."""

    name = "zenodo"
    api_display_name = "Zenodo"

    def is_enabled(self, config: Any, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when a DOI is present to drive Zenodo lookups.

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
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield Zenodo hosted PDFs for the supplied work.

        Args:
            client: HTTPX client for issuing HTTP calls.
            config: Resolver configuration providing retry policies.
            artifact: Work metadata containing DOI information.

        Yields:
            ResolverResult: Candidate download URLs or diagnostic events.
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
            "https://zenodo.org/api/records/",
            config=config,
            params={"q": f'doi:"{doi}"', "size": 3, "sort": "mostrecent"},
        )
        if error:
            if error.event_reason is ResolverEventReason.CONNECTION_ERROR:
                error.event_reason = ResolverEventReason.REQUEST_ERROR
            yield error
            return

        hits = data.get("hits", {})
        if not isinstance(hits, dict):
            LOGGER.warning(
                "Zenodo API returned malformed hits payload: %s",
                type(hits).__name__,
            )
            return
        hits_list = hits.get("hits", [])
        if not isinstance(hits_list, list):
            LOGGER.warning(
                "Zenodo API returned malformed hits list: %s",
                type(hits_list).__name__,
            )
            return
        for record in hits_list or []:
            if not isinstance(record, dict):
                LOGGER.warning("Skipping malformed Zenodo record: %r", record)
                continue
            files = record.get("files", []) or []
            if not isinstance(files, list):
                LOGGER.warning("Skipping Zenodo record with invalid files payload: %r", files)
                continue
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    LOGGER.warning(
                        "Skipping non-dict Zenodo file entry in record %s",
                        record.get("id"),
                    )
                    continue
                file_type = (file_entry.get("type") or "").lower()
                file_key = (file_entry.get("key") or "").lower()
                if file_type == "pdf" or file_key.endswith(".pdf"):
                    links = file_entry.get("links")
                    url = links.get("self") if isinstance(links, dict) else None
                    if url:
                        yield ResolverResult(
                            url=url,
                            metadata={
                                "source": "zenodo",
                                "record_id": record.get("id"),
                                "filename": file_entry.get("key"),
                            },
                        )
