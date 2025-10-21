# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.unpaywall",
#   "purpose": "Unpaywall resolver implementation",
#   "sections": [
#     {
#       "id": "unpaywallresolver",
#       "name": "UnpaywallResolver",
#       "anchor": "class-unpaywallresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the Unpaywall API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

import httpx

from DocsToKG.ContentDownload.core import dedupe, normalize_doi
from DocsToKG.ContentDownload.urls import canonical_for_index

from .base import (
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    _fetch_unpaywall_data,
)
from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


@register_v2("unpaywall")
class UnpaywallResolver(RegisteredResolver):
    """Resolve PDFs via the Unpaywall API."""

    name = "unpaywall"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        """Return ``True`` when unpaywall credentials and a DOI are provided.

        Args:
            config: Resolver configuration containing the Unpaywall contact email.
            artifact: Work record offering DOI metadata.

        Returns:
            bool: Whether the resolver should attempt to fetch data.
        """
        return bool(config.unpaywall_email and artifact.doi)

    def iter_urls(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield Unpaywall-sourced PDF URLs for ``artifact``.

        Args:
            client: HTTPX client for outbound HTTP calls.
            config: Resolver configuration with API parameters.
            artifact: Work metadata used to build the lookup.

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
        try:
            data = _fetch_unpaywall_data(client, config, doi)
        except httpx.TimeoutException as exc:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.TIMEOUT,
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except httpx.TransportError as exc:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.CONNECTION_ERROR,
                metadata={"error": str(exc)},
            )
            return
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.HTTP_ERROR,
                http_status=status,
                metadata={"error_detail": str(exc)},
            )
            return
        except httpx.RequestError as exc:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.REQUEST_ERROR,
                metadata={"error": str(exc)},
            )
            return
        except ValueError as json_err:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.JSON_ERROR,
                metadata={"error_detail": str(json_err)},
            )
            return
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Unexpected error in Unpaywall resolver session path")
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        best = (data or {}).get("best_oa_location") or {}
        url = best.get("url_for_pdf")
        if url:
            candidates.append((url, {"source": "best_oa_location"}))

        for loc in (data or {}).get("oa_locations", []) or []:
            if not isinstance(loc, dict):
                continue
            url = loc.get("url_for_pdf")
            if url:
                candidates.append((url, {"source": "oa_location"}))

        unique_urls = dedupe([candidate_url for candidate_url, _ in candidates])
        for unique_url in unique_urls:
            for candidate_url, metadata in candidates:
                if candidate_url == unique_url:
                    # Explicitly compute canonical URL for RFC 3986 compliance and deduplication
                    try:
                        canonical_url = canonical_for_index(unique_url)
                    except Exception:
                        canonical_url = unique_url
                    yield ResolverResult(
                        url=unique_url,
                        canonical_url=canonical_url,
                        metadata=metadata,
                    )
                    break
