# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.semantic_scholar",
#   "purpose": "Semantic Scholar resolver implementation",
#   "sections": [
#     {
#       "id": "semanticscholarresolver",
#       "name": "SemanticScholarResolver",
#       "anchor": "class-semanticscholarresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the Semantic Scholar Graph API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

import httpx

from DocsToKG.ContentDownload.core import normalize_doi

from .base import (
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    _fetch_semantic_scholar_data,
)
from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


@register_v2("semantic_scholar")
class SemanticScholarResolver(RegisteredResolver):
    """Resolve PDFs via the Semantic Scholar Graph API."""

    name = "semantic_scholar"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        """Return ``True`` when a DOI is available for Semantic Scholar queries.

        Args:
            config: Resolver configuration containing API credentials.
            artifact: Work record being processed.

        Returns:
            bool: Whether the resolver should attempt to fetch metadata.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield Semantic Scholar hosted PDFs linked to ``artifact``.

        Args:
            client: HTTPX client for outbound HTTP calls.
            config: Resolver configuration with API keys and limits.
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
        try:
            data = _fetch_semantic_scholar_data(client, config, doi)
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
        except ValueError as exc:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.JSON_ERROR,
                metadata={"error_detail": str(exc)},
            )
            return
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Unexpected Semantic Scholar resolver error")
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return

        open_access = (data.get("openAccessPdf") or {}) if isinstance(data, dict) else {}
        url = open_access.get("url") if isinstance(open_access, dict) else None
        if url:
            yield ResolverResult(url=url, metadata={"source": "semantic-scholar"})
        else:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_OPENACCESS_PDF,
                metadata={"doi": doi},
            )
