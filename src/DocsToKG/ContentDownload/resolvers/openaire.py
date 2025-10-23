# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.openaire",
#   "purpose": "OpenAIRE resolver implementation",
#   "sections": [
#     {
#       "id": "openaireresolver",
#       "name": "OpenAireResolver",
#       "anchor": "class-openaireresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver implementation for the OpenAIRE API."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import httpx

from DocsToKG.ContentDownload.core import dedupe, normalize_doi
from DocsToKG.ContentDownload.networking import BreakerOpenError
from DocsToKG.ContentDownload.resolvers.base import (
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    _collect_candidate_urls,
    request_with_retries,
)

from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact


LOGGER = logging.getLogger(__name__)


@register_v2("openaire")
class OpenAireResolver:
    """Resolve URLs using the OpenAIRE API."""

    name = "openaire"

    def is_enabled(self, config: Any, artifact: WorkArtifact) -> bool:
        """Return ``True`` when the work includes a DOI for OpenAIRE queries.

        Args:
            config: Resolver configuration (unused for enablement).
            artifact: Work record under evaluation.

        Returns:
            bool: Whether this resolver should attempt the work.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        client: httpx.Client,
        config: Any,
        artifact: WorkArtifact,
    ) -> Iterable[ResolverResult]:
        """Yield OpenAIRE URLs that point to downloadable PDFs.

        Args:
            client: HTTPX client for issuing HTTP requests.
            config: Resolver configuration providing timeouts and headers.
            artifact: Work metadata containing the DOI lookup.

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
            resp = request_with_retries(
                client,
                "get",
                "https://api.openaire.eu/search/publications",
                role="metadata",
                params={"doi": doi},
                headers=config.polite_headers,
                timeout=config.get_timeout(self.name),
                retry_after_cap=config.retry_after_cap,
            )
            resp.raise_for_status()
        except BreakerOpenError as exc:
            meta = {"doi": doi, "error": str(exc)}
            breaker_meta = getattr(exc, "breaker_meta", None)
            if isinstance(breaker_meta, Mapping):
                meta["breaker"] = dict(breaker_meta)
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.BREAKER_OPEN,
                metadata=meta,
            )
            return
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
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Unexpected error in OpenAIRE resolver")
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        try:
            try:
                data = resp.json()
            except ValueError:
                try:
                    data = json.loads(resp.text)
                except ValueError as json_err:
                    preview = resp.text[:200] if hasattr(resp, "text") else ""
                    yield ResolverResult(
                        url=None,
                        event=ResolverEvent.ERROR,
                        event_reason=ResolverEventReason.JSON_ERROR,
                        metadata={"error_detail": str(json_err), "content_preview": preview},
                    )
                    return
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Unexpected JSON error in OpenAIRE resolver")
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                )
                return
            results: list[str] = []
            _collect_candidate_urls(data, results)
            for url in dedupe(results):
                if url.lower().endswith(".pdf"):
                    yield ResolverResult(url=url, metadata={"source": "openaire"})
        finally:
            resp.close()
