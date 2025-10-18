# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.openaire",
#   "purpose": "OpenAIRE resolver implementation",
#   "sections": [
#     {
#       "id": "openaire-resolver",
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
from typing import TYPE_CHECKING, Iterable, List

import requests as _requests

from DocsToKG.ContentDownload.core import dedupe, normalize_doi

from .base import (
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    _collect_candidate_urls,
    request_with_retries,
)

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


class OpenAireResolver(RegisteredResolver):
    """Resolve URLs using the OpenAIRE API."""

    name = "openaire"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        return artifact.doi is not None

    def iter_urls(
        self,
        session: _requests.Session,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
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
                session,
                "get",
                "https://api.openaire.eu/search/publications",
                params={"doi": doi},
                headers=config.polite_headers,
                timeout=config.get_timeout(self.name),
            )
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.TIMEOUT,
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.CONNECTION_ERROR,
                metadata={"error": str(exc)},
            )
            return
        except _requests.RequestException as exc:
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
        if resp.status_code != 200:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.HTTP_ERROR,
                http_status=resp.status_code,
                metadata={"error_detail": f"OpenAIRE API returned {resp.status_code}"},
            )
            return
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
        results: List[str] = []
        _collect_candidate_urls(data, results)
        for url in dedupe(results):
            if url.lower().endswith(".pdf"):
                yield ResolverResult(url=url, metadata={"source": "openaire"})
