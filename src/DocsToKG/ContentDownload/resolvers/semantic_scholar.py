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
import re
from typing import TYPE_CHECKING, Iterable

import requests as _requests

from DocsToKG.ContentDownload.core import normalize_doi

from .base import (
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    _fetch_semantic_scholar_data,
)

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


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
        session: _requests.Session,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield Semantic Scholar hosted PDFs linked to ``artifact``.

        Args:
            session: Requests session for outbound HTTP calls.
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
            data = _fetch_semantic_scholar_data(session, config, doi)
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
        except _requests.HTTPError as exc:
            status = None
            response_obj = getattr(exc, "response", None)
            if response_obj is not None and hasattr(response_obj, "status_code"):
                status = response_obj.status_code
            if status is None:
                match = re.search(r"(\d{3})", str(exc))
                if match:
                    status = int(match.group(1))
            yield ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.HTTP_ERROR,
                http_status=status,
                metadata={"error_detail": str(exc)},
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
