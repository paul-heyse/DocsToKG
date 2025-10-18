# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.wayback",
#   "purpose": "Wayback Machine resolver",
#   "sections": [
#     {
#       "id": "wayback-resolver",
#       "name": "WaybackResolver",
#       "anchor": "class-waybackresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver that queries the Internet Archive Wayback Machine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

import requests as _requests

from .base import RegisteredResolver, ResolverEvent, ResolverEventReason, ResolverResult
from .base import request_with_retries

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


class WaybackResolver(RegisteredResolver):
    """Fallback resolver that queries the Internet Archive Wayback Machine."""

    name = "wayback"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        return bool(artifact.failed_pdf_urls)

    def iter_urls(
        self,
        session: _requests.Session,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        if not artifact.failed_pdf_urls:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_FAILED_URLS,
            )
            return

        for original in artifact.failed_pdf_urls:
            try:
                resp = request_with_retries(
                    session,
                    "get",
                    "https://archive.org/wayback/available",
                    params={"url": original},
                    timeout=config.get_timeout(self.name),
                    headers=config.polite_headers,
                )
            except _requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.TIMEOUT,
                    metadata={
                        "original": original,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                continue
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.CONNECTION_ERROR,
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except _requests.RequestException as exc:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.REQUEST_ERROR,
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Unexpected Wayback resolver error")
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                    metadata={
                        "original": original,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            if resp.status_code != 200:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.HTTP_ERROR,
                    http_status=resp.status_code,
                    metadata={
                        "original": original,
                        "error_detail": f"Wayback returned {resp.status_code}",
                    },
                )
                continue
            try:
                data = resp.json()
            except ValueError as json_err:
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.JSON_ERROR,
                    metadata={"original": original, "error_detail": str(json_err)},
                )
                continue
            closest = (data.get("archived_snapshots") or {}).get("closest") or {}
            if closest.get("available") and closest.get("url"):
                metadata = {"original": original}
                if closest.get("timestamp"):
                    metadata["timestamp"] = closest["timestamp"]
                yield ResolverResult(url=closest["url"], metadata=metadata)
