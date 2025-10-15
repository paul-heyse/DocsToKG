"""Resolver targeting the Open Science Framework API for preprint downloads."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, List

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.utils import dedupe, normalize_doi

from ..types import ResolverConfig, ResolverResult

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


LOGGER = logging.getLogger(__name__)


class OsfResolver:
    """Resolve artefacts hosted on the Open Science Framework.

    Attributes:
        name: Resolver identifier shared with the pipeline.

    Examples:
        >>> resolver = OsfResolver()
        >>> resolver.name
        'osf'
    """

    name = "osf"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when the artifact has a DOI for OSF lookup.

        Args:
            config: Resolver configuration providing OSF request details.
            artifact: Work artifact potentially containing a DOI value.

        Returns:
            Boolean indicating whether the resolver should run.
        """

        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate download URLs from the OSF API.

        Args:
            session: HTTP session available for API requests.
            config: Resolver configuration including polite headers and timeouts.
            artifact: Work artifact describing the record under consideration.

        Returns:
            Iterable of resolver results containing candidate download URLs.
        """

        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://api.osf.io/v2/preprints/",
                params={"filter[doi]": doi},
                headers=config.polite_headers,
                timeout=config.get_timeout(self.name),
            )
        except requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except requests.RequestException as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"error": str(exc)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error in OSF resolver")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        if resp.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=resp.status_code,
                metadata={"error_detail": f"OSF API returned {resp.status_code}"},
            )
            return
        try:
            data = resp.json()
        except ValueError as json_err:
            preview = resp.text[:200] if hasattr(resp, "text") else ""
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(json_err), "content_preview": preview},
            )
            return
        urls: List[str] = []
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


__all__ = ["OsfResolver"]
