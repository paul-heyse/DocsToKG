# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.wayback",
#   "purpose": "Wayback Machine resolver with CDX-first discovery algorithm",
#   "sections": [
#     {
#       "id": "waybackresolver",
#       "name": "WaybackResolver",
#       "anchor": "class-waybackresolver",
#       "kind": "class"
#     },
#     {
#       "id": "wayback-override-http",
#       "name": "_override_wayback_http",
#       "anchor": "function-wayback-override-http",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-discovery",
#       "name": "_discover_snapshots",
#       "anchor": "function-wayback-discovery",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-availability",
#       "name": "_check_availability",
#       "anchor": "function-wayback-availability",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-cdx",
#       "name": "_query_cdx",
#       "anchor": "function-wayback-cdx",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-html-parse",
#       "name": "_parse_html_for_pdf",
#       "anchor": "function-wayback-html-parse",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-verify-pdf",
#       "name": "_verify_pdf_snapshot",
#       "anchor": "function-wayback-verify-pdf",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver that queries the Internet Archive Wayback Machine with CDX-first discovery."""

from __future__ import annotations

import itertools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import httpx

from DocsToKG.ContentDownload.networking import BreakerOpenError, request_with_retries
from DocsToKG.ContentDownload.urls import canonical_for_index, canonical_for_request

from DocsToKG.ContentDownload.resolvers.base import (
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    find_pdf_via_anchor,
    find_pdf_via_link,
    find_pdf_via_meta,
)
from .registry_v2 import register_v2

try:  # pragma: no cover - optional dependency
    from waybackpy import WaybackMachineAvailabilityAPI, WaybackMachineCDXServerAPI
    from waybackpy.exceptions import ArchiveNotInAvailabilityAPIResponse, WaybackError
except ImportError:  # pragma: no cover - handled gracefully at runtime
    WaybackMachineAvailabilityAPI = None  # type: ignore[assignment]
    WaybackMachineCDXServerAPI = None  # type: ignore[assignment]
    ArchiveNotInAvailabilityAPIResponse = None  # type: ignore[assignment]
    WaybackError = Exception  # type: ignore[assignment]

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


class _HTTPXResponseAdapter:
    """Adapter that mimics the subset of ``requests.Response`` used by waybackpy."""

    __slots__ = ("_response", "status_code", "headers", "text", "content", "url")

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status_code = response.status_code
        self.headers = dict(response.headers)
        self.text = response.text
        self.content = response.content
        self.url = str(response.url)

    def json(self) -> Any:
        return self._response.json()

    def close(self) -> None:
        self._response.close()


@register_v2("wayback")
class WaybackResolver:
    """Fallback resolver that queries the Internet Archive Wayback Machine."""

    name = "wayback"

    def is_enabled(self, config: Any, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when prior resolver attempts have failed."""

        return bool(artifact.failed_pdf_urls)

    def iter_urls(
        self,
        client: httpx.Client,
        config: Any,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield archived URLs discovered via availability/CDX lookups."""

        if not artifact.failed_pdf_urls:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_FAILED_URLS,
                metadata={"reason": "no-failed-urls"},
            )
            return

        wayback_config = getattr(config, "wayback_config", {})
        year_window = int(wayback_config.get("year_window", 2))
        max_snapshots = int(wayback_config.get("max_snapshots", 8))
        min_pdf_bytes = int(wayback_config.get("min_pdf_bytes", 4096))
        html_parse_enabled = bool(wayback_config.get("html_parse", True))
        availability_first = bool(wayback_config.get("availability_first", True))

        for original_url in artifact.failed_pdf_urls:
            try:
                canonical_url = canonical_for_index(original_url)
                publication_year = getattr(artifact, "publication_year", None)

                snapshot_url, metadata = self._discover_snapshots(
                    client,
                    config,
                    original_url,
                    canonical_url,
                    publication_year,
                    availability_first,
                    year_window,
                    max_snapshots,
                    min_pdf_bytes,
                    html_parse_enabled,
                )

                if snapshot_url:
                    # When coming from HTML parsing we may need to canonicalise the URL again.
                    resolved_url = canonical_for_request(snapshot_url, role="artifact")
                    yield ResolverResult(
                        url=resolved_url,
                        metadata={
                            "source": "wayback",
                            "original": original_url,
                            "canonical": canonical_url,
                            **metadata,
                        },
                    )
                else:
                    yield ResolverResult(
                        url=None,
                        event=ResolverEvent.SKIPPED,
                        event_reason=ResolverEventReason.NO_OPENACCESS_PDF,
                        metadata={
                            "source": "wayback",
                            "original": original_url,
                            "canonical": canonical_url,
                            "reason": "no_snapshot",
                            **metadata,
                        },
                    )

            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Unexpected Wayback resolver error for %s", original_url)
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                    metadata={
                        "source": "wayback",
                        "original": original_url,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

    def _discover_snapshots(
        self,
        client: httpx.Client,
        config: Any,
        original_url: str,
        canonical_url: str,
        publication_year: Optional[int],
        availability_first: bool,
        year_window: int,
        max_snapshots: int,
        min_pdf_bytes: int,
        html_parse_enabled: bool,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Discover a suitable snapshot using availability followed by CDX search."""

        metadata: Dict[str, Any] = {
            "discovery_method": "none",
            "original": original_url,
        }

        if availability_first:
            availability_url, availability_metadata = self._check_availability(
                client, config, original_url, min_pdf_bytes
            )
            metadata.update(availability_metadata)
            if availability_url:
                ok, pdf_metadata = self._verify_pdf_snapshot(
                    client, config, availability_url, min_pdf_bytes
                )
                metadata.update(pdf_metadata)
                if ok:
                    metadata["discovery_method"] = "availability"
                    return availability_url, metadata

        cdx_snapshots = self._query_cdx(
            client,
            config,
            original_url,
            publication_year,
            year_window,
            max_snapshots,
        )

        for snapshot in cdx_snapshots:
            snapshot_url = snapshot.get("archive_url")
            if not snapshot_url:
                continue

            snapshot_mimetype = (snapshot.get("mimetype") or "").lower()
            snapshot_status = snapshot.get("statuscode")
            snapshot_timestamp = snapshot.get("timestamp")
            metadata.update(
                {
                    "memento_ts": snapshot_timestamp,
                    "statuscode": snapshot_status,
                    "mimetype": snapshot_mimetype,
                }
            )

            if snapshot_timestamp and publication_year:
                try:
                    metadata["distance_years"] = abs(
                        int(snapshot_timestamp[:4]) - int(publication_year)
                    )
                except ValueError:
                    metadata["distance_years"] = None

            if snapshot_mimetype == "application/pdf" and snapshot_status == "200":
                ok, pdf_metadata = self._verify_pdf_snapshot(
                    client, config, snapshot_url, min_pdf_bytes
                )
                metadata.update(pdf_metadata)
                if ok:
                    metadata["discovery_method"] = "cdx_pdf_direct"
                    return snapshot_url, metadata

            if html_parse_enabled and snapshot_mimetype in {"text/html", "application/xhtml+xml"}:
                pdf_url, html_metadata = self._parse_html_for_pdf(client, config, snapshot_url)
                metadata.update(html_metadata)
                if pdf_url:
                    ok, pdf_metadata = self._verify_pdf_snapshot(
                        client, config, pdf_url, min_pdf_bytes
                    )
                    metadata.update(pdf_metadata)
                    if ok:
                        metadata["discovery_method"] = "cdx_html_parse"
                        metadata["discovered_pdf_url"] = pdf_url
                        return pdf_url, metadata

        return None, metadata

    def _check_availability(
        self,
        client: httpx.Client,
        config: Any,
        url: str,
        min_pdf_bytes: int,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Use the Wayback availability API as a fast-path for direct PDFs."""

        metadata: Dict[str, Any] = {
            "availability_checked": True,
            "availability_available": False,
        }

        if WaybackMachineAvailabilityAPI is None:  # pragma: no cover - optional dependency
            metadata["availability_error"] = "waybackpy-unavailable"
            return None, metadata

        try:
            with self._override_wayback_http(client, config):
                api = WaybackMachineAvailabilityAPI(
                    url,
                    user_agent=self._user_agent(config),
                    max_tries=1,
                )
                api.api_call_time_gap = 0
                availability_url = api.archive_url
                timestamp = api.timestamp()

                if availability_url:
                    metadata.update(
                        {
                            "availability_available": True,
                            "availability_timestamp": timestamp.strftime("%Y%m%d%H%M%S"),
                        }
                    )
                    return availability_url, metadata

        except ArchiveNotInAvailabilityAPIResponse as exc:  # pragma: no cover - defensive
            metadata["availability_error"] = str(exc)
        except Exception as exc:
            LOGGER.debug("Availability check failed for %s: %s", url, exc)
            metadata["availability_error"] = str(exc)

        return None, metadata

    def _query_cdx(
        self,
        client: httpx.Client,
        config: Any,
        url: str,
        publication_year: Optional[int],
        year_window: int,
        max_snapshots: int,
    ) -> List[Dict[str, Any]]:
        """Query the Wayback CDX API for snapshot metadata."""

        if WaybackMachineCDXServerAPI is None:  # pragma: no cover - optional dependency
            LOGGER.debug("waybackpy not available; skipping CDX query for %s", url)
            return []

        snapshots: List[Dict[str, Any]] = []

        try:
            start_ts: Optional[str] = None
            end_ts: Optional[str] = None
            if publication_year:
                start_year = max(1994, publication_year - year_window)
                end_year = publication_year + year_window
                start_ts = f"{start_year:04d}0101000000"
                end_ts = f"{end_year:04d}1231235959"

            with self._override_wayback_http(client, config):
                cdx_api = WaybackMachineCDXServerAPI(
                    url,
                    user_agent=self._user_agent(config),
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    filters=["statuscode:200"],
                    limit=str(max(1, max_snapshots)),
                )
                for snapshot in itertools.islice(cdx_api.snapshots(), max_snapshots):
                    snapshots.append(
                        {
                            "archive_url": getattr(snapshot, "archive_url", ""),
                            "timestamp": getattr(snapshot, "timestamp", ""),
                            "mimetype": getattr(snapshot, "mimetype", ""),
                            "statuscode": getattr(snapshot, "statuscode", ""),
                        }
                    )

        except WaybackError as exc:  # pragma: no cover - defensive logging
            LOGGER.debug("CDX query returned an error for %s: %s", url, exc)
        except Exception as exc:
            LOGGER.debug("CDX query failed for %s: %s", url, exc)

        snapshots.sort(key=lambda snap: snap.get("timestamp", ""), reverse=True)
        return snapshots

    def _parse_html_for_pdf(
        self,
        client: httpx.Client,
        config: Any,
        html_url: str,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Fetch an archived HTML page and attempt to recover a PDF link from it."""

        metadata: Dict[str, Any] = {
            "html_snapshot_url": html_url,
        }

        try:
            resp = request_with_retries(
                client,
                "get",
                html_url,
                role="landing",
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            resp.raise_for_status()

            metadata.update(
                {
                    "html_http_status": resp.status_code,
                    "html_content_type": resp.headers.get("content-type"),
                    "html_bytes": len(resp.content),
                    "from_cache": bool(
                        resp.extensions.get("docs_network_meta", {}).get("from_cache")
                    ),
                }
            )

            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None, metadata

            html_content = resp.text

            try:
                from bs4 import BeautifulSoup
            except ImportError:  # pragma: no cover - optional dependency
                metadata["html_parse_error"] = "beautifulsoup-unavailable"
                return None, metadata

            soup = BeautifulSoup(html_content, "html.parser")

            pdf_url = find_pdf_via_meta(soup, html_url)
            if pdf_url:
                metadata["pdf_discovery_method"] = "meta"
                return canonical_for_request(pdf_url, role="artifact"), metadata

            pdf_url = find_pdf_via_link(soup, html_url)
            if pdf_url:
                metadata["pdf_discovery_method"] = "link"
                return canonical_for_request(pdf_url, role="artifact"), metadata

            pdf_url = find_pdf_via_anchor(soup, html_url)
            if pdf_url:
                metadata["pdf_discovery_method"] = "anchor"
                return canonical_for_request(pdf_url, role="artifact"), metadata

        except BreakerOpenError as exc:
            metadata["breaker_error"] = str(exc)
            breaker_meta = getattr(exc, "breaker_meta", None)
            if isinstance(breaker_meta, Mapping):
                metadata["breaker"] = dict(breaker_meta)
        except Exception as exc:
            LOGGER.debug("HTML parsing failed for %s: %s", html_url, exc)
            metadata["html_parse_error"] = str(exc)
        finally:
            if "resp" in locals():
                resp.close()

        return None, metadata

    def _verify_pdf_snapshot(
        self,
        client: httpx.Client,
        config: Any,
        url: str,
        min_bytes: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sanity-check that an archived URL points to a plausible PDF."""

        metadata: Dict[str, Any] = {
            "candidate_url": url,
        }

        try:
            resp = request_with_retries(
                client,
                "head",
                url,
                role="artifact",
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            resp.raise_for_status()

            metadata["head_status"] = resp.status_code
            metadata["content_type"] = resp.headers.get("content-type")
            metadata["content_length"] = resp.headers.get("content-length")

            content_type = (resp.headers.get("content-type") or "").lower()
            if "application/pdf" not in content_type:
                return False, metadata

            if metadata["content_length"]:
                try:
                    size = int(metadata["content_length"])
                    metadata["min_bytes_pass"] = size >= min_bytes
                    if size < min_bytes:
                        return False, metadata
                except ValueError:
                    metadata["min_bytes_pass"] = None

            if metadata["content_length"] and metadata["content_length"].isdigit():
                size = int(metadata["content_length"])
                if size < max(min_bytes * 2, 8192):
                    pdf_resp = request_with_retries(
                        client,
                        "get",
                        url,
                        role="artifact",
                        timeout=config.get_timeout(self.name),
                        headers=config.polite_headers,
                        retry_after_cap=config.retry_after_cap,
                    )
                    pdf_resp.raise_for_status()
                    metadata["pdf_signature"] = pdf_resp.content[:8].startswith(b"%PDF-")
                    pdf_resp.close()
                    if not metadata["pdf_signature"]:
                        return False, metadata

            metadata.setdefault("pdf_signature", True)
            return True, metadata

        except BreakerOpenError as exc:
            metadata["breaker_error"] = str(exc)
            breaker_meta = getattr(exc, "breaker_meta", None)
            if isinstance(breaker_meta, Mapping):
                metadata["breaker"] = dict(breaker_meta)
            return False, metadata
        except Exception as exc:
            LOGGER.debug("PDF verification failed for %s: %s", url, exc)
            metadata["verification_error"] = str(exc)
            return False, metadata
        finally:
            if "resp" in locals():
                resp.close()

    def _user_agent(self, config: Any) -> str:
        header = config.polite_headers.get("User-Agent")
        if header:
            return header
        return "DocsToKG-WaybackResolver/1.0"

    @contextmanager
    def _override_wayback_http(
        self,
        client: httpx.Client,
        config: Any,
    ) -> Iterator[None]:
        """Monkeypatch waybackpy HTTP helpers so they use our shared HTTPX client."""

        if WaybackMachineAvailabilityAPI is None or WaybackMachineCDXServerAPI is None:
            yield
            return

        from waybackpy import availability_api, cdx_utils  # type: ignore

        timeout = config.get_timeout(self.name)
        retry_after_cap = config.retry_after_cap
        default_headers = dict(config.polite_headers)

        original_cdx_get = cdx_utils.get_response
        original_availability_get = availability_api.requests.get

        def _fetch(
            target_url: str,
            *,
            params: Optional[Dict[str, Any]] = None,
            req_headers: Optional[Dict[str, str]] = None,
        ) -> _HTTPXResponseAdapter:
            response = request_with_retries(
                client,
                "get",
                target_url,
                role="metadata",
                params=params,
                headers=req_headers if req_headers is not None else default_headers,
                timeout=timeout,
                retry_after_cap=retry_after_cap,
            )
            return _HTTPXResponseAdapter(response)

        def _cdx_get_response(
            target_url: str,
            req_headers: Optional[Dict[str, str]] = None,
            retries: int = 5,  # noqa: D417 - signature required by waybackpy
            backoff_factor: float = 0.5,
        ) -> _HTTPXResponseAdapter:
            del retries, backoff_factor
            return _fetch(target_url, req_headers=req_headers)

        def _availability_get(
            target_url: str,
            params: Optional[Dict[str, Any]] = None,
            req_headers: Optional[Dict[str, str]] = None,
        ) -> _HTTPXResponseAdapter:
            return _fetch(target_url, params=params, req_headers=req_headers)

        cdx_utils.get_response = _cdx_get_response  # type: ignore[assignment]
        availability_api.requests.get = _availability_get  # type: ignore[assignment]
        try:
            yield
        finally:
            cdx_utils.get_response = original_cdx_get  # type: ignore[assignment]
            availability_api.requests.get = original_availability_get  # type: ignore[assignment]
