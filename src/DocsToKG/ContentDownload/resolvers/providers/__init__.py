"""
Resolver Providers

This module consolidates all resolver provider implementations into a single
registry-backed module. Providers register themselves upon subclassing the
``RegisteredResolver`` base, allowing ``default_resolvers`` to materialise the
prioritised resolver stack without manual bookkeeping.

The consolidation centralises shared imports, caching helpers, and optional
third-party dependencies while preserving the public API expected by the
resolver pipeline and caching utilities.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
from urllib.parse import quote, urljoin, urlparse

import requests

from DocsToKG.ContentDownload.network import request_with_retries
from DocsToKG.ContentDownload.utils import (
    dedupe,
    normalize_doi,
    normalize_pmcid,
    strip_prefix,
)

from ..headers import headers_cache_key
from ..types import DEFAULT_RESOLVER_ORDER, Resolver, ResolverConfig, ResolverResult

_headers_cache_key = headers_cache_key

try:  # Optional dependency guarded at runtime
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    BeautifulSoup = None

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


LOGGER = logging.getLogger(__name__)


class ResolverRegistry:
    """Registry tracking resolver classes by their ``name`` attribute."""

    _providers: Dict[str, Type[Resolver]] = {}

    @classmethod
    def register(cls, resolver_cls: Type[Resolver]) -> Type[Resolver]:
        """Register a resolver class under its declared ``name`` attribute.
        
        Args:
            resolver_cls: Resolver implementation to register.
        
        Returns:
            Type[Resolver]: The registered resolver class for chaining.
        """
        name = getattr(resolver_cls, "name", None)
        if not name:
            raise ValueError(f"Resolver class {resolver_cls.__name__} missing 'name' attribute")
        cls._providers[name] = resolver_cls
        return resolver_cls

    @classmethod
    def create_default(cls) -> List[Resolver]:
        """Instantiate resolver instances in priority order.
        
        Args:
            None
        
        Returns:
            List[Resolver]: Resolver instances ordered by default priority.
        """
        instances: List[Resolver] = []
        seen: set[str] = set()
        for name in DEFAULT_RESOLVER_ORDER:
            resolver_cls = cls._providers.get(name)
            if resolver_cls is not None:
                instances.append(resolver_cls())
                seen.add(name)
        for name in sorted(cls._providers):
            if name not in seen:
                instances.append(cls._providers[name]())
        return instances


class RegisteredResolver:
    """Mixin ensuring subclasses register with :class:`ResolverRegistry`."""

    def __init_subclass__(cls, register: bool = True, **kwargs: Any) -> None:  # type: ignore[override]
        super().__init_subclass__(**kwargs)
        if register:
            ResolverRegistry.register(cls)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helper utilities reused across providers


def _absolute_url(base: str, href: str) -> str:
    """Resolve relative ``href`` values against ``base`` to obtain absolute URLs."""

    parsed = urlparse(href)
    if parsed.scheme and parsed.netloc:
        return href
    return urljoin(base, href)


def _collect_candidate_urls(node: object, results: List[str]) -> None:
    """Recursively collect HTTP(S) URLs from nested response payloads."""

    if isinstance(node, dict):
        for value in node.values():
            _collect_candidate_urls(value, results)
    elif isinstance(node, list):
        for item in node:
            _collect_candidate_urls(item, results)
    elif isinstance(node, str) and node.lower().startswith("http"):
        results.append(node)


@lru_cache(maxsize=1000)
def _fetch_crossref_data(
    doi: str,
    mailto: Optional[str],
    timeout: float,
    headers_key: Tuple[Tuple[str, str], ...],
) -> Dict[str, Any]:
    """Retrieve Crossref metadata for ``doi`` with polite header caching."""

    headers = dict(headers_key)
    params = {"mailto": mailto} if mailto else None
    response = requests.get(
        f"https://api.crossref.org/works/{quote(doi)}",
        params=params,
        timeout=timeout,
        headers=headers,
    )
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1000)
def _fetch_unpaywall_data(
    doi: str,
    email: Optional[str],
    timeout: float,
    headers_key: Tuple[Tuple[str, str], ...],
) -> Dict[str, Any]:
    """Fetch Unpaywall metadata for ``doi`` using polite caching."""

    headers = dict(headers_key)
    response = requests.get(
        f"https://api.unpaywall.org/v2/{quote(doi)}",
        params={"email": email} if email else None,
        timeout=timeout,
        headers=headers,
    )
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1000)
def _fetch_semantic_scholar_data(
    doi: str,
    api_key: Optional[str],
    timeout: float,
    headers_key: Tuple[Tuple[str, str], ...],
) -> Dict[str, Any]:
    """Fetch Semantic Scholar Graph API metadata for ``doi`` with caching."""

    headers = dict(headers_key)
    if api_key:
        headers["x-api-key"] = api_key
    response = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}",
        params={"fields": "title,openAccessPdf"},
        timeout=timeout,
        headers=headers,
    )
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Resolver implementations


class ArxivResolver(RegisteredResolver):
    """Resolve arXiv preprints using arXiv identifier lookups."""

    name = "arxiv"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.arxiv_id)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        arxiv_id = artifact.arxiv_id
        if not arxiv_id:
            yield ResolverResult(url=None, event="skipped", event_reason="no-arxiv-id")
            return
        arxiv_id = strip_prefix(arxiv_id, "arxiv:")
        yield ResolverResult(
            url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            metadata={"identifier": arxiv_id},
        )


class CoreResolver(RegisteredResolver):
    """Resolve PDFs using the CORE API."""

    name = "core"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(config.core_api_key and artifact.doi)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        headers = dict(config.polite_headers)
        headers["Authorization"] = f"Bearer {config.core_api_key}"
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://api.core.ac.uk/v3/search/works",
                params={"q": f'doi:"{doi}"', "page": 1, "pageSize": 3},
                headers=headers,
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
            LOGGER.exception("Unexpected error in CORE resolver")
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
                metadata={"error_detail": f"CORE API returned {resp.status_code}"},
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
        for hit in data.get("results", []) or []:
            if not isinstance(hit, dict):
                continue
            url = hit.get("downloadUrl") or hit.get("pdfDownloadLink")
            if url:
                yield ResolverResult(url=url, metadata={"source": "core"})
            for entry in hit.get("fullTextLinks") or []:
                if isinstance(entry, dict):
                    href = entry.get("url") or entry.get("link")
                    if href and href.lower().endswith(".pdf"):
                        yield ResolverResult(url=href, metadata={"source": "core"})


class CrossrefResolver(RegisteredResolver):
    """Resolve candidate URLs from the Crossref metadata API."""

    name = "crossref"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        email = config.mailto or config.unpaywall_email
        endpoint = f"https://api.crossref.org/works/{quote(doi)}"
        params = {"mailto": email} if email else None
        headers = dict(config.polite_headers)
        data: Optional[Dict[str, Any]] = None
        if hasattr(session, "get"):
            response: Optional[requests.Response] = None
            try:
                response = request_with_retries(
                    session,
                    "GET",
                    endpoint,
                    params=params,
                    timeout=config.get_timeout(self.name),
                    headers=headers,
                    allow_redirects=True,
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
                LOGGER.exception("Unexpected error in Crossref resolver")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                )
                return

            status = response.status_code if response is not None else 200
            if response is not None and status != 200:
                response.close()
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=status,
                    metadata={"error_detail": f"Crossref API returned {status}"},
                )
                return

            try:
                if response is not None:
                    data = response.json()
            except ValueError as json_err:
                preview = (
                    response.text[:200]
                    if response is not None and hasattr(response, "text")
                    else ""
                )
                if response is not None:
                    response.close()
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
                    metadata={"error_detail": str(json_err), "content_preview": preview},
                )
                return
            finally:
                if response is not None:
                    response.close()
        else:
            try:
                data = _fetch_crossref_data(
                    doi,
                    email,
                    config.get_timeout(self.name),
                    headers_cache_key(config.polite_headers),
                )
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=status,
                    metadata={"error_detail": f"Crossref HTTPError: {status}"},
                )
                return
            except requests.RequestException as exc:  # pragma: no cover - network errors
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"error": str(exc)},
                )
                return
            except ValueError as json_err:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
                    metadata={"error_detail": str(json_err)},
                )
                return
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected cached request error in Crossref resolver")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                )
                return

        message = ((data or {}).get("message") or {}) if isinstance(data, dict) else {}
        link_section = message.get("link") or []
        if not isinstance(link_section, list):
            link_section = []

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for entry in link_section:
            if not isinstance(entry, dict):
                continue
            url = entry.get("URL")
            content_type = entry.get("content-type")
            if url and (content_type or "").lower() in {"application/pdf", "text/html"}:
                candidates.append((url, {"content_type": content_type}))

        for url in dedupe([candidate_url for candidate_url, _ in candidates]):
            for candidate_url, metadata in candidates:
                if candidate_url == url:
                    yield ResolverResult(url=url, metadata=metadata)
                    break


class DoajResolver(RegisteredResolver):
    """Resolve Open Access links using the DOAJ API."""

    name = "doaj"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        headers = dict(config.polite_headers)
        if config.doaj_api_key:
            headers["X-API-KEY"] = config.doaj_api_key
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://doaj.org/api/v2/search/articles/",
                params={"pageSize": 3, "q": f'doi:"{doi}"'},
                headers=headers,
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
            LOGGER.exception("Unexpected error in DOAJ resolver")
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
                metadata={"error_detail": f"DOAJ API returned {resp.status_code}"},
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
        candidates = []
        for result in data.get("results", []) or []:
            bibjson = (result or {}).get("bibjson", {})
            for link in bibjson.get("link", []) or []:
                if not isinstance(link, dict):
                    continue
                url = link.get("url")
                if url and url.lower().endswith(".pdf"):
                    candidates.append(url)
        for url in dedupe(candidates):
            yield ResolverResult(url=url, metadata={"source": "doaj"})


class EuropePmcResolver(RegisteredResolver):
    """Resolve Open Access links via the Europe PMC REST API."""

    name = "europe_pmc"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": f'DOI:"{doi}"', "format": "json", "pageSize": 3},
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
            )
        except requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
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
            LOGGER.exception("Unexpected error in Europe PMC resolver")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        if resp.status_code != 200:
            LOGGER.warning("Europe PMC API returned %s for DOI %s", resp.status_code, doi)
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
        candidates: List[str] = []
        for result in (data.get("resultList", {}) or {}).get("result", []) or []:
            full_text = result.get("fullTextUrlList", {}) or {}
            for entry in full_text.get("fullTextUrl", []) or []:
                if not isinstance(entry, dict):
                    continue
                if (entry.get("documentStyle") or "").lower() != "pdf":
                    continue
                url = entry.get("url")
                if url:
                    candidates.append(url)
        for url in dedupe(candidates):
            yield ResolverResult(url=url, metadata={"source": "europe_pmc"})


class FigshareResolver(RegisteredResolver):
    """Resolve Figshare repository metadata into download URLs."""

    name = "figshare"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return

        headers = dict(config.polite_headers)
        headers.setdefault("Content-Type", "application/json")

        try:
            response = request_with_retries(
                session,
                "post",
                "https://api.figshare.com/v2/articles/search",
                json={
                    "search_for": f':doi: "{doi}"',
                    "page": 1,
                    "page_size": 3,
                },
                timeout=config.get_timeout(self.name),
                headers=headers,
            )
        except requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={
                    "timeout": config.get_timeout(self.name),
                    "error": str(exc),
                },
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
            LOGGER.exception("Unexpected error in Figshare resolver")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return

        if response.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=response.status_code,
                metadata={
                    "error_detail": f"Figshare API returned {response.status_code}",
                },
            )
            return

        try:
            articles = response.json()
        except ValueError as json_err:
            preview = response.text[:200] if hasattr(response, "text") else ""
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(json_err), "content_preview": preview},
            )
            return

        if not isinstance(articles, list):
            LOGGER.warning(
                "Figshare API returned non-list articles payload: %s", type(articles).__name__
            )
            return

        for article in articles:
            if not isinstance(article, dict):
                LOGGER.warning("Skipping malformed Figshare article: %r", article)
                continue
            files = article.get("files", []) or []
            if not isinstance(files, list):
                LOGGER.warning("Skipping Figshare article with invalid files payload: %r", files)
                continue
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    LOGGER.warning("Skipping non-dict Figshare file entry: %r", file_entry)
                    continue
                filename = (file_entry.get("name") or "").lower()
                download_url = file_entry.get("download_url")

                if filename.endswith(".pdf") and download_url:
                    yield ResolverResult(
                        url=download_url,
                        metadata={
                            "source": "figshare",
                            "article_id": article.get("id"),
                            "filename": file_entry.get("name"),
                        },
                    )


class HalResolver(RegisteredResolver):
    """Resolve publications from the HAL open archive."""

    name = "hal"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://api.archives-ouvertes.fr/search/",
                params={"q": f"doiId_s:{doi}", "fl": "fileMain_s,file_s"},
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
            LOGGER.exception("Unexpected error in HAL resolver")
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
                metadata={"error_detail": f"HAL API returned {resp.status_code}"},
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
        docs = (data.get("response") or {}).get("docs") or []
        urls: List[str] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            main = doc.get("fileMain_s")
            if isinstance(main, str):
                urls.append(main)
            files = doc.get("file_s")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, str):
                        urls.append(item)
        for url in dedupe(urls):
            if url.lower().endswith(".pdf"):
                yield ResolverResult(url=url, metadata={"source": "hal"})


class LandingPageResolver(RegisteredResolver):
    """Attempt to scrape landing pages when explicit PDFs are unavailable."""

    name = "landing_page"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.landing_urls)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        if BeautifulSoup is None:
            yield ResolverResult(url=None, event="skipped", event_reason="no-beautifulsoup")
            return
        for landing in artifact.landing_urls:
            try:
                resp = request_with_retries(
                    session,
                    "get",
                    landing,
                    headers=config.polite_headers,
                    timeout=config.get_timeout(self.name),
                )
            except requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={
                        "landing": landing,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                continue
            except requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"landing": landing, "error": str(exc)},
                )
                continue
            except requests.RequestException as exc:  # pragma: no cover - network errors
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"landing": landing, "error": str(exc)},
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected error scraping landing page")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={
                        "landing": landing,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                continue

            if resp.status_code != 200:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=resp.status_code,
                    metadata={
                        "landing": landing,
                        "error_detail": f"Landing page returned {resp.status_code}",
                    },
                )
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if meta and meta.get("content"):
                url = _absolute_url(landing, meta["content"].strip())
                yield ResolverResult(url=url, referer=landing, metadata={"pattern": "meta"})
                continue

            for link in soup.find_all("link"):
                rel = " ".join(link.get("rel") or []).lower()
                typ = (link.get("type") or "").lower()
                href = link.get("href") or ""
                if "alternate" in rel and "application/pdf" in typ and href:
                    url = _absolute_url(landing, href.strip())
                    yield ResolverResult(url=url, referer=landing, metadata={"pattern": "link"})
                    break

            for anchor in soup.find_all("a"):
                href = (anchor.get("href") or "").strip()
                if not href:
                    continue
                text = (anchor.get_text() or "").strip().lower()
                href_lower = href.lower()
                if href_lower.endswith(".pdf") or "pdf" in text:
                    candidate = _absolute_url(landing, href)
                    if candidate.lower().endswith(".pdf"):
                        yield ResolverResult(
                            url=candidate,
                            referer=landing,
                            metadata={"pattern": "anchor"},
                        )
                        break


class OpenAireResolver(RegisteredResolver):
    """Resolve URLs using the OpenAIRE API."""

    name = "openaire"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
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
            LOGGER.exception("Unexpected error in OpenAIRE resolver")
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
                    event="error",
                    event_reason="json-error",
                    metadata={"error_detail": str(json_err), "content_preview": preview},
                )
                return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected JSON error in OpenAIRE resolver")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        results: List[str] = []
        _collect_candidate_urls(data, results)
        for url in dedupe(results):
            if url.lower().endswith(".pdf"):
                yield ResolverResult(url=url, metadata={"source": "openaire"})


class OpenAlexResolver(RegisteredResolver):
    """Resolve OpenAlex work metadata into candidate download URLs."""

    name = "openalex"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.pdf_urls or artifact.open_access_url)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        candidates = list(dedupe(artifact.pdf_urls))
        if getattr(artifact, "open_access_url", None):
            candidates.append(artifact.open_access_url)

        if not candidates:
            yield ResolverResult(url=None, event="skipped", event_reason="no-openalex-urls")
            return

        for url in dedupe(candidates):
            if not url:
                continue
            yield ResolverResult(url=url, metadata={"source": "openalex_metadata"})


class OsfResolver(RegisteredResolver):
    """Resolve artefacts hosted on the Open Science Framework."""

    name = "osf"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
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


class PmcResolver(RegisteredResolver):
    """Resolve PubMed Central articles via identifiers and lookups."""

    name = "pmc"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.pmcid or artifact.pmid or artifact.doi)

    def _lookup_pmcids(
        self, session: requests.Session, identifiers: List[str], config: ResolverConfig
    ) -> List[str]:
        if not identifiers:
            return []
        try:
            resp = request_with_retries(
                session,
                "get",
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={
                    "ids": ",".join(identifiers),
                    "format": "json",
                    "tool": "docs-to-kg",
                    "email": config.unpaywall_email or "",
                },
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
            )
        except requests.Timeout as exc:
            LOGGER.debug("PMC ID lookup timed out: %s", exc)
            return []
        except requests.ConnectionError as exc:
            LOGGER.debug("PMC ID lookup connection error: %s", exc)
            return []
        except requests.RequestException as exc:
            LOGGER.debug("PMC ID lookup request error: %s", exc)
            return []
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error looking up PMC IDs")
            return []
        if resp.status_code != 200:
            LOGGER.debug("PMC ID lookup HTTP error status: %s", resp.status_code)
            return []
        try:
            data = resp.json()
        except ValueError as json_err:
            LOGGER.debug("PMC ID lookup JSON error: %s", json_err)
            return []
        results: List[str] = []
        for record in data.get("records", []) or []:
            pmcid = record.get("pmcid")
            if pmcid:
                results.append(normalize_pmcid(pmcid))
        return [pmc for pmc in results if pmc]

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        pmcids: List[str] = []
        if artifact.pmcid:
            pmcids.append(normalize_pmcid(artifact.pmcid))
        identifiers: List[str] = []
        doi = normalize_doi(artifact.doi)
        if doi:
            identifiers.append(doi)
        elif artifact.pmid:
            identifiers.append(artifact.pmid)
        if not pmcids:
            pmcids.extend(self._lookup_pmcids(session, identifiers, config))

        if not pmcids:
            yield ResolverResult(url=None, event="skipped", event_reason="no-pmcid")
            return

        for pmcid in dedupe(pmcids):
            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
            fallback_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
            try:
                resp = request_with_retries(
                    session,
                    "get",
                    oa_url,
                    timeout=config.get_timeout(self.name),
                    headers=config.polite_headers,
                )
            except requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={
                        "pmcid": pmcid,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"pmcid": pmcid, "error": str(exc)},
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except requests.RequestException as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"pmcid": pmcid, "error": str(exc)},
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected PMC OA lookup error")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={"pmcid": pmcid, "error": str(exc), "error_type": type(exc).__name__},
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            if resp.status_code != 200:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=resp.status_code,
                    metadata={
                        "pmcid": pmcid,
                        "error_detail": f"OA endpoint returned {resp.status_code}",
                    },
                )
                yield ResolverResult(
                    url=fallback_url,
                    metadata={"pmcid": pmcid, "source": "pdf-fallback"},
                )
                continue
            for match in re.finditer(r'href="([^"]+\.pdf)"', resp.text, flags=re.I):
                href = match.group(1)
                url = _absolute_url(oa_url, href)
                yield ResolverResult(
                    url=url,
                    metadata={"pmcid": pmcid, "source": "oa"},
                )
            yield ResolverResult(
                url=fallback_url,
                metadata={"pmcid": pmcid, "source": "pdf-fallback"},
            )


class SemanticScholarResolver(RegisteredResolver):
    """Resolve PDFs via the Semantic Scholar Graph API."""

    name = "semantic_scholar"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        try:
            data = _fetch_semantic_scholar_data(
                doi,
                config.semantic_scholar_api_key,
                config.get_timeout(self.name),
                headers_cache_key(config.polite_headers),
            )
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            detail = status if status is not None else "unknown"
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=status,
                metadata={"error_detail": f"Semantic Scholar HTTPError: {detail}"},
            )
            return
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
        except ValueError as json_err:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(json_err)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected Semantic Scholar resolver error")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
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
                event="skipped",
                event_reason="no-openaccess-pdf",
                metadata={"doi": doi},
            )


class UnpaywallResolver(RegisteredResolver):
    """Resolve PDFs via the Unpaywall API."""

    name = "unpaywall"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(config.unpaywall_email and artifact.doi)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return
        endpoint = f"https://api.unpaywall.org/v2/{quote(doi)}"
        headers = dict(config.polite_headers)
        params = {"email": config.unpaywall_email} if config.unpaywall_email else None
        if hasattr(session, "get"):
            try:
                response = session.get(
                    endpoint,
                    params=params,
                    timeout=config.get_timeout(self.name),
                    headers=headers,
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
            except Exception as exc:  # pragma: no cover - safety
                LOGGER.exception("Unexpected error in Unpaywall resolver session path")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                )
                return

            status = getattr(response, "status_code", 200)
            if status != 200:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=status,
                    metadata={"error_detail": f"Unpaywall returned {status}"},
                )
                return

            try:
                data = response.json()
            except ValueError as json_err:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
                    metadata={
                        "error_detail": str(json_err),
                        "content_preview": response.text[:200] if hasattr(response, "text") else "",
                    },
                )
                return
        else:
            try:
                data = _fetch_unpaywall_data(
                    doi,
                    config.unpaywall_email,
                    config.get_timeout(self.name),
                    headers_cache_key(config.polite_headers),
                )
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=status,
                    metadata={"error_detail": f"Unpaywall HTTPError: {status}"},
                )
                return
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
            except requests.RequestException as exc:  # pragma: no cover - network errors
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"error": str(exc)},
                )
                return
            except ValueError as json_err:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="json-error",
                    metadata={"error_detail": str(json_err)},
                )
                return
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected cached request error in Unpaywall resolver")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
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
                    yield ResolverResult(url=unique_url, metadata=metadata)
                    break


class WaybackResolver(RegisteredResolver):
    """Fallback resolver that queries the Internet Archive Wayback Machine."""

    name = "wayback"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return bool(artifact.failed_pdf_urls)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        if not artifact.failed_pdf_urls:
            yield ResolverResult(url=None, event="skipped", event_reason="no-failed-urls")
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
            except requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={
                        "original": original,
                        "timeout": config.get_timeout(self.name),
                        "error": str(exc),
                    },
                )
                continue
            except requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except requests.RequestException as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected Wayback resolver error")
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
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
                    event="error",
                    event_reason="http-error",
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
                    event="error",
                    event_reason="json-error",
                    metadata={"original": original, "error_detail": str(json_err)},
                )
                continue
            closest = (data.get("archived_snapshots") or {}).get("closest") or {}
            if closest.get("available") and closest.get("url"):
                metadata = {"original": original}
                if closest.get("timestamp"):
                    metadata["timestamp"] = closest["timestamp"]
                yield ResolverResult(url=closest["url"], metadata=metadata)


class ZenodoResolver(RegisteredResolver):
    """Resolve Zenodo records into downloadable open-access PDF URLs."""

    name = "zenodo"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` when resolver prerequisites are met for the artifact.
        
        Args:
            config: Resolver configuration containing runtime toggles and credentials.
            artifact: Work artifact capturing document metadata and identifiers.
        
        Returns:
            bool: ``True`` when the resolver should attempt to resolve the artifact.
        """
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield resolver results discovered for the supplied artifact.
        
        Args:
            session: Requests session used to communicate with upstream providers.
            config: Resolver configuration supplying timeouts and headers.
            artifact: Work artifact describing the document under resolution.
        
        Returns:
            Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.
        """
        doi = normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(url=None, event="skipped", event_reason="no-doi")
            return

        try:
            response = request_with_retries(
                session,
                "get",
                "https://zenodo.org/api/records/",
                params={"q": f'doi:"{doi}"', "size": 3, "sort": "mostrecent"},
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
            )
        except requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={
                    "timeout": config.get_timeout(self.name),
                    "error": str(exc),
                },
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
            LOGGER.exception("Unexpected error contacting Zenodo API")
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="unexpected-error",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return

        if response.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=response.status_code,
                metadata={
                    "error_detail": f"Zenodo API returned {response.status_code}",
                },
            )
            return

        try:
            data = response.json()
        except ValueError as json_err:
            preview = response.text[:200] if hasattr(response, "text") else ""
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(json_err), "content_preview": preview},
            )
            return

        hits = data.get("hits", {})
        if not isinstance(hits, dict):
            LOGGER.warning("Zenodo API returned malformed hits payload: %s", type(hits).__name__)
            return
        hits_list = hits.get("hits", [])
        if not isinstance(hits_list, list):
            LOGGER.warning("Zenodo API returned malformed hits list: %s", type(hits_list).__name__)
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
                        "Skipping non-dict Zenodo file entry in record %s", record.get("id")
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


# ---------------------------------------------------------------------------
# Public API helpers


def default_resolvers() -> List[Resolver]:
    """Return default resolver instances in priority order.
    
    Args:
        None
    
    Returns:
        List[Resolver]: Resolver instances ordered according to defaults.
    """

    return ResolverRegistry.create_default()


__all__ = [
    "ResolverRegistry",
    "RegisteredResolver",
    "default_resolvers",
    "ArxivResolver",
    "CoreResolver",
    "CrossrefResolver",
    "DoajResolver",
    "EuropePmcResolver",
    "FigshareResolver",
    "HalResolver",
    "LandingPageResolver",
    "OpenAireResolver",
    "OpenAlexResolver",
    "OsfResolver",
    "PmcResolver",
    "SemanticScholarResolver",
    "UnpaywallResolver",
    "WaybackResolver",
    "ZenodoResolver",
    "_fetch_crossref_data",
    "_fetch_unpaywall_data",
    "_fetch_semantic_scholar_data",
    "headers_cache_key",
    "_headers_cache_key",
]
