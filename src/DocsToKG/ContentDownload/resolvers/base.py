# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.base",
#   "purpose": "Resolver shared types, registry primitives, and helper utilities",
#   "sections": [
#     {
#       "id": "resolverresult",
#       "name": "ResolverResult",
#       "anchor": "class-resolverresult",
#       "kind": "class"
#     },
#     {
#       "id": "resolverevent",
#       "name": "ResolverEvent",
#       "anchor": "class-resolverevent",
#       "kind": "class"
#     },
#     {
#       "id": "resolvereventreason",
#       "name": "ResolverEventReason",
#       "anchor": "class-resolvereventreason",
#       "kind": "class"
#     },
#     {
#       "id": "resolver",
#       "name": "Resolver",
#       "anchor": "class-resolver",
#       "kind": "class"
#     },
#     {
#       "id": "resolverregistry",
#       "name": "ResolverRegistry",
#       "anchor": "class-resolverregistry",
#       "kind": "class"
#     },
#     {
#       "id": "registeredresolver",
#       "name": "RegisteredResolver",
#       "anchor": "class-registeredresolver",
#       "kind": "class"
#     },
#     {
#       "id": "apiresolverbase",
#       "name": "ApiResolverBase",
#       "anchor": "class-apiresolverbase",
#       "kind": "class"
#     },
#     {
#       "id": "absolute-url",
#       "name": "_absolute_url",
#       "anchor": "function-absolute-url",
#       "kind": "function"
#     },
#     {
#       "id": "collect-candidate-urls",
#       "name": "_collect_candidate_urls",
#       "anchor": "function-collect-candidate-urls",
#       "kind": "function"
#     },
#     {
#       "id": "find-pdf-via-meta",
#       "name": "find_pdf_via_meta",
#       "anchor": "function-find-pdf-via-meta",
#       "kind": "function"
#     },
#     {
#       "id": "find-pdf-via-link",
#       "name": "find_pdf_via_link",
#       "anchor": "function-find-pdf-via-link",
#       "kind": "function"
#     },
#     {
#       "id": "find-pdf-via-anchor",
#       "name": "find_pdf_via_anchor",
#       "anchor": "function-find-pdf-via-anchor",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-semantic-scholar-data",
#       "name": "_fetch_semantic_scholar_data",
#       "anchor": "function-fetch-semantic-scholar-data",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-unpaywall-data",
#       "name": "_fetch_unpaywall_data",
#       "anchor": "function-fetch-unpaywall-data",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===
"""Shared resolver primitives and helpers for the content download pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)
from urllib.parse import quote, urljoin, urlparse, urlsplit

import httpx

from DocsToKG.ContentDownload.networking import BreakerOpenError, request_with_retries
from DocsToKG.ContentDownload.urls import canonical_for_index

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig

try:  # Optional dependency guarded at runtime
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    BeautifulSoup = None
    XMLParsedAsHTMLWarning = None

LOGGER = logging.getLogger(__name__)

DEFAULT_RESOLVER_ORDER = [
    "openalex",
    "unpaywall",
    "crossref",
    "landing_page",
    "arxiv",
    "pmc",
    "europe_pmc",
    "core",
    "semantic_scholar",
    "doaj",
    "zenodo",
    "figshare",
    "osf",
    "openaire",
    "hal",
    "wayback",
]

_DEFAULT_RESOLVER_TOGGLES: Dict[str, bool] = {
    name: name not in {"openaire", "hal", "osf"} for name in DEFAULT_RESOLVER_ORDER
}
DEFAULT_RESOLVER_TOGGLES = MappingProxyType(_DEFAULT_RESOLVER_TOGGLES)


@dataclass
class ResolverResult:
    """Either a candidate download URL or an informational resolver event."""

    url: Optional[str]
    referer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event: Optional["ResolverEvent"] = None
    event_reason: Optional["ResolverEventReason"] = None
    http_status: Optional[int] = None
    original_url: Optional[str] = None
    canonical_url: Optional[str] = None
    origin_host: Optional[str] = None

    def __post_init__(self) -> None:
        if self.event is not None and not isinstance(self.event, ResolverEvent):
            self.event = ResolverEvent.from_wire(self.event)
        if self.event_reason is not None and not isinstance(self.event_reason, ResolverEventReason):
            self.event_reason = ResolverEventReason.from_wire(self.event_reason)
        if self.url:
            original = self.original_url or self.url
            canonical = self.canonical_url
            if canonical is None:
                try:
                    canonical = canonical_for_index(original)
                except Exception:
                    canonical = original
            object.__setattr__(self, "original_url", original)
            object.__setattr__(self, "canonical_url", canonical)
            object.__setattr__(self, "url", canonical)
        if self.origin_host is None and self.referer:
            parsed = urlsplit(self.referer)
            host = parsed.hostname
            if host:
                object.__setattr__(self, "origin_host", host.lower())

    @property
    def is_event(self) -> bool:
        """Return ``True`` when this result represents an informational event."""

        return self.url is None


class ResolverEvent(Enum):
    """Structured event taxonomy emitted by resolvers."""

    ERROR = "error"
    HTML_ONLY = "html-only"
    INFO = "info"
    RETRY = "retry"
    SKIPPED = "skipped"

    @classmethod
    def from_wire(cls, value: Union[str, "ResolverEvent", None]) -> Optional["ResolverEvent"]:
        """Coerce serialized event values into :class:`ResolverEvent` members."""

        if value is None:
            return None
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        if not text:
            return None
        for member in cls:
            if member.value == text:
                return member
        raise ValueError(f"Unknown resolver event '{value}'")


class ResolverEventReason(Enum):
    """Structured reason taxonomy for resolver events."""

    CONNECTION_ERROR = "connection-error"
    HTTP_ERROR = "http-error"
    JSON_ERROR = "json-error"
    REQUEST_ERROR = "request-error"
    UNEXPECTED_ERROR = "unexpected-error"
    TIMEOUT = "timeout"
    NO_ARXIV_ID = "no-arxiv-id"
    NO_DOI = "no-doi"
    NO_BEAUTIFULSOUP = "no-beautifulsoup"
    NO_FAILED_URLS = "no-failed-urls"
    NO_WAYBACK_SNAPSHOT = "no-wayback-snapshot"
    NO_OPENACCESS_PDF = "no-openaccess-pdf"
    NO_OPENALEX_URLS = "no-openalex-urls"
    NO_PMCID = "no-pmcid"
    RESOLVER_EXCEPTION = "resolver-exception"
    RATE_LIMIT = "rate-limit"
    BREAKER_OPEN = "breaker-open"
    NOT_APPLICABLE = "not-applicable"

    @classmethod
    def from_wire(
        cls, value: Union[str, "ResolverEventReason", None]
    ) -> Optional["ResolverEventReason"]:
        """Coerce serialized reason values into :class:`ResolverEventReason` members."""

        if value is None:
            return None
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        if not text:
            return None
        for member in cls:
            if member.value == text:
                return member
        raise ValueError(f"Unknown resolver event reason '{value}'")


class Resolver(Protocol):
    """Protocol that resolver implementations must follow."""

    name: str

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        """Return ``True`` if this resolver should run for the given artifact."""

    def iter_urls(
        self,
        session: httpx.Client,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs or events for the given artifact."""


class ResolverRegistry:
    """Registry tracking resolver classes by their ``name`` attribute."""

    _providers: Dict[str, Type[Resolver]] = {}

    @classmethod
    def register(cls, resolver_cls: Type[Resolver]) -> Type[Resolver]:
        """Register a resolver class under its declared ``name`` attribute."""

        name = getattr(resolver_cls, "name", None)
        if not name:
            raise ValueError(f"Resolver class {resolver_cls.__name__} missing 'name' attribute")
        cls._providers[name] = resolver_cls
        return resolver_cls

    @classmethod
    def create_default(cls) -> list[Resolver]:
        """Instantiate resolver instances in priority order."""

        instances: list[Resolver] = []
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


class ApiResolverBase(RegisteredResolver, register=False):
    """Shared helper for resolvers interacting with JSON-based HTTP APIs."""

    def _request_json(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        *,
        config: "ResolverConfig",
        timeout: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[Any], Optional[ResolverResult]]:
        timeout_value = timeout if timeout is not None else config.get_timeout(self.name)
        request_headers: Dict[str, str] = dict(config.polite_headers)
        if headers:
            request_headers.update(headers)

        kwargs.setdefault("allow_redirects", True)

        try:
            response = request_with_retries(
                client,
                method,
                url,
                role="metadata",
                params=params,
                json=json,
                headers=request_headers,
                timeout=timeout_value,
                retry_after_cap=config.retry_after_cap,
                **kwargs,
            )
        except BreakerOpenError as exc:
            meta: Dict[str, Any] = {"url": url, "error": str(exc)}
            breaker_meta = getattr(exc, "breaker_meta", None)
            if isinstance(breaker_meta, Mapping):
                meta["breaker"] = dict(breaker_meta)
            return (
                None,
                ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.BREAKER_OPEN,
                    metadata=meta,
                ),
            )
        except httpx.TimeoutException as exc:
            return (
                None,
                ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.TIMEOUT,
                    metadata={
                        "url": url,
                        "timeout": timeout_value,
                        "error": str(exc),
                    },
                ),
            )
        except httpx.RequestError as exc:
            return (
                None,
                ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.CONNECTION_ERROR,
                    metadata={"url": url, "error": str(exc)},
                ),
            )
        except httpx.HTTPError as exc:
            return (
                None,
                ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.REQUEST_ERROR,
                    metadata={"url": url, "error": str(exc)},
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Unexpected error issuing %s request to %s", method, url)
            return (
                None,
                ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                    metadata={
                        "url": url,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                ),
            )

        try:
            if response.status_code != 200:
                return (
                    None,
                    ResolverResult(
                        url=None,
                        event=ResolverEvent.ERROR,
                        event_reason=ResolverEventReason.HTTP_ERROR,
                        http_status=response.status_code,
                        metadata={
                            "url": url,
                            "error_detail": f"{getattr(self, 'api_display_name', self.name)} API returned {response.status_code}",
                        },
                    ),
                )
            data = response.json()
        except ValueError as json_err:
            preview = response.text[:200] if hasattr(response, "text") else ""
            return (
                None,
                ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.JSON_ERROR,
                    metadata={
                        "url": url,
                        "error_detail": str(json_err),
                        "content_preview": preview,
                    },
                ),
            )
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        return data, None


def _absolute_url(base: str, href: str) -> str:
    """Resolve relative ``href`` values against ``base`` to obtain absolute URLs."""

    parsed = urlparse(href)
    if parsed.scheme and parsed.netloc:
        return href
    base_parts = urlparse(base)
    if base_parts.scheme and base_parts.netloc:
        origin = f"{base_parts.scheme}://{base_parts.netloc}/"
        return urljoin(origin, href)
    return urljoin(base, href)


def _collect_candidate_urls(node: object, results: list[str]) -> None:
    """Recursively collect HTTP(S) URLs from nested response payloads."""

    if isinstance(node, dict):
        for value in node.values():
            _collect_candidate_urls(value, results)
    elif isinstance(node, list):
        for item in node:
            _collect_candidate_urls(item, results)
    elif isinstance(node, str) and node.lower().startswith("http"):
        results.append(node)


def find_pdf_via_meta(soup: "BeautifulSoup", base_url: str) -> Optional[str]:
    """Return PDF URL declared via ``citation_pdf_url`` meta tags."""

    tag = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if tag and tag.get("content"):
        return _absolute_url(base_url, tag["content"])  # type: ignore[index]
    return None


def find_pdf_via_link(soup: "BeautifulSoup", base_url: str) -> Optional[str]:
    """Return PDF URL advertised through alternate link tags."""

    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel") or []).lower()
        typ = (link.get("type") or "").lower()
        href = (link.get("href") or "").strip()
        if "alternate" in rel and "application/pdf" in typ and href:
            return _absolute_url(base_url, href)
    return None


def find_pdf_via_anchor(soup: "BeautifulSoup", base_url: str) -> Optional[str]:
    """Return PDF URL inferred from anchor elements mentioning PDFs."""

    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        if not href:
            continue
        text = (anchor.get_text() or "").strip().lower()
        href_lower = href.lower()
        if href_lower.endswith(".pdf") or "pdf" in text:
            candidate = _absolute_url(base_url, href)
            if candidate.lower().endswith(".pdf"):
                return candidate
    return None


def _fetch_semantic_scholar_data(
    client: httpx.Client,
    config: "ResolverConfig",
    doi: str,
) -> Dict[str, Any]:
    """Return Semantic Scholar metadata for ``doi`` using configured headers."""

    headers = dict(config.polite_headers)
    if getattr(config, "semantic_scholar_api_key", None):
        headers["x-api-key"] = config.semantic_scholar_api_key
    response = request_with_retries(
        client,
        "GET",
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}",
        role="metadata",
        params={"fields": "title,openAccessPdf"},
        timeout=config.get_timeout("semantic_scholar"),
        headers=headers,
        retry_after_cap=config.retry_after_cap,
    )
    try:
        response.raise_for_status()
        return response.json()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _fetch_unpaywall_data(
    client: httpx.Client,
    config: "ResolverConfig",
    doi: str,
) -> Dict[str, Any]:
    """Return Unpaywall metadata for ``doi`` using configured headers."""

    endpoint = f"https://api.unpaywall.org/v2/{quote(doi)}"
    headers = dict(config.polite_headers)
    params = {"email": config.unpaywall_email} if getattr(config, "unpaywall_email", None) else None
    response = request_with_retries(
        client,
        "GET",
        endpoint,
        role="metadata",
        params=params,
        timeout=config.get_timeout("unpaywall"),
        headers=headers,
        retry_after_cap=config.retry_after_cap,
    )
    try:
        response.raise_for_status()
        return response.json()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


__all__ = [
    "ApiResolverBase",
    "BeautifulSoup",
    "DEFAULT_RESOLVER_ORDER",
    "DEFAULT_RESOLVER_TOGGLES",
    "Resolver",
    "ResolverEvent",
    "ResolverEventReason",
    "ResolverRegistry",
    "ResolverResult",
    "XMLParsedAsHTMLWarning",
    "_absolute_url",
    "_collect_candidate_urls",
    "_fetch_semantic_scholar_data",
    "_fetch_unpaywall_data",
    "find_pdf_via_anchor",
    "find_pdf_via_link",
    "find_pdf_via_meta",
    "request_with_retries",
]
