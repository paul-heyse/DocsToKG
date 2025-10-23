# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.base",
#   "purpose": "Shared resolver primitives and helpers.",
#   "sections": [
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
#       "id": "resolverresult",
#       "name": "ResolverResult",
#       "anchor": "class-resolverresult",
#       "kind": "class"
#     },
#     {
#       "id": "resolver",
#       "name": "Resolver",
#       "anchor": "class-resolver",
#       "kind": "class"
#     },
#     {
#       "id": "absolute-url",
#       "name": "_absolute_url",
#       "anchor": "function-absolute-url",
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
#       "id": "collect-candidate-urls",
#       "name": "_collect_candidate_urls",
#       "anchor": "function-collect-candidate-urls",
#       "kind": "function"
#     },
#     {
#       "id": "apiresolverbase",
#       "name": "ApiResolverBase",
#       "anchor": "class-apiresolverbase",
#       "kind": "class"
#     },
#     {
#       "id": "fetch-unpaywall-data",
#       "name": "_fetch_unpaywall_data",
#       "anchor": "function-fetch-unpaywall-data",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-semantic-scholar-data",
#       "name": "_fetch_semantic_scholar_data",
#       "anchor": "function-fetch-semantic-scholar-data",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Shared resolver primitives and helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urljoin

import httpx

try:  # pragma: no cover - optional dependency
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning  # type: ignore
except ImportError:  # pragma: no cover - gracefully degrade when bs4 is missing
    BeautifulSoup = None  # type: ignore[assignment]
    XMLParsedAsHTMLWarning = None  # type: ignore[assignment]

from DocsToKG.ContentDownload.networking import request_with_retries

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolver registry (legacy compatibility)
# ---------------------------------------------------------------------------


class ResolverRegistry:
    """Light-weight registry retaining backwards compatibility with legacy code."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, resolver_cls: type) -> None:
        name = getattr(resolver_cls, "name", resolver_cls.__name__).lower()
        cls._registry[name] = resolver_cls

    @classmethod
    def create_default(cls) -> list[Any]:
        """Instantiate all registered resolvers (used by some legacy tests)."""

        return [resolver_cls() for resolver_cls in cls._registry.values()]


class RegisteredResolver:
    """Compatibility shim for legacy resolver inheritance pattern."""

    def __init_subclass__(cls, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if register:
            ResolverRegistry.register(cls)


# ---------------------------------------------------------------------------
# Resolver events
# ---------------------------------------------------------------------------


class ResolverEvent(str):
    """Event emitted by a resolver when yielding a non-download result."""

    ERROR = "error"
    INFO = "info"
    SKIPPED = "skipped"

    def is_event(self) -> bool:
        return True

    @classmethod
    def from_wire(cls, value: Any) -> ResolverEvent:
        if isinstance(value, ResolverEvent):
            return value
        if isinstance(value, str):
            normalized = value.replace("-", "_").upper()
            if hasattr(cls, normalized):
                return getattr(cls, normalized)
        raise ValueError(f"Unknown resolver event value: {value!r}")


class ResolverEventReason(str):
    """Structured reason taxonomy for resolver events."""

    BREAKER_OPEN = "breaker-open"
    CONNECTION_ERROR = "connection-error"
    HTTP_ERROR = "http-error"
    JSON_ERROR = "json-error"
    NO_ARXIV_ID = "no-arxiv-id"
    NO_BEAUTIFULSOUP = "no-beautifulsoup"
    NO_DOI = "no-doi"
    NO_FAILED_URLS = "no-failed-urls"
    NO_OPENACCESS_PDF = "no-openaccess-pdf"
    NO_OPENALEX_URLS = "no-openalex-urls"
    NO_PMCID = "no-pmcid"
    RATE_LIMIT = "rate-limit"
    REQUEST_ERROR = "request-error"
    RESOLVER_EXCEPTION = "resolver-exception"
    ROBOTS_DISALLOWED = "robots-disallowed"
    TIMEOUT = "timeout"
    UNEXPECTED_ERROR = "unexpected-error"

    @classmethod
    def from_wire(cls, value: Any) -> ResolverEventReason:
        if isinstance(value, ResolverEventReason):
            return value
        if isinstance(value, str):
            normalized = value.replace("-", "_").upper()
            if hasattr(cls, normalized):
                return getattr(cls, normalized)
        raise ValueError(f"Unknown resolver event reason: {value!r}")


# ---------------------------------------------------------------------------
# Resolver result payload
# ---------------------------------------------------------------------------


class ResolverResult:
    """Simple container for resolver outcomes."""

    def __init__(
        self,
        url: str | None = None,
        *,
        referer: str | None = None,
        canonical_url: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        event: ResolverEvent | None = None,
        event_reason: ResolverEventReason | None = None,
        http_status: int | None = None,
        **extra: Any,
    ) -> None:
        self.url = url
        self.referer = referer
        self.canonical_url = canonical_url
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.event = event
        self.event_reason = event_reason
        self.http_status = http_status
        for key, value in extra.items():
            setattr(self, key, value)

    def is_event(self) -> bool:
        return self.event is not None


# ---------------------------------------------------------------------------
# Resolver protocol (public interface used by legacy imports)
# ---------------------------------------------------------------------------


class Resolver(Protocol):
    """Protocol describing the minimal resolver interface."""

    name: str

    def resolve(
        self,
        artifact: Any,
        session: Any,
        ctx: Any,
        telemetry: Any | None,
        run_id: str | None,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Default resolver toggles (legacy surface expected by tests/tooling)
# ---------------------------------------------------------------------------


_DEFAULT_RESOLVER_TOGGLES = {
    "unpaywall": True,
    "crossref": True,
    "arxiv": True,
    "europe_pmc": True,
    "core": True,
    "doaj": True,
    "semantic_scholar": True,
    "landing_page": True,
    "wayback": True,
    "openalex": True,
    "zenodo": True,
    "osf": True,
    "openaire": True,
    "hal": True,
    "figshare": True,
}

DEFAULT_RESOLVER_TOGGLES = MappingProxyType(_DEFAULT_RESOLVER_TOGGLES)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _absolute_url(base: str, href: str) -> str:
    return urljoin(base, href)


def find_pdf_via_meta(soup: Any, base_url: str) -> str | None:
    if soup is None:
        return None
    tag = getattr(soup, "find", lambda *args, **kwargs: None)(
        "meta", attrs={"name": "citation_pdf_url"}
    )
    content = tag.get("content") if tag else None
    if isinstance(content, str) and content.strip():
        return _absolute_url(base_url, content.strip())
    return None


def find_pdf_via_link(soup: Any, base_url: str) -> str | None:
    if soup is None:
        return None
    for tag in getattr(soup, "find_all", lambda *args, **kwargs: [])("link"):
        rel = tag.get("rel") or []
        rel_tokens = {token.lower() for token in rel} if isinstance(rel, Sequence) else set()
        link_type = (tag.get("type") or "").lower()
        href = tag.get("href")
        if (
            href
            and isinstance(href, str)
            and ("alternate" in rel_tokens or rel_tokens == set())
            and "application/pdf" in link_type
        ):
            return _absolute_url(base_url, href)
    return None


def find_pdf_via_anchor(soup: Any, base_url: str) -> str | None:
    if soup is None:
        return None
    for anchor in getattr(soup, "find_all", lambda *args, **kwargs: [])("a"):
        href = anchor.get("href")
        if isinstance(href, str) and href.lower().endswith(".pdf"):
            return _absolute_url(base_url, href)
    return None


def _collect_candidate_urls(node: Any, results: list[str]) -> None:
    if isinstance(node, str):
        value = node.strip()
        if value.startswith("http://") or value.startswith("https://"):
            results.append(value)
        return
    if isinstance(node, Mapping):
        for value in node.values():
            _collect_candidate_urls(value, results)
        return
    if isinstance(node, Sequence) and not isinstance(node, (bytes, bytearray)):
        for item in node:
            _collect_candidate_urls(item, results)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


class ApiResolverBase(RegisteredResolver):
    """Shared helper for HTTP API based resolvers."""

    name: str = "resolver"

    def _request_json(
        self,
        client: httpx.Client | None,
        method: str,
        url: str,
        *,
        config: Any,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        role: str = "metadata",
        **kwargs: Any,
    ) -> tuple[Any | None, ResolverResult | None]:
        polite_headers: dict[str, str] = {}
        base_headers = getattr(config, "polite_headers", None)
        if isinstance(base_headers, Mapping):
            polite_headers.update({k: str(v) for k, v in base_headers.items()})
        if headers:
            polite_headers.update({k: str(v) for k, v in headers.items()})

        timeout: float | None = None
        get_timeout = getattr(config, "get_timeout", None)
        if callable(get_timeout):
            try:
                timeout = float(get_timeout(self.name))
            except Exception:  # pragma: no cover - defensive
                timeout = None

        retry_after_cap = getattr(config, "retry_after_cap", None)

        try:
            response = request_with_retries(
                client,
                method,
                url,
                resolver=self.name,
                headers=polite_headers or None,
                params=params,
                json=json,
                timeout=timeout,
                retry_after_cap=retry_after_cap,
                role=role,
                **kwargs,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            metadata = {"error": str(exc)}
            if timeout is not None:
                metadata["timeout"] = timeout
            return None, ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.TIMEOUT,
                metadata=metadata,
            )
        except httpx.TransportError as exc:
            return None, ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.CONNECTION_ERROR,
                metadata={"error": str(exc)},
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            return None, ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.HTTP_ERROR,
                http_status=status,
                metadata={"error_detail": str(exc)},
            )
        except httpx.RequestError as exc:
            return None, ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.REQUEST_ERROR,
                metadata={"error": str(exc)},
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Unexpected error during %s resolver request", self.name)
            return None, ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )

        try:
            data = response.json()
        except ValueError as exc:
            preview = response.text[:200] if hasattr(response, "text") else ""
            response.close()
            metadata = {"error_detail": str(exc)}
            if preview:
                metadata["content_preview"] = preview
            return None, ResolverResult(
                url=None,
                event=ResolverEvent.ERROR,
                event_reason=ResolverEventReason.JSON_ERROR,
                metadata=metadata,
            )
        else:
            response.close()
            return data, None


def _fetch_unpaywall_data(
    client: httpx.Client | None,
    config: Any,
    doi: str,
) -> Any:
    headers: dict[str, str] = {}
    base_headers = getattr(config, "polite_headers", None)
    if isinstance(base_headers, Mapping):
        headers.update({k: str(v) for k, v in base_headers.items()})
    email = getattr(config, "unpaywall_email", None) or getattr(config, "mailto", None)
    if email and "mailto" not in headers:
        headers["mailto"] = str(email)
    timeout = None
    get_timeout = getattr(config, "get_timeout", None)
    if callable(get_timeout):
        try:
            timeout = float(get_timeout("unpaywall"))
        except Exception:  # pragma: no cover - defensive
            timeout = None

    response = request_with_retries(
        client,
        "get",
        f"https://api.unpaywall.org/v2/{doi}",
        resolver="unpaywall",
        headers=headers or None,
        params={"email": email} if email else None,
        timeout=timeout,
        retry_after_cap=getattr(config, "retry_after_cap", None),
    )
    try:
        response.raise_for_status()
        return response.json()
    finally:
        response.close()


def _fetch_semantic_scholar_data(
    client: httpx.Client | None,
    config: Any,
    doi: str,
) -> Any:
    headers: dict[str, str] = {}
    base_headers = getattr(config, "polite_headers", None)
    if isinstance(base_headers, Mapping):
        headers.update({k: str(v) for k, v in base_headers.items()})
    api_key = getattr(config, "semantic_scholar_api_key", None)
    if api_key:
        headers["x-api-key"] = str(api_key)

    timeout = None
    get_timeout = getattr(config, "get_timeout", None)
    if callable(get_timeout):
        try:
            timeout = float(get_timeout("semantic_scholar"))
        except Exception:  # pragma: no cover - defensive
            timeout = None

    response = request_with_retries(
        client,
        "get",
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
        resolver="semantic_scholar",
        headers=headers or None,
        params={"fields": "openAccessPdf"},
        timeout=timeout,
        retry_after_cap=getattr(config, "retry_after_cap", None),
    )
    try:
        response.raise_for_status()
        return response.json()
    finally:
        response.close()


__all__ = [
    "ApiResolverBase",
    "BeautifulSoup",
    "DEFAULT_RESOLVER_TOGGLES",
    "RegisteredResolver",
    "ResolverEvent",
    "ResolverEventReason",
    "ResolverRegistry",
    "ResolverResult",
    "XMLParsedAsHTMLWarning",
    "_DEFAULT_RESOLVER_TOGGLES",
    "_collect_candidate_urls",
    "_fetch_semantic_scholar_data",
    "_fetch_unpaywall_data",
    "find_pdf_via_anchor",
    "find_pdf_via_link",
    "find_pdf_via_meta",
    "request_with_retries",
]
