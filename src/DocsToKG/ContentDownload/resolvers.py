"""
Content Download Resolver Orchestration

This module centralises resolver configuration, provider registration,
pipeline orchestration, and cache helpers for the DocsToKG content download
stack. Resolver classes encapsulate provider-specific discovery logic while
the shared pipeline coordinates rate limiting, concurrency, and polite HTTP
behaviour.

Key Features:
- Resolver registry supplying default provider ordering and toggles.
- Shared retry helper integration to ensure consistent network backoff.
- Manifest and attempt bookkeeping for detailed diagnostics.
- Utility functions for cache invalidation and signature normalisation.

Dependencies:
- requests: Outbound HTTP traffic and session management.
- BeautifulSoup: Optional HTML parsing for resolver implementations.
- DocsToKG.ContentDownload.network: Shared retry and session helpers.

Usage:
    from DocsToKG.ContentDownload import resolvers

    config = resolvers.ResolverConfig()
    active_resolvers = resolvers.default_resolvers()
    pipeline = resolvers.ResolverPipeline(
        resolvers=active_resolvers,
        config=config,
    )
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time as _time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
)
from urllib.parse import quote, urljoin, urlparse, urlsplit

import requests as _requests

from DocsToKG.ContentDownload.utils import (
    dedupe,
    normalize_doi,
    normalize_pmcid,
    strip_prefix,
)

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact

try:  # Optional dependency guarded at runtime
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    BeautifulSoup = None

LOGGER = logging.getLogger(__name__)

_time_alias = _time
_requests_alias = _requests

DEFAULT_RESOLVER_ORDER: List[str] = [
    "openalex",
    "unpaywall",
    "crossref",
    "landing_page",
    "arxiv",
    "pmc",
    "europe_pmc",
    "core",
    "zenodo",
    "figshare",
    "doaj",
    "semantic_scholar",
    "openaire",
    "hal",
    "osf",
    "wayback",
]

_DEFAULT_RESOLVER_TOGGLES: Dict[str, bool] = {
    name: name not in {"openaire", "hal", "osf"} for name in DEFAULT_RESOLVER_ORDER
}


@dataclass
class ResolverResult:
    """Either a candidate download URL or an informational resolver event.

    Attributes:
        url: Candidate download URL emitted by the resolver (``None`` for events).
        referer: Optional referer header to accompany the download request.
        metadata: Arbitrary metadata recorded alongside the result.
        event: Optional event label (e.g., ``"error"`` or ``"skipped"``).
        event_reason: Human-readable reason describing the event.
        http_status: HTTP status associated with the event, when available.

    Examples:
        >>> ResolverResult(url="https://example.org/file.pdf", metadata={"resolver": "core"})
    """

    url: Optional[str]
    referer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event: Optional[str] = None
    event_reason: Optional[str] = None
    http_status: Optional[int] = None

    @property
    def is_event(self) -> bool:
        """Return ``True`` when this result represents an informational event.

        Args:
            self: Resolver result instance under inspection.

        Returns:
            bool: ``True`` if the resolver emitted an event instead of a URL.
        """

        return self.url is None


@dataclass
class ResolverConfig:
    """Runtime configuration options applied across resolvers.

    Attributes:
        resolver_order: Ordered list of resolver names to execute.
        resolver_toggles: Mapping toggling individual resolvers on/off.
        max_attempts_per_work: Maximum number of resolver attempts per work item.
        timeout: Default HTTP timeout applied to resolvers.
        sleep_jitter: Random jitter added between retries.
        polite_headers: HTTP headers to apply for polite crawling.
        unpaywall_email: Contact email registered with Unpaywall.
        core_api_key: API key used for the CORE resolver.
        semantic_scholar_api_key: API key for Semantic Scholar resolver.
        doaj_api_key: API key for DOAJ resolver.
        resolver_timeouts: Resolver-specific timeout overrides.
        resolver_min_interval_s: Minimum interval between resolver _requests.
        domain_min_interval_s: Optional per-domain rate limits overriding resolver settings.
        resolver_rate_limits: Deprecated rate limit configuration retained for compat.
        enable_head_precheck: Toggle applying HEAD filtering before downloads.
        resolver_head_precheck: Per-resolver overrides for HEAD filtering behaviour.
        mailto: Contact email appended to polite headers and user agent string.
        max_concurrent_resolvers: Upper bound on concurrent resolver threads per work.
        enable_global_url_dedup: Enable global URL deduplication across works when True.

    Notes:
        ``enable_head_precheck`` toggles inexpensive HEAD lookups before downloads
        to filter obvious HTML responses. ``resolver_head_precheck`` allows
        per-resolver overrides when specific providers reject HEAD _requests.
        ``max_concurrent_resolvers`` bounds the number of resolver threads used
        per work while still respecting configured rate limits.

    Examples:
        >>> config = ResolverConfig()
        >>> config.max_attempts_per_work
        25
    """

    resolver_order: List[str] = field(default_factory=lambda: list(DEFAULT_RESOLVER_ORDER))
    resolver_toggles: Dict[str, bool] = field(
        default_factory=lambda: dict(_DEFAULT_RESOLVER_TOGGLES)
    )
    max_attempts_per_work: int = 25
    timeout: float = 30.0
    sleep_jitter: float = 0.35
    polite_headers: Dict[str, str] = field(default_factory=dict)
    unpaywall_email: Optional[str] = None
    core_api_key: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    doaj_api_key: Optional[str] = None
    resolver_timeouts: Dict[str, float] = field(default_factory=dict)
    resolver_min_interval_s: Dict[str, float] = field(default_factory=dict)
    domain_min_interval_s: Dict[str, float] = field(default_factory=dict)
    resolver_rate_limits: Dict[str, float] = field(default_factory=dict)
    enable_head_precheck: bool = True
    resolver_head_precheck: Dict[str, bool] = field(default_factory=dict)
    mailto: Optional[str] = None
    max_concurrent_resolvers: int = 1
    enable_global_url_dedup: bool = False

    def get_timeout(self, resolver_name: str) -> float:
        """Return the timeout to use for a resolver, falling back to defaults.

        Args:
            resolver_name: Name of the resolver requesting a timeout.

        Returns:
            float: Timeout value in seconds.
        """

        return self.resolver_timeouts.get(resolver_name, self.timeout)

    def is_enabled(self, resolver_name: str) -> bool:
        """Return ``True`` when the resolver is enabled for the current run.

        Args:
            resolver_name: Name of the resolver.

        Returns:
            bool: ``True`` if the resolver is enabled.
        """

        return self.resolver_toggles.get(resolver_name, True)

    def __post_init__(self) -> None:
        """Validate configuration fields and apply defaults for missing values.

        Args:
            self: Configuration instance requiring validation.

        Returns:
            None
        """

        if self.max_concurrent_resolvers < 1:
            raise ValueError(
                f"max_concurrent_resolvers must be >= 1, got {self.max_concurrent_resolvers}"
            )
        if self.max_concurrent_resolvers > 10:
            warnings.warn(
                (
                    "max_concurrent_resolvers="
                    f"{self.max_concurrent_resolvers} > 10 may violate rate limits. "
                    "Ensure resolver_min_interval_s is configured appropriately for all resolvers."
                ),
                UserWarning,
                stacklevel=2,
            )

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        for resolver_name, timeout_val in self.resolver_timeouts.items():
            if timeout_val <= 0:
                raise ValueError(
                    ("resolver_timeouts['{name}'] must be positive, got {value}").format(
                        name=resolver_name, value=timeout_val
                    )
                )

        for resolver_name, interval in self.resolver_min_interval_s.items():
            if interval < 0:
                raise ValueError(
                    ("resolver_min_interval_s['{name}'] must be non-negative, got {value}").format(
                        name=resolver_name, value=interval
                    )
                )

        if self.resolver_rate_limits:
            for name, value in self.resolver_rate_limits.items():
                self.resolver_min_interval_s.setdefault(name, value)

        normalized_domain_limits: Dict[str, float] = {}
        for host, interval in self.domain_min_interval_s.items():
            if interval < 0:
                raise ValueError(
                    ("domain_min_interval_s['{name}'] must be non-negative, got {value}").format(
                        name=host, value=interval
                    )
                )
            normalized_domain_limits[host.lower()] = interval
        if normalized_domain_limits:
            self.domain_min_interval_s = normalized_domain_limits

        if self.max_attempts_per_work < 1:
            raise ValueError(
                f"max_attempts_per_work must be >= 1, got {self.max_attempts_per_work}"
            )


@dataclass
class AttemptRecord:
    """Structured log record describing a resolver attempt.

    Attributes:
        work_id: Identifier of the work being processed.
        resolver_name: Name of the resolver that produced the record.
        resolver_order: Ordinal position of the resolver in the pipeline.
        url: Candidate URL that was attempted.
        status: Classification or status string for the attempt.
        http_status: HTTP status code (when available).
        content_type: Response content type.
        elapsed_ms: Approximate elapsed time for the attempt in milliseconds.
        resolver_wall_time_ms: Wall-clock time spent inside the resolver including
            rate limiting, measured in milliseconds.
        reason: Optional descriptive reason for failures or skips.
        metadata: Arbitrary metadata supplied by the resolver.
        sha256: SHA-256 digest of downloaded content, when available.
        content_length: Size of the downloaded content in bytes.
        dry_run: Flag indicating whether the attempt occurred in dry-run mode.

    Examples:
        >>> AttemptRecord(
        ...     work_id="W1",
        ...     resolver_name="unpaywall",
        ...     resolver_order=1,
        ...     url="https://example.org/pdf",
        ...     status="pdf",
        ...     http_status=200,
        ...     content_type="application/pdf",
        ...     elapsed_ms=120.5,
        ... )
    """

    work_id: str
    resolver_name: str
    resolver_order: Optional[int]
    url: Optional[str]
    status: str
    http_status: Optional[int]
    content_type: Optional[str]
    elapsed_ms: Optional[float]
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sha256: Optional[str] = None
    content_length: Optional[int] = None
    dry_run: bool = False
    resolver_wall_time_ms: Optional[float] = None


class AttemptLogger(Protocol):
    """Protocol for logging resolver attempts.

    Attributes:
        None: The protocol formalises the callable surface without storing state.

    Examples:
        >>> class Collector:
        ...     def __init__(self):
        ...         self.records = []
        ...     def log(self, record: AttemptRecord) -> None:
        ...         self.records.append(record)
        >>> collector = Collector()
        >>> isinstance(collector, AttemptLogger)
        True
    """

    def log(self, record: AttemptRecord) -> None:
        """Log a resolver attempt.

        Args:
            record: Attempt record describing the resolver execution.

        Returns:
            None
        """


@dataclass
class DownloadOutcome:
    """Outcome of a resolver download attempt.

    Attributes:
        classification: Classification label describing the outcome (e.g., 'pdf').
        path: Local filesystem path to the stored artifact.
        http_status: HTTP status code when available.
        content_type: Content type reported by the server.
        elapsed_ms: Time spent downloading in milliseconds.
        error: Optional error string describing failures.
        sha256: SHA-256 digest of the downloaded content.
        content_length: Size of the downloaded content in bytes.
        etag: HTTP ETag header value when provided.
        last_modified: HTTP Last-Modified timestamp.
        extracted_text_path: Optional path to extracted text artefacts.

    Examples:
        >>> DownloadOutcome(classification="pdf", path="pdfs/sample.pdf", http_status=200,
        ...                 content_type="application/pdf", elapsed_ms=150.0)
    """

    classification: str
    path: Optional[str] = None
    http_status: Optional[int] = None
    content_type: Optional[str] = None
    elapsed_ms: Optional[float] = None
    error: Optional[str] = None
    sha256: Optional[str] = None
    content_length: Optional[int] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    extracted_text_path: Optional[str] = None

    @property
    def is_pdf(self) -> bool:
        """Return ``True`` when the classification represents a PDF.

        Args:
            self: Download outcome to evaluate.

        Returns:
            bool: ``True`` if the outcome corresponds to a PDF download.
        """

        return self.classification in {"pdf", "pdf_unknown"}


@dataclass
class PipelineResult:
    """Aggregate result returned by the resolver pipeline.

    Attributes:
        success: Indicates whether the pipeline found a suitable asset.
        resolver_name: Resolver that produced the successful result.
        url: URL that was ultimately fetched.
        outcome: Download outcome associated with the result.
        html_paths: Collected HTML artefacts from the pipeline.
        failed_urls: Candidate URLs that failed during this run.
        reason: Optional reason string explaining failures.

    Examples:
        >>> PipelineResult(success=True, resolver_name="unpaywall", url="https://example")
    """

    success: bool
    resolver_name: Optional[str] = None
    url: Optional[str] = None
    outcome: Optional[DownloadOutcome] = None
    html_paths: List[str] = field(default_factory=list)
    failed_urls: List[str] = field(default_factory=list)
    reason: Optional[str] = None


class Resolver(Protocol):
    """Protocol that resolver implementations must follow.

    Attributes:
        name: Resolver identifier used within configuration.

    Examples:
        Concrete implementations are provided in classes such as
        :class:`UnpaywallResolver` and :class:`CrossrefResolver` below.
    """

    name: str

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        """Return ``True`` if this resolver should run for the given artifact.

        Args:
            config: Resolver configuration options.
            artifact: Work artifact under consideration.

        Returns:
            bool: ``True`` when the resolver should run for the artifact.
        """

    def iter_urls(
        self,
        session: _requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs or events for the given artifact.

        Args:
            session: HTTP session used for outbound _requests.
            config: Resolver configuration.
            artifact: Work artifact describing the current item.

        Returns:
            Iterable[ResolverResult]: Stream of download candidates or events.
        """


@dataclass
class ResolverMetrics:
    """Lightweight metrics collector for resolver execution.

    Attributes:
        attempts: Counter of attempts per resolver.
        successes: Counter of successful PDF downloads per resolver.
        html: Counter of HTML-only results per resolver.
        skips: Counter of skip events keyed by resolver and reason.

    Examples:
        >>> metrics = ResolverMetrics()
        >>> metrics.record_attempt("unpaywall", DownloadOutcome("pdf", None, 200, None, 10.0))
        >>> metrics.summary()["attempts"]["unpaywall"]
        1
    """

    attempts: Counter = field(default_factory=Counter)
    successes: Counter = field(default_factory=Counter)
    html: Counter = field(default_factory=Counter)
    skips: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)

    def record_attempt(self, resolver_name: str, outcome: DownloadOutcome) -> None:
        """Record a resolver attempt and update success/html counters.

        Args:
            resolver_name: Name of the resolver that executed.
            outcome: Download outcome produced by the resolver.

        Returns:
            None
        """

        self.attempts[resolver_name] += 1
        if outcome.classification == "html":
            self.html[resolver_name] += 1
        if outcome.is_pdf:
            self.successes[resolver_name] += 1

    def record_skip(self, resolver_name: str, reason: str) -> None:
        """Record a skip event for a resolver with a reason tag.

        Args:
            resolver_name: Resolver that was skipped.
            reason: Short description explaining the skip.

        Returns:
            None
        """

        key = f"{resolver_name}:{reason}"
        self.skips[key] += 1

    def record_failure(self, resolver_name: str) -> None:
        """Record a resolver failure occurrence.

        Args:
            resolver_name: Resolver that raised an exception during execution.

        Returns:
            None
        """

        self.failures[resolver_name] += 1

    def summary(self) -> Dict[str, Any]:
        """Return aggregated metrics summarizing resolver behaviour.

        Args:
            self: Metrics collector instance aggregating resolver statistics.

        Returns:
            Dict[str, Any]: Snapshot of attempts, successes, HTML hits, and skips.

        Examples:
            >>> metrics = ResolverMetrics()
            >>> metrics.record_attempt("unpaywall", DownloadOutcome("pdf", None, 200, None, 10.0))
            >>> metrics.summary()["attempts"]["unpaywall"]
            1
        """

        return {
            "attempts": dict(self.attempts),
            "successes": dict(self.successes),
            "html": dict(self.html),
            "skips": dict(self.skips),
            "failures": dict(self.failures),
        }


DownloadFunc = Callable[..., DownloadOutcome]


def headers_cache_key(headers: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """Return a deterministic cache key for HTTP header dictionaries.

    Args:
        headers: Mapping of header names to values.

    Returns:
        Tuple of lowercase header names paired with their original values,
        sorted alphabetically for stable hashing.
    """

    items: Iterable[Tuple[str, str]] = (
        (key.lower(), value) for key, value in (headers or {}).items()
    )
    return tuple(sorted(items))


_headers_cache_key = headers_cache_key


def request_with_retries(
    session: _requests.Session,
    method: str,
    url: str,
    **kwargs: Any,
) -> _requests.Response:
    """Proxy to :func:`DocsToKG.ContentDownload.network.request_with_retries`.

    Args:
        session: Requests session used to perform the HTTP call.
        method: HTTP method such as ``"GET"`` or ``"HEAD"``.
        url: Fully-qualified URL for the request.
        **kwargs: Additional parameters forwarded to the network layer helper.

    Returns:
        requests.Response: Response returned by the shared network helper.

    Notes:
        The indirection keeps resolver providers compatible with tests that patch the
        network-layer helper while avoiding circular imports during module initialisation.
    """

    from DocsToKG.ContentDownload.network import request_with_retries as _request_with_retries

    return _request_with_retries(session, method, url, **kwargs)


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
    response = _requests.get(
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
    response = _requests.get(
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
    response = _requests.get(
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
        session: _requests.Session,
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
        session: _requests.Session,
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
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except _requests.RequestException as exc:
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
        session: _requests.Session,
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
            response: Optional[_requests.Response] = None
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
            except _requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
                )
                return
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"error": str(exc)},
                )
                return
            except _requests.RequestException as exc:
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
            except _requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=status,
                    metadata={"error_detail": f"Crossref HTTPError: {status}"},
                )
                return
            except _requests.RequestException as exc:  # pragma: no cover - network errors
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
        session: _requests.Session,
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
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except _requests.RequestException as exc:
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
        session: _requests.Session,
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
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.RequestException as exc:
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
        session: _requests.Session,
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
        except _requests.Timeout as exc:
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
        except _requests.RequestException as exc:
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
        session: _requests.Session,
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
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except _requests.RequestException as exc:
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
        session: _requests.Session,
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
            except _requests.Timeout as exc:
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
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"landing": landing, "error": str(exc)},
                )
                continue
            except _requests.RequestException as exc:  # pragma: no cover - network errors
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
        session: _requests.Session,
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
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except _requests.RequestException as exc:
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
        session: _requests.Session,
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
        session: _requests.Session,
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
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except _requests.RequestException as exc:
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
        self, session: _requests.Session, identifiers: List[str], config: ResolverConfig
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
        except _requests.Timeout as exc:
            LOGGER.debug("PMC ID lookup timed out: %s", exc)
            return []
        except _requests.ConnectionError as exc:
            LOGGER.debug("PMC ID lookup connection error: %s", exc)
            return []
        except _requests.RequestException as exc:
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
        session: _requests.Session,
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
            except _requests.Timeout as exc:
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
            except _requests.ConnectionError as exc:
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
            except _requests.RequestException as exc:
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
        session: _requests.Session,
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
        except _requests.HTTPError as exc:
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
        except _requests.Timeout as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="timeout",
                metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
            )
            return
        except _requests.ConnectionError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="connection-error",
                metadata={"error": str(exc)},
            )
            return
        except _requests.RequestException as exc:
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
        session: _requests.Session,
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
            except _requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
                )
                return
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"error": str(exc)},
                )
                return
            except _requests.RequestException as exc:
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
            except _requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=status,
                    metadata={"error_detail": f"Unpaywall HTTPError: {status}"},
                )
                return
            except _requests.Timeout as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={"timeout": config.get_timeout(self.name), "error": str(exc)},
                )
                return
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"error": str(exc)},
                )
                return
            except _requests.RequestException as exc:  # pragma: no cover - network errors
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
        session: _requests.Session,
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
            except _requests.Timeout as exc:
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
            except _requests.ConnectionError as exc:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"original": original, "error": str(exc)},
                )
                continue
            except _requests.RequestException as exc:
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
        session: _requests.Session,
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
        except _requests.Timeout as exc:
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
        except _requests.RequestException as exc:
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
    """Instantiate the default resolver stack in priority order.

    Args:
        None

    Returns:
        List[Resolver]: Resolver instances ordered according to
        ``DEFAULT_RESOLVER_ORDER``.

    Examples:
        >>> from DocsToKG.ContentDownload import resolvers
        >>> [resolver.name for resolver in resolvers.default_resolvers()]  # doctest: +ELLIPSIS
        ['openalex', 'unpaywall', ...]
    """

    return ResolverRegistry.create_default()


def _callable_accepts_argument(func: DownloadFunc, name: str) -> bool:
    """Return ``True`` when ``func`` accepts an argument named ``name``.

    Args:
        func: Download function whose call signature should be inspected.
        name: Argument name whose presence should be detected.

    Returns:
        bool: ``True`` when ``func`` accepts the argument or variable parameters.
    """

    try:
        from inspect import Parameter, signature
    except ImportError:  # pragma: no cover - inspect always available
        return True

    try:
        func_signature = signature(func)
    except (TypeError, ValueError):
        return True

    for parameter in func_signature.parameters.values():
        if parameter.kind in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            return True
        if parameter.name == name:
            return True
    return False


class _RunState:
    """Mutable pipeline execution state shared across resolvers.

    Args:
        dry_run: Indicates whether downloads should be skipped.

    Attributes:
        dry_run: Indicates whether downloads should be skipped.
        seen_urls: Set of URLs already attempted.
        html_paths: Collected HTML fallback paths.
        failed_urls: URLs that failed during resolution.
        attempt_counter: Total number of resolver attempts performed.

    Examples:
        >>> state = _RunState(dry_run=True)
        >>> state.dry_run
        True
    """

    __slots__ = (
        "dry_run",
        "seen_urls",
        "html_paths",
        "failed_urls",
        "attempt_counter",
    )

    def __init__(self, dry_run: bool) -> None:
        """Initialise run-state bookkeeping for a pipeline execution.

        Args:
            dry_run: Flag indicating whether downloads should be skipped.

        Returns:
            None
        """

        self.dry_run = dry_run
        self.seen_urls: set[str] = set()
        self.html_paths: List[str] = []
        self.failed_urls: List[str] = []
        self.attempt_counter = 0


class ResolverPipeline:
    """Executes resolvers in priority order until a PDF download succeeds.

    The pipeline is safe to reuse across worker threads when
    :attr:`ResolverConfig.max_concurrent_resolvers` is greater than one. All
    mutable shared state is protected by :class:`threading.Lock` instances and
    only read concurrently without mutation. HTTP ``_requests.Session`` objects
    are treated as read-only; callers must avoid mutating shared sessions after
    handing them to the pipeline.

    Attributes:
        config: Resolver configuration containing ordering and rate limits.
        download_func: Callable responsible for downloading resolved URLs.
        logger: Structured attempt logger capturing resolver telemetry.
        metrics: Metrics collector tracking resolver performance.

    Examples:
        >>> pipeline = ResolverPipeline([], ResolverConfig(), lambda *args, **kwargs: None, None)  # doctest: +SKIP
        >>> isinstance(pipeline.metrics, ResolverMetrics)  # doctest: +SKIP
        True
    """

    def __init__(
        self,
        resolvers: Sequence[Resolver],
        config: ResolverConfig,
        download_func: DownloadFunc,
        logger: AttemptLogger,
        metrics: Optional[ResolverMetrics] = None,
    ) -> None:
        """Create a resolver pipeline with ordering, download, and metric hooks.

        Args:
            resolvers: Resolver instances available for execution.
            config: Pipeline configuration controlling ordering and limits.
            download_func: Callable responsible for downloading resolved URLs.
            logger: Logger that records resolver attempt metadata.
            metrics: Optional metrics collector used for resolver telemetry.

        Returns:
            None
        """

        self._resolver_map = {resolver.name: resolver for resolver in resolvers}
        self.config = config
        self.download_func = download_func
        self.logger = logger
        self.metrics = metrics or ResolverMetrics()
        self._last_invocation: Dict[str, float] = defaultdict(lambda: 0.0)
        self._lock = threading.Lock()
        self._global_seen_urls: set[str] = set()
        self._global_lock = threading.Lock()
        self._last_host_hit: defaultdict[str, float] = defaultdict(float)
        self._host_lock = threading.Lock()
        self._download_accepts_context = _callable_accepts_argument(download_func, "context")
        self._download_accepts_head_flag = _callable_accepts_argument(
            download_func, "head_precheck_passed"
        )

    def _respect_rate_limit(self, resolver_name: str) -> None:
        """Sleep as required to respect per-resolver rate limiting policies.

        The method performs an atomic read-modify-write on
        :attr:`_last_invocation` guarded by :attr:`_lock` to ensure that
        concurrent threads honour resolver spacing requirements.

        Args:
            resolver_name: Name of the resolver to rate limit.

        Returns:
            None
        """

        limit = self.config.resolver_min_interval_s.get(resolver_name)
        if not limit:
            limit = self.config.resolver_rate_limits.get(resolver_name)
        if not limit:
            return
        wait = 0.0
        with self._lock:
            last = self._last_invocation[resolver_name]
            now = _time.monotonic()
            delta = now - last
            if delta < limit:
                wait = limit - delta
            self._last_invocation[resolver_name] = now + wait
        if wait > 0:
            _time.sleep(wait)

    def _respect_domain_limit(self, url: str) -> None:
        """Enforce per-domain throttling when configured.

        Args:
            url: Resolver URL whose host may be subject to throttling.

        Returns:
            None
        """

        if not url or not self.config.domain_min_interval_s:
            return
        host = urlsplit(url).netloc.lower()
        if not host:
            return
        interval = self.config.domain_min_interval_s.get(host)
        if not interval:
            return
        now = _time.monotonic()
        wait = 0.0
        with self._host_lock:
            last = self._last_host_hit[host]
            if last <= 0.0:
                self._last_host_hit[host] = now if now > 0.0 else 1e-9
                return
            elapsed = now - last
            if elapsed >= interval:
                self._last_host_hit[host] = now
                return
            wait = interval - elapsed
        if wait > 0:
            _time.sleep(wait)
            with self._host_lock:
                self._last_host_hit[host] = _time.monotonic()

    def _jitter_sleep(self) -> None:
        """Introduce a small delay to avoid stampeding downstream services.

        Args:
            self: Pipeline instance executing resolver scheduling logic.

        Returns:
            None
        """

        if self.config.sleep_jitter <= 0:
            return
        _time.sleep(self.config.sleep_jitter + random.random() * 0.1)

    def _should_attempt_head_check(self, resolver_name: str) -> bool:
        """Return ``True`` when a resolver should perform a HEAD preflight request.

        Args:
            resolver_name: Name of the resolver under consideration.

        Returns:
            Boolean indicating whether the resolver should issue a HEAD request.
        """

        if resolver_name in self.config.resolver_head_precheck:
            return self.config.resolver_head_precheck[resolver_name]
        return self.config.enable_head_precheck

    def _head_precheck_url(
        self,
        session: _requests.Session,
        url: str,
        timeout: float,
    ) -> bool:
        """Issue a HEAD request to validate that ``url`` plausibly returns a PDF.

        Args:
            session: Requests session used for issuing the HEAD request.
            url: Candidate URL whose response should be inspected.
            timeout: Timeout budget for the preflight request.

        Returns:
            ``True`` when the response appears to represent a PDF download.
        """

        try:
            response = request_with_retries(
                session,
                "HEAD",
                url,
                max_retries=1,
                timeout=min(timeout, 5.0),
                allow_redirects=True,
            )
        except Exception:
            return True

        try:
            if response.status_code not in {200, 302, 304}:
                return False

            content_type = (response.headers.get("Content-Type") or "").lower()
            content_length = response.headers.get("Content-Length", "")

            if "text/html" in content_type:
                return False
            if content_length == "0":
                return False

            return True
        finally:
            response.close()

    def run(
        self,
        session: _requests.Session,
        artifact: "WorkArtifact",
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute resolvers until a PDF is obtained or resolvers are exhausted.

        Args:
            session: Requests session used for resolver HTTP calls.
            artifact: Work artifact describing the document to resolve.
            context: Optional execution context containing flags such as ``dry_run``.

        Returns:
            PipelineResult capturing resolver attempts and successful downloads.
        """

        context_data: Dict[str, Any] = context or {}
        state = _RunState(dry_run=bool(context_data.get("dry_run", False)))

        if self.config.max_concurrent_resolvers == 1:
            return self._run_sequential(session, artifact, context_data, state)

        return self._run_concurrent(session, artifact, context_data, state)

    def _run_sequential(
        self,
        session: _requests.Session,
        artifact: "WorkArtifact",
        context_data: Dict[str, Any],
        state: _RunState,
    ) -> PipelineResult:
        """Execute resolvers in order using the current thread.

        Args:
            session: Shared requests session for resolver HTTP calls.
            artifact: Work artifact describing the document being processed.
            context_data: Execution context dictionary.
            state: Mutable run state tracking attempts and duplicates.

        Returns:
            PipelineResult summarising the sequential run outcome.
        """

        for order_index, resolver_name in enumerate(self.config.resolver_order, start=1):
            resolver = self._prepare_resolver(resolver_name, order_index, artifact, state)
            if resolver is None:
                continue

            results, wall_ms = self._collect_resolver_results(
                resolver_name,
                resolver,
                session,
                artifact,
            )

            for result in results:
                pipeline_result = self._process_result(
                    session,
                    artifact,
                    resolver_name,
                    order_index,
                    result,
                    context_data,
                    state,
                    resolver_wall_time_ms=wall_ms,
                )
                if pipeline_result is not None:
                    return pipeline_result

        return PipelineResult(
            success=False,
            html_paths=list(state.html_paths),
            failed_urls=list(state.failed_urls),
        )

    def _run_concurrent(
        self,
        session: _requests.Session,
        artifact: "WorkArtifact",
        context_data: Dict[str, Any],
        state: _RunState,
    ) -> PipelineResult:
        """Execute resolvers concurrently using a thread pool.

        Args:
            session: Shared requests session for resolver HTTP calls.
            artifact: Work artifact describing the document being processed.
            context_data: Execution context dictionary.
            state: Mutable run state tracking attempts and duplicates.

        Returns:
            PipelineResult summarising the concurrent run outcome.
        """

        max_workers = self.config.max_concurrent_resolvers
        active_futures: Dict[Future[Tuple[List[ResolverResult], float]], Tuple[str, int]] = {}

        def submit_next(
            executor: ThreadPoolExecutor,
            start_index: int,
        ) -> int:
            """Queue additional resolvers until reaching concurrency limits.

            Args:
                executor: Thread pool responsible for executing resolver calls.
                start_index: Index in ``resolver_order`` where submission should resume.

            Returns:
                Updated index pointing to the next resolver candidate that has not been submitted.
            """
            index = start_index
            while len(active_futures) < max_workers and index < len(self.config.resolver_order):
                resolver_name = self.config.resolver_order[index]
                order_index = index + 1
                index += 1
                resolver = self._prepare_resolver(resolver_name, order_index, artifact, state)
                if resolver is None:
                    continue
                future = executor.submit(
                    self._collect_resolver_results,
                    resolver_name,
                    resolver,
                    session,
                    artifact,
                )
                active_futures[future] = (resolver_name, order_index)
            return index

        next_index = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            next_index = submit_next(executor, next_index)

            while active_futures:
                done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    resolver_name, order_index = active_futures.pop(future)
                    try:
                        results, wall_ms = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        results = [
                            ResolverResult(
                                url=None,
                                event="error",
                                event_reason="resolver-exception",
                                metadata={"message": str(exc)},
                            )
                        ]
                        wall_ms = 0.0

                    for result in results:
                        pipeline_result = self._process_result(
                            session,
                            artifact,
                            resolver_name,
                            order_index,
                            result,
                            context_data,
                            state,
                            resolver_wall_time_ms=wall_ms,
                        )
                        if pipeline_result is not None:
                            executor.shutdown(wait=False, cancel_futures=True)
                            return pipeline_result

                next_index = submit_next(executor, next_index)

        return PipelineResult(
            success=False,
            html_paths=list(state.html_paths),
            failed_urls=list(state.failed_urls),
        )

    def _prepare_resolver(
        self,
        resolver_name: str,
        order_index: int,
        artifact: "WorkArtifact",
        state: _RunState,
    ) -> Optional[Resolver]:
        """Return a prepared resolver or log skip events when unavailable.

        Args:
            resolver_name: Name of the resolver to prepare.
            order_index: Execution order index for the resolver.
            artifact: Work artifact being processed.
            state: Mutable run state tracking skips and duplicates.

        Returns:
            Resolver instance when available and enabled, otherwise ``None``.
        """

        resolver = self._resolver_map.get(resolver_name)
        if resolver is None:
            self.logger.log(
                AttemptRecord(
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=None,
                    status="skipped",
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason="resolver-missing",
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=0.0,
                )
            )
            self.metrics.record_skip(resolver_name, "missing")
            return None

        if not self.config.is_enabled(resolver_name):
            self.logger.log(
                AttemptRecord(
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=None,
                    status="skipped",
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason="resolver-disabled",
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=0.0,
                )
            )
            self.metrics.record_skip(resolver_name, "disabled")
            return None

        if not resolver.is_enabled(self.config, artifact):
            self.logger.log(
                AttemptRecord(
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=None,
                    status="skipped",
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason="resolver-not-applicable",
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=0.0,
                )
            )
            self.metrics.record_skip(resolver_name, "not-applicable")
            return None

        return resolver

    def _collect_resolver_results(
        self,
        resolver_name: str,
        resolver: Resolver,
        session: _requests.Session,
        artifact: "WorkArtifact",
    ) -> Tuple[List[ResolverResult], float]:
        """Collect resolver results while applying rate limits and error handling.

        Args:
            resolver_name: Name of the resolver being executed (for logging and limits).
            resolver: Resolver instance that will generate candidate URLs.
            session: Requests session forwarded to the resolver.
            artifact: Work artifact describing the current document.

        Returns:
            Tuple of resolver results and the resolver wall time (ms).
        """

        results: List[ResolverResult] = []
        self._respect_rate_limit(resolver_name)
        start = _time.monotonic()
        try:
            for result in resolver.iter_urls(session, self.config, artifact):
                results.append(result)
        except Exception as exc:
            self.metrics.record_failure(resolver_name)
            results.append(
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="resolver-exception",
                    metadata={"message": str(exc)},
                )
            )
        wall_ms = (_time.monotonic() - start) * 1000.0
        return results, wall_ms

    def _process_result(
        self,
        session: _requests.Session,
        artifact: "WorkArtifact",
        resolver_name: str,
        order_index: int,
        result: ResolverResult,
        context_data: Dict[str, Any],
        state: _RunState,
        *,
        resolver_wall_time_ms: Optional[float] = None,
    ) -> Optional[PipelineResult]:
        """Process a single resolver result and return a terminal pipeline outcome.

        Args:
            session: Requests session used for follow-up download calls.
            artifact: Work artifact describing the document being processed.
            resolver_name: Name of the resolver that produced the result.
            order_index: 1-based index of the resolver in the execution order.
            result: Resolver result containing either a URL or event metadata.
            context_data: Execution context dictionary.
            state: Mutable run state tracking attempts and duplicates.
            resolver_wall_time_ms: Wall-clock time spent in the resolver.

        Returns:
            PipelineResult when resolution succeeds, otherwise ``None``.
        """

        if result.is_event:
            self.logger.log(
                AttemptRecord(
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=None,
                    status=result.event or "event",
                    http_status=result.http_status,
                    content_type=None,
                    elapsed_ms=None,
                    reason=result.event_reason,
                    metadata=result.metadata,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=resolver_wall_time_ms,
                )
            )
            if result.event_reason:
                self.metrics.record_skip(resolver_name, result.event_reason)
            return None

        url = result.url
        if not url:
            return None
        if self.config.enable_global_url_dedup:
            with self._global_lock:
                duplicate = url in self._global_seen_urls
                if not duplicate:
                    self._global_seen_urls.add(url)
            if duplicate:
                self.logger.log(
                    AttemptRecord(
                        work_id=artifact.work_id,
                        resolver_name=resolver_name,
                        resolver_order=order_index,
                        url=url,
                        status="skipped",
                        http_status=None,
                        content_type=None,
                        elapsed_ms=None,
                        reason="duplicate-url-global",
                        metadata=result.metadata,
                        dry_run=state.dry_run,
                        resolver_wall_time_ms=resolver_wall_time_ms,
                    )
                )
                self.metrics.record_skip(resolver_name, "duplicate-url-global")
                return None
        if url in state.seen_urls:
            self.logger.log(
                AttemptRecord(
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=url,
                    status="skipped",
                    http_status=None,
                    content_type=None,
                    elapsed_ms=None,
                    reason="duplicate-url",
                    metadata=result.metadata,
                    dry_run=state.dry_run,
                    resolver_wall_time_ms=resolver_wall_time_ms,
                )
            )
            self.metrics.record_skip(resolver_name, "duplicate-url")
            return None

        state.seen_urls.add(url)
        download_context = dict(context_data)
        head_precheck_passed = False
        if self._should_attempt_head_check(resolver_name):
            head_precheck_passed = self._head_precheck_url(
                session,
                url,
                self.config.get_timeout(resolver_name),
            )
            if not head_precheck_passed:
                self.logger.log(
                    AttemptRecord(
                        work_id=artifact.work_id,
                        resolver_name=resolver_name,
                        resolver_order=order_index,
                        url=url,
                        status="skipped",
                        http_status=None,
                        content_type=None,
                        elapsed_ms=None,
                        reason="head-precheck-failed",
                        metadata=result.metadata,
                        dry_run=state.dry_run,
                        resolver_wall_time_ms=resolver_wall_time_ms,
                    )
                )
                self.metrics.record_skip(resolver_name, "head-precheck-failed")
                return None
            head_precheck_passed = True

        state.attempt_counter += 1
        self._respect_domain_limit(url)
        kwargs: Dict[str, Any] = {}
        if self._download_accepts_head_flag:
            kwargs["head_precheck_passed"] = head_precheck_passed

        if self._download_accepts_context:
            outcome = self.download_func(
                session,
                artifact,
                url,
                result.referer,
                self.config.get_timeout(resolver_name),
                download_context,
                **kwargs,
            )
        else:
            outcome = self.download_func(
                session,
                artifact,
                url,
                result.referer,
                self.config.get_timeout(resolver_name),
                **kwargs,
            )

        self.logger.log(
            AttemptRecord(
                work_id=artifact.work_id,
                resolver_name=resolver_name,
                resolver_order=order_index,
                url=url,
                status=outcome.classification,
                http_status=outcome.http_status,
                content_type=outcome.content_type,
                elapsed_ms=outcome.elapsed_ms,
                reason=outcome.error,
                metadata=result.metadata,
                sha256=outcome.sha256,
                content_length=outcome.content_length,
                dry_run=state.dry_run,
                resolver_wall_time_ms=resolver_wall_time_ms,
            )
        )
        self.metrics.record_attempt(resolver_name, outcome)

        if outcome.classification == "html" and outcome.path:
            state.html_paths.append(outcome.path)

        if not outcome.is_pdf and url:
            if url not in state.failed_urls:
                state.failed_urls.append(url)
            if url not in artifact.failed_pdf_urls:
                artifact.failed_pdf_urls.append(url)

        if outcome.is_pdf:
            return PipelineResult(
                success=True,
                resolver_name=resolver_name,
                url=url,
                outcome=outcome,
                html_paths=list(state.html_paths),
                failed_urls=list(state.failed_urls),
            )

        if state.attempt_counter >= self.config.max_attempts_per_work:
            return PipelineResult(
                success=False,
                resolver_name=resolver_name,
                url=url,
                outcome=outcome,
                html_paths=list(state.html_paths),
                failed_urls=list(state.failed_urls),
                reason="max-attempts-reached",
            )

        self._jitter_sleep()
        return None


def clear_resolver_caches() -> None:
    """Clear resolver-level HTTP caches to force fresh lookups.

    This utility resets the internal LRU caches used by the Unpaywall,
    Crossref, and Semantic Scholar resolvers. It should be called before
    executing resolver pipelines when deterministic behaviour across runs is
    required (for example, in unit tests or benchmarking scenarios).

    Args:
        None

    Returns:
        None
    """

    _fetch_unpaywall_data.cache_clear()
    _fetch_crossref_data.cache_clear()
    _fetch_semantic_scholar_data.cache_clear()


_LEGACY_EXPORTS = {
    "time": _time_alias,
    "requests": _requests_alias,
}

_DEPRECATION_MESSAGES = {
    "time": (
        "DocsToKG.ContentDownload.resolvers.time is deprecated; import 'time' "
        "directly. This alias will be removed in a future release."
    ),
    "requests": (
        "DocsToKG.ContentDownload.resolvers.requests is deprecated; import the "
        "'requests' package directly. This alias will be removed in a future release."
    ),
}


def __getattr__(name: str):
    """Return legacy exports while emitting :class:`DeprecationWarning`."""

    if name in _LEGACY_EXPORTS:
        warnings.warn(
            _DEPRECATION_MESSAGES.get(
                name,
                f"DocsToKG.ContentDownload.resolvers.{name} is deprecated",
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return _LEGACY_EXPORTS[name]
    raise AttributeError(name)


__all__ = [
    "AttemptLogger",
    "AttemptRecord",
    "DownloadFunc",
    "DownloadOutcome",
    "PipelineResult",
    "Resolver",
    "ResolverConfig",
    "ResolverMetrics",
    "ResolverPipeline",
    "ResolverRegistry",
    "RegisteredResolver",
    "ResolverResult",
    "DEFAULT_RESOLVER_ORDER",
    "default_resolvers",
    "clear_resolver_caches",
    "headers_cache_key",
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
    "time",
    "requests",
]
