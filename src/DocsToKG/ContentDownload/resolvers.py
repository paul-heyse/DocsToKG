# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers",
#   "purpose": "Resolver definitions and pipeline orchestration",
#   "sections": [
#     {"id": "ResolverResult", "name": "ResolverResult", "anchor": "class-ResolverResult", "kind": "class"},
#     {"id": "ResolverConfig", "name": "ResolverConfig", "anchor": "class-ResolverConfig", "kind": "class"},
#     {"id": "AttemptRecord", "name": "AttemptRecord", "anchor": "class-AttemptRecord", "kind": "class"},
#     {"id": "DownloadOutcome", "name": "DownloadOutcome", "anchor": "class-DownloadOutcome", "kind": "class"},
#     {"id": "PipelineResult", "name": "PipelineResult", "anchor": "class-PipelineResult", "kind": "class"},
#     {"id": "Resolver", "name": "Resolver", "anchor": "class-Resolver", "kind": "class"},
#     {"id": "ResolverMetrics", "name": "ResolverMetrics", "anchor": "class-ResolverMetrics", "kind": "class"},
#     {"id": "_fetch_semantic_scholar_data", "name": "_fetch_semantic_scholar_data", "anchor": "function-_fetch_semantic_scholar_data", "kind": "function"},
#     {"id": "_fetch_unpaywall_data", "name": "_fetch_unpaywall_data", "anchor": "function-_fetch_unpaywall_data", "kind": "function"},
#     {"id": "ResolverRegistry", "name": "ResolverRegistry", "anchor": "class-ResolverRegistry", "kind": "class"},
#     {"id": "RegisteredResolver", "name": "RegisteredResolver", "anchor": "class-RegisteredResolver", "kind": "class"},
#     {"id": "ApiResolverBase", "name": "ApiResolverBase", "anchor": "class-ApiResolverBase", "kind": "class"},
#     {"id": "_absolute_url", "name": "_absolute_url", "anchor": "function-_absolute_url", "kind": "function"},
#     {"id": "_collect_candidate_urls", "name": "_collect_candidate_urls", "anchor": "function-_collect_candidate_urls", "kind": "function"},
#     {"id": "find_pdf_via_meta", "name": "find_pdf_via_meta", "anchor": "function-find_pdf_via_meta", "kind": "function"},
#     {"id": "find_pdf_via_link", "name": "find_pdf_via_link", "anchor": "function-find_pdf_via_link", "kind": "function"},
#     {"id": "find_pdf_via_anchor", "name": "find_pdf_via_anchor", "anchor": "function-find_pdf_via_anchor", "kind": "function"},
#     {"id": "default_resolvers", "name": "default_resolvers", "anchor": "function-default_resolvers", "kind": "function"},
#     {"id": "ArxivResolver", "name": "ArxivResolver", "anchor": "class-ArxivResolver", "kind": "class"},
#     {"id": "CoreResolver", "name": "CoreResolver", "anchor": "class-CoreResolver", "kind": "class"},
#     {"id": "CrossrefResolver", "name": "CrossrefResolver", "anchor": "class-CrossrefResolver", "kind": "class"},
#     {"id": "DoajResolver", "name": "DoajResolver", "anchor": "class-DoajResolver", "kind": "class"},
#     {"id": "EuropePmcResolver", "name": "EuropePmcResolver", "anchor": "class-EuropePmcResolver", "kind": "class"},
#     {"id": "FigshareResolver", "name": "FigshareResolver", "anchor": "class-FigshareResolver", "kind": "class"},
#     {"id": "HalResolver", "name": "HalResolver", "anchor": "class-HalResolver", "kind": "class"},
#     {"id": "LandingPageResolver", "name": "LandingPageResolver", "anchor": "class-LandingPageResolver", "kind": "class"},
#     {"id": "OpenAireResolver", "name": "OpenAireResolver", "anchor": "class-OpenAireResolver", "kind": "class"},
#     {"id": "OpenAlexResolver", "name": "OpenAlexResolver", "anchor": "class-OpenAlexResolver", "kind": "class"},
#     {"id": "OsfResolver", "name": "OsfResolver", "anchor": "class-OsfResolver", "kind": "class"},
#     {"id": "PmcResolver", "name": "PmcResolver", "anchor": "class-PmcResolver", "kind": "class"},
#     {"id": "SemanticScholarResolver", "name": "SemanticScholarResolver", "anchor": "class-SemanticScholarResolver", "kind": "class"},
#     {"id": "UnpaywallResolver", "name": "UnpaywallResolver", "anchor": "class-UnpaywallResolver", "kind": "class"},
#     {"id": "WaybackResolver", "name": "WaybackResolver", "anchor": "class-WaybackResolver", "kind": "class"},
#     {"id": "ZenodoResolver", "name": "ZenodoResolver", "anchor": "class-ZenodoResolver", "kind": "class"},
#     {"id": "_callable_accepts_argument", "name": "_callable_accepts_argument", "anchor": "function-_callable_accepts_argument", "kind": "function"},
#     {"id": "_RunState", "name": "_RunState", "anchor": "class-_RunState", "kind": "class"},
#     {"id": "ResolverPipeline", "name": "ResolverPipeline", "anchor": "class-ResolverPipeline", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

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
import math
import random
import re
import threading
import time as _time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
)
from urllib.parse import quote, urljoin, urlparse, urlsplit

import requests as _requests

from DocsToKG.ContentDownload.network import head_precheck, request_with_retries
from DocsToKG.ContentDownload.classifications import Classification, PDF_LIKE
from DocsToKG.ContentDownload.utils import (
    dedupe,
    normalize_doi,
    normalize_pmcid,
    normalize_url,
    strip_prefix,
)
from DocsToKG.ContentDownload.telemetry import AttemptSink

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact

try:  # Optional dependency guarded at runtime
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    BeautifulSoup = None

# --- Globals ---

LOGGER = logging.getLogger(__name__)

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
DEFAULT_RESOLVER_TOGGLES = MappingProxyType(_DEFAULT_RESOLVER_TOGGLES)

__all__ = [
    "AttemptRecord",
    "AttemptSink",
    "ApiResolverBase",
    "ArxivResolver",
    "CoreResolver",
    "CrossrefResolver",
    "DEFAULT_RESOLVER_ORDER",
    "DEFAULT_RESOLVER_TOGGLES",
    "DoajResolver",
    "DownloadFunc",
    "DownloadOutcome",
    "EuropePmcResolver",
    "FigshareResolver",
    "HalResolver",
    "LandingPageResolver",
    "OpenAireResolver",
    "OpenAlexResolver",
    "OsfResolver",
    "PmcResolver",
    "PipelineResult",
    "Resolver",
    "ResolverConfig",
    "ResolverMetrics",
    "ResolverPipeline",
    "ResolverRegistry",
    "ResolverResult",
    "RegisteredResolver",
    "SemanticScholarResolver",
    "UnpaywallResolver",
    "WaybackResolver",
    "ZenodoResolver",
    "default_resolvers",
    "find_pdf_via_anchor",
    "find_pdf_via_link",
    "find_pdf_via_meta",
]

# --- Resolver Configuration ---

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
    enable_head_precheck: bool = True
    resolver_head_precheck: Dict[str, bool] = field(default_factory=dict)
    head_precheck_host_overrides: Dict[str, bool] = field(default_factory=dict)
    mailto: Optional[str] = None
    max_concurrent_resolvers: int = 1
    enable_global_url_dedup: bool = False
    # Heuristic knobs (defaults preserve current CLI behaviour)
    sniff_bytes: int = 64 * 1024
    min_pdf_bytes: int = 1024
    tail_check_bytes: int = 2048

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

    classification: Classification | str
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

        classification = Classification.from_wire(self.classification)
        return classification in PDF_LIKE


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
    latency_ms: defaultdict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    status_counts: defaultdict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    error_reasons: defaultdict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record_attempt(self, resolver_name: str, outcome: DownloadOutcome) -> None:
        """Record a resolver attempt and update success/html counters.

        Args:
            resolver_name: Name of the resolver that executed.
            outcome: Download outcome produced by the resolver.

        Returns:
            None
        """

        classification = Classification.from_wire(outcome.classification)
        with self._lock:
            self.attempts[resolver_name] += 1
            if classification is Classification.HTML:
                self.html[resolver_name] += 1
            if outcome.is_pdf:
                self.successes[resolver_name] += 1
            classification_key = (
                classification.value if isinstance(classification, Classification) else str(outcome.classification)
            )
            if classification_key:
                self.status_counts[resolver_name][classification_key] += 1
            if outcome.elapsed_ms is not None:
                self.latency_ms[resolver_name].append(float(outcome.elapsed_ms))
            reason = outcome.error
            if not reason and classification_key and classification_key not in {
                Classification.PDF.value,
                Classification.PDF_UNKNOWN.value,
            }:
                reason = classification_key
            if reason:
                self.error_reasons[resolver_name][reason] += 1

    def record_skip(self, resolver_name: str, reason: str) -> None:
        """Record a skip event for a resolver with a reason tag.

        Args:
            resolver_name: Resolver that was skipped.
            reason: Short description explaining the skip.

        Returns:
            None
        """

        with self._lock:
            key = f"{resolver_name}:{reason}"
            self.skips[key] += 1

    def record_failure(self, resolver_name: str) -> None:
        """Record a resolver failure occurrence.

        Args:
            resolver_name: Resolver that raised an exception during execution.

        Returns:
            None
        """

        with self._lock:
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

        def _percentile(sorted_values: List[float], percentile: float) -> float:
            if not sorted_values:
                return 0.0
            if len(sorted_values) == 1:
                return sorted_values[0]
            k = (len(sorted_values) - 1) * (percentile / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_values[int(k)]
            d0 = sorted_values[f] * (c - k)
            d1 = sorted_values[c] * (k - f)
            return d0 + d1

        with self._lock:
            latency_summary: Dict[str, Dict[str, float]] = {}
            for resolver_name, samples in self.latency_ms.items():
                if not samples:
                    continue
                ordered = sorted(samples)
                count = len(samples)
                total = sum(samples)
                latency_summary[resolver_name] = {
                    "count": count,
                    "mean_ms": total / count,
                    "min_ms": ordered[0],
                    "p50_ms": _percentile(ordered, 50.0),
                    "p95_ms": _percentile(ordered, 95.0),
                    "max_ms": ordered[-1],
                }

            status_summary = {resolver: dict(counter) for resolver, counter in self.status_counts.items() if counter}
            reason_summary = {
                resolver: [
                    {"reason": reason, "count": count}
                    for reason, count in counter.most_common(5)
                ]
                for resolver, counter in self.error_reasons.items()
                if counter
            }

            return {
                "attempts": dict(self.attempts),
                "successes": dict(self.successes),
                "html": dict(self.html),
                "skips": dict(self.skips),
                "failures": dict(self.failures),
                "latency_ms": latency_summary,
                "status_counts": status_summary,
                "error_reasons": reason_summary,
            }


# --- Shared Helpers ---

def _fetch_semantic_scholar_data(
    session: _requests.Session,
    config: ResolverConfig,
    doi: str,
) -> Dict[str, Any]:
    """Return Semantic Scholar metadata for ``doi`` using configured headers."""

    headers = dict(config.polite_headers)
    if config.semantic_scholar_api_key:
        headers["x-api-key"] = config.semantic_scholar_api_key
    response = request_with_retries(
        session,
        "GET",
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}",
        params={"fields": "title,openAccessPdf"},
        timeout=config.get_timeout("semantic_scholar"),
        headers=headers,
    )
    try:
        if response.status_code != 200:
            error = _requests.HTTPError(
                f"Semantic Scholar HTTPError: {response.status_code}"
            )
            error.response = response
            raise error
        return response.json()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _fetch_unpaywall_data(
    session: _requests.Session,
    config: ResolverConfig,
    doi: str,
) -> Dict[str, Any]:
    """Return Unpaywall metadata for ``doi`` using configured headers."""

    endpoint = f"https://api.unpaywall.org/v2/{quote(doi)}"
    headers = dict(config.polite_headers)
    params = {"email": config.unpaywall_email} if config.unpaywall_email else None
    response = request_with_retries(
        session,
        "GET",
        endpoint,
        params=params,
        timeout=config.get_timeout("unpaywall"),
        headers=headers,
    )
    try:
        if response.status_code != 200:
            error = _requests.HTTPError(f"Unpaywall HTTPError: {response.status_code}")
            error.response = response
            raise error
        return response.json()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


DownloadFunc = Callable[..., DownloadOutcome]


# --- Resolver Registry ---


class ResolverRegistry:
    """Registry tracking resolver classes by their ``name`` attribute.

    Attributes:
        _providers: Mapping of resolver names to resolver classes.

    Examples:
        >>> ResolverRegistry.register(type("Tmp", (RegisteredResolver,), {"name": "tmp"}))  # doctest: +SKIP
        <class 'Tmp'>
    """

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

        Raises:
            None.
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
    """Mixin ensuring subclasses register with :class:`ResolverRegistry`.

    Attributes:
        None: Subclasses inherit registration behaviour automatically.

    Examples:
        >>> class ExampleResolver(RegisteredResolver):
        ...     name = "example"
        ...     def is_enabled(self, config, artifact):
        ...         return True
        ...     def iter_urls(self, session, config, artifact):
        ...         yield ResolverResult(url="https://example.org")
    """

    def __init_subclass__(cls, register: bool = True, **kwargs: Any) -> None:  # type: ignore[override]
        super().__init_subclass__(**kwargs)
        if register:
            ResolverRegistry.register(cls)  # type: ignore[arg-type]


class ApiResolverBase(RegisteredResolver, register=False):
    """Shared helper for resolvers interacting with JSON-based HTTP APIs.

    Attributes:
        name: Resolver identifier used for logging and registration.

    Examples:
        >>> class ExampleApiResolver(ApiResolverBase):
        ...     name = "example-api"
        ...     def iter_urls(self, session, config, artifact):
        ...         payload, result = self._request_json(
        ...             session,
        ...             \"GET\",
        ...             \"https://example.org/api\",
        ...             config=config,
        ...         )
        ...         if payload:
        ...             yield result  # doctest: +SKIP
    """

    def _request_json(
        self,
        session: _requests.Session,
        method: str,
        url: str,
        *,
        config: ResolverConfig,
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
                session,
                method,
                url,
                params=params,
                json=json,
                headers=request_headers,
                timeout=timeout_value,
                **kwargs,
            )
        except _requests.Timeout as exc:
            return (
                None,
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="timeout",
                    metadata={
                        "url": url,
                        "timeout": timeout_value,
                        "error": str(exc),
                    },
                ),
            )
        except _requests.ConnectionError as exc:
            return (
                None,
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="connection-error",
                    metadata={"url": url, "error": str(exc)},
                ),
            )
        except _requests.RequestException as exc:
            return (
                None,
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"url": url, "error": str(exc)},
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Unexpected error issuing %s request to %s", method, url)
            return (
                None,
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="unexpected-error",
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
                        event="error",
                        event_reason="http-error",
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
                    event="error",
                    event_reason="json-error",
                    metadata={
                        "url": url,
                        "error_detail": str(json_err),
                        "content_preview": preview,
                    },
                ),
            )
        finally:
            close = getattr(response, 'close', None)
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


def find_pdf_via_meta(soup: "BeautifulSoup", base_url: str) -> Optional[str]:
    """Return PDF URL declared via ``citation_pdf_url`` meta tags.

    Args:
        soup: Parsed HTML document to inspect for metadata tags.
        base_url: URL used to resolve relative PDF links.

    Returns:
        Optional[str]: Absolute PDF URL when one is advertised; otherwise ``None``.
    """

    tag = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if tag is None:
        return None
    href = (tag.get("content") or "").strip()
    if not href:
        return None
    return _absolute_url(base_url, href)


def find_pdf_via_link(soup: "BeautifulSoup", base_url: str) -> Optional[str]:
    """Return PDF URL referenced via ``<link rel=\"alternate\" type=\"application/pdf\">``.

    Args:
        soup: Parsed HTML document to inspect for ``<link>`` elements.
        base_url: URL used to resolve relative ``href`` attributes.

    Returns:
        Optional[str]: Absolute PDF URL if link metadata is present; otherwise ``None``.
    """

    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel") or []).lower()
        typ = (link.get("type") or "").lower()
        href = (link.get("href") or "").strip()
        if "alternate" in rel and "application/pdf" in typ and href:
            return _absolute_url(base_url, href)
    return None


def find_pdf_via_anchor(soup: "BeautifulSoup", base_url: str) -> Optional[str]:
    """Return PDF URL inferred from anchor elements mentioning PDFs.

    Args:
        soup: Parsed HTML document to search for anchor tags referencing PDFs.
        base_url: URL used to resolve relative anchor targets.

    Returns:
        Optional[str]: Absolute PDF URL when an anchor appears to reference one.
    """

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


def default_resolvers() -> List[Resolver]:
    """Instantiate the default resolver stack in priority order.

    Args:
        None

    Returns:
        List[Resolver]: Resolver instances ordered according to ``DEFAULT_RESOLVER_ORDER``.
    """

    return ResolverRegistry.create_default()


# --- Resolver implementations ---

class ArxivResolver(RegisteredResolver):
    """Resolve arXiv preprints using arXiv identifier lookups.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> ArxivResolver().name
        'arxiv'
    """

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


class CoreResolver(ApiResolverBase):
    """Resolve PDFs using the CORE API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> CoreResolver().name
        'core'
    """

    name = "core"
    api_display_name = "CORE"

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
        headers = {"Authorization": f"Bearer {config.core_api_key}"}
        data, error = self._request_json(
            session,
            "GET",
            "https://api.core.ac.uk/v3/search/works",
            config=config,
            params={"q": f'doi:"{doi}"', "page": 1, "pageSize": 3},
            headers=headers,
        )
        if error:
            yield error
            return
        results = (data.get("results") if isinstance(data, dict) else None) or []
        for hit in results:
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


class CrossrefResolver(ApiResolverBase):
    """Resolve candidate URLs from the Crossref metadata API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> CrossrefResolver().name
        'crossref'
    """

    name = "crossref"
    api_display_name = "Crossref"

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
        params = {"mailto": email} if email else None
        data, error = self._request_json(
            session,
            "GET",
            f"https://api.crossref.org/works/{quote(doi)}",
            config=config,
            params=params,
        )
        if error:
            yield error
            return

        message = (data.get("message") if isinstance(data, dict) else None) or {}
        link_section = message.get("link") or []
        if not isinstance(link_section, list):
            link_section = []

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for entry in link_section:
            if not isinstance(entry, dict):
                continue
            url = entry.get("URL")
            content_type = entry.get("content-type")
            ctype = (content_type or "").lower()
            if url and ctype in {"application/pdf", "application/x-pdf", "text/html"}:
                candidates.append((url, {"content_type": content_type}))

        for url in dedupe([candidate_url for candidate_url, _ in candidates]):
            for candidate_url, metadata in candidates:
                if candidate_url == url:
                    yield ResolverResult(url=url, metadata=metadata)
                    break


class DoajResolver(ApiResolverBase):
    """Resolve Open Access links using the DOAJ API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> DoajResolver().name
        'doaj'
    """

    name = "doaj"
    api_display_name = "DOAJ"

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
        extra_headers: Dict[str, str] = {}
        if config.doaj_api_key:
            extra_headers["X-API-KEY"] = config.doaj_api_key
        data, error = self._request_json(
            session,
            "GET",
            "https://doaj.org/api/v2/search/articles/",
            config=config,
            params={"pageSize": 3, "q": f'doi:"{doi}"'},
            headers=extra_headers,
        )
        if error:
            yield error
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


class EuropePmcResolver(ApiResolverBase):
    """Resolve Open Access links via the Europe PMC REST API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> EuropePmcResolver().name
        'europe_pmc'
    """

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
        data, error = self._request_json(
            session,
            "GET",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            config=config,
            params={"query": f'DOI:"{doi}"', "format": "json", "pageSize": 3},
        )
        if error:
            if error.event_reason == "http-error":
                return
            yield error
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


class FigshareResolver(ApiResolverBase):
    """Resolve Figshare repository metadata into download URLs.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> FigshareResolver().name
        'figshare'
    """

    name = "figshare"
    api_display_name = "Figshare"

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

        extra_headers = {"Content-Type": "application/json"}
        data, error = self._request_json(
            session,
            "POST",
            "https://api.figshare.com/v2/articles/search",
            config=config,
            json={"search_for": f':doi: "{doi}"', "page": 1, "page_size": 3},
            headers=extra_headers,
        )
        if error:
            yield error
            return

        if isinstance(data, list):
            articles = data
        else:
            LOGGER.warning(
                "Figshare API returned non-list articles payload: %s",
                type(data).__name__ if data is not None else "None",
            )
            articles = []

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


class HalResolver(ApiResolverBase):
    """Resolve publications from the HAL open archive.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> HalResolver().name
        'hal'
    """

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
        data, error = self._request_json(
            session,
            "GET",
            "https://api.archives-ouvertes.fr/search/",
            config=config,
            params={"q": f"doiId_s:{doi}", "fl": "fileMain_s,file_s"},
        )
        if error:
            yield error
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
    """Attempt to scrape landing pages when explicit PDFs are unavailable.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> LandingPageResolver().name
        'landing_page'
    """

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
            for pattern, finder in (
                ("meta", find_pdf_via_meta),
                ("link", find_pdf_via_link),
                ("anchor", find_pdf_via_anchor),
            ):
                candidate = finder(soup, landing)
                if candidate:
                    yield ResolverResult(
                        url=candidate,
                        referer=landing,
                        metadata={"pattern": pattern},
                    )
                    break


class OpenAireResolver(RegisteredResolver):
    """Resolve URLs using the OpenAIRE API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> OpenAireResolver().name
        'openaire'
    """

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
    """Resolve OpenAlex work metadata into candidate download URLs.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> OpenAlexResolver().name
        'openalex'
    """

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


class OsfResolver(ApiResolverBase):
    """Resolve artefacts hosted on the Open Science Framework.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> OsfResolver().name
        'osf'
    """

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
        data, error = self._request_json(
            session,
            "GET",
            "https://api.osf.io/v2/preprints/",
            config=config,
            params={"filter[doi]": doi},
        )
        if error:
            yield error
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
    """Resolve PubMed Central articles via identifiers and lookups.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> PmcResolver().name
        'pmc'
    """

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
            pdf_links_emitted = False
            try:
                root = ET.fromstring(resp.text)
            except ET.ParseError as exc:
                LOGGER.debug("PMC OA XML parse error for %s: %s", pmcid, exc)
            else:
                for link in root.iter():
                    tag = link.tag.rsplit("}", 1)[-1].lower()
                    if tag != "link":
                        continue
                    href = (
                        link.attrib.get("href")
                        or link.attrib.get("{http://www.w3.org/1999/xlink}href")
                        or ""
                    ).strip()
                    fmt = (link.attrib.get("format") or "").lower()
                    mime = (link.attrib.get("type") or "").lower()
                    if not href:
                        continue
                    if fmt == "pdf" or mime == "application/pdf" or href.lower().endswith(".pdf"):
                        pdf_links_emitted = True
                        yield ResolverResult(
                            url=_absolute_url(oa_url, href),
                            metadata={"pmcid": pmcid, "source": "oa"},
                        )
            yield ResolverResult(
                url=fallback_url,
                metadata={"pmcid": pmcid, "source": "pdf-fallback"},
            )


class SemanticScholarResolver(RegisteredResolver):
    """Resolve PDFs via the Semantic Scholar Graph API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> SemanticScholarResolver().name
        'semantic_scholar'
    """

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
            data = _fetch_semantic_scholar_data(session, config, doi)
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
        except _requests.HTTPError as exc:
            status = None
            response_obj = getattr(exc, "response", None)
            if response_obj is not None and hasattr(response_obj, "status_code"):
                status = response_obj.status_code
            if status is None:
                match = re.search(r"(\\d{3})", str(exc))
                if match:
                    status = int(match.group(1))
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=status,
                metadata={"error_detail": str(exc)},
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
        except ValueError as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
                metadata={"error_detail": str(exc)},
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
    """Resolve PDFs via the Unpaywall API.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> UnpaywallResolver().name
        'unpaywall'
    """

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
        try:
            data = _fetch_unpaywall_data(session, config, doi)
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
        except _requests.HTTPError as exc:
            status = None
            response_obj = getattr(exc, "response", None)
            if response_obj is not None and hasattr(response_obj, "status_code"):
                status = response_obj.status_code
            if status is None:
                match = re.search(r"(\\d{3})", str(exc))
                if match:
                    status = int(match.group(1))
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=status,
                metadata={"error_detail": str(exc)},
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
            LOGGER.exception("Unexpected error in Unpaywall resolver session path")
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
    """Fallback resolver that queries the Internet Archive Wayback Machine.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> WaybackResolver().name
        'wayback'
    """

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


class ZenodoResolver(ApiResolverBase):
    """Resolve Zenodo records into downloadable open-access PDF URLs.

    Attributes:
        name: Resolver identifier registered with the pipeline.

    Examples:
        >>> ZenodoResolver().name
        'zenodo'
    """

    name = "zenodo"
    api_display_name = "Zenodo"

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

        data, error = self._request_json(
            session,
            "GET",
            "https://zenodo.org/api/records/",
            config=config,
            params={"q": f'doi:"{doi}"', "size": 3, "sort": "mostrecent"},
        )
        if error:
            if error.event_reason == "connection-error":
                error.event_reason = "request-error"
            yield error
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


# --- Resolver Pipeline ---

# --- Resolver Pipeline ---


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
        logger: AttemptSink,
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
        self._last_host_hit: Dict[str, float] = {}
        self._host_lock = threading.Lock()
        self._download_accepts_context = _callable_accepts_argument(download_func, "context")
        self._download_accepts_head_flag = _callable_accepts_argument(
            download_func, "head_precheck_passed"
        )

    def _emit_attempt(
        self,
        record: AttemptRecord,
        *,
        timestamp: Optional[str] = None,
    ) -> None:
        """Invoke the configured logger using the structured attempt contract.

        Args:
            record: Attempt payload emitted by the pipeline.
            timestamp: Optional ISO timestamp forwarded to sinks that accept it.

        Returns:
            None
        """

        if not hasattr(self.logger, "log_attempt") or not callable(
            getattr(self.logger, "log_attempt")
        ):
            raise AttributeError("ResolverPipeline logger must provide log_attempt().")
        self.logger.log_attempt(record, timestamp=timestamp)

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
            last = self._last_host_hit.get(host)
            if last is None:
                self._last_host_hit[host] = now if now > 0.0 else 1e-9
                return
            elapsed = now - last
            if elapsed >= interval:
                self._last_host_hit[host] = now
                return
            wait = interval - elapsed
        if wait > 0:
            jitter = random.random() * 0.05
            _time.sleep(wait + jitter)
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

    def _should_attempt_head_check(self, resolver_name: str, url: Optional[str]) -> bool:
        """Return ``True`` when a resolver should perform a HEAD preflight request.

        Args:
            resolver_name: Name of the resolver under consideration.

        Returns:
            Boolean indicating whether the resolver should issue a HEAD request.
        """

        if resolver_name in self.config.resolver_head_precheck:
            return self.config.resolver_head_precheck[resolver_name]
        if url:
            parsed = urlparse(url)
            host = (parsed.hostname or parsed.netloc or "").lower()
            if host:
                override = self.config.head_precheck_host_overrides.get(host)
                if override is None and host.startswith("www."):
                    override = self.config.head_precheck_host_overrides.get(host[4:])
                if override is not None:
                    return override
        return self.config.enable_head_precheck

    def _head_precheck_url(
        self,
        session: _requests.Session,
        url: str,
        timeout: float,
    ) -> bool:
        """Delegate to the shared network-layer preflight helper.

        Args:
            session: Requests session used to issue the HEAD preflight.
            url: Candidate URL that may host a downloadable document.
            timeout: Maximum duration in seconds to wait for the HEAD response.

        Returns:
            bool: ``True`` when the URL passes preflight checks, ``False`` otherwise.
        """

        return head_precheck(session, url, timeout)

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
            self._emit_attempt(
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
            self._emit_attempt(
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
            self._emit_attempt(
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
            self._emit_attempt(
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
        url = normalize_url(url)
        if self.config.enable_global_url_dedup:
            with self._global_lock:
                duplicate = url in self._global_seen_urls
                if not duplicate:
                    self._global_seen_urls.add(url)
            if duplicate:
                self._emit_attempt(
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
            self._emit_attempt(
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
        if context_data.get("list_only"):
            self._emit_attempt(
                AttemptRecord(
                    work_id=artifact.work_id,
                    resolver_name=resolver_name,
                    resolver_order=order_index,
                    url=url,
                    status="listed",
                    http_status=None,
                    content_type=None,
                    elapsed_ms=0.0,
                    reason="list-only",
                    metadata=result.metadata,
                    dry_run=True,
                    resolver_wall_time_ms=resolver_wall_time_ms,
                )
            )
            self.metrics.record_skip(resolver_name, "list-only")
            return None

        download_context = dict(context_data)
        # Ensure downstream download heuristics are present with config defaults.
        download_context.setdefault("sniff_bytes", self.config.sniff_bytes)
        download_context.setdefault("min_pdf_bytes", self.config.min_pdf_bytes)
        download_context.setdefault("tail_check_bytes", self.config.tail_check_bytes)
        head_precheck_passed = False
        if self._should_attempt_head_check(resolver_name, url):
            head_precheck_passed = self._head_precheck_url(
                session,
                url,
                self.config.get_timeout(resolver_name),
            )
            if not head_precheck_passed:
                self._emit_attempt(
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

        self._emit_attempt(
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

        classification = Classification.from_wire(outcome.classification)
        if classification is Classification.HTML and outcome.path:
            state.html_paths.append(outcome.path)

        if classification not in PDF_LIKE and url:
            if url not in state.failed_urls:
                state.failed_urls.append(url)
            if url not in artifact.failed_pdf_urls:
                artifact.failed_pdf_urls.append(url)

        if classification in PDF_LIKE:
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
