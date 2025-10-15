"""Type definitions and protocols for the resolver pipeline."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Protocol

import requests

if TYPE_CHECKING:
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact

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
            None

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
        resolver_min_interval_s: Minimum interval between resolver requests.
        resolver_rate_limits: Deprecated rate limit configuration retained for compat.
        enable_head_precheck: Toggle applying HEAD filtering before downloads.
        resolver_head_precheck: Per-resolver overrides for HEAD filtering behaviour.
        mailto: Contact email appended to polite headers and user agent string.

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
    resolver_rate_limits: Dict[str, float] = field(default_factory=dict)
    enable_head_precheck: bool = True
    resolver_head_precheck: Dict[str, bool] = field(default_factory=dict)
    mailto: Optional[str] = None
    max_concurrent_resolvers: int = 1

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
        """Validate configuration fields and apply defaults."""

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


class AttemptLogger(Protocol):
    """Protocol for logging resolver attempts.

    Examples:
        >>> class Collector:
        ...     def __init__(self):
        ...         self.records = []
        ...     def log(self, record: AttemptRecord) -> None:
        ...         self.records.append(record)
        >>> collector = Collector()
        >>> isinstance(collector, AttemptLogger)
        True

    Attributes:
        None
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
            None

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
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs or events for the given artifact.

        Args:
            session: HTTP session used for outbound requests.
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

    def summary(self) -> Dict[str, Any]:
        """Return aggregated metrics summarizing resolver behaviour.

        Args:
            None

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
        }


DownloadFunc = Callable[..., DownloadOutcome]

__all__ = [
    "AttemptLogger",
    "AttemptRecord",
    "DEFAULT_RESOLVER_ORDER",
    "DownloadFunc",
    "DownloadOutcome",
    "PipelineResult",
    "Resolver",
    "ResolverConfig",
    "ResolverMetrics",
    "ResolverResult",
]
