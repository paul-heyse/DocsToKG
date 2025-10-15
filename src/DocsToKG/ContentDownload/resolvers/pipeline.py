"""
Resolver Pipeline Execution Engine

This module coordinates the execution of resolver providers that discover
downloadable artefacts for scholarly works. It encapsulates sequential and
concurrent strategies, rate limiting, duplicate detection, and callback hooks
for logging and metrics collection.

Key Features:
- Sequential and concurrent resolver scheduling with configurable concurrency.
- Rate-limiting enforcement, optional domain-level throttling, and HEAD preflight checks.
- State tracking for seen URLs, global deduplication, HTML fallbacks, and failure metrics.

Usage:
    from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline

    pipeline = ResolverPipeline(
        resolvers=[],
        config=ResolverConfig(),
        download_func=lambda *args, **kwargs: DownloadOutcome("miss"),
        logger=lambda record: None,
        metrics=ResolverMetrics(),
    )
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import requests

from .types import (
    AttemptLogger,
    AttemptRecord,
    DownloadFunc,
    PipelineResult,
    Resolver,
    ResolverConfig,
    ResolverMetrics,
    ResolverResult,
)

if TYPE_CHECKING:
    from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs: Any,
) -> requests.Response:
    """Proxy to :func:`DocsToKG.ContentDownload.http.request_with_retries`.

    The indirection keeps this module compatible with unit tests that monkeypatch
    either the pipeline-level attribute or the underlying HTTP helper while also
    deferring imports to avoid circular dependencies during runtime initialisation.

    Args:
        session: Requests session used to execute the outbound HTTP call.
        method: HTTP verb such as ``"GET"`` or ``"HEAD"``.
        url: Fully qualified URL to fetch.
        **kwargs: Additional keyword arguments forwarded to the HTTP helper.

    Returns:
        requests.Response: Response object produced by the proxied helper.

    Raises:
        requests.RequestException: Propagated from the underlying retry helper.
    """

    from DocsToKG.ContentDownload.http import request_with_retries as _request_with_retries

    return _request_with_retries(session, method, url, **kwargs)


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
    only read concurrently without mutation. HTTP ``requests.Session`` objects
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
            now = time.monotonic()
            delta = now - last
            if delta < limit:
                wait = limit - delta
            self._last_invocation[resolver_name] = now + wait
        if wait > 0:
            time.sleep(wait)

    def _respect_domain_limit(self, url: str) -> None:
        """Enforce per-domain throttling when configured."""

        if not url or not self.config.domain_min_interval_s:
            return
        host = urlsplit(url).netloc.lower()
        if not host:
            return
        interval = self.config.domain_min_interval_s.get(host)
        if not interval:
            return
        now = time.monotonic()
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
            time.sleep(wait)
            with self._host_lock:
                self._last_host_hit[host] = time.monotonic()

    def _jitter_sleep(self) -> None:
        """Introduce a small delay to avoid stampeding downstream services.

        Args:
            self: Pipeline instance executing resolver scheduling logic.

        Returns:
            None
        """

        if self.config.sleep_jitter <= 0:
            return
        time.sleep(self.config.sleep_jitter + random.random() * 0.1)

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
        session: requests.Session,
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
        session: requests.Session,
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
        session: requests.Session,
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
        session: requests.Session,
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
        session: requests.Session,
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
        start = time.monotonic()
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
        wall_ms = (time.monotonic() - start) * 1000.0
        return results, wall_ms

    def _process_result(
        self,
        session: requests.Session,
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
        if self._should_attempt_head_check(resolver_name):
            if not self._head_precheck_url(
                session,
                url,
                self.config.get_timeout(resolver_name),
            ):
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

        state.attempt_counter += 1
        self._respect_domain_limit(url)
        if self._download_accepts_context:
            outcome = self.download_func(
                session,
                artifact,
                url,
                result.referer,
                self.config.get_timeout(resolver_name),
                context_data,
            )
        else:
            outcome = self.download_func(
                session,
                artifact,
                url,
                result.referer,
                self.config.get_timeout(resolver_name),
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


__all__ = ["ResolverPipeline"]
