"""Resolver pipeline orchestration and execution logic."""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

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


def _callable_accepts_argument(func: DownloadFunc, name: str) -> bool:
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
    """Mutable pipeline execution state shared across resolvers."""

    __slots__ = (
        "dry_run",
        "seen_urls",
        "html_paths",
        "failed_urls",
        "attempt_counter",
    )

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.seen_urls: set[str] = set()
        self.html_paths: List[str] = []
        self.failed_urls: List[str] = []
        self.attempt_counter = 0


class ResolverPipeline:
    """Executes resolvers in priority order until a PDF download succeeds.

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
        self._resolver_map = {resolver.name: resolver for resolver in resolvers}
        self.config = config
        self.download_func = download_func
        self.logger = logger
        self.metrics = metrics or ResolverMetrics()
        self._last_invocation: Dict[str, float] = defaultdict(lambda: 0.0)
        self._lock = threading.Lock()
        self._download_accepts_context = _callable_accepts_argument(download_func, "context")

    def _respect_rate_limit(self, resolver_name: str) -> None:
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

    def _jitter_sleep(self) -> None:
        if self.config.sleep_jitter <= 0:
            return
        time.sleep(self.config.sleep_jitter + random.random() * 0.1)

    def _should_attempt_head_check(self, resolver_name: str) -> bool:
        if resolver_name in self.config.resolver_head_precheck:
            return self.config.resolver_head_precheck[resolver_name]
        return self.config.enable_head_precheck

    def _head_precheck_url(
        self,
        session: requests.Session,
        url: str,
        timeout: float,
    ) -> bool:
        try:
            from DocsToKG.ContentDownload.http import request_with_retries

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
        """Execute resolvers until a PDF is obtained or resolvers are exhausted."""

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
        for order_index, resolver_name in enumerate(self.config.resolver_order, start=1):
            resolver = self._prepare_resolver(resolver_name, order_index, artifact, state)
            if resolver is None:
                continue

            self._respect_rate_limit(resolver_name)
            with self._lock:
                self._last_invocation[resolver_name] = time.monotonic()

            for result in resolver.iter_urls(session, self.config, artifact):
                pipeline_result = self._process_result(
                    session,
                    artifact,
                    resolver_name,
                    order_index,
                    result,
                    context_data,
                    state,
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
        max_workers = self.config.max_concurrent_resolvers
        active_futures: Dict[Future[List[ResolverResult]], Tuple[str, int]] = {}

        def submit_next(
            executor: ThreadPoolExecutor,
            start_index: int,
        ) -> int:
            index = start_index
            while (
                len(active_futures) < max_workers
                and index < len(self.config.resolver_order)
            ):
                resolver_name = self.config.resolver_order[index]
                order_index = index + 1
                index += 1
                resolver = self._prepare_resolver(
                    resolver_name, order_index, artifact, state
                )
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
                        results = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        results = [
                            ResolverResult(
                                url=None,
                                event="error",
                                event_reason="resolver-exception",
                                metadata={"message": str(exc)},
                            )
                        ]

                    for result in results:
                        pipeline_result = self._process_result(
                            session,
                            artifact,
                            resolver_name,
                            order_index,
                            result,
                            context_data,
                            state,
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
    ) -> List[ResolverResult]:
        results: List[ResolverResult] = []
        try:
            self._respect_rate_limit(resolver_name)
            with self._lock:
                self._last_invocation[resolver_name] = time.monotonic()
            for result in resolver.iter_urls(session, self.config, artifact):
                results.append(result)
        except Exception as exc:
            results.append(
                ResolverResult(
                    url=None,
                    event="error",
                    event_reason="resolver-exception",
                    metadata={"message": str(exc)},
                )
            )
        return results

    def _process_result(
        self,
        session: requests.Session,
        artifact: "WorkArtifact",
        resolver_name: str,
        order_index: int,
        result: ResolverResult,
        context_data: Dict[str, Any],
        state: _RunState,
    ) -> Optional[PipelineResult]:
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
                )
            )
            if result.event_reason:
                self.metrics.record_skip(resolver_name, result.event_reason)
            return None

        url = result.url
        if not url:
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
                    )
                )
                self.metrics.record_skip(resolver_name, "head-precheck-failed")
                return None

        state.attempt_counter += 1
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
