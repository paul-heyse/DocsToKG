"""Resolver pipeline orchestration and execution logic."""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import requests

from .types import (
    AttemptLogger,
    AttemptRecord,
    DownloadFunc,
    PipelineResult,
    Resolver,
    ResolverConfig,
    ResolverMetrics,
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

    def run(
        self,
        session: requests.Session,
        artifact: "WorkArtifact",
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute resolvers sequentially until a PDF is obtained or exhausted.

        Args:
            session: HTTP session shared across resolver invocations.
            artifact: Work artifact representing the item being resolved.
            context: Optional dictionary carrying pipeline execution context.

        Returns:
            PipelineResult describing the final outcome of resolver execution.
        """

        context_data: Dict[str, Any] = context or {}
        dry_run = bool(context_data.get("dry_run", False))
        seen_urls: set[str] = set()
        html_paths: List[str] = []
        attempt_counter = 0

        for order_index, resolver_name in enumerate(self.config.resolver_order, start=1):
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
                        dry_run=dry_run,
                    )
                )
                self.metrics.record_skip(resolver_name, "missing")
                continue

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
                        dry_run=dry_run,
                    )
                )
                self.metrics.record_skip(resolver_name, "disabled")
                continue

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
                        dry_run=dry_run,
                    )
                )
                self.metrics.record_skip(resolver_name, "not-applicable")
                continue

            self._respect_rate_limit(resolver_name)
            with self._lock:
                self._last_invocation[resolver_name] = time.monotonic()

            for result in resolver.iter_urls(session, self.config, artifact):
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
                            dry_run=dry_run,
                        )
                    )
                    if result.event_reason:
                        self.metrics.record_skip(resolver_name, result.event_reason)
                    continue

                url = result.url
                if not url:
                    continue
                if url in seen_urls:
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
                            dry_run=dry_run,
                        )
                    )
                    self.metrics.record_skip(resolver_name, "duplicate-url")
                    continue

                seen_urls.add(url)
                attempt_counter += 1
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
                        dry_run=dry_run,
                    )
                )
                self.metrics.record_attempt(resolver_name, outcome)

                if outcome.classification == "html" and outcome.path:
                    html_paths.append(outcome.path)

                if outcome.is_pdf:
                    return PipelineResult(
                        success=True,
                        resolver_name=resolver_name,
                        url=url,
                        outcome=outcome,
                        html_paths=html_paths,
                    )

                if attempt_counter >= self.config.max_attempts_per_work:
                    return PipelineResult(
                        success=False,
                        resolver_name=resolver_name,
                        url=url,
                        outcome=outcome,
                        html_paths=html_paths,
                        reason="max-attempts-reached",
                    )

                self._jitter_sleep()

        return PipelineResult(success=False, html_paths=html_paths)


__all__ = ["ResolverPipeline"]
