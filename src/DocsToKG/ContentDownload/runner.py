# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.runner",
#   "purpose": "Run orchestration for the content download CLI and scripting APIs",
#   "sections": [
#     {"id": "downloadrunstate", "name": "DownloadRunState", "anchor": "class-downloadrunstate", "kind": "class"},
#     {"id": "downloadrun", "name": "DownloadRun", "anchor": "class-downloadrun", "kind": "class"},
#     {"id": "iterate-openalex", "name": "iterate_openalex", "anchor": "function-iterate-openalex", "kind": "function"},
#     {"id": "run-helper", "name": "run", "anchor": "function-run", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""Execution harness coordinating DocsToKG content download runs end-to-end.

Responsibilities
----------------
- Translate a :class:`~DocsToKG.ContentDownload.args.ResolvedConfig` snapshot
  into an executable workflow by wiring providers, resolver pipelines, telemetry
  sinks, and resumable state.
- Manage lifecycle hooks for manifest streams (JSONL/CSV/SQLite),
  concurrency-bound worker pools, and resource cleanup via
  :class:`DownloadRun` context management.
- Provide resumable execution by hydrating prior manifest/index snapshots and
  skipping already-processed works before dispatching to the resolver pipeline.
- Hydrate global URL dedupe sets, robots caches, and the shared HTTPX client so
  retry/caching policies (Hishel, Tenacity, polite headers) are honoured across
  workers without needing per-thread session pools.
- Coordinate sequential ``--sleep`` throttling while leaving concurrent worker
  pools free from the default delay unless operators request it explicitly.
- Surface convenience helpers (:func:`run`, :func:`iterate_openalex`) that are
  reused by tests, smoke scripts, and the CLI while returning a
  :class:`~DocsToKG.ContentDownload.summary.RunResult`.

Key Components
--------------
- ``DownloadRun`` – encapsulates setup, execution, and teardown for a single run.
- ``DownloadRunState`` – aggregates counters thread-safely for telemetry output.
- ``get_http_client`` – acquires the cached HTTPX/Hishel client shared across
  resolver workers.
- ``iterate_openalex`` – generator that pages through OpenAlex Works queries,
  respecting CLI throttles and polite headers.
- ``run`` – top-level helper that owns the context-manager lifetime of
  :class:`DownloadRun` for simple scripting integrations.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import httpx
from pyalex import Works
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from DocsToKG.ContentDownload import locks
from DocsToKG.ContentDownload.args import ResolvedConfig
from DocsToKG.ContentDownload.core import (
    Classification,
    ReasonCode,
    WorkArtifact,
    atomic_write_text,
)
from DocsToKG.ContentDownload.download import (
    DownloadConfig,
    RobotsCache,
    create_artifact,
    download_candidate,
    ensure_dir,
    process_one_work,
)
from DocsToKG.ContentDownload.httpx_transport import get_http_client
from DocsToKG.ContentDownload.ratelimit import (
    get_rate_limiter_manager,
    serialize_policy,
)
from DocsToKG.ContentDownload.pipeline import (
    DownloadOutcome,
    ResolverMetrics,
    ResolverPipeline,
)
from DocsToKG.ContentDownload.providers import OpenAlexWorkProvider, WorkProvider
from DocsToKG.ContentDownload.summary import RunResult, build_summary_record
from DocsToKG.ContentDownload.networking import (
    DEFAULT_RETRYABLE_STATUSES,
    RetryAfterJitterWait,
    request_with_retries,
    set_breaker_registry,
)
from DocsToKG.ContentDownload.telemetry import (
    AttemptSink,
    CsvSink,
    JsonlResumeLookup,
    JsonlSink,
    LastAttemptCsvSink,
    ManifestIndexSink,
    MultiSink,
    RotatingJsonlSink,
    RunTelemetry,
    SqliteResumeLookup,
    SqliteSink,
    SummarySink,
    load_resume_completed_from_sqlite,
    looks_like_csv_resume_target,
    looks_like_sqlite_resume_target,
)

__all__ = ["DownloadRun", "iterate_openalex", "run"]

LOGGER = logging.getLogger("DocsToKG.ContentDownload")


_OPENALEX_RETRYABLE_EXCEPTIONS = (httpx.HTTPError,)


def _openalex_headers_from_config() -> Dict[str, str]:
    """Return polite headers derived from the active pyalex configuration."""

    try:
        from pyalex import api as pyalex_api  # type: ignore
    except Exception:
        return {}

    config = getattr(pyalex_api, "config", None)
    if config is None:
        return {}

    headers: Dict[str, str] = {}
    api_key = getattr(config, "api_key", None)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    email = getattr(config, "email", None)
    if email:
        headers["From"] = email

    user_agent = getattr(config, "user_agent", None)
    if user_agent:
        headers["User-Agent"] = user_agent

    return headers


class _OpenAlexTenacitySession:
    """Adapter that executes pyalex session calls via `request_with_retries`."""

    __slots__ = (
        "_client",
        "_max_retries",
        "_backoff_factor",
        "_backoff_max",
        "_retry_after_cap",
    )

    def __init__(
        self,
        *,
        client: httpx.Client,
        max_retries: int,
        backoff_factor: float,
        backoff_max: Optional[float],
        retry_after_cap: Optional[float],
    ) -> None:
        self._client = client
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._backoff_max = backoff_max
        self._retry_after_cap = retry_after_cap

    def get(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        request_headers: Dict[str, str] = dict(_openalex_headers_from_config())
        if headers:
            request_headers.update(headers)

        # Discard requests-specific kwargs that httpx does not understand.
        kwargs.pop("allow_redirects", None)
        kwargs.pop("stream", None)
        kwargs.pop("auth", None)

        return request_with_retries(
            self._client,
            "GET",
            url,
            role="metadata",
            max_retries=self._max_retries,
            backoff_factor=self._backoff_factor,
            backoff_max=self._backoff_max,
            retry_after_cap=self._retry_after_cap,
            respect_retry_after=True,
            retry_statuses=DEFAULT_RETRYABLE_STATUSES,
            params=params,
            headers=request_headers or None,
            timeout=timeout,
            **kwargs,
        )


@dataclass
class DownloadRunState:
    """Mutable run-time state shared across the runner lifecycle."""

    http_client: httpx.Client
    options: DownloadConfig
    resume_lookup: Mapping[str, Dict[str, Any]]
    resume_completed: Set[str]
    resume_cleanup: Optional[Callable[[], None]] = field(default=None, repr=False)
    processed: int = 0
    saved: int = 0
    html_only: int = 0
    xml_only: int = 0
    skipped: int = 0
    downloaded_bytes: int = 0
    worker_failures: int = 0
    _lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)

    def update_from_result(self, result: Dict[str, Any]) -> None:
        """Update aggregate counters from an individual work result."""

        with self._lock:
            self.processed += 1
            if result.get("saved"):
                self.saved += 1
            if result.get("html_only"):
                self.html_only += 1
            if result.get("xml_only"):
                self.xml_only += 1
            if result.get("skipped"):
                self.skipped += 1
            downloaded = result.get("downloaded_bytes") or 0
            try:
                self.downloaded_bytes += int(downloaded)
            except (TypeError, ValueError):
                pass

    def record_worker_failure(self) -> None:
        """Increment the worker failure counter in a thread-safe manner."""

        with self._lock:
            self.worker_failures += 1


class DownloadRun:
    """Stage-oriented orchestration for executing a content download run."""

    def __init__(self, resolved: ResolvedConfig) -> None:
        self.resolved = resolved
        self.args = resolved.args
        self.multi_sink: Optional[MultiSink] = None
        self.attempt_logger: Optional[RunTelemetry] = None
        self.metrics: Optional[ResolverMetrics] = None
        self.pipeline: Optional[ResolverPipeline] = None
        self.provider: Optional[WorkProvider] = None
        self.state: Optional[DownloadRunState] = None
        self._ephemeral_stack: Optional[contextlib.ExitStack] = None
        # Sink factories (overridable for tests/embedding)
        self.jsonl_sink_factory = JsonlSink
        self.rotating_jsonl_sink_factory = RotatingJsonlSink
        self.manifest_index_sink_factory = ManifestIndexSink
        self.last_attempt_sink_factory = LastAttemptCsvSink
        self.sqlite_sink_factory = SqliteSink
        self.summary_sink_factory = SummarySink
        self.csv_sink_factory = CsvSink
        self.multi_sink_factory = MultiSink
        self.run_telemetry_factory = RunTelemetry
        self.process_one_work_func = process_one_work
        self.iterate_openalex_func = iterate_openalex
        self.download_candidate_func = download_candidate

    def __enter__(self) -> "DownloadRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release resources owned by the run instance."""

        stack = self._ephemeral_stack
        self._ephemeral_stack = None
        if stack is not None:
            with contextlib.suppress(Exception):
                stack.close()

        state = self.state
        if state is None:
            return

        self.state = None

        resume_cleanup = state.resume_cleanup
        state.resume_cleanup = None

        if resume_cleanup is not None:
            with contextlib.suppress(Exception):
                resume_cleanup()

    def setup_sinks(self, stack: Optional[contextlib.ExitStack] = None) -> MultiSink:
        """Initialise telemetry sinks responsible for manifest and summary data."""

        if stack is None:
            if self._ephemeral_stack is not None:
                raise RuntimeError("Telemetry sinks are already initialised.")
            stack = contextlib.ExitStack()
            stack.__enter__()
            self._ephemeral_stack = stack

        sinks: List[AttemptSink] = []
        manifest_path = self.resolved.manifest_path
        locks.configure_lock_root(manifest_path.parent)
        log_format = getattr(self.args, "log_format", None)
        if isinstance(log_format, str):
            log_format = log_format.lower()
        else:
            log_format = "jsonl"

        if log_format == "jsonl":
            if self.args.log_rotate:
                jsonl_sink = stack.enter_context(
                    self.rotating_jsonl_sink_factory(manifest_path, max_bytes=self.args.log_rotate)
                )
            else:
                jsonl_sink = stack.enter_context(self.jsonl_sink_factory(manifest_path))
            sinks.append(jsonl_sink)
        elif log_format != "csv":
            LOGGER.warning("Unknown log format '%s'; defaulting to JSONL.", log_format)
            jsonl_sink = stack.enter_context(self.jsonl_sink_factory(manifest_path))
            sinks.append(jsonl_sink)

        index_path = manifest_path.with_suffix(".index.json")
        index_sink = stack.enter_context(self.manifest_index_sink_factory(index_path))
        sinks.append(index_sink)

        last_attempt_path = manifest_path.with_suffix(".last.csv")
        last_attempt_sink = stack.enter_context(self.last_attempt_sink_factory(last_attempt_path))
        sinks.append(last_attempt_sink)

        sqlite_sink = stack.enter_context(self.sqlite_sink_factory(self.resolved.sqlite_path))
        sinks.append(sqlite_sink)

        summary_path = manifest_path.with_suffix(".summary.json")
        summary_sink = stack.enter_context(self.summary_sink_factory(summary_path))
        sinks.append(summary_sink)

        csv_requested = log_format == "csv"
        if csv_requested and not self.resolved.csv_path:
            raise ValueError("--log-format csv selected but no CSV path was resolved.")

        if self.resolved.csv_path and (csv_requested or getattr(self.args, "log_csv", None)):
            csv_sink = stack.enter_context(self.csv_sink_factory(self.resolved.csv_path))
            sinks.append(csv_sink)

        combined_sink = self.multi_sink_factory(sinks)
        self.multi_sink = combined_sink
        self.attempt_logger = self.run_telemetry_factory(combined_sink)
        return combined_sink

    def setup_resolver_pipeline(self) -> ResolverPipeline:
        """Create the resolver pipeline backed by telemetry and metrics."""

        if self.attempt_logger is None:
            raise RuntimeError("setup_sinks() must be called before setup_resolver_pipeline().")

        metrics = ResolverMetrics()
        pipeline = ResolverPipeline(
            resolvers=self.resolved.resolver_instances,
            config=self.resolved.resolver_config,
            download_func=self.download_candidate_func,
            logger=self.attempt_logger,
            metrics=metrics,
            initial_seen_urls=self.resolved.persistent_seen_urls or None,
            global_manifest_index=self.resolved.previous_url_index,
            run_id=self.resolved.run_id,
        )
        self.metrics = metrics
        self.pipeline = pipeline
        return pipeline

    def setup_work_provider(self) -> WorkProvider:
        """Construct the OpenAlex work provider used to yield artefacts."""

        iterate_kwargs = {
            "per_page": self.args.per_page,
            "max_results": self.args.max,
            "retry_attempts": self.resolved.openalex_retry_attempts,
            "retry_backoff": self.resolved.openalex_retry_backoff,
            "retry_max_delay": self.resolved.openalex_retry_max_delay,
            "retry_after_cap": self.resolved.retry_after_cap,
        }
        try:
            signature = inspect.signature(self.iterate_openalex_func)
        except (TypeError, ValueError):
            supported_kwargs = iterate_kwargs
        else:
            if any(
                param.kind is inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            ):
                supported_kwargs = iterate_kwargs
            else:
                supported_kwargs = {
                    key: value
                    for key, value in iterate_kwargs.items()
                    if key in signature.parameters
                }
        work_iterable = self.iterate_openalex_func(
            self.resolved.query,
            **supported_kwargs,
        )
        provider = OpenAlexWorkProvider(
            query=self.resolved.query,
            works_iterable=work_iterable,
            artifact_factory=create_artifact,
            pdf_dir=self.resolved.pdf_dir,
            html_dir=self.resolved.html_dir,
            xml_dir=self.resolved.xml_dir,
            per_page=self.args.per_page,
            max_results=self.args.max,
            retry_attempts=self.resolved.openalex_retry_attempts,
            retry_backoff=self.resolved.openalex_retry_backoff,
            retry_max_delay=self.resolved.openalex_retry_max_delay,
            retry_after_cap=self.resolved.retry_after_cap,
            iterate_openalex_func=self.iterate_openalex_func,
        )
        self.provider = provider
        return provider

    def setup_download_state(
        self,
        http_client: Optional[httpx.Client] = None,
        robots_cache: Optional[RobotsCache] = None,
        breaker_registry: Optional[Any] = None,
    ) -> DownloadRunState:
        """Initialise download options and counters for the run."""

        resume_lookup: Mapping[str, Dict[str, Any]]
        resume_completed: Set[str]
        resume_path_raw = self.args.resume_from
        sqlite_path = self.resolved.sqlite_path
        if resume_path_raw is not None:
            resume_path = Path(resume_path_raw).expanduser()
            resume_lookup, resume_completed, resume_cleanup = self._load_resume_state(resume_path)
        else:
            manifest_path = self.resolved.manifest_path
            prefix = f"{manifest_path.name}."
            has_rotated = any(
                candidate.is_file() and candidate.name[len(prefix) :].isdigit()
                for candidate in manifest_path.parent.glob(f"{manifest_path.name}.*")
            )
            sqlite_available = sqlite_path and sqlite_path.exists()
            if manifest_path.exists() or has_rotated or sqlite_available:
                resume_lookup, resume_completed, resume_cleanup = self._load_resume_state(
                    manifest_path
                )
            else:
                resume_lookup, resume_completed, resume_cleanup = {}, set(), None
        options = DownloadConfig(
            dry_run=self.args.dry_run,
            list_only=self.args.list_only,
            extract_html_text=self.resolved.extract_html_text,
            run_id=self.resolved.run_id,
            previous_lookup=resume_lookup,
            resume_completed=resume_completed,
            sniff_bytes=self.args.sniff_bytes,
            min_pdf_bytes=self.args.min_pdf_bytes,
            tail_check_bytes=self.args.tail_check_bytes,
            robots_checker=(
                robots_cache if robots_cache is not None else self.resolved.robots_checker
            ),
            content_addressed=self.args.content_addressed,
            verify_cache_digest=self.args.verify_cache_digest,
        )
        retry_after_cap = getattr(self.resolved.resolver_config, "retry_after_cap", None)
        if retry_after_cap is not None:
            options.extra["retry_after_cap"] = retry_after_cap
        options.previous_lookup = resume_lookup

        # Set global breaker registry for networking layer
        if breaker_registry is not None:
            from DocsToKG.ContentDownload.networking import set_breaker_registry

            set_breaker_registry(breaker_registry)

        if self.pipeline is not None:
            # Legacy pipeline breaker registry call removed - now handled by networking layer
            pass

        client = http_client or get_http_client()
        state = DownloadRunState(
            http_client=client,
            options=options,
            resume_lookup=resume_lookup,
            resume_completed=resume_completed,
            resume_cleanup=resume_cleanup,
        )
        self.state = state
        return state

    def _load_resume_state(
        self, resume_path: Path
    ) -> Tuple[Mapping[str, Dict[str, Any]], Set[str], Optional[Callable[[], None]]]:
        """Load resume metadata from JSON manifests with SQLite fallback."""

        resolved_sqlite_path = self.resolved.sqlite_path
        sqlite_candidates: List[Path] = []
        if looks_like_csv_resume_target(resume_path) and not resolved_sqlite_path:
            LOGGER.debug("Resume target %s appears to be CSV without SQLite cache.", resume_path)
        for suffix in (".sqlite3", ".sqlite"):
            candidate = resume_path.with_suffix(suffix)
            if candidate not in sqlite_candidates:
                sqlite_candidates.append(candidate)
        if looks_like_sqlite_resume_target(resume_path):
            sqlite_candidates.insert(0, resume_path)
        sqlite_path = next(
            (candidate for candidate in sqlite_candidates if candidate.is_file()), None
        )
        if sqlite_path is None:
            sqlite_path = resolved_sqlite_path
        elif resolved_sqlite_path and sqlite_path != resolved_sqlite_path:
            LOGGER.debug(
                "Using SQLite cache %s located alongside resume target %s.",
                sqlite_path,
                resume_path,
            )
        resume_path_exists = resume_path.exists()
        has_rotated = False
        if not resume_path_exists:
            prefix = f"{resume_path.name}."
            has_rotated = any(
                candidate.is_file() and candidate.name[len(prefix) :].isdigit()
                for candidate in resume_path.parent.glob(f"{resume_path.name}.*")
            )

        used_sqlite = False
        resume_lookup: Mapping[str, Dict[str, Any]]
        resume_completed: Set[str]
        cleanup_callback: Optional[Callable[[], None]] = None

        def _build_json_lookup() -> Tuple[
            Mapping[str, Dict[str, Any]], Set[str], Optional[Callable[[], None]]
        ]:
            json_lookup = JsonlResumeLookup(resume_path)
            completed_ids = set(json_lookup.completed_work_ids)
            return json_lookup, completed_ids, getattr(json_lookup, "close", None)

        if sqlite_path and sqlite_path.exists():
            sqlite_lookup = SqliteResumeLookup(sqlite_path)
            cleanup_callback = getattr(sqlite_lookup, "close", None)
            try:
                resume_completed = load_resume_completed_from_sqlite(sqlite_path)
                row_count = len(sqlite_lookup)
                if row_count == 0 and (resume_path_exists or has_rotated):
                    if cleanup_callback is not None:
                        with contextlib.suppress(Exception):
                            cleanup_callback()
                    cleanup_callback = None
                    resume_lookup, resume_completed, cleanup_callback = _build_json_lookup()
                else:
                    resume_lookup = sqlite_lookup
                    used_sqlite = True
            except Exception:
                if cleanup_callback is not None:
                    with contextlib.suppress(Exception):
                        cleanup_callback()
                raise
        else:
            resume_lookup, resume_completed, cleanup_callback = _build_json_lookup()

        if (
            not resume_path_exists
            and not has_rotated
            and sqlite_path
            and sqlite_path.exists()
            and (
                len(resume_completed) > 0
                or (hasattr(resume_lookup, "__len__") and len(resume_lookup) > 0)
            )
        ):
            LOGGER.warning(
                "Resume manifest %s is missing; loading resume metadata from SQLite %s.",
                resume_path,
                sqlite_path,
            )

        if used_sqlite and looks_like_csv_resume_target(resume_path) and resume_path_exists:
            LOGGER.debug(
                "Resume target %s is CSV; using SQLite cache %s for resume lookup.",
                resume_path,
                sqlite_path,
            )

        return resume_lookup, resume_completed, cleanup_callback

    def setup_worker_pool(self) -> ThreadPoolExecutor:
        """Create a thread pool when concurrency is enabled."""

        return ThreadPoolExecutor(max_workers=max(self.args.workers, 1))

    def process_work_item(
        self,
        work: WorkArtifact,
        options: DownloadConfig,
        *,
        client: Optional[httpx.Client] = None,
    ) -> Dict[str, Any]:
        """Process a single work artefact and update aggregate counters."""

        if self.pipeline is None or self.attempt_logger is None or self.metrics is None:
            raise RuntimeError("Resolver pipeline not initialised.")
        if self.state is None:
            raise RuntimeError("Download state not initialised.")

        active_client = client or get_http_client()
        result = self.process_one_work_func(
            work,
            active_client,
            self.resolved.pdf_dir,
            self.resolved.html_dir,
            self.resolved.xml_dir,
            self.pipeline,
            self.attempt_logger,
            self.metrics,
            options=options,
        )
        self.state.update_from_result(result)
        return result

    def _record_worker_crash_manifest(
        self,
        artifact_context: Optional[Tuple[WorkArtifact, bool, Optional[str]]],
        exc: Exception,
    ) -> None:
        """Record a manifest entry for a worker crash if telemetry is available."""

        if self.attempt_logger is None or artifact_context is None:
            return

        artifact, dry_run_flag, run_id_token = artifact_context
        normalized_run_id = run_id_token or self.resolved.run_id
        crash_url = f"worker-crash://{normalized_run_id or 'unknown-run'}/{artifact.work_id}"
        try:
            outcome = DownloadOutcome(
                classification=Classification.SKIPPED,
                reason=ReasonCode.WORKER_EXCEPTION,
                reason_detail="worker-crash",
                error=str(exc),
            )
            self.attempt_logger.record_manifest(
                artifact,
                resolver=None,
                url=crash_url,
                outcome=outcome,
                html_paths=(),
                dry_run=dry_run_flag,
                run_id=normalized_run_id,
                reason=ReasonCode.WORKER_EXCEPTION,
                reason_detail="worker-crash",
            )
        except Exception:
            LOGGER.warning(
                "Failed to record manifest after worker crash",
                exc_info=True,
            )

    def _handle_worker_exception(
        self,
        state: DownloadRunState,
        exc: Exception,
        *,
        work_id: Optional[str] = None,
        artifact_context: Optional[Tuple[WorkArtifact, bool, Optional[str]]] = None,
    ) -> None:
        """Apply consistent crash handling for sequential and threaded workers."""

        state.record_worker_failure()
        extra_fields: Dict[str, Any] = {"error": str(exc)}
        if work_id:
            extra_fields["work_id"] = work_id
        LOGGER.exception(
            "worker_crash",
            extra={"extra_fields": extra_fields},
        )
        self._record_worker_crash_manifest(artifact_context, exc)
        state.update_from_result({"skipped": True})

    def run(self) -> RunResult:
        """Execute the content download pipeline and return the aggregate result."""

        summary: Dict[str, Any] = {}
        summary_record: Dict[str, Any] = {}
        state: Optional[DownloadRunState] = None

        policy_snapshot = {
            host: serialize_policy(policy) for host, policy in self.resolved.rate_policies.items()
        }
        LOGGER.info(
            "Rate limiter configured with backend=%s options=%s policies=%s",
            self.resolved.rate_backend.backend,
            dict(self.resolved.rate_backend.options),
            json.dumps(policy_snapshot, sort_keys=True),
        )

        try:
            with contextlib.ExitStack() as stack:
                self.setup_sinks(stack)
                self.setup_resolver_pipeline()
                provider = self.setup_work_provider()

                http_client = get_http_client()

                # Initialize breaker registry
                breaker_registry = None
                try:
                    from DocsToKG.ContentDownload.breakers import BreakerRegistry, BreakerConfig
                    from DocsToKG.ContentDownload.networking_breaker_listener import (
                        NetworkBreakerListener,
                        BreakerListenerConfig,
                    )
                except ImportError:
                    LOGGER.debug("pybreaker not available, circuit breakers disabled")
                else:
                    breaker_config_obj = getattr(
                        self.resolved.resolver_config, "breaker_config", None
                    )
                    if not isinstance(breaker_config_obj, BreakerConfig):
                        breaker_config_obj = BreakerConfig()

                    def listener_factory(host: str, scope: str, resolver: Optional[str]):
                        if self.attempt_logger is not None:
                            return NetworkBreakerListener(
                                self.attempt_logger,
                                BreakerListenerConfig(
                                    run_id=self.resolved.run_id,
                                    host=host,
                                    scope=scope,
                                    resolver=resolver,
                                ),
                            )
                        return None

                    try:
                        breaker_registry = BreakerRegistry(
                            breaker_config_obj, listener_factory=listener_factory
                        )
                        # Register breaker registry with networking layer
                        from DocsToKG.ContentDownload.networking import set_breaker_registry

                        set_breaker_registry(breaker_registry)
                    except Exception as e:
                        LOGGER.warning("Failed to initialize circuit breakers: %s", e)

                state = self.setup_download_state(
                    http_client, self.resolved.robots_checker, breaker_registry
                )

                if self.args.workers == 1:
                    client = state.http_client
                    for artifact in provider.iter_artifacts():
                        try:
                            self.process_work_item(artifact, state.options, client=client)
                        except Exception as exc:
                            self._handle_worker_exception(
                                state,
                                exc,
                                work_id=getattr(artifact, "work_id", None),
                                artifact_context=(
                                    artifact,
                                    bool(state.options.dry_run),
                                    state.options.run_id or self.resolved.run_id,
                                ),
                            )
                        if self.args.sleep > 0:
                            time.sleep(self.args.sleep)
                else:
                    executor = self.setup_worker_pool()
                    with executor:
                        in_flight: List[Future[Dict[str, Any]]] = []
                        max_in_flight = max(self.args.workers * 2, 1)
                        future_work_ids: Dict[Future[Dict[str, Any]], Optional[str]] = {}
                        future_context: Dict[
                            Future[Dict[str, Any]],
                            Tuple[WorkArtifact, bool, Optional[str]],
                        ] = {}
                        raw_sleep = getattr(self.args, "sleep", 0.0)
                        sleep_interval = float(raw_sleep or 0.0)
                        if sleep_interval < 0.0:
                            sleep_interval = 0.0
                        last_submit_at: Optional[float] = None

                        def _wait_for_submit_slot() -> None:
                            nonlocal last_submit_at
                            if sleep_interval <= 0.0:
                                return
                            if last_submit_at is None:
                                return
                            target_time = last_submit_at + sleep_interval
                            now = time.monotonic()
                            if now < target_time:
                                time.sleep(target_time - now)

                        def _submit(work_item: WorkArtifact) -> Future[Dict[str, Any]]:
                            def _runner() -> Dict[str, Any]:
                                return self.process_work_item(
                                    work_item,
                                    state.options,
                                    client=state.http_client,
                                )

                            future = executor.submit(_runner)
                            future_work_ids[future] = getattr(work_item, "work_id", None)
                            future_context[future] = (
                                work_item,
                                bool(state.options.dry_run),
                                state.options.run_id or self.resolved.run_id,
                            )
                            return future

                        def _handle_future(completed_future: Future[Dict[str, Any]]) -> None:
                            work_id = future_work_ids.pop(completed_future, None)
                            artifact_context = future_context.pop(completed_future, None)
                            try:
                                completed_future.result()
                            except Exception as exc:
                                self._handle_worker_exception(
                                    state,
                                    exc,
                                    work_id=work_id,
                                    artifact_context=artifact_context,
                                )
                                return

                        for artifact in provider.iter_artifacts():
                            if len(in_flight) >= max_in_flight:
                                done, pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
                                for completed_future in done:
                                    _handle_future(completed_future)
                                in_flight = list(pending)
                            _wait_for_submit_slot()
                            future = _submit(artifact)
                            last_submit_at = time.monotonic()
                            in_flight.append(future)

                        if in_flight:
                            for future in as_completed(list(in_flight)):
                                _handle_future(future)

                metrics = self.metrics or ResolverMetrics()
                summary = metrics.summary()
                limiter_manager = get_rate_limiter_manager()
                summary["rate_limiter"] = {
                    "backend": self.resolved.rate_backend.backend,
                    "backend_options": dict(self.resolved.rate_backend.options),
                    "policies": {
                        host: serialize_policy(policy)
                        for host, policy in self.resolved.rate_policies.items()
                    },
                    "metrics": limiter_manager.metrics_snapshot(),
                }
                summary_record = build_summary_record(
                    run_id=self.resolved.run_id,
                    processed=state.processed if state else 0,
                    saved=state.saved if state else 0,
                    html_only=state.html_only if state else 0,
                    xml_only=state.xml_only if state else 0,
                    skipped=state.skipped if state else 0,
                    worker_failures=state.worker_failures if state else 0,
                    bytes_downloaded=state.downloaded_bytes if state else 0,
                    summary=summary,
                )
                if self.attempt_logger is not None:
                    try:
                        self.attempt_logger.log_summary(summary_record)
                    except Exception:
                        LOGGER.warning("Failed to log summary record", exc_info=True)

        except Exception:
            raise
        else:
            metrics_path = self.resolved.manifest_path.with_suffix(".metrics.json")
            try:
                ensure_dir(metrics_path.parent)
                with locks.summary_lock(metrics_path):
                    atomic_write_text(
                        metrics_path,
                        json.dumps(summary_record, indent=2, sort_keys=True) + "\n",
                    )
            except Exception:
                LOGGER.warning("Failed to write metrics sidecar %s", metrics_path, exc_info=True)
        finally:
            self.close()

        return RunResult(
            run_id=self.resolved.run_id,
            processed=state.processed if state else 0,
            saved=state.saved if state else 0,
            html_only=state.html_only if state else 0,
            xml_only=state.xml_only if state else 0,
            skipped=state.skipped if state else 0,
            worker_failures=state.worker_failures if state else 0,
            bytes_downloaded=state.downloaded_bytes if state else 0,
            summary=summary,
            summary_record=summary_record,
        )


def iterate_openalex(
    query: Works,
    per_page: int,
    max_results: Optional[int],
    *,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
    retry_max_delay: float = 75.0,
    retry_after_cap: Optional[float] = None,
) -> Iterable[Dict[str, Any]]:
    """Iterate over OpenAlex works respecting pagination, limits, and retry policy.

    Pagination runs through pyalex while delegating retry cadence to the shared
    Tenacity policy used elsewhere in ContentDownload. When running against the
    real pyalex client the paginator's internal requests session is replaced
    with a shim that calls :func:`request_with_retries`; lightweight test stubs
    fall back to a Retrying controller wrapped around the iterator itself.
    """

    max_retries = max(0, int(retry_attempts))
    backoff_factor = max(0.0, float(retry_backoff))
    backoff_cap = None if retry_max_delay is None else max(0.0, float(retry_max_delay))
    retry_after_limit = None if retry_after_cap is None else max(0.0, float(retry_after_cap))

    pager = query.paginate(
        per_page=per_page, n_max=max_results if max_results is not None else None
    )
    pager_iter = iter(pager)

    client = get_http_client()
    if hasattr(pager, "_session"):
        pager._session = _OpenAlexTenacitySession(
            client=client,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            backoff_max=backoff_cap,
            retry_after_cap=retry_after_limit,
        )

    fallback_wait = wait_random_exponential(multiplier=backoff_factor, max=backoff_cap)
    wait_strategy = RetryAfterJitterWait(
        respect_retry_after=True,
        retry_after_cap=retry_after_limit,
        backoff_max=backoff_cap,
        retry_statuses=DEFAULT_RETRYABLE_STATUSES,
        fallback_wait=fallback_wait,
    )
    retrying = Retrying(
        retry=retry_if_exception_type(_OPENALEX_RETRYABLE_EXCEPTIONS),
        wait=wait_strategy,
        stop=stop_after_attempt(max_retries + 1),
        sleep=time.sleep,
        reraise=True,
    )

    retrieved = 0
    while True:
        try:
            page = retrying(lambda: next(pager_iter))
        except StopIteration:
            break
        except _OPENALEX_RETRYABLE_EXCEPTIONS as exc:
            LOGGER.error(
                "OpenAlex pagination failed after %s attempt(s): %s",
                max_retries + 1,
                exc,
                exc_info=True,
            )
            raise

        page_iterable = page if isinstance(page, Iterable) else [page]
        for work in page_iterable:
            yield work
            retrieved += 1
            if max_results is not None and retrieved >= max_results:
                return


def run(resolved: ResolvedConfig) -> RunResult:
    """Execute the download pipeline using a :class:`DownloadRun` orchestration."""

    download_run = DownloadRun(resolved)
    return download_run.run()
