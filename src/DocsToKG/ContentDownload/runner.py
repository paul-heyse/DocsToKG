"""Execution harness for DocsToKG content download runs."""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import random
import sqlite3
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import requests
from pyalex import Works

from DocsToKG.ContentDownload.args import ResolvedConfig
from DocsToKG.ContentDownload.core import (
    PDF_LIKE,
    Classification,
    ReasonCode,
    WorkArtifact,
    atomic_write_text,
    normalize_url,
)
from DocsToKG.ContentDownload.download import (
    DownloadConfig,
    RobotsCache,
    create_artifact,
    download_candidate,
    ensure_dir,
    process_one_work,
)
from DocsToKG.ContentDownload.networking import ThreadLocalSessionFactory, create_session
from DocsToKG.ContentDownload.pipeline import (
    DownloadOutcome,
    ResolverMetrics,
    ResolverPipeline,
)
from DocsToKG.ContentDownload.providers import OpenAlexWorkProvider, WorkProvider
from DocsToKG.ContentDownload.summary import RunResult, build_summary_record
from DocsToKG.ContentDownload.telemetry import (
    MANIFEST_SCHEMA_VERSION,
    AttemptSink,
    CsvSink,
    JsonlResumeLookup,
    JsonlSink,
    LastAttemptCsvSink,
    ManifestIndexSink,
    MultiSink,
    RotatingJsonlSink,
    RunTelemetry,
    SqliteSink,
    SummarySink,
    load_resume_completed_from_sqlite,
    looks_like_csv_resume_target,
    looks_like_sqlite_resume_target,
    SqliteResumeLookup,
)

__all__ = ["DownloadRun", "iterate_openalex", "run"]

LOGGER = logging.getLogger("DocsToKG.ContentDownload")


@dataclass
class DownloadRunState:
    """Mutable run-time state shared across the runner lifecycle."""

    session_factory: ThreadLocalSessionFactory
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

        if self._ephemeral_stack is not None:
            with contextlib.suppress(Exception):
                self._ephemeral_stack.close()
            self._ephemeral_stack = None
        if self.state is not None:
            with contextlib.suppress(Exception):
                self.state.session_factory.close_all()
            if self.state.resume_cleanup is not None:
                with contextlib.suppress(Exception):
                    self.state.resume_cleanup()

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
        )
        self.provider = provider
        return provider

    def setup_download_state(
        self,
        session_factory: ThreadLocalSessionFactory,
        robots_cache: Optional[RobotsCache] = None,
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
        state = DownloadRunState(
            session_factory=session_factory,
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

        def _build_json_lookup() -> Tuple[Mapping[str, Dict[str, Any]], Set[str], Optional[Callable[[], None]]]:
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
                    # Enable preloading for offline access after close (when using SQLite fallback)
                    sqlite_lookup.enable_preload_on_close()
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
        session: Optional[requests.Session] = None,
    ) -> Dict[str, Any]:
        """Process a single work artefact and update aggregate counters."""

        if self.pipeline is None or self.attempt_logger is None or self.metrics is None:
            raise RuntimeError("Resolver pipeline not initialised.")
        if self.state is None:
            raise RuntimeError("Download state not initialised.")

        session_factory = self.state.session_factory
        active_session = session or session_factory()
        result = self.process_one_work_func(
            work,
            active_session,
            self.resolved.pdf_dir,
            self.resolved.html_dir,
            self.resolved.xml_dir,
            self.pipeline,
            self.attempt_logger,
            self.metrics,
            options=options,
            session_factory=session_factory,
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

        try:
            with contextlib.ExitStack() as stack:
                self.setup_sinks(stack)
                self.setup_resolver_pipeline()
                provider = self.setup_work_provider()

                concurrency_product = self.resolved.concurrency_product
                pool_connections = min(max(64, concurrency_product * 4), 512)
                pool_maxsize = min(max(128, concurrency_product * 8), 1024)

                def _build_thread_session() -> requests.Session:
                    return create_session(
                        self.resolved.resolver_config.polite_headers,
                        pool_connections=pool_connections,
                        pool_maxsize=pool_maxsize,
                    )

                session_factory = ThreadLocalSessionFactory(_build_thread_session)
                state = self.setup_download_state(session_factory, self.resolved.robots_checker)

                if self.args.workers == 1:
                    session = state.session_factory()
                    for artifact in provider.iter_artifacts():
                        if session is None:
                            session = state.session_factory()
                        try:
                            self.process_work_item(artifact, state.options, session=session)
                        except Exception as exc:
                            state.session_factory.close_current()
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
                            session = None
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
                        future_thread_ids: Dict[
                            Future[Dict[str, Any]],
                            Dict[str, int],
                        ] = {}

                        def _submit(work_item: WorkArtifact) -> Future[Dict[str, Any]]:
                            thread_info: Dict[str, int] = {}

                            def _runner() -> Dict[str, Any]:
                                thread_info["thread_id"] = threading.get_ident()
                                try:
                                    return self.process_work_item(
                                        work_item, state.options
                                    )
                                except Exception:
                                    state.session_factory.close_current()
                                    raise

                            future = executor.submit(_runner)
                            future_work_ids[future] = getattr(work_item, "work_id", None)
                            future_context[future] = (
                                work_item,
                                bool(state.options.dry_run),
                                state.options.run_id or self.resolved.run_id,
                            )
                            future_thread_ids[future] = thread_info
                            return future

                        def _handle_future(completed_future: Future[Dict[str, Any]]) -> None:
                            work_id = future_work_ids.pop(completed_future, None)
                            artifact_context = future_context.pop(completed_future, None)
                            thread_info = future_thread_ids.pop(completed_future, None)
                            thread_id = (
                                thread_info.get("thread_id")
                                if thread_info is not None
                                else None
                            )
                            try:
                                completed_future.result()
                            except Exception as exc:
                                self._handle_worker_exception(
                                    state,
                                    exc,
                                    work_id=work_id,
                                    artifact_context=artifact_context,
                                )
                                if thread_id is not None:
                                    state.session_factory.close_for_thread(thread_id)
                                return

                        for artifact in provider.iter_artifacts():
                            if len(in_flight) >= max_in_flight:
                                done, pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
                                for completed_future in done:
                                    _handle_future(completed_future)
                                in_flight = list(pending)
                            in_flight.append(_submit(artifact))

                        if in_flight:
                            for future in as_completed(list(in_flight)):
                                _handle_future(future)

                metrics = self.metrics or ResolverMetrics()
                summary = metrics.summary()
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
                atomic_write_text(
                    metrics_path,
                    json.dumps(summary_record, indent=2, sort_keys=True) + "\n",
                )
            except Exception:
                LOGGER.warning("Failed to write metrics sidecar %s", metrics_path, exc_info=True)
        finally:
            if state is not None:
                state.session_factory.close_all()
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


def _calculate_equal_jitter_delay(
    attempt: int,
    *,
    backoff_factor: float,
    backoff_max: float,
) -> float:
    """Return an exponential backoff delay using equal jitter."""

    if backoff_factor <= 0 or attempt < 0:
        return 0.0

    base_delay = backoff_factor * (2**attempt)
    if base_delay <= 0:
        return 0.0

    capped_base = min(base_delay, backoff_max)
    if capped_base <= 0:
        return 0.0

    half = capped_base / 2.0
    return half + random.uniform(0.0, half)


def iterate_openalex(
    query: Works,
    per_page: int,
    max_results: Optional[int],
    *,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
    retry_max_delay: float = 60.0,
) -> Iterable[Dict[str, Any]]:
    """Iterate over OpenAlex works respecting pagination, limits, and retry policy.

    Retries honour ``Retry-After`` headers while applying an equal-jitter
    exponential backoff capped by ``retry_max_delay`` to avoid unbounded sleeps.
    """

    def _retry_after_seconds(exc: Exception) -> Optional[float]:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        try:
            header_value = response.headers.get("Retry-After")
        except Exception:
            return None
        if not header_value:
            return None
        text = str(header_value).strip()
        if not text:
            return None
        try:
            seconds = float(text)
        except (TypeError, ValueError):
            try:
                retry_dt = parsedate_to_datetime(text)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_dt is None:
                return None
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            seconds = (retry_dt - now).total_seconds()
        return max(0.0, float(seconds))

    max_retries = max(0, int(retry_attempts))
    base_backoff = max(0.0, float(retry_backoff))
    max_delay = max(0.0, float(retry_max_delay))

    pager = query.paginate(
        per_page=per_page, n_max=max_results if max_results is not None else None
    )
    pager_iter = iter(pager)
    retrieved = 0

    while True:
        attempt = 0
        while True:
            try:
                page = next(pager_iter)
            except StopIteration:
                return
            except (requests.HTTPError, requests.RequestException) as exc:
                if attempt >= max_retries:
                    LOGGER.error(
                        "OpenAlex pagination failed with no retries remaining (allowed=%s).",
                        max_retries,
                        exc_info=True,
                    )
                    raise
                attempt += 1
                retry_after = _retry_after_seconds(exc)
                base_delay = base_backoff * (2 ** (attempt - 1)) if base_backoff else 0.0
                jitter_delay = (
                    _calculate_equal_jitter_delay(
                        attempt - 1,
                        backoff_factor=base_backoff,
                        backoff_max=max_delay if max_delay > 0 else base_delay,
                    )
                    if base_delay > 0
                    else 0.0
                )
                if max_delay <= 0:
                    jitter_delay = 0.0
                retry_after_effective = 0.0
                if retry_after is not None:
                    retry_after_effective = retry_after
                    if max_delay > 0:
                        retry_after_effective = min(retry_after_effective, max_delay)
                    else:
                        retry_after_effective = 0.0
                delay = max(jitter_delay, retry_after_effective)
                if max_delay > 0:
                    delay = min(delay, max_delay)
                LOGGER.warning(
                    "OpenAlex pagination error (%s/%s retries): %s. Retrying in %.2fs.",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            else:
                break

        for work in page:
            yield work
            retrieved += 1
            if max_results is not None and retrieved >= max_results:
                return


def run(resolved: ResolvedConfig) -> RunResult:
    """Execute the download pipeline using a :class:`DownloadRun` orchestration."""

    download_run = DownloadRun(resolved)
    return download_run.run()
