"""Execution harness for DocsToKG content download runs."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
from DocsToKG.ContentDownload.pipeline import ResolverMetrics, ResolverPipeline
from DocsToKG.ContentDownload.providers import OpenAlexWorkProvider, WorkProvider
from DocsToKG.ContentDownload.summary import RunResult, build_summary_record
from DocsToKG.ContentDownload.telemetry import (
    MANIFEST_SCHEMA_VERSION,
    AttemptSink,
    CsvSink,
    JsonlSink,
    LastAttemptCsvSink,
    ManifestIndexSink,
    MultiSink,
    RotatingJsonlSink,
    RunTelemetry,
    SqliteSink,
    SummarySink,
    load_previous_manifest,
)

__all__ = ["DownloadRun", "iterate_openalex", "run"]

LOGGER = logging.getLogger("DocsToKG.ContentDownload")


@dataclass
class DownloadRunState:
    """Mutable run-time state shared across the runner lifecycle."""

    session_factory: ThreadLocalSessionFactory
    options: DownloadConfig
    resume_lookup: Dict[str, Dict[str, Any]]
    resume_completed: Set[str]
    processed: int = 0
    saved: int = 0
    html_only: int = 0
    xml_only: int = 0
    skipped: int = 0
    downloaded_bytes: int = 0
    worker_failures: int = 0

    def update_from_result(self, result: Dict[str, Any]) -> None:
        """Update aggregate counters from an individual work result."""

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
                    self.rotating_jsonl_sink_factory(
                        manifest_path, max_bytes=self.args.log_rotate
                    )
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
            raise ValueError(
                "--log-format csv selected but no CSV path was resolved."
            )

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
            download_func=download_candidate,
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

        work_iterable = iterate_openalex(
            self.resolved.query,
            per_page=self.args.per_page,
            max_results=self.args.max,
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

        resume_lookup: Dict[str, Dict[str, Any]]
        resume_completed: Set[str]
        resume_path_raw = self.args.resume_from
        sqlite_path = self.resolved.sqlite_path
        if resume_path_raw is not None:
            resume_path = Path(resume_path_raw)
            resume_lookup, resume_completed = self._load_resume_state(resume_path)
        else:
            manifest_path = self.resolved.manifest_path
            if manifest_path.exists():
                resume_lookup, resume_completed = self._load_resume_state(manifest_path)
            elif sqlite_path and sqlite_path.exists():
                LOGGER.warning(
                    "Resume manifest %s is missing; loading resume metadata from SQLite %s.",
                    manifest_path,
                    sqlite_path,
                )
                resume_lookup, resume_completed = load_previous_manifest(
                    manifest_path,
                    sqlite_path=sqlite_path,
                    allow_sqlite_fallback=True,
                )
            else:
                resume_lookup, resume_completed = {}, set()
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
        state = DownloadRunState(
            session_factory=session_factory,
            options=options,
            resume_lookup=resume_lookup,
            resume_completed=resume_completed,
        )
        self.state = state
        return state

    def _load_resume_state(
        self, resume_path: Path
    ) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
        """Load resume metadata from JSON manifests with SQLite fallback."""

        sqlite_path = self.resolved.sqlite_path
        needs_warning = False
        if not resume_path.exists():
            prefix = f"{resume_path.name}."
            has_rotated = any(
                candidate.is_file() and candidate.name[len(prefix) :].isdigit()
                for candidate in resume_path.parent.glob(f"{resume_path.name}.*")
            )
            needs_warning = not has_rotated and sqlite_path and sqlite_path.exists()

        if needs_warning:
            LOGGER.warning(
                "Resume manifest %s is missing; loading resume metadata from SQLite %s.",
                resume_path,
                sqlite_path,
            )

        return load_previous_manifest(
            resume_path,
            sqlite_path=sqlite_path,
            allow_sqlite_fallback=True,
        )

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
        result = process_one_work(
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
                        self.process_work_item(artifact, state.options, session=session)
                        if self.args.sleep > 0:
                            time.sleep(self.args.sleep)
                else:
                    executor = self.setup_worker_pool()
                    with executor:
                        in_flight: List[Future[Dict[str, Any]]] = []
                        max_in_flight = max(self.args.workers * 2, 1)
                        future_work_ids: Dict[Future[Dict[str, Any]], Optional[str]] = {}

                        def _submit(work_item: WorkArtifact) -> Future[Dict[str, Any]]:
                            future = executor.submit(
                                self.process_work_item, work_item, state.options
                            )
                            future_work_ids[future] = getattr(work_item, "work_id", None)
                            return future

                        def _handle_future(completed_future: Future[Dict[str, Any]]) -> None:
                            work_id = future_work_ids.pop(completed_future, None)
                            try:
                                completed_future.result()
                            except Exception as exc:
                                state.worker_failures += 1
                                extra_fields: Dict[str, Any] = {"error": str(exc)}
                                if work_id:
                                    extra_fields["work_id"] = work_id
                                LOGGER.exception(
                                    "worker_crash",
                                    extra={"extra_fields": extra_fields},
                                )
                                state.update_from_result({"skipped": True})
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

        except Exception:
            raise
        else:
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


def iterate_openalex(
    query: Works, per_page: int, max_results: Optional[int]
) -> Iterable[Dict[str, Any]]:
    """Iterate over OpenAlex works respecting pagination and limits."""

    pager = query.paginate(
        per_page=per_page, n_max=max_results if max_results is not None else None
    )
    retrieved = 0
    for page in pager:
        for work in page:
            yield work
            retrieved += 1
            if max_results is not None and retrieved >= max_results:
                return


def run(resolved: ResolvedConfig) -> RunResult:
    """Execute the download pipeline using a :class:`DownloadRun` orchestration."""

    download_run = DownloadRun(resolved)
    return download_run.run()
