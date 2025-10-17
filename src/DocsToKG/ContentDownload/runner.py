from __future__ import annotations

import contextlib
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from typing import Any, Dict, Iterable, List, Optional

import requests
from pyalex import Works

from DocsToKG.ContentDownload.args import ResolvedConfig
from DocsToKG.ContentDownload.download import (
    DownloadOptions,
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
    AttemptSink,
    CsvSink,
    JsonlSink,
    LastAttemptCsvSink,
    ManifestIndexSink,
    MultiSink,
    RotatingJsonlSink,
    RunTelemetry,
    SummarySink,
    SqliteSink,
    load_previous_manifest,
)
from DocsToKG.ContentDownload.core import WorkArtifact, atomic_write_text

__all__ = ['iterate_openalex', 'run']

LOGGER = logging.getLogger('DocsToKG.ContentDownload')

def iterate_openalex(
    query: Works, per_page: int, max_results: Optional[int]
) -> Iterable[Dict[str, Any]]:
    """Iterate over OpenAlex works respecting pagination and limits.

    Args:
        query: Configured Works query instance.
        per_page: Number of results to request per page.
        max_results: Optional maximum number of works to yield.

    Yields:
        Work payload dictionaries returned by the OpenAlex API.

    Returns:
        Iterable yielding the same work payload dictionaries for convenience.
    """
    pager = query.paginate(per_page=per_page, n_max=None)
    retrieved = 0
    for page in pager:
        for work in page:
            yield work
            retrieved += 1
            if max_results and retrieved >= max_results:
                return


def run(resolved: ResolvedConfig) -> RunResult:
    """Execute the download pipeline using the provided resolved configuration."""

    args = resolved.args
    run_id = resolved.run_id
    pdf_dir = resolved.pdf_dir
    html_dir = resolved.html_dir
    xml_dir = resolved.xml_dir
    manifest_path = resolved.manifest_path
    csv_path = resolved.csv_path
    sqlite_path = resolved.sqlite_path
    budget_requests = resolved.budget_requests
    budget_bytes = resolved.budget_bytes

    summary: Dict[str, Any] = {}
    summary_record: Dict[str, Any] = {}

    processed = 0
    saved = 0
    html_only = 0
    xml_only = 0
    skipped = 0
    total_downloaded_bytes = 0
    stop_due_to_budget = False
    worker_failures = 0

    with contextlib.ExitStack() as stack:
        sinks: List[AttemptSink] = []
        if args.log_rotate:
            jsonl_sink = stack.enter_context(
                RotatingJsonlSink(manifest_path, max_bytes=args.log_rotate)
            )
        else:
            jsonl_sink = stack.enter_context(JsonlSink(manifest_path))
        sinks.append(jsonl_sink)

        index_path = manifest_path.with_suffix('.index.json')
        index_sink = stack.enter_context(ManifestIndexSink(index_path))
        sinks.append(index_sink)

        last_attempt_path = manifest_path.with_suffix('.last.csv')
        last_attempt_sink = stack.enter_context(LastAttemptCsvSink(last_attempt_path))
        sinks.append(last_attempt_sink)

        sqlite_sink = stack.enter_context(SqliteSink(sqlite_path))
        sinks.append(sqlite_sink)

        summary_path = manifest_path.with_suffix('.summary.json')
        summary_sink = stack.enter_context(SummarySink(summary_path))
        sinks.append(summary_sink)

        if csv_path:
            csv_sink = stack.enter_context(CsvSink(csv_path))
            sinks.append(csv_sink)

        attempt_logger: AttemptSink = sinks[0] if len(sinks) == 1 else MultiSink(sinks)
        attempt_logger = RunTelemetry(attempt_logger)

        resume_lookup, resume_completed = load_previous_manifest(args.resume_from)
        download_options = DownloadOptions(
            dry_run=args.dry_run,
            list_only=args.list_only,
            extract_html_text=args.extract_html_text,
            run_id=run_id,
            previous_lookup=resume_lookup,
            resume_completed=resume_completed,
            max_bytes=args.max_bytes,
            sniff_bytes=args.sniff_bytes,
            min_pdf_bytes=args.min_pdf_bytes,
            tail_check_bytes=args.tail_check_bytes,
            robots_checker=resolved.robots_checker,
            content_addressed=args.content_addressed,
        )

        metrics = ResolverMetrics()
        pipeline = ResolverPipeline(
            resolvers=resolved.resolver_instances,
            config=resolved.resolver_config,
            download_func=download_candidate,
            logger=attempt_logger,
            metrics=metrics,
            initial_seen_urls=resolved.persistent_seen_urls or None,
            global_manifest_index=resolved.previous_url_index,
            run_id=run_id,
        )

        work_iterable = iterate_openalex(
            resolved.query,
            per_page=args.per_page,
            max_results=args.max,
        )

        provider: WorkProvider = OpenAlexWorkProvider(
            query=resolved.query,
            works_iterable=work_iterable,
            artifact_factory=create_artifact,
            pdf_dir=pdf_dir,
            html_dir=html_dir,
            xml_dir=xml_dir,
            per_page=args.per_page,
            max_results=args.max,
        )

        concurrency_product = resolved.concurrency_product
        pool_connections = min(max(64, concurrency_product * 4), 512)
        pool_maxsize = min(max(128, concurrency_product * 8), 1024)

        def _build_thread_session() -> requests.Session:
            return create_session(
                resolved.resolver_config.polite_headers,
                pool_connections=pool_connections,
                pool_maxsize=pool_maxsize,
            )

        session_factory = ThreadLocalSessionFactory(_build_thread_session)

        def _record_result(res: Dict[str, Any]) -> None:
            nonlocal processed, saved, html_only, xml_only, skipped, total_downloaded_bytes
            processed += 1
            if res.get('saved'):
                saved += 1
            if res.get('html_only'):
                html_only += 1
            if res.get('xml_only'):
                xml_only += 1
            if res.get('skipped'):
                skipped += 1
            downloaded = res.get('downloaded_bytes') or 0
            try:
                total_downloaded_bytes += int(downloaded)
            except (TypeError, ValueError):
                pass

        def _should_stop() -> bool:
            if budget_requests is not None and processed >= budget_requests:
                return True
            if budget_bytes is not None and total_downloaded_bytes >= budget_bytes:
                return True
            return False

        try:
            if args.workers == 1:
                session = session_factory()
                for artifact in provider.iter_artifacts():
                    if stop_due_to_budget:
                        break
                    result = process_one_work(
                        artifact,
                        session,
                        pdf_dir,
                        html_dir,
                        xml_dir,
                        pipeline,
                        attempt_logger,
                        metrics,
                        options=download_options,
                        session_factory=session_factory,
                    )
                    _record_result(result)
                    if not stop_due_to_budget and _should_stop():
                        stop_due_to_budget = True
                        break
                    if args.sleep > 0:
                        time.sleep(args.sleep)
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    in_flight: List[Future[Dict[str, Any]]] = []
                    max_in_flight = max(args.workers * 2, 1)
                    future_work_ids: Dict[Future[Dict[str, Any]], Optional[str]] = {}

                    def _runner(work_item: WorkArtifact) -> Dict[str, Any]:
                        session = session_factory()
                        return process_one_work(
                            work_item,
                            session,
                            pdf_dir,
                            html_dir,
                            xml_dir,
                            pipeline,
                            attempt_logger,
                            metrics,
                            options=download_options,
                            session_factory=session_factory,
                        )

                    def _submit(work_item: WorkArtifact) -> Future[Dict[str, Any]]:
                        future = executor.submit(_runner, work_item)
                        future_work_ids[future] = getattr(work_item, 'work_id', None)
                        return future

                    def _handle_future(completed_future: Future[Dict[str, Any]]) -> None:
                        nonlocal stop_due_to_budget, worker_failures
                        work_id = future_work_ids.pop(completed_future, None)
                        try:
                            result = completed_future.result()
                        except Exception as exc:
                            worker_failures += 1
                            extra_fields: Dict[str, Any] = {'error': str(exc)}
                            if work_id:
                                extra_fields['work_id'] = work_id
                            LOGGER.exception(
                                'worker_crash',
                                extra={'extra_fields': extra_fields},
                            )
                            _record_result({'skipped': True})
                            if not stop_due_to_budget and _should_stop():
                                stop_due_to_budget = True
                            return

                        _record_result(result)
                        if not stop_due_to_budget and _should_stop():
                            stop_due_to_budget = True

                    for artifact in provider.iter_artifacts():
                        if stop_due_to_budget:
                            break
                        if len(in_flight) >= max_in_flight:
                            done, pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
                            for completed_future in done:
                                _handle_future(completed_future)
                            in_flight = list(pending)
                            if stop_due_to_budget:
                                break
                        future = _submit(artifact)
                        in_flight.append(future)

                    if in_flight:
                        for future in as_completed(list(in_flight)):
                            _handle_future(future)
        except Exception:
            raise
        else:
            if stop_due_to_budget:
                LOGGER.info(
                    'Stopping due to budget exhaustion',
                    extra={
                        'extra_fields': {
                            'budget_requests': budget_requests,
                            'budget_bytes': budget_bytes,
                            'processed': processed,
                            'bytes_downloaded': total_downloaded_bytes,
                        }
                    },
                )
            summary = metrics.summary()
            summary_record = build_summary_record(
                run_id=run_id,
                processed=processed,
                saved=saved,
                html_only=html_only,
                xml_only=xml_only,
                skipped=skipped,
                worker_failures=worker_failures,
                bytes_downloaded=total_downloaded_bytes,
                summary=summary,
                budget_requests=budget_requests,
                budget_bytes=budget_bytes,
                stop_due_to_budget=stop_due_to_budget,
            )
            try:
                attempt_logger.log_summary(summary_record)
            except Exception:
                LOGGER.warning('Failed to log summary record', exc_info=True)
            metrics_path = manifest_path.with_suffix('.metrics.json')
            try:
                ensure_dir(metrics_path.parent)
                atomic_write_text(
                    metrics_path,
                    json.dumps(summary_record, indent=2, sort_keys=True) + "\n",
                )
            except Exception:
                LOGGER.warning('Failed to write metrics sidecar %s', metrics_path, exc_info=True)
        finally:
            session_factory.close_all()

    return RunResult(
        run_id=run_id,
        processed=processed,
        saved=saved,
        html_only=html_only,
        xml_only=xml_only,
        skipped=skipped,
        worker_failures=worker_failures,
        bytes_downloaded=total_downloaded_bytes,
        budget_requests=budget_requests,
        budget_bytes=budget_bytes,
        stop_due_to_budget=stop_due_to_budget,
        summary=summary,
        summary_record=summary_record,
    )
