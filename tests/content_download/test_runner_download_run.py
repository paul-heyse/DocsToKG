import argparse
import contextlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
import pytest

from DocsToKG.ContentDownload.args import ResolvedConfig, bootstrap_run_environment
from DocsToKG.ContentDownload.core import WorkArtifact
from DocsToKG.ContentDownload.networking import ThreadLocalSessionFactory
from DocsToKG.ContentDownload.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.providers import WorkProvider
from DocsToKG.ContentDownload.runner import DownloadRun
from DocsToKG.ContentDownload.telemetry import (
    CsvSink,
    JsonlSink,
    MANIFEST_SCHEMA_VERSION,
    ManifestUrlIndex,
    MultiSink,
    RunTelemetry,
    SummarySink,
)


class DummyWorks:
    """Minimal Works stub yielding no results by default."""

    def __init__(self, pages: Optional[Sequence[Iterable[Dict[str, object]]]] = None) -> None:
        self._pages = list(pages or [])

    def paginate(
        self, per_page: int, n_max: Optional[int] = None
    ) -> Iterable[Iterable[Dict[str, object]]]:
        return iter(self._pages)


def _build_args(
    overrides: Optional[Dict[str, object]] = None, **extra: object
) -> argparse.Namespace:
    defaults: Dict[str, object] = {
        "log_rotate": None,
        "resume_from": None,
        "log_format": "jsonl",
        "log_csv": None,
        "dry_run": False,
        "list_only": False,
        "sniff_bytes": 4096,
        "min_pdf_bytes": 1024,
        "tail_check_bytes": 2048,
        "content_addressed": False,
        "verify_cache_digest": False,
        "warm_manifest_cache": False,
        "per_page": 25,
        "max": None,
        "workers": 1,
        "sleep": 0.0,
    }
    if overrides:
        defaults.update(overrides)
    defaults.update(extra)
    return argparse.Namespace(**defaults)


def make_resolved_config(
    tmp_path,
    *,
    workers: int = 1,
    csv: bool = True,
    works_pages: Optional[Sequence[Iterable[Dict[str, object]]]] = None,
) -> ResolvedConfig:
    args = _build_args(workers=workers)
    pdf_dir = tmp_path / "pdfs"
    html_dir = tmp_path / "html"
    xml_dir = tmp_path / "xml"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.jsonl"
    csv_path = manifest_dir / "manifest.csv" if csv else None
    sqlite_path = manifest_dir / "manifest.sqlite"
    resolver_config = SimpleNamespace(polite_headers={})
    query = DummyWorks(works_pages)
    return ResolvedConfig(
        args=args,
        run_id="run-123",
        query=query,
        pdf_dir=pdf_dir,
        html_dir=html_dir,
        xml_dir=xml_dir,
        manifest_path=manifest_path,
        csv_path=csv_path,
        sqlite_path=sqlite_path,
        resolver_instances=[],
        resolver_config=resolver_config,
        previous_url_index=ManifestUrlIndex(None),
        persistent_seen_urls=set(),
        robots_checker=None,
        concurrency_product=max(workers, 1),
        extract_html_text=False,
        verify_cache_digest=False,
    )


def _manifest_entry(work_id: str, *, run_id: str = "resume-run") -> str:
    return json.dumps(
        {
            "record_type": "manifest",
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "work_id": work_id,
            "url": f"https://example.org/{work_id}.pdf",
            "classification": "pdf",
        }
    )


def _make_artifact(resolved: ResolvedConfig, work_id: str) -> WorkArtifact:
    return WorkArtifact(
        work_id=work_id,
        title=f"Title {work_id}",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem=work_id.lower(),
        pdf_dir=resolved.pdf_dir,
        html_dir=resolved.html_dir,
        xml_dir=resolved.xml_dir,
    )


def test_setup_sinks_returns_multisink(tmp_path):
    resolved = make_resolved_config(tmp_path)
    bootstrap_run_environment(resolved)
    download_run = DownloadRun(resolved)

    with contextlib.ExitStack() as stack:
        sink = download_run.setup_sinks(stack)
        assert isinstance(sink, MultiSink)
        assert isinstance(download_run.attempt_logger, RunTelemetry)
        assert any(isinstance(member, JsonlSink) for member in sink._sinks)
        assert any(isinstance(member, SummarySink) for member in sink._sinks)

    download_run.close()


def test_setup_sinks_csv_format_skips_jsonl(tmp_path):
    resolved = make_resolved_config(tmp_path)
    resolved.args.log_format = "csv"
    resolved.args.log_csv = resolved.csv_path
    bootstrap_run_environment(resolved)
    download_run = DownloadRun(resolved)

    with contextlib.ExitStack() as stack:
        sink = download_run.setup_sinks(stack)
        assert isinstance(sink, MultiSink)
        assert not any(isinstance(member, JsonlSink) for member in sink._sinks)
        assert any(isinstance(member, CsvSink) for member in sink._sinks)

    download_run.close()


def test_setup_resolver_pipeline_returns_pipeline(tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)
    download_run = DownloadRun(resolved)

    with contextlib.ExitStack() as stack:
        download_run.setup_sinks(stack)
        pipeline = download_run.setup_resolver_pipeline()

    assert isinstance(pipeline, ResolverPipeline)
    assert download_run.metrics is not None
    download_run.close()


def test_setup_work_provider_returns_provider(tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    download_run = DownloadRun(resolved)

    provider = download_run.setup_work_provider()

    assert isinstance(provider, WorkProvider)
    download_run.close()


def test_setup_download_state_records_manifest_data(patcher, tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)
    download_run = DownloadRun(resolved)

    dummy_lookup = {"W1": {"path": "foo"}}
    dummy_completed = {"W2"}
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.load_previous_manifest",
        lambda _: (dummy_lookup, dummy_completed),
    )

    factory = ThreadLocalSessionFactory(requests.Session)
    state = download_run.setup_download_state(factory, robots_cache="robots")

    assert state.options.previous_lookup == dummy_lookup
    assert state.options.resume_completed == dummy_completed
    assert state.options.robots_checker == "robots"

    factory.close_all()
    download_run.close()


def test_setup_download_state_raises_when_resume_manifest_missing(tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)
    missing_manifest = resolved.manifest_path
    resolved.args.resume_from = missing_manifest
    download_run = DownloadRun(resolved)

    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        with pytest.raises(ValueError) as excinfo:
            download_run.setup_download_state(factory)
    finally:
        factory.close_all()
        download_run.close()

    message = str(excinfo.value)
    assert str(missing_manifest) in message
    assert "--resume-from" in message


def test_setup_download_state_falls_back_to_sqlite_when_manifest_missing(tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)
    sqlite_path = resolved.sqlite_path
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            """
            CREATE TABLE manifests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                run_id TEXT,
                schema_version INTEGER,
                work_id TEXT,
                title TEXT,
                publication_year INTEGER,
                resolver TEXT,
                url TEXT,
                normalized_url TEXT,
                path TEXT,
                path_mtime_ns INTEGER,
                classification TEXT,
                content_type TEXT,
                reason TEXT,
                reason_detail TEXT,
                html_paths TEXT,
                sha256 TEXT,
                content_length INTEGER,
                etag TEXT,
                last_modified TEXT,
                extracted_text_path TEXT,
                dry_run INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO manifests (
                timestamp, run_id, schema_version, work_id, title, publication_year,
                resolver, url, normalized_url, path, path_mtime_ns, classification,
                content_type, reason, reason_detail, html_paths, sha256,
                content_length, etag, last_modified, extracted_text_path, dry_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2025-01-01T00:00:00Z",
                "resume-run",
                MANIFEST_SCHEMA_VERSION,
                "W-SQLITE",
                "SQLite Resume",
                2024,
                "openalex",
                "https://example.org/W-SQLITE.pdf",
                "https://example.org/w-sqlite.pdf",
                str(resolved.pdf_dir / "stored.pdf"),
                None,
                "pdf",
                "application/pdf",
                None,
                None,
                None,
                "deadbeef",
                1024,
                None,
                None,
                None,
                0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    missing_manifest = resolved.manifest_path
    resolved.args.resume_from = missing_manifest
    download_run = DownloadRun(resolved)

    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        state = download_run.setup_download_state(factory)
    finally:
        factory.close_all()
        download_run.close()

    assert "W-SQLITE" in state.options.resume_completed
    previous_lookup = state.options.previous_lookup.get("W-SQLITE")
    assert previous_lookup is not None and previous_lookup
    sqlite_entry = next(iter(previous_lookup.values()))
    assert sqlite_entry["classification"] == "pdf"
    assert sqlite_entry["path"].endswith("stored.pdf")


def test_setup_worker_pool_creates_executor_when_parallel(tmp_path):
    resolved = make_resolved_config(tmp_path, workers=3)
    download_run = DownloadRun(resolved)

    pool = download_run.setup_worker_pool()
    try:
        assert isinstance(pool, ThreadPoolExecutor)
    finally:
        pool.shutdown(wait=True)
        download_run.close()


def test_download_run_run_processes_artifacts(patcher, tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)

    artifacts = [
        WorkArtifact(
            work_id="W1",
            title="First",
            publication_year=2024,
            doi=None,
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="w1",
            pdf_dir=resolved.pdf_dir,
            html_dir=resolved.html_dir,
            xml_dir=resolved.xml_dir,
        ),
        WorkArtifact(
            work_id="W2",
            title="Second",
            publication_year=2024,
            doi=None,
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="w2",
            pdf_dir=resolved.pdf_dir,
            html_dir=resolved.html_dir,
            xml_dir=resolved.xml_dir,
        ),
    ]

    class StubProvider:
        def __init__(self, batch: List[WorkArtifact]) -> None:
            self._batch = batch

        def iter_artifacts(self) -> Iterable[WorkArtifact]:
            yield from self._batch

    patcher.setattr(
        "DocsToKG.ContentDownload.runner.load_previous_manifest",
        lambda _: ({}, set()),
    )

    def fake_setup_work_provider(self: DownloadRun) -> WorkProvider:
        provider = StubProvider(artifacts)
        self.provider = provider
        return provider

    patcher.setattr(DownloadRun, "setup_work_provider", fake_setup_work_provider)

    def fake_process_one_work(
        work: WorkArtifact,
        session: requests.Session,
        pdf_dir,
        html_dir,
        xml_dir,
        pipeline,
        logger,
        metrics,
        *,
        options,
        session_factory=None,
    ) -> Dict[str, Any]:
        return {"saved": True, "downloaded_bytes": 42}

    patcher.setattr(
        "DocsToKG.ContentDownload.runner.process_one_work",
        fake_process_one_work,
    )

    download_run = DownloadRun(resolved)
    result = download_run.run()

    assert result.processed == 2
    assert result.saved == 2
    assert result.bytes_downloaded == 84

    download_run.close()


def test_run_auto_resume_uses_manifest_path_when_resume_flag_absent(patcher, tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)
    resolved.manifest_path.write_text(_manifest_entry("W_SKIP") + "\n", encoding="utf-8")

    artifacts = [
        _make_artifact(resolved, "W_SKIP"),
        _make_artifact(resolved, "W_RUN"),
    ]

    class StubProvider:
        def __init__(self, batch: List[WorkArtifact]) -> None:
            self._batch = batch

        def iter_artifacts(self) -> Iterable[WorkArtifact]:
            yield from self._batch

    def fake_setup_work_provider(self: DownloadRun) -> WorkProvider:
        provider = StubProvider(artifacts)
        self.provider = provider
        return provider

    def fake_process_one_work(
        work: WorkArtifact,
        session: requests.Session,
        pdf_dir,
        html_dir,
        xml_dir,
        pipeline,
        logger,
        metrics,
        *,
        options,
        session_factory=None,
    ) -> Dict[str, Any]:
        skipped = work.work_id in options.resume_completed
        return {
            "skipped": skipped,
            "saved": not skipped,
            "downloaded_bytes": 0,
        }

    patcher.setattr(DownloadRun, "setup_work_provider", fake_setup_work_provider)
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.process_one_work",
        fake_process_one_work,
    )

    download_run = DownloadRun(resolved)
    result = download_run.run()

    assert result.processed == 2
    assert result.skipped == 1
    assert result.saved == 1


def test_run_respects_explicit_resume_from_override(patcher, tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)
    resolved.manifest_path.write_text(_manifest_entry("W_BASE") + "\n", encoding="utf-8")

    override_path = resolved.manifest_path.with_name("override.jsonl")
    override_path.write_text(_manifest_entry("W_OVERRIDE") + "\n", encoding="utf-8")
    resolved.args.resume_from = override_path

    artifacts = [
        _make_artifact(resolved, "W_BASE"),
        _make_artifact(resolved, "W_OVERRIDE"),
    ]

    class StubProvider:
        def __init__(self, batch: List[WorkArtifact]) -> None:
            self._batch = batch

        def iter_artifacts(self) -> Iterable[WorkArtifact]:
            yield from self._batch

    def fake_setup_work_provider(self: DownloadRun) -> WorkProvider:
        provider = StubProvider(artifacts)
        self.provider = provider
        return provider

    def fake_process_one_work(
        work: WorkArtifact,
        session: requests.Session,
        pdf_dir,
        html_dir,
        xml_dir,
        pipeline,
        logger,
        metrics,
        *,
        options,
        session_factory=None,
    ) -> Dict[str, Any]:
        skipped = work.work_id in options.resume_completed
        return {
            "skipped": skipped,
            "saved": not skipped,
            "downloaded_bytes": 0,
        }

    patcher.setattr(DownloadRun, "setup_work_provider", fake_setup_work_provider)
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.process_one_work",
        fake_process_one_work,
    )

    download_run = DownloadRun(resolved)
    result = download_run.run()

    assert result.processed == 2
    assert result.skipped == 1
    assert result.saved == 1
    # Explicit override should prefer override.jsonl, leaving W_BASE unskipped.
    assert resolved.args.resume_from == override_path
