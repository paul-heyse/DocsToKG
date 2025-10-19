import argparse
import contextlib
import itertools
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pytest
import requests

from DocsToKG.ContentDownload.args import ResolvedConfig, bootstrap_run_environment
from DocsToKG.ContentDownload.core import Classification, ReasonCode, WorkArtifact
from DocsToKG.ContentDownload.download import handle_resume_logic
from DocsToKG.ContentDownload.networking import ThreadLocalSessionFactory
from DocsToKG.ContentDownload.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.providers import WorkProvider
from DocsToKG.ContentDownload.runner import DownloadRun, iterate_openalex
from DocsToKG.ContentDownload.telemetry import (
    MANIFEST_SCHEMA_VERSION,
    CsvSink,
    JsonlSink,
    ManifestEntry,
    ManifestUrlIndex,
    MultiSink,
    RunTelemetry,
    SqliteResumeLookup,
    SqliteSink,
    SummarySink,
    load_previous_manifest,
)


class DummyWorks:
    """Minimal Works stub yielding no results by default."""

    def __init__(self, pages: Optional[Sequence[Iterable[Dict[str, object]]]] = None) -> None:
        self._pages = list(pages or [])

    def paginate(
        self, per_page: int, n_max: Optional[int] = None
    ) -> Iterable[Iterable[Dict[str, object]]]:
        return iter(self._pages)


class FlakyWorks:
    """Works stub that raises once before yielding pages."""

    def __init__(self, pages: Sequence[Iterable[Dict[str, object]]]) -> None:
        self._pages = [list(page) for page in pages]

    def paginate(
        self, per_page: int, n_max: Optional[int] = None
    ) -> Iterable[Iterable[Dict[str, object]]]:
        return _FlakyPager(self._pages)


class _FlakyPager:
    def __init__(self, pages: Sequence[Iterable[Dict[str, object]]]) -> None:
        self._pages = [list(page) for page in pages]
        self._index = 0
        self._raised = False

    def __iter__(self) -> "_FlakyPager":
        return self

    def __next__(self) -> Iterable[Dict[str, object]]:
        if not self._raised:
            self._raised = True
            response = requests.Response()
            response.status_code = 429
            response.headers["Retry-After"] = "1"
            error = requests.HTTPError("rate limited")
            error.response = response
            raise error
        if self._index >= len(self._pages):
            raise StopIteration
        page = self._pages[self._index]
        self._index += 1
        return list(page)


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
        "openalex_retry_attempts": 3,
        "openalex_retry_backoff": 0.0,
        "retry_after_cap": None,
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
        openalex_retry_attempts=args.openalex_retry_attempts,
        openalex_retry_backoff=args.openalex_retry_backoff,
        retry_after_cap=args.retry_after_cap,
    )


def test_iterate_openalex_retries_then_succeeds(monkeypatch):
    works = FlakyWorks([[{"id": "W1"}, {"id": "W2"}]])
    slept: List[float] = []

    def _record_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr("DocsToKG.ContentDownload.runner.time.sleep", _record_sleep)

    results = list(
        iterate_openalex(
            works,
            per_page=2,
            max_results=None,
            retry_attempts=2,
            retry_backoff=0.2,
        )
    )

    assert [item["id"] for item in results] == ["W1", "W2"]
    assert slept and slept[0] >= 1.0


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


def _count_open_fds() -> int:
    fd_root = Path("/proc/self/fd")
    if not fd_root.exists():
        raise FileNotFoundError("/proc/self/fd is not available")
    return len(list(fd_root.iterdir()))


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
    resolved.manifest_path.write_text("", encoding="utf-8")
    download_run = DownloadRun(resolved)

    dummy_lookup = {"W1": {"path": "foo"}}
    dummy_completed = {"W2"}

    class _DummyJsonLookup(dict):
        def __init__(self) -> None:
            super().__init__(dummy_lookup)
            self.completed_work_ids = frozenset(dummy_completed)

        def close(self) -> None:
            pass

    lookup_instance = _DummyJsonLookup()

    patcher.setattr(
        "DocsToKG.ContentDownload.runner.JsonlResumeLookup",
        lambda *args, **kwargs: lookup_instance,
    )

    factory = ThreadLocalSessionFactory(requests.Session)
    state = download_run.setup_download_state(factory, robots_cache="robots")

    assert state.resume_lookup is lookup_instance
    assert state.options.previous_lookup is lookup_instance
    assert state.options.resume_completed == dummy_completed
    assert state.options.robots_checker == "robots"

    factory.close_all()
    download_run.close()


def test_setup_download_state_uses_lazy_sqlite_lookup(tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)
    sqlite_path = resolved.sqlite_path
    assert sqlite_path is not None

    with SqliteSink(sqlite_path) as sink:
        for index in range(2000):
            work_id = f"W{index}"
            entry = ManifestEntry(
                schema_version=MANIFEST_SCHEMA_VERSION,
                timestamp="2024-01-01T00:00:00Z",
                work_id=work_id,
                title=f"Work {index}",
                publication_year=None,
                resolver="resolver",
                url=f"https://example.org/{work_id}.pdf",
                path=f"/tmp/{work_id}.pdf",
                classification=Classification.PDF.value,
                content_type="application/pdf",
                reason=None,
                html_paths=[],
                sha256=None,
                content_length=1024,
                etag=None,
                last_modified=None,
                extracted_text_path=None,
                dry_run=False,
                path_mtime_ns=None,
                run_id="resume",
            )
            sink.log_manifest(entry)

    download_run = DownloadRun(resolved)
    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        state = download_run.setup_download_state(factory)
        lookup = state.options.previous_lookup
        assert isinstance(lookup, SqliteResumeLookup)
        assert getattr(lookup, "_cache") == {}
        sample = lookup.get("W100")
        assert sample is not None and sample
        assert len(getattr(lookup, "_cache")) == 1
    finally:
        factory.close_all()
        download_run.close()


def test_setup_download_state_fresh_csv_run_has_no_resume_warning(caplog, tmp_path):
    resolved = make_resolved_config(tmp_path)
    resolved.args.log_format = "csv"
    resolved.args.log_csv = resolved.csv_path
    bootstrap_run_environment(resolved)
    sqlite_path = resolved.sqlite_path
    assert sqlite_path is not None
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_path.touch(exist_ok=True)

    download_run = DownloadRun(resolved)

    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        with caplog.at_level(logging.WARNING):
            download_run.setup_download_state(factory)
        messages = [record.getMessage() for record in caplog.records]
        assert not any("Resume manifest" in message for message in messages)
    finally:
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
        message = str(excinfo.value)
        assert str(missing_manifest) in message
        assert "--resume-from" in message
    finally:
        factory.close_all()
        download_run.close()


def test_setup_download_state_expands_user_resume_path(tmp_path, patcher, monkeypatch):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    expected_resume = home_dir / "manifests" / "resume.jsonl"
    expected_resume.parent.mkdir(parents=True, exist_ok=True)

    captured: Dict[str, object] = {}

    def _capture_resume_path(_self, resume_path):
        captured["path"] = resume_path
        return {}, set(), None

    patcher.setattr(DownloadRun, "_load_resume_state", _capture_resume_path)

    resolved.args.resume_from = "~/manifests/resume.jsonl"
    download_run = DownloadRun(resolved)

    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        download_run.setup_download_state(factory)
        assert captured.get("path") == expected_resume
    finally:
        factory.close_all()
        download_run.close()


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
        assert "W-SQLITE" in state.options.resume_completed
        previous_lookup = state.options.previous_lookup.get("W-SQLITE")
        assert previous_lookup is not None and previous_lookup
        sqlite_entry = next(iter(previous_lookup.values()))
        assert sqlite_entry["classification"] == "pdf"
        assert sqlite_entry["path"].endswith("stored.pdf")
    finally:
        factory.close_all()
        download_run.close()


def test_download_run_closes_sqlite_resume_handles(tmp_path):
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
                "W-HANDLES",
                "SQLite Resume",
                2024,
                "openalex",
                "https://example.org/W-HANDLES.pdf",
                "https://example.org/w-handles.pdf",
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

    download_run = DownloadRun(resolved)
    factory = ThreadLocalSessionFactory(requests.Session)

    during = None
    after = None
    try:
        try:
            before = _count_open_fds()
        except FileNotFoundError:
            pytest.skip("/proc/self/fd is required for handle tracking")
        state = download_run.setup_download_state(factory)
        during = _count_open_fds()
        assert isinstance(state.resume_lookup, SqliteResumeLookup)
        assert during > before
    finally:
        download_run.close()
        if during is not None:
            after = _count_open_fds()
        factory.close_all()


def test_sqlite_resume_lookup_preload_allows_post_close_access(tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)
    sqlite_path = resolved.sqlite_path
    assert sqlite_path is not None

    with SqliteSink(sqlite_path) as sink:
        for index in range(5):
            work_id = f"W{index}"
            entry = ManifestEntry(
                schema_version=MANIFEST_SCHEMA_VERSION,
                timestamp="2024-02-01T00:00:00Z",
                work_id=work_id,
                title=f"Work {index}",
                publication_year=None,
                resolver="resolver",
                url=f"https://example.org/{work_id}.pdf",
                path=f"/tmp/{work_id}.pdf",
                classification=Classification.PDF.value,
                content_type="application/pdf",
                reason=None,
                html_paths=[],
                sha256=None,
                content_length=2048,
                etag=None,
                last_modified=None,
                extracted_text_path=None,
                dry_run=False,
                path_mtime_ns=None,
                run_id="resume",
            )
            sink.log_manifest(entry)

    download_run = DownloadRun(resolved)
    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        state = download_run.setup_download_state(factory)
        lookup = state.resume_lookup
        assert isinstance(lookup, SqliteResumeLookup)

        lookup.preload_all_entries()
        assert state.resume_cleanup is not None
        state.resume_cleanup()
        state.resume_cleanup = None

        cached_entry = next(iter(lookup["W3"].values()))
        assert cached_entry["path"].endswith("W3.pdf")
    finally:
        factory.close_all()
        download_run.close()


def test_resume_cleanup_avoids_bulk_selects_for_large_resume(tmp_path):
    resolved = make_resolved_config(tmp_path, csv=False)
    bootstrap_run_environment(resolved)
    sqlite_path = resolved.sqlite_path
    assert sqlite_path is not None

    with SqliteSink(sqlite_path) as sink:
        for index in range(5000):
            work_id = f"W{index}"
            entry = ManifestEntry(
                schema_version=MANIFEST_SCHEMA_VERSION,
                timestamp="2024-03-01T00:00:00Z",
                work_id=work_id,
                title=f"Work {index}",
                publication_year=None,
                resolver="resolver",
                url=f"https://example.org/{work_id}.pdf",
                path=f"/tmp/{work_id}.pdf",
                classification=Classification.PDF.value,
                content_type="application/pdf",
                reason=None,
                html_paths=[],
                sha256=None,
                content_length=4096,
                etag=None,
                last_modified=None,
                extracted_text_path=None,
                dry_run=False,
                path_mtime_ns=None,
                run_id="resume",
            )
            sink.log_manifest(entry)

    download_run = DownloadRun(resolved)
    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        state = download_run.setup_download_state(factory)
        lookup = state.resume_lookup
        assert isinstance(lookup, SqliteResumeLookup)
        sample_entry = next(iter(lookup["W1024"].values()))
        assert sample_entry["path"].endswith("W1024.pdf")

        traced: List[str] = []
        lookup._conn.set_trace_callback(traced.append)  # type: ignore[attr-defined]
        traced.clear()

        assert state.resume_cleanup is not None
        state.resume_cleanup()
        state.resume_cleanup = None

        select_queries = [
            sql for sql in traced if sql and sql.strip().lower().startswith("select")
        ]
        assert select_queries == []
    finally:
        factory.close_all()
        download_run.close()


def test_setup_download_state_resumes_with_csv_only_logs(tmp_path):
    resolved = make_resolved_config(tmp_path)
    bootstrap_run_environment(resolved)
    resolved.args.log_format = "csv"
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
                "2025-01-03T00:00:00Z",
                "resume-run",
                MANIFEST_SCHEMA_VERSION,
                "W-CSV",
                "CSV Resume",
                2024,
                "openalex",
                "https://example.org/W-CSV.pdf",
                "https://example.org/w-csv.pdf",
                str(resolved.pdf_dir / "stored.pdf"),
                None,
                "pdf",
                "application/pdf",
                None,
                None,
                None,
                "feedface",
                512,
                None,
                None,
                None,
                0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    csv_path = resolved.csv_path
    assert csv_path is not None
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("run_id,work_id\nresume-run,W-CSV\n", encoding="utf-8")

    if resolved.manifest_path.exists():
        resolved.manifest_path.unlink()

    download_run = DownloadRun(resolved)

    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        state = download_run.setup_download_state(factory)
        assert "W-CSV" in state.options.resume_completed
        csv_lookup = state.options.previous_lookup.get("W-CSV")
        assert csv_lookup is not None and csv_lookup
        resume_entry = next(iter(csv_lookup.values()))
        assert resume_entry["classification"] == "pdf"
        assert resume_entry["path"].endswith("stored.pdf")
    finally:
        factory.close_all()
        download_run.close()


def test_setup_download_state_accepts_explicit_csv_resume(tmp_path):
    resolved = make_resolved_config(tmp_path)
    bootstrap_run_environment(resolved)
    sqlite_path = resolved.sqlite_path
    assert sqlite_path is not None
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
                "2025-01-04T00:00:00Z",
                "resume-run",
                MANIFEST_SCHEMA_VERSION,
                "W-CSV",
                "CSV Resume",
                2024,
                "openalex",
                "https://example.org/W-CSV.pdf",
                "https://example.org/w-csv.pdf",
                str(resolved.pdf_dir / "stored.pdf"),
                None,
                "pdf",
                "application/pdf",
                None,
                None,
                None,
                "feedface",
                512,
                None,
                None,
                None,
                0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    csv_path = resolved.csv_path
    assert csv_path is not None
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("run_id,work_id\nresume-run,W-CSV\n", encoding="utf-8")

    if resolved.manifest_path.exists():
        resolved.manifest_path.unlink()

    resolved.args.resume_from = csv_path

    download_run = DownloadRun(resolved)

    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        state = download_run.setup_download_state(factory)
        assert "W-CSV" in state.options.resume_completed
        previous_lookup = state.options.previous_lookup.get("W-CSV")
        assert previous_lookup is not None and previous_lookup
        resume_entry = next(iter(previous_lookup.values()))
        assert resume_entry["classification"] == "pdf"
        assert resume_entry["path"].endswith("stored.pdf")
    finally:
        factory.close_all()
        download_run.close()


def test_setup_download_state_prefers_adjacent_sqlite_for_external_csv(tmp_path):
    resolved = make_resolved_config(tmp_path)
    bootstrap_run_environment(resolved)

    external_dir = tmp_path / "external_resume"
    external_dir.mkdir(parents=True, exist_ok=True)
    csv_path = external_dir / "external_attempts.csv"
    sqlite_path = csv_path.with_suffix(".sqlite3")

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
                "2025-02-01T00:00:00Z",
                "resume-run",
                MANIFEST_SCHEMA_VERSION,
                "W-EXTERNAL",
                "External CSV Resume",
                2024,
                "openalex",
                "https://example.org/W-EXTERNAL.pdf",
                "https://example.org/w-external.pdf",
                str(resolved.pdf_dir / "external.pdf"),
                None,
                "pdf",
                "application/pdf",
                None,
                None,
                None,
                "feedface",
                4096,
                None,
                None,
                None,
                0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    csv_path.write_text("run_id,work_id\nresume-run,W-EXTERNAL\n", encoding="utf-8")

    manifest_sqlite = resolved.sqlite_path
    if manifest_sqlite is not None and manifest_sqlite.exists():
        manifest_sqlite.unlink()

    resolved.args.resume_from = csv_path

    download_run = DownloadRun(resolved)

    factory = ThreadLocalSessionFactory(requests.Session)
    try:
        state = download_run.setup_download_state(factory)
        assert "W-EXTERNAL" in state.options.resume_completed
        previous_lookup = state.options.previous_lookup.get("W-EXTERNAL")
        assert previous_lookup is not None and previous_lookup
        resume_entry = next(iter(previous_lookup.values()))
        assert resume_entry["classification"] == "pdf"
        assert resume_entry["path"].endswith("external.pdf")
    finally:
        factory.close_all()
        download_run.close()


def test_setup_download_state_detects_cached_artifact_from_other_cwd(tmp_path, monkeypatch):
    run_root = tmp_path / "first"
    pdf_dir = run_root / "pdfs"
    html_dir = run_root / "html"
    xml_dir = run_root / "xml"
    for directory in (pdf_dir, html_dir, xml_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cached_pdf = pdf_dir / "w-abs.pdf"
    cached_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    manifest_path = run_root / "manifest.jsonl"
    manifest_entry = {
        "record_type": "manifest",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "resume-run",
        "work_id": "W-ABS",
        "url": "https://example.org/w-abs.pdf",
        "classification": "pdf",
        "path": "pdfs/w-abs.pdf",
    }
    manifest_path.write_text(json.dumps(manifest_entry) + "\n", encoding="utf-8")

    other_root = tmp_path / "second"
    other_root.mkdir()
    monkeypatch.chdir(other_root)

    resume_target = Path(os.path.relpath(manifest_path, other_root))
    relative_pdf_dir = Path(os.path.relpath(pdf_dir, other_root))
    relative_html_dir = Path(os.path.relpath(html_dir, other_root))
    relative_xml_dir = Path(os.path.relpath(xml_dir, other_root))

    args = _build_args()
    args.resume_from = resume_target

    resolved = ResolvedConfig(
        args=args,
        run_id="resume-run",
        query=DummyWorks(),
        pdf_dir=relative_pdf_dir,
        html_dir=relative_html_dir,
        xml_dir=relative_xml_dir,
        manifest_path=resume_target,
        csv_path=None,
        sqlite_path=resume_target.with_suffix(".sqlite3"),
        resolver_instances=[],
        resolver_config=SimpleNamespace(polite_headers={}),
        previous_url_index=ManifestUrlIndex(None),
        persistent_seen_urls=set(),
        robots_checker=None,
        concurrency_product=1,
        extract_html_text=False,
        verify_cache_digest=False,
        openalex_retry_attempts=args.openalex_retry_attempts,
        openalex_retry_backoff=args.openalex_retry_backoff,
        retry_after_cap=args.retry_after_cap,
    )

    bootstrap_run_environment(resolved)

    download_run = DownloadRun(resolved)
    factory = ThreadLocalSessionFactory(requests.Session)
    previous_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    cached_path: Optional[str] = None
    try:
        state = download_run.setup_download_state(factory)
        assert "W-ABS" in state.options.resume_completed
        previous_lookup = state.options.previous_lookup.get("W-ABS")
        assert previous_lookup is not None and previous_lookup
        resume_entry = next(iter(previous_lookup.values()))
        cached_path = resume_entry["path"]
    finally:
        factory.close_all()
        download_run.close()

    assert previous_lookup is not None and previous_lookup
    assert cached_path is not None
    # Paths in manifests are stored as relative paths
    assert not Path(cached_path).is_absolute()
    # The path should exist relative to the manifest directory (first), not current working directory (second)
    manifest_dir = resume_target.parent
    resolved_path = manifest_dir / cached_path
    assert resolved_path.exists()

    artifact = _make_artifact(resolved, "W-ABS")
    decision = handle_resume_logic(
        artifact,
        previous_lookup,
        state.options,
    )
    assert decision.should_skip is True
    assert decision.reason is ReasonCode.RESUME_COMPLETE
    assert decision.outcome is not None
    assert decision.outcome.classification is Classification.SKIPPED


def test_manifest_url_index_resolves_relative_paths(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    pdf_dir = run_root / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    cached_file = pdf_dir / "cached.pdf"
    cached_file.write_bytes(b"binary")

    sqlite_path = run_root / "manifest.sqlite3"
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            """
            CREATE TABLE manifests (
                timestamp TEXT,
                url TEXT,
                normalized_url TEXT,
                path TEXT,
                sha256 TEXT,
                classification TEXT,
                etag TEXT,
                last_modified TEXT,
                content_length INTEGER,
                path_mtime_ns INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO manifests (
                timestamp,
                url,
                normalized_url,
                path,
                sha256,
                classification,
                etag,
                last_modified,
                content_length,
                path_mtime_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2025-01-01T00:00:00Z",
                "https://example.org/cached.pdf",
                "https://example.org/cached.pdf",
                "pdfs/cached.pdf",
                "deadbeef",
                "pdf",
                None,
                None,
                cached_file.stat().st_size,
                cached_file.stat().st_mtime_ns,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    monkeypatch.chdir(other_root)

    relative_sqlite = Path(os.path.relpath(sqlite_path, other_root))
    index = ManifestUrlIndex(relative_sqlite)

    record = index.get("https://example.org/cached.pdf")
    assert record is not None
    assert record["path"] == str(cached_file.resolve())
    assert Path(record["path"]).exists()

    existing = list(index.iter_existing())
    assert existing
    _, meta = existing[0]
    assert meta["path"] == record["path"]


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
        DownloadRun,
        "_load_resume_state",
        lambda self, resume_path: ({}, set(), None),
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

    summary_path = resolved.manifest_path.with_suffix(".summary.json")
    metrics_path = resolved.manifest_path.with_suffix(".metrics.json")

    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload == result.summary_record

    assert metrics_path.exists()
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload == result.summary_record

    download_run.close()


def test_run_parallel_workers_aggregates_state(patcher, tmp_path):
    worker_count = 4
    total_artifacts = 40
    failure_ids = {f"W{index}" for index in range(0, total_artifacts, 7)}
    payload_size = 13

    resolved = make_resolved_config(tmp_path, csv=False, workers=worker_count)
    bootstrap_run_environment(resolved)

    artifacts = [_make_artifact(resolved, f"W{index}") for index in range(total_artifacts)]

    class StubProvider:
        def __init__(self, batch: List[WorkArtifact]) -> None:
            self._batch = batch

        def iter_artifacts(self) -> Iterable[WorkArtifact]:
            yield from self._batch

    patcher.setattr(
        DownloadRun,
        "_load_resume_state",
        lambda self, resume_path: ({}, set(), None),
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
        time.sleep(0.002)
        if work.work_id in failure_ids:
            raise RuntimeError("boom")
        return {"saved": True, "downloaded_bytes": payload_size}

    patcher.setattr(
        "DocsToKG.ContentDownload.runner.process_one_work",
        fake_process_one_work,
    )

    download_run = DownloadRun(resolved)
    result = download_run.run()

    assert result.processed == total_artifacts
    assert result.saved == total_artifacts - len(failure_ids)
    assert result.skipped == len(failure_ids)
    assert result.worker_failures == len(failure_ids)
    assert result.bytes_downloaded == (total_artifacts - len(failure_ids)) * payload_size

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


@pytest.mark.parametrize("worker_count", [1, 2])
def test_worker_crash_records_manifest_reason(patcher, tmp_path, worker_count):
    resolved = make_resolved_config(tmp_path, workers=worker_count, csv=False)
    bootstrap_run_environment(resolved)

    failure_id = "W_FAIL"
    artifacts = [
        _make_artifact(resolved, failure_id),
        _make_artifact(resolved, "W_OK"),
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
        if work.work_id == failure_id:
            raise RuntimeError("boom")
        return {"saved": True, "downloaded_bytes": 5}

    patcher.setattr(DownloadRun, "setup_work_provider", fake_setup_work_provider)
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.process_one_work",
        fake_process_one_work,
    )

    download_run = DownloadRun(resolved)
    result = download_run.run()

    assert result.worker_failures == 1
    per_work, completed = load_previous_manifest(resolved.manifest_path)
    assert failure_id not in completed

    crash_entries = per_work.get(failure_id)
    assert crash_entries is not None and crash_entries
    failure_record = next(iter(crash_entries.values()))

    assert failure_record["classification"] == Classification.SKIPPED.value
    assert failure_record["reason"] == ReasonCode.WORKER_EXCEPTION.value
    assert failure_record["reason_detail"] == "worker-crash"


def test_worker_tls_failure_discards_thread_session(patcher, tmp_path):
    resolved = make_resolved_config(tmp_path, workers=2, csv=False)
    bootstrap_run_environment(resolved)

    failure_id = "W_TLS_FAIL"
    recovery_id = "W_RECOVER"
    artifacts = [
        _make_artifact(resolved, failure_id),
        _make_artifact(resolved, recovery_id),
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

    session_counter = itertools.count()
    created_sessions: List["RecordingSession"] = []
    closed_tokens: List[int] = []

    class RecordingSession:
        def __init__(self, token: int) -> None:
            self.token = token

        def close(self) -> None:
            closed_tokens.append(self.token)

    def fake_create_session(*args, **kwargs):
        token = next(session_counter)
        session = RecordingSession(token)
        created_sessions.append(session)
        return session

    observed_sessions: List[RecordingSession] = []

    def fake_process_one_work(
        work: WorkArtifact,
        session: RecordingSession,
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
        observed_sessions.append(session)
        if work.work_id == failure_id:
            raise requests.exceptions.SSLError("tls handshake failed")
        return {"saved": True, "downloaded_bytes": 1}

    patcher.setattr(DownloadRun, "setup_work_provider", fake_setup_work_provider)
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.create_session", fake_create_session
    )
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.process_one_work",
        fake_process_one_work,
    )
    patcher.setattr(
        DownloadRun,
        "setup_worker_pool",
        lambda self: ThreadPoolExecutor(max_workers=1),
    )

    download_run = DownloadRun(resolved)
    result = download_run.run()

    assert result.worker_failures == 1
    assert len(observed_sessions) == 2
    assert observed_sessions[0] is not observed_sessions[1]
    assert created_sessions[0].token in closed_tokens
    assert closed_tokens[0] == created_sessions[0].token
