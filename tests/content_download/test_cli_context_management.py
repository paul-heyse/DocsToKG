# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_cli_context_management",
#   "purpose": "Pytest coverage for content download cli context management scenarios",
#   "sections": [
#     {
#       "id": "recordingsink",
#       "name": "_RecordingSink",
#       "anchor": "class-recordingsink",
#       "kind": "class"
#     },
#     {
#       "id": "recordinglastattemptsink",
#       "name": "_RecordingLastAttemptSink",
#       "anchor": "class-recordinglastattemptsink",
#       "kind": "class"
#     },
#     {
#       "id": "recordingcsvsink",
#       "name": "_RecordingCsvSink",
#       "anchor": "class-recordingcsvsink",
#       "kind": "class"
#     },
#     {
#       "id": "recordingindexsink",
#       "name": "_RecordingIndexSink",
#       "anchor": "class-recordingindexsink",
#       "kind": "class"
#     },
#     {
#       "id": "base-args",
#       "name": "_base_args",
#       "anchor": "function-base-args",
#       "kind": "function"
#     },
#     {
#       "id": "single-work",
#       "name": "_single_work",
#       "anchor": "function-single-work",
#       "kind": "function"
#     },
#     {
#       "id": "test-main-closes-sinks-when-pipeline-raises",
#       "name": "test_main_closes_sinks_when_pipeline_raises",
#       "anchor": "function-test-main-closes-sinks-when-pipeline-raises",
#       "kind": "function"
#     },
#     {
#       "id": "test-main-with-csv-releases-attempt-files",
#       "name": "test_main_with_csv_releases_attempt_files",
#       "anchor": "function-test-main-with-csv-releases-attempt-files",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader


class _RecordingSink:
    instances: List["_RecordingSink"] = []

    def __init__(self, path: Path):
        self.path = Path(path)
        self.closed = False
        self.logged: List[Dict[str, Any]] = []
        _RecordingSink.instances.append(self)

    def __enter__(self) -> "_RecordingSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()

    def log_attempt(self, *args, **kwargs) -> None:  # noqa: D401
        self.logged.append({"type": "attempt", "args": args, "kwargs": kwargs})

    def log_manifest(self, *args, **kwargs) -> None:  # noqa: D401
        self.logged.append({"type": "manifest", "args": args, "kwargs": kwargs})

    def log_summary(self, *args, **kwargs) -> None:  # noqa: D401
        self.logged.append({"type": "summary", "args": args, "kwargs": kwargs})

    def close(self) -> None:  # noqa: D401
        self.closed = True


class _RecordingLastAttemptSink(_RecordingSink):
    pass


class _RecordingCsvSink(_RecordingSink):
    pass


class _RecordingIndexSink(_RecordingSink):
    pass
# --- Helper Functions ---


def _base_args(tmp_path: Path) -> list[str]:
    manifest = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    return [
        "download_pyalex_pdfs.py",
        "--topic",
        "context",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(out_dir),
        "--manifest",
        str(manifest),
        "--max",
        "1",
    ]


def _single_work() -> Dict[str, Any]:
    return {
        "id": "https://openalex.org/WC1",
        "title": "Context Work",
        "publication_year": 2020,
        "ids": {"doi": "10.1000/context"},
        "open_access": {"oa_url": None},
        "best_oa_location": {"pdf_url": "https://oa.example/context.pdf"},
        "primary_location": {},
        "locations": [],
    }
# --- Test Cases ---


def test_main_closes_sinks_when_pipeline_raises(monkeypatch, tmp_path):
    _RecordingSink.instances = []

    monkeypatch.setattr(downloader, "JsonlSink", _RecordingSink)
    monkeypatch.setattr(downloader, "ManifestIndexSink", _RecordingIndexSink)
    monkeypatch.setattr(downloader, "CsvSink", _RecordingCsvSink)
    monkeypatch.setattr(downloader, "LastAttemptCsvSink", _RecordingLastAttemptSink)

    def boom(*_args, **_kwargs):
        raise RuntimeError("resolver boom")

    monkeypatch.setattr(downloader, "process_one_work", boom)
    monkeypatch.setattr(downloader, "default_resolvers", lambda: [])
    monkeypatch.setattr(
        downloader, "iterate_openalex", lambda *args, **kwargs: iter([_single_work()])
    )
    monkeypatch.setattr(downloader, "resolve_topic_id_if_needed", lambda topic: topic)

    argv = _base_args(tmp_path)
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(RuntimeError) as excinfo:
        downloader.main()

    assert "resolver boom" in str(excinfo.value)

    assert _RecordingSink.instances, "expected Jsonl/Index sinks to be created"
    assert all(inst.closed for inst in _RecordingSink.instances)


def test_main_with_csv_releases_attempt_files(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "default_resolvers", lambda: [])
    monkeypatch.setattr(
        downloader, "iterate_openalex", lambda *args, **kwargs: iter([_single_work()])
    )
    monkeypatch.setattr(downloader, "resolve_topic_id_if_needed", lambda topic: topic)

    pdf_dir = tmp_path / "out" / "pdfs"
    pdf_dir.mkdir(parents=True)
    pdf_path = pdf_dir / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    outcome = downloader.resolvers.DownloadOutcome(
        classification="pdf",
        path=str(pdf_path),
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=10.0,
    )

    class StubPipeline:
        def __init__(self, *_, logger=None, metrics=None, **kwargs):
            self.logger = logger
            self.metrics = metrics

        def run(self, session, artifact, context=None):
            self.logger.log_attempt(
                downloader.resolvers.AttemptRecord(
                    work_id=artifact.work_id,
                    resolver_name="stub",
                    resolver_order=1,
                    url="https://oa.example/context.pdf",
                    status=outcome.classification,
                    http_status=outcome.http_status,
                    content_type=outcome.content_type,
                    elapsed_ms=outcome.elapsed_ms,
                    dry_run=False,
                )
            )
            self.metrics.record_attempt("stub", outcome)
            return downloader.resolvers.PipelineResult(
                success=True,
                resolver_name="stub",
                url="https://oa.example/context.pdf",
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    monkeypatch.setattr(downloader, "ResolverPipeline", StubPipeline)

    argv = _base_args(tmp_path)
    argv.extend(["--log-format", "csv"])
    monkeypatch.setattr("sys.argv", argv)

    downloader.main()

    manifest = Path(argv[argv.index("--manifest") + 1])
    csv_path = manifest.with_suffix(".csv")
    last_path = manifest.with_name("manifest.last.csv")

    assert csv_path.exists()
    assert last_path.exists()

    moved = csv_path.with_name(csv_path.name + ".moved")
    csv_path.rename(moved)
    assert moved.exists()
