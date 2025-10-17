import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

requests = pytest.importorskip("requests")

from DocsToKG.ContentDownload.core import Classification, WorkArtifact
from DocsToKG.ContentDownload.pipeline import (  # noqa: E402
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    ResolverResult,
    ResolverEvent,
    ResolverEventReason,
    find_pdf_via_anchor,
    find_pdf_via_link,
    find_pdf_via_meta,
)
from DocsToKG.ContentDownload.telemetry import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    JsonlSink,
    ManifestEntry,
    SqliteSink,
    load_manifest_url_index,
    load_previous_manifest,
)

bs4 = pytest.importorskip("bs4")
from bs4 import BeautifulSoup  # type: ignore  # noqa: E402


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def log_attempt(self, record, *, timestamp: Optional[str] = None) -> None:  # noqa: D401
        self.records.append(record)

    def log_manifest(self, entry) -> None:  # noqa: D401
        return None

    def log_summary(self, summary: Dict[str, Any]) -> None:  # noqa: D401
        return None

    def close(self) -> None:  # noqa: D401
        return None


class _EventResolver:
    name = "event"

    def is_enabled(self, config: ResolverConfig, artifact: WorkArtifact) -> bool:
        return True

    def iter_urls(
        self,
        session,  # pragma: no cover - interface compatibility only
        config: ResolverConfig,
        artifact: WorkArtifact,
    ):
        yield ResolverResult(
            url=None,
            event=ResolverEvent.ERROR,
            event_reason=ResolverEventReason.HTTP_ERROR,
            metadata={"detail": "simulated"},
            http_status=503,
        )


def test_manifest_entry_roundtrip_matches_golden(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "manifest.jsonl"
    sqlite_path = tmp_path / "manifest.sqlite3"

    entry = ManifestEntry(
        schema_version=MANIFEST_SCHEMA_VERSION,
        timestamp="2025-01-01T00:00:00Z",
        run_id="run-contract",
        work_id="W-contract",
        title="Contract Test Work",
        publication_year=2024,
        resolver="contract-resolver",
        url="https://pub.example/paper.pdf",
        path="/tmp/paper.pdf",
        classification=Classification.PDF.value,
        content_type="application/pdf",
        reason=None,
        reason_detail=None,
        html_paths=["/tmp/paper.html"],
        sha256="deadbeefcafebabe",
        content_length=12345,
        etag='"etag"',
        last_modified="Wed, 01 Jan 2025 00:00:00 GMT",
        extracted_text_path="/tmp/paper.txt",
        dry_run=False,
    )

    jsonl_sink = JsonlSink(jsonl_path)
    jsonl_sink.log_manifest(entry)
    jsonl_sink.close()

    sqlite_sink = SqliteSink(sqlite_path)
    sqlite_sink.log_manifest(entry)
    sqlite_sink.close()

    lines = [line.strip() for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    jsonl_record = json.loads(lines[0])
    per_work, completed = load_previous_manifest(jsonl_path)
    sqlite_index = load_manifest_url_index(sqlite_path)

    actual = {
        "jsonl": jsonl_record,
        "per_work": per_work,
        "completed": sorted(completed),
        "sqlite": sqlite_index,
    }

    golden_path = Path("tests/data/content_download/golden/manifest_roundtrip.json")
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    assert actual == expected


def test_resolver_event_attempt_contract(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    xml_dir = tmp_path / "xml"
    pdf_dir.mkdir()
    html_dir.mkdir()
    xml_dir.mkdir()

    artifact = WorkArtifact(
        work_id="W-contract",
        title="Contracted Resolver Event",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="contracted",
        pdf_dir=pdf_dir,
        html_dir=html_dir,
        xml_dir=xml_dir,
    )

    config = ResolverConfig(
        resolver_order=["event"],
        resolver_toggles={"event": True},
        enable_global_url_dedup=False,
    )
    metrics = ResolverMetrics()
    logger = _CaptureLogger()

    def _fail_download(*args, **kwargs):
        raise AssertionError("download should not be invoked for resolver events")

    pipeline = ResolverPipeline(
        [_EventResolver()],
        config,
        _fail_download,
        logger,
        metrics,
        run_id="contract-run",
    )

    result = pipeline.run(requests.Session(), artifact)
    assert result.success is False
    assert len(logger.records) == 1
    record = logger.records[0]
    assert record.status == "error"
    assert record.reason == "http-error"
    assert record.reason_detail == "http-error"
    assert record.http_status == 503
    assert record.url is None
    assert record.metadata == {"detail": "simulated"}
    assert record.resolver_name == "event"
    assert record.resolver_order == 1
    assert record.dry_run is False
    assert metrics.summary()["skips"] == {"event:http-error": 1}


FIXTURE_HTML_DIR = Path("tests/data/content_download/html")


def _load_html(name: str) -> BeautifulSoup:
    return BeautifulSoup(
        (FIXTURE_HTML_DIR / name).read_text(encoding="utf-8"),
        "html.parser",
    )


def test_find_pdf_via_meta_with_canned_html() -> None:
    soup = _load_html("with_meta.html")
    base = "https://journal.example/articles/volume-1/record/"
    expected = "https://journal.example/files/meta-paper.pdf"
    assert find_pdf_via_meta(soup, base) == expected


def test_find_pdf_via_link_with_canned_html() -> None:
    soup = _load_html("with_link.html")
    base = "https://journal.example/issues/2024/42/article"
    expected = "https://journal.example/downloads/link-paper.pdf"
    assert find_pdf_via_link(soup, base) == expected


def test_find_pdf_via_anchor_with_canned_html() -> None:
    soup = _load_html("with_anchor.html")
    base = "https://journal.example/view/anchor"
    expected = "https://journal.example/pdfs/anchor-paper.pdf"
    assert find_pdf_via_anchor(soup, base) == expected
