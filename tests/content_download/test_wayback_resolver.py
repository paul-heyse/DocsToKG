"""Unit tests for the Wayback resolver."""

from pathlib import Path
import pytest

from DocsToKG.ContentDownload.core import WorkArtifact
from DocsToKG.ContentDownload.pipeline import ResolverConfig
from DocsToKG.ContentDownload.resolvers.base import ResolverEvent, ResolverEventReason
from DocsToKG.ContentDownload.urls import canonical_for_request
from DocsToKG.ContentDownload.resolvers.wayback import WaybackResolver


@pytest.fixture
def resolver() -> WaybackResolver:
    return WaybackResolver()


@pytest.fixture
def config() -> ResolverConfig:
    cfg = ResolverConfig()
    cfg.wayback_config = {
        "year_window": 2,
        "max_snapshots": 8,
        "min_pdf_bytes": 4096,
        "html_parse": True,
        "availability_first": True,
    }
    return cfg


@pytest.fixture
def artifact() -> WorkArtifact:
    return WorkArtifact(
        work_id="W-test",
        title="Test Work",
        failed_pdf_urls=["https://example.com/paper.pdf"],
        publication_year=2023,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="test-work",
        pdf_dir=Path("/tmp/pdf"),
        html_dir=Path("/tmp/html"),
        xml_dir=Path("/tmp/xml"),
    )


def test_is_enabled(resolver: WaybackResolver, config: ResolverConfig, artifact: WorkArtifact):
    assert resolver.is_enabled(config, artifact) is True
    artifact.failed_pdf_urls = []
    assert resolver.is_enabled(config, artifact) is False


def test_iter_urls_when_no_failed_urls(resolver: WaybackResolver, config: ResolverConfig):
    art = WorkArtifact(
        work_id="foo",
        title="Foo",
        failed_pdf_urls=[],
        publication_year=None,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="foo",
        pdf_dir=Path("/tmp/pdf"),
        html_dir=Path("/tmp/html"),
        xml_dir=Path("/tmp/xml"),
    )

    results = list(resolver.iter_urls(object(), config, art))
    assert len(results) == 1
    event = results[0]
    assert event.is_event
    assert event.event == ResolverEvent.SKIPPED
    assert event.event_reason == ResolverEventReason.NO_FAILED_URLS


def test_availability_fast_path(monkeypatch, resolver, config, artifact):
    archive_url = "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf"

    def fake_availability(self, client, cfg, url, min_bytes):
        assert url == artifact.failed_pdf_urls[0]
        return archive_url, {
            "availability_checked": True,
            "availability_timestamp": "20230101000000",
        }

    def fake_query(self, client, cfg, url, year, window, max_snapshots):
        return []

    def fake_verify(self, client, cfg, url, min_bytes):
        return True, {
            "head_status": 200,
            "content_type": "application/pdf",
            "content_length": "50000",
            "pdf_signature": True,
        }

    monkeypatch.setattr(WaybackResolver, "_check_availability", fake_availability, raising=False)
    monkeypatch.setattr(WaybackResolver, "_query_cdx", fake_query, raising=False)
    monkeypatch.setattr(WaybackResolver, "_verify_pdf_snapshot", fake_verify, raising=False)

    results = list(resolver.iter_urls(object(), config, artifact))
    assert len(results) == 1
    result = results[0]
    assert result.url == canonical_for_request(archive_url, role="artifact")
    assert result.metadata["discovery_method"] == "availability"
    assert result.metadata["head_status"] == 200


def test_cdx_pdf_direct(monkeypatch, resolver, config, artifact):
    pdf_url = "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf"

    def fake_availability(self, client, cfg, url, min_bytes):
        return None, {"availability_checked": True}

    def fake_query(self, client, cfg, url, year, window, max_snapshots):
        return [
            {
                "archive_url": pdf_url,
                "timestamp": "20230101000000",
                "mimetype": "application/pdf",
                "statuscode": "200",
            }
        ]

    def fake_verify(self, client, cfg, url, min_bytes):
        return True, {"head_status": 200, "content_type": "application/pdf"}

    monkeypatch.setattr(WaybackResolver, "_check_availability", fake_availability, raising=False)
    monkeypatch.setattr(WaybackResolver, "_query_cdx", fake_query, raising=False)
    monkeypatch.setattr(WaybackResolver, "_verify_pdf_snapshot", fake_verify, raising=False)

    results = list(resolver.iter_urls(object(), config, artifact))
    assert len(results) == 1
    result = results[0]
    assert result.url == canonical_for_request(pdf_url, role="artifact")
    assert result.metadata["discovery_method"] == "cdx_pdf_direct"
    assert result.metadata["head_status"] == 200


def test_cdx_html_parse(monkeypatch, resolver, config, artifact):
    html_snapshot = "https://web.archive.org/web/20230101000000/https://example.com/page"
    pdf_candidate = "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf"

    def fake_availability(self, client, cfg, url, min_bytes):
        return None, {"availability_checked": True}

    def fake_query(self, client, cfg, url, year, window, max_snapshots):
        return [
            {
                "archive_url": html_snapshot,
                "timestamp": "20230101000000",
                "mimetype": "text/html",
                "statuscode": "200",
            }
        ]

    def fake_parse(self, client, cfg, html_url):
        assert html_url == html_snapshot
        return pdf_candidate, {"pdf_discovery_method": "meta"}

    def fake_verify(self, client, cfg, url, min_bytes):
        return True, {"head_status": 200, "content_type": "application/pdf"}

    monkeypatch.setattr(WaybackResolver, "_check_availability", fake_availability, raising=False)
    monkeypatch.setattr(WaybackResolver, "_query_cdx", fake_query, raising=False)
    monkeypatch.setattr(WaybackResolver, "_parse_html_for_pdf", fake_parse, raising=False)
    monkeypatch.setattr(WaybackResolver, "_verify_pdf_snapshot", fake_verify, raising=False)

    results = list(resolver.iter_urls(object(), config, artifact))
    assert len(results) == 1
    result = results[0]
    assert result.url == canonical_for_request(pdf_candidate, role="artifact")
    assert result.metadata["discovery_method"] == "cdx_html_parse"
    assert result.metadata["pdf_discovery_method"] == "meta"


def test_no_snapshot_found(monkeypatch, resolver, config, artifact):
    def fake_availability(self, client, cfg, url, min_bytes):
        return None, {"availability_checked": True}

    def fake_query(self, client, cfg, url, year, window, max_snapshots):
        return []

    monkeypatch.setattr(WaybackResolver, "_check_availability", fake_availability, raising=False)
    monkeypatch.setattr(WaybackResolver, "_query_cdx", fake_query, raising=False)

    results = list(resolver.iter_urls(object(), config, artifact))
    assert len(results) == 1
    result = results[0]
    assert result.is_event
    assert result.event == ResolverEvent.SKIPPED
    assert result.event_reason == ResolverEventReason.NOT_APPLICABLE
    assert result.metadata["reason"] == "no_snapshot"
