import hashlib
import sys
import types
from pathlib import Path
from typing import Callable, Dict, Tuple

import pytest
import requests

if "pyalex" not in sys.modules:
    pyalex_stub = types.ModuleType("pyalex")
    pyalex_stub.Topics = object
    pyalex_stub.Works = object
    config_stub = types.ModuleType("pyalex.config")
    config_stub.mailto = None
    pyalex_stub.config = config_stub
    sys.modules["pyalex"] = pyalex_stub
    sys.modules["pyalex.config"] = config_stub

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload.resolvers import DownloadOutcome


class FakeResponse:
    def __init__(self, status_code: int, headers=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = list(chunks or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def iter_content(self, chunk_size: int):
        for chunk in self._chunks:
            yield chunk

    def close(self):
        pass


def make_artifact(tmp_path: Path) -> downloader.WorkArtifact:
    artifact = downloader.WorkArtifact(
        work_id="W-DOWNLOAD",
        title="Outcome Example",
        publication_year=2024,
        doi="10.1000/example",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="outcome-example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
    artifact.html_dir.mkdir(parents=True, exist_ok=True)
    return artifact


def stub_requests(monkeypatch, mapping: Dict[Tuple[str, str], Callable[[], FakeResponse] | FakeResponse]):
    def fake_request(session, method, url, **kwargs):
        key = (method.upper(), url)
        if key not in mapping:
            raise AssertionError(f"Unexpected request {key}")
        response = mapping[key]
        return response() if callable(response) else response

    monkeypatch.setattr(downloader, "request_with_retries", fake_request)


def test_successful_pdf_download_populates_metadata(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"
    pdf_bytes = b"%PDF-1.4\n1 0 obj<<>>\nendobj\n%%EOF"
    expected_sha = hashlib.sha256(pdf_bytes).hexdigest()

    mapping = {
        ("HEAD", url): FakeResponse(200, headers={"Content-Type": "application/pdf"}),
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/pdf",
                "ETag": '"etag-123"',
                "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            },
            chunks=[pdf_bytes],
        ),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(session, artifact, url, None, timeout=10.0)

    assert outcome.classification == "pdf"
    assert outcome.path is not None
    stored = Path(outcome.path)
    assert stored.exists()
    assert outcome.sha256 == expected_sha
    assert outcome.content_length == stored.stat().st_size
    assert outcome.etag == '"etag-123"'
    assert outcome.last_modified == "Mon, 01 Jan 2024 00:00:00 GMT"
    assert outcome.error is None
    assert outcome.extracted_text_path is None


def test_cached_response_preserves_prior_metadata(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"
    cached_path = str(artifact.pdf_dir / "cached.pdf")
    context = {
        "previous": {
            url: {
                "path": cached_path,
                "sha256": "cached-sha",
                "content_length": 1024,
                "etag": '"etag-cached"',
                "last_modified": "Tue, 02 Jan 2024 00:00:00 GMT",
            }
        }
    }

    mapping = {
        ("HEAD", url): FakeResponse(200, headers={"Content-Type": "application/pdf"}),
        ("GET", url): lambda: FakeResponse(
            304,
            headers={"Content-Type": "application/pdf"},
        ),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(session, artifact, url, None, timeout=10.0, context=context)

    assert outcome.classification == "cached"
    assert outcome.path == cached_path
    assert outcome.sha256 == "cached-sha"
    assert outcome.content_length == 1024
    assert outcome.etag == '"etag-cached"'
    assert outcome.last_modified == "Tue, 02 Jan 2024 00:00:00 GMT"
    assert outcome.error is None


def test_http_error_sets_metadata_to_none(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"

    mapping = {
        ("HEAD", url): FakeResponse(200, headers={"Content-Type": "application/pdf"}),
        ("GET", url): lambda: FakeResponse(404, headers={"Content-Type": "text/html"}),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(session, artifact, url, None, timeout=10.0)

    assert outcome.classification == "http_error"
    assert outcome.path is None
    assert outcome.sha256 is None
    assert outcome.content_length is None
    assert outcome.etag is None
    assert outcome.last_modified is None
    assert outcome.error is None


def test_html_download_with_text_extraction(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/page.html"
    html_bytes = b"<!DOCTYPE html><html><body><p>Hello</p></body></html>"

    html_extractor = types.SimpleNamespace(extract=lambda text: "Hello")
    monkeypatch.setitem(sys.modules, "trafilatura", html_extractor)

    mapping = {
        ("HEAD", url): FakeResponse(200, headers={"Content-Type": "text/html"}),
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "text/html",
                "ETag": '"etag-html"',
                "Last-Modified": "Wed, 03 Jan 2024 00:00:00 GMT",
            },
            chunks=[html_bytes],
        ),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"extract_html_text": True},
    )

    assert outcome.classification == "html"
    assert outcome.path is not None and outcome.path.endswith(".html")
    assert outcome.extracted_text_path is not None
    extracted = Path(outcome.extracted_text_path)
    assert extracted.exists()
    assert extracted.read_text(encoding="utf-8") == "Hello"
    assert outcome.sha256 is not None
    assert outcome.etag == '"etag-html"'
    assert outcome.last_modified == "Wed, 03 Jan 2024 00:00:00 GMT"


def test_dry_run_preserves_metadata_without_files(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"
    pdf_bytes = b"%PDF-1.4\n1 0 obj<<>>\nendobj\n%%EOF"

    mapping = {
        ("HEAD", url): FakeResponse(200, headers={"Content-Type": "application/pdf"}),
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/pdf",
                "ETag": '"etag-dry"',
                "Last-Modified": "Thu, 04 Jan 2024 00:00:00 GMT",
            },
            chunks=[pdf_bytes],
        ),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"dry_run": True},
    )

    assert outcome.classification == "pdf"
    assert outcome.path is None
    assert outcome.sha256 is None
    assert outcome.content_length is None
    assert outcome.extracted_text_path is None
    assert outcome.etag == '"etag-dry"'
    assert outcome.last_modified == "Thu, 04 Jan 2024 00:00:00 GMT"


def test_build_manifest_entry_includes_download_metadata(tmp_path):
    artifact = make_artifact(tmp_path)
    download_path = str(artifact.pdf_dir / "saved.pdf")
    outcome = DownloadOutcome(
        classification="pdf",
        path=download_path,
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=150.0,
        error=None,
        sha256="abc123",
        content_length=4096,
        etag='"etag-manifest"',
        last_modified="Fri, 05 Jan 2024 00:00:00 GMT",
        extracted_text_path=str(artifact.html_dir / "saved.txt"),
    )

    entry = downloader.build_manifest_entry(
        artifact,
        "figshare",
        "https://example.org/paper.pdf",
        outcome,
        html_paths=["/tmp/example.html"],
        dry_run=False,
    )

    assert entry.sha256 == "abc123"
    assert entry.content_length == 4096
    assert entry.etag == '"etag-manifest"'
    assert entry.last_modified == "Fri, 05 Jan 2024 00:00:00 GMT"
    assert entry.extracted_text_path == str(artifact.html_dir / "saved.txt")
