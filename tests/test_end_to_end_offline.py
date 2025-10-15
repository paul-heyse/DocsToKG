"""
Offline Resolver Pipeline Tests

This module validates the resolver pipeline in an offline, fully mocked
environment to ensure PDFs are downloaded, verified, and classified
without reaching external services.

Key Scenarios:
- Executes an end-to-end pipeline run using Unpaywall metadata responses
- Detects corrupt PDF payloads lacking EOF markers during download

Dependencies:
- pytest/responses: HTTP mocking for Unpaywall and content servers
- DocsToKG.ContentDownload: Resolver pipeline and artifact helpers

Usage:
    pytest tests/test_end_to_end_offline.py
"""

from pathlib import Path

import pytest
import requests

pytest.importorskip("pyalex")

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload import resolvers

responses = pytest.importorskip("responses")


class MemoryLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, record):
        self.records.append(record)


@responses.activate
def test_resolver_pipeline_downloads_pdf_end_to_end(tmp_path):
    work = {
        "id": "https://openalex.org/W123",
        "title": "Resolver Demo",
        "publication_year": 2021,
        "ids": {"doi": "10.1000/resolver-demo"},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }
    artifact = downloader.create_artifact(
        work, pdf_dir=tmp_path / "pdf", html_dir=tmp_path / "html"
    )

    pdf_url = "https://cdn.example/resolver-demo.pdf"
    responses.add(
        responses.GET,
        "https://api.unpaywall.org/v2/10.1000/resolver-demo",
        json={"best_oa_location": {"url_for_pdf": pdf_url}},
        status=200,
    )
    responses.add(
        responses.HEAD,
        pdf_url,
        headers={"Content-Type": "application/pdf"},
        status=200,
    )
    responses.add(
        responses.GET,
        pdf_url,
        body=b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n%%EOF",
        headers={"Content-Type": "application/pdf"},
        status=200,
    )

    config = resolvers.ResolverConfig(
        resolver_order=["unpaywall"],
        resolver_toggles={"unpaywall": True},
        unpaywall_email="team@example.org",
    )
    logger = MemoryLogger()
    session = requests.Session()
    session.headers.update({"User-Agent": "pytest"})
    pipeline = resolvers.ResolverPipeline(
        resolvers=[resolvers.UnpaywallResolver()],
        config=config,
        download_func=downloader.download_candidate,
        logger=logger,
    )
    result = pipeline.run(session, artifact)
    assert result.success is True
    assert result.outcome and result.outcome.path
    pdf_path = Path(result.outcome.path)
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")


@responses.activate
def test_download_candidate_marks_corrupt_without_eof(tmp_path):
    work = {
        "id": "https://openalex.org/W456",
        "title": "Corrupt PDF",
        "publication_year": 2022,
        "ids": {"doi": "10.1000/corrupt"},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }
    artifact = downloader.create_artifact(
        work, pdf_dir=tmp_path / "pdf", html_dir=tmp_path / "html"
    )
    pdf_url = "https://cdn.example/corrupt.pdf"
    responses.add(
        responses.GET,
        pdf_url,
        body=b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n",
        headers={"Content-Type": "application/pdf"},
        status=200,
    )
    responses.add(responses.HEAD, pdf_url, headers={"Content-Type": "application/pdf"}, status=200)
    session = requests.Session()
    outcome = downloader.download_candidate(session, artifact, pdf_url, referer=None, timeout=5.0)
    assert outcome.classification == "pdf_corrupt"
    assert outcome.path is None
