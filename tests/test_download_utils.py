"""
Content Download Utility Tests

This module covers the helper primitives used by the OpenAlex downloader
for slug generation, payload classification, and identifier normalization
to guarantee consistent artifact naming and metadata hygiene.

Key Scenarios:
- Normalizes DOIs, PMIDs, PMCIDs, and arXiv identifiers
- Classifies payload content types across PDF and HTML responses
- Deduplicates candidate URLs while preserving provenance metadata

Dependencies:
- pytest: Parametrization and assertions
- DocsToKG.ContentDownload: Utility functions under test

Usage:
    pytest tests/test_download_utils.py
"""

import pytest

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload.utils import normalize_doi, normalize_pmcid


def test_slugify_truncates_and_normalises():
    assert downloader.slugify("Hello, World!", keep=8) == "Hello_Wo"
    assert downloader.slugify("   ", keep=10) == "untitled"
    assert downloader.slugify("Study: B-cells & growth", keep=40) == "Study_Bcells_growth"


@pytest.mark.parametrize(
    "payload,ctype,url,expected",
    [
        (b"%PDF-sample", "application/pdf", "https://example.org/file.pdf", "pdf"),
        (b"   %PDF-1.4", "text/plain", "https://example.org/file.bin", "pdf"),
        (b"<html><head></head>", "text/html", "https://example.org", "html"),
        (b"", "application/pdf", "https://example.org/file.pdf", "pdf"),
        (b"", "text/plain", "https://example.org/foo.pdf", "pdf"),
    ],
)
def test_classify_payload_variants(payload, ctype, url, expected):
    assert downloader.classify_payload(payload, ctype, url) == expected


def test_collect_location_urls_dedupes_and_tracks_sources():
    work = {
        "best_oa_location": {
            "landing_page_url": "https://host.example/landing",
            "pdf_url": "https://host.example/paper.pdf",
            "source": {"display_name": "Host"},
        },
        "primary_location": {
            "landing_page_url": "https://host.example/landing",
            "pdf_url": "https://cdn.example/paper.pdf",
            "source": {"display_name": "Mirror"},
        },
        "locations": [
            {
                "landing_page_url": "https://mirror.example/landing",
                "pdf_url": "https://cdn.example/paper.pdf",
                "source": {"display_name": "Mirror"},
            }
        ],
        "open_access": {"oa_url": "https://oa.example/paper.pdf"},
    }
    collected = downloader._collect_location_urls(work)
    assert collected["landing"] == [
        "https://host.example/landing",
        "https://mirror.example/landing",
    ]
    assert collected["pdf"] == [
        "https://host.example/paper.pdf",
        "https://cdn.example/paper.pdf",
        "https://oa.example/paper.pdf",
    ]
    assert collected["sources"] == ["Host", "Mirror"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://doi.org/10.1000/foo", "10.1000/foo"),
        (" 10.1000/bar ", "10.1000/bar"),
        (None, None),
    ],
)
def test_normalize_doi(value, expected):
    assert normalize_doi(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("PMID:123456", "123456"),
        ("https://pubmed.ncbi.nlm.nih.gov/98765/", "98765"),
        (None, None),
    ],
)
def test_normalize_pmid(value, expected):
    assert downloader._normalize_pmid(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("PMC12345", "PMC12345"),
        ("pmc9876", "PMC9876"),
        ("9876", "PMC9876"),
        (None, None),
    ],
)
def test_normalize_pmcid(value, expected):
    assert normalize_pmcid(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("arXiv:2101.12345", "2101.12345"),
        ("https://arxiv.org/abs/2010.00001", "2010.00001"),
        ("2101.99999", "2101.99999"),
    ],
)
def test_normalize_arxiv(value, expected):
    assert downloader._normalize_arxiv(value) == expected
