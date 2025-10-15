"""
Content Download CLI Tests

This module validates the OpenAlex PDF download command line entry point,
covering configuration loading, mailto propagation, and manifest output
generation for downstream analytics.

Key Scenarios:
- Enforces optional PyYAML dependency for YAML configuration files
- Confirms CLI arguments map to resolver configuration correctly
- Verifies manifest output and topic resolution side effects

Dependencies:
- pytest: Fixture and assertion framework
- DocsToKG.ContentDownload.download_pyalex_pdfs: CLI implementation

Usage:
    pytest tests/test_cli.py
"""

from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("pyalex")

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload import resolvers


def test_read_resolver_config_yaml_requires_pyyaml(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("resolver_order: ['a']", encoding="utf-8")

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError):
        downloader.read_resolver_config(config_path)


def test_load_resolver_config_applies_mailto():
    args = type(
        "Args",
        (),
        {
            "resolver_config": None,
            "unpaywall_email": None,
            "core_api_key": None,
            "semantic_scholar_api_key": None,
            "doaj_api_key": None,
            "max_resolver_attempts": None,
            "resolver_timeout": None,
            "disable_resolver": [],
            "mailto": "team@example.org",
            "resolver_order": None,
        },
    )()
    config = downloader.load_resolver_config(args, ["alpha", "beta", "gamma"], ["beta"])
    assert config.mailto == "team@example.org"
    assert "mailto:team@example.org" in config.polite_headers["User-Agent"]
    assert config.resolver_order == ["beta", "alpha", "gamma"]


def test_main_writes_manifest_and_sets_mailto(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    work = {
        "id": "https://openalex.org/W1",
        "title": "Sample Work",
        "publication_year": 2020,
        "ids": {"doi": "10.1000/example"},
        "open_access": {"oa_url": None},
        "best_oa_location": {"pdf_url": "https://oa.example/direct.pdf"},
        "primary_location": {},
        "locations": [],
    }

    def fake_iterate(query, per_page, max_results):
        yield work

    outcome = resolvers.DownloadOutcome(
        classification="pdf",
        path=str(pdf_dir / "out.pdf"),
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=12.0,
        error=None,
    )
    (pdf_dir / "out.pdf").write_bytes(b"%PDF-1.4\n...%%EOF")

    monkeypatch.setattr(downloader, "iterate_openalex", fake_iterate)
    calls = []

    def fake_resolve(topic):
        calls.append(topic)
        return "https://openalex.org/T1"

    monkeypatch.setattr(downloader, "resolve_topic_id_if_needed", fake_resolve)
    monkeypatch.setattr(
        downloader,
        "attempt_openalex_candidates",
        lambda *args, **kwargs: (outcome, "https://oa.example/direct.pdf"),
    )

    argv = [
        "download_pyalex_pdfs.py",
        "--topic",
        "machine learning",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(pdf_dir),
        "--manifest",
        str(manifest_path),
        "--mailto",
        "team@example.org",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    downloader.main()
    content = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    record = json.loads(content[0])
    assert record["resolver"] == "openalex"
    assert record["url"] == "https://oa.example/direct.pdf"
    assert record["classification"] == "pdf"
    assert downloader.oa_config.email == "team@example.org"
    assert calls == ["machine learning"]


def test_main_requires_topic_or_topic_id(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["download_pyalex_pdfs.py", "--year-start", "2020", "--year-end", "2020"]
    )
    with pytest.raises(SystemExit):
        downloader.main()
