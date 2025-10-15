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
            "concurrent_resolvers": None,
            "head_precheck": None,
            "accept": None,
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
    monkeypatch.setattr(downloader, "default_resolvers", lambda: [])

    class StubPipeline:
        def __init__(
            self, resolvers=None, config=None, download_func=None, logger=None, metrics=None, **_
        ):
            self.logger = logger
            self.metrics = metrics

        def run(self, session, artifact, context=None):
            self.logger.log(
                resolvers.AttemptRecord(
                    work_id=artifact.work_id,
                    resolver_name="openalex",
                    resolver_order=1,
                    url="https://oa.example/direct.pdf",
                    status=outcome.classification,
                    http_status=outcome.http_status,
                    content_type=outcome.content_type,
                    elapsed_ms=outcome.elapsed_ms,
                    reason=outcome.error,
                    sha256=outcome.sha256,
                    content_length=outcome.content_length,
                    dry_run=False,
                )
            )
            self.metrics.record_attempt("openalex", outcome)
            return resolvers.PipelineResult(
                success=True,
                resolver_name="openalex",
                url="https://oa.example/direct.pdf",
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    monkeypatch.setattr(downloader, "ResolverPipeline", StubPipeline)

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
    entries = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    attempts = [entry for entry in entries if entry.get("record_type") == "attempt"]
    manifests = [entry for entry in entries if entry.get("record_type") == "manifest"]

    assert len(attempts) == 1
    assert attempts[0]["status"] == "pdf"
    assert attempts[0]["resolver_name"] == "openalex"

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest["resolver"] == "openalex"
    assert manifest["url"] == "https://oa.example/direct.pdf"
    assert manifest["classification"] == "pdf"
    assert downloader.oa_config.email == "team@example.org"
    assert calls == ["machine learning"]


def test_main_requires_topic_or_topic_id(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["download_pyalex_pdfs.py", "--year-start", "2020", "--year-end", "2020"]
    )
    with pytest.raises(SystemExit):
        downloader.main()


def test_cli_flag_propagation_and_metrics_export(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    monkeypatch.setattr(downloader, "iterate_openalex", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(downloader, "default_resolvers", lambda: [])
    monkeypatch.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)

    captured = {}

    class _StubPipeline:
        def __init__(self, *, config=None, logger=None, metrics=None, **kwargs):
            captured["config"] = config
            self.logger = logger
            self.metrics = metrics

        def run(self, *args, **kwargs):  # pragma: no cover - no work processed
            return resolvers.PipelineResult(
                success=False,
                resolver_name="stub",
                url=None,
                outcome=None,
                html_paths=[],
                failed_urls=[],
            )

    monkeypatch.setattr(downloader, "ResolverPipeline", _StubPipeline)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_pyalex_pdfs.py",
            "--topic",
            "graphs",
            "--year-start",
            "2020",
            "--year-end",
            "2020",
            "--out",
            str(out_dir),
            "--manifest",
            str(manifest_path),
            "--concurrent-resolvers",
            "4",
            "--no-head-precheck",
            "--accept",
            "application/pdf",
        ],
    )

    downloader.main()

    config = captured["config"]
    assert config.max_concurrent_resolvers == 4
    assert config.enable_head_precheck is False
    assert config.polite_headers.get("Accept") == "application/pdf"

    metrics_path = manifest_path.with_suffix(".metrics.json")
    assert metrics_path.exists()
    metrics_doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics_doc) == {"processed", "saved", "html_only", "skipped", "resolvers"}
    assert metrics_doc["processed"] == 0
    assert metrics_doc["resolvers"] == {
        "attempts": {},
        "successes": {},
        "html": {},
        "skips": {},
        "failures": {},
    }
