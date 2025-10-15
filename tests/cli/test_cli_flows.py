"""CLI, dry-run, resume, and environment validation for OpenAlex downloads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVRC = REPO_ROOT / ".envrc"
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_env.sh"
README = REPO_ROOT / "README.md"
DOCS_SETUP = REPO_ROOT / "docs" / "02-setup" / "index.md"
AGENTS = REPO_ROOT / "openspec" / "AGENTS.md"


@pytest.fixture
def download_modules():
    """Provide downloader/resolver modules guarded by optional dependencies."""

    pytest.importorskip("pyalex")
    requests = pytest.importorskip("requests")
    from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
    from DocsToKG.ContentDownload import resolvers

    return SimpleNamespace(downloader=downloader, resolvers=resolvers, requests=requests)


def test_read_resolver_config_yaml_requires_pyyaml(download_modules, monkeypatch, tmp_path):
    downloader = download_modules.downloader
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


def test_load_resolver_config_applies_mailto(download_modules):
    downloader = download_modules.downloader
    args = SimpleNamespace(
        resolver_config=None,
        unpaywall_email=None,
        core_api_key=None,
        semantic_scholar_api_key=None,
        doaj_api_key=None,
        max_resolver_attempts=None,
        resolver_timeout=None,
        disable_resolver=[],
        mailto="team@example.org",
        resolver_order=None,
        concurrent_resolvers=None,
        head_precheck=None,
        accept=None,
        enable_resolver=[],
        global_url_dedup=None,
        domain_min_interval=[],
    )
    config = downloader.load_resolver_config(args, ["alpha", "beta", "gamma"], ["beta"])
    assert config.mailto == "team@example.org"
    assert "mailto:team@example.org" in config.polite_headers["User-Agent"]
    assert config.resolver_order == ["beta", "alpha", "gamma"]


def test_main_writes_manifest_and_sets_mailto(download_modules, monkeypatch, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers

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
    monkeypatch.setattr("sys.argv", argv)

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


def test_main_requires_topic_or_topic_id(download_modules, monkeypatch):
    downloader = download_modules.downloader
    monkeypatch.setattr(
        "sys.argv", ["download_pyalex_pdfs.py", "--year-start", "2020", "--year-end", "2020"]
    )
    with pytest.raises(SystemExit):
        downloader.main()


def test_cli_flag_propagation_and_metrics_export(download_modules, monkeypatch, tmp_path):
    downloader = download_modules.downloader

    manifest_path = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    monkeypatch.setattr(downloader, "iterate_openalex", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(downloader, "default_resolvers", lambda: [])
    monkeypatch.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)

    captured: Dict[str, object] = {}

    class _StubPipeline:
        def __init__(self, *, config=None, logger=None, metrics=None, **kwargs):
            captured["config"] = config
            self.logger = logger
            self.metrics = metrics

        def run(self, *args, **kwargs):  # pragma: no cover - no work processed
            return download_modules.resolvers.PipelineResult(
                success=False,
                resolver_name="stub",
                url=None,
                outcome=None,
                html_paths=[],
                failed_urls=[],
            )

    monkeypatch.setattr(downloader, "ResolverPipeline", _StubPipeline)

    monkeypatch.setattr(
        "sys.argv",
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


def test_download_candidate_dry_run_does_not_create_files(download_modules, tmp_path):
    downloader = download_modules.downloader
    requests = download_modules.requests
    responses = pytest.importorskip("responses")

    artifact = downloader.WorkArtifact(
        work_id="W1",
        title="Dry Run Example",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.org/paper.pdf"],
        open_access_url=None,
        source_display_names=[],
        base_stem="dry-run-example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )

    session = requests.Session()
    url = artifact.pdf_urls[0]
    call_methods: list[str] = []
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mocked:
        mocked.add(responses.HEAD, url, status=200, headers={"Content-Type": "application/pdf"})
        mocked.add(responses.GET, url, status=200, body=b"%PDF-1.4\n%%EOF\n")

        context = {"dry_run": True, "extract_html_text": False, "previous": {}}
        outcome = downloader.download_candidate(
            session, artifact, url, None, timeout=30.0, context=context
        )
        call_methods = [call.request.method for call in mocked.calls]

    assert outcome.classification == "pdf"
    assert outcome.path is None
    assert list((artifact.pdf_dir).glob("*.pdf")) == []
    assert call_methods.count("GET") == 1


def test_process_one_work_logs_manifest_in_dry_run(download_modules, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers
    requests = download_modules.requests

    artifact = downloader.WorkArtifact(
        work_id="W1",
        title="Dry Run Example",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.org/paper.pdf"],
        open_access_url=None,
        source_display_names=[],
        base_stem="dry-run-example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    work = {
        "id": "https://openalex.org/W1",
        "title": artifact.title,
        "publication_year": artifact.publication_year,
        "ids": {},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }

    session = requests.Session()
    logger_path = tmp_path / "attempts.jsonl"
    logger = downloader.JsonlLogger(logger_path)
    metrics = resolvers.ResolverMetrics()

    class _StubPipeline:
        def run(self, session, artifact, context=None):  # pragma: no cover - interface shim
            return resolvers.PipelineResult(
                success=True,
                resolver_name="stub",
                url="https://example.org/paper.pdf",
                outcome=resolvers.DownloadOutcome(
                    classification="pdf",
                    path=None,
                    http_status=200,
                    content_type="application/pdf",
                    elapsed_ms=12.0,
                ),
                html_paths=[],
            )

    result = downloader.process_one_work(
        work,
        session,
        artifact.pdf_dir,
        artifact.html_dir,
        pipeline=_StubPipeline(),
        logger=logger,
        metrics=metrics,
        dry_run=True,
        extract_html_text=False,
        previous_lookup={},
        resume_completed=set(),
    )

    logger.close()

    assert result["saved"] is True
    contents = [
        json.loads(line) for line in logger_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    manifest_records = [entry for entry in contents if entry["record_type"] == "manifest"]
    assert manifest_records, "Expected at least one manifest record"
    assert all(record["dry_run"] is True for record in manifest_records)


class _NoopPipeline:
    def run(self, session, artifact, context=None):  # pragma: no cover - should not run
        raise AssertionError("Pipeline should not execute when resume skips work")


def test_resume_skips_completed_work(download_modules, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers
    requests = download_modules.requests

    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    session = requests.Session()
    logger_path = tmp_path / "attempts.jsonl"
    logger = downloader.JsonlLogger(logger_path)
    metrics = resolvers.ResolverMetrics()

    work = {
        "id": "https://openalex.org/W-RESUME",
        "title": "Resume Example",
        "publication_year": 2020,
        "ids": {},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }

    result = downloader.process_one_work(
        work,
        session,
        pdf_dir,
        html_dir,
        pipeline=_NoopPipeline(),
        logger=logger,
        metrics=metrics,
        dry_run=False,
        extract_html_text=False,
        previous_lookup={},
        resume_completed={"W-RESUME"},
    )

    logger.close()

    assert result["skipped"] is True
    entries = [
        json.loads(line) for line in logger_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    manifest_entries = [entry for entry in entries if entry["record_type"] == "manifest"]
    assert len(manifest_entries) == 1
    manifest = manifest_entries[0]
    assert manifest["work_id"] == "W-RESUME"
    assert manifest["classification"] == "skipped"
    assert manifest["dry_run"] is False


def test_envrc_configures_virtualenv_and_pythonpath() -> None:
    text = ENVRC.read_text(encoding="utf-8")
    assert 'export VIRTUAL_ENV="$VENVP"' in text
    assert 'PATH_add "$VIRTUAL_ENV/bin"' in text
    assert 'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"' in text


def test_bootstrap_script_installs_project() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'python -m venv "$VENV_PATH"' in text
    assert '"$VENV_PATH/bin/pip" install -e .' in text
    assert os.access(BOOTSTRAP, os.X_OK), "bootstrap script must be executable"


def test_documentation_mentions_bootstrap_and_direnv() -> None:
    readme = README.read_text(encoding="utf-8")
    docs_setup = DOCS_SETUP.read_text(encoding="utf-8")
    agents = AGENTS.read_text(encoding="utf-8")

    for content in (readme, docs_setup):
        assert "./scripts/bootstrap_env.sh" in content
        assert "direnv allow" in content

    assert "## Environment Activation" in agents
    assert "direnv exec . python" in agents
