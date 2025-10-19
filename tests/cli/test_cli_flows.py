# === NAVMAP v1 ===
# {
#   "module": "tests.cli.test_cli_flows",
#   "purpose": "Pytest coverage for cli cli flows scenarios",
#   "sections": [
#     {
#       "id": "download-modules",
#       "name": "download_modules",
#       "anchor": "function-download-modules",
#       "kind": "function"
#     },
#     {
#       "id": "test-read-resolver-config-yaml-requires-pyyaml",
#       "name": "test_read_resolver_config_yaml_requires_pyyaml",
#       "anchor": "function-test-read-resolver-config-yaml-requires-pyyaml",
#       "kind": "function"
#     },
#     {
#       "id": "test-load-resolver-config-applies-mailto",
#       "name": "test_load_resolver_config_applies_mailto",
#       "anchor": "function-test-load-resolver-config-applies-mailto",
#       "kind": "function"
#     },
#     {
#       "id": "test-main-writes-manifest-and-sets-mailto",
#       "name": "test_main_writes_manifest_and_sets_mailto",
#       "anchor": "function-test-main-writes-manifest-and-sets-mailto",
#       "kind": "function"
#     },
#     {
#       "id": "test-main-with-csv-writes-last-attempt-csv",
#       "name": "test_main_with_csv_writes_last_attempt_csv",
#       "anchor": "function-test-main-with-csv-writes-last-attempt-csv",
#       "kind": "function"
#     },
#     {
#       "id": "test-main-with-staging-creates-timestamped-directories",
#       "name": "test_main_with_staging_creates_timestamped_directories",
#       "anchor": "function-test-main-with-staging-creates-timestamped-directories",
#       "kind": "function"
#     },
#     {
#       "id": "test-main-dry-run-skips-writing-files",
#       "name": "test_main_dry_run_skips_writing_files",
#       "anchor": "function-test-main-dry-run-skips-writing-files",
#       "kind": "function"
#     },
#     {
#       "id": "test-main-requires-topic-or-topic-id",
#       "name": "test_main_requires_topic_or_topic_id",
#       "anchor": "function-test-main-requires-topic-or-topic-id",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-flag-propagation-and-metrics-export",
#       "name": "test_cli_flag_propagation_and_metrics_export",
#       "anchor": "function-test-cli-flag-propagation-and-metrics-export",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-candidate-dry-run-does-not-create-files",
#       "name": "test_download_candidate_dry_run_does_not_create_files",
#       "anchor": "function-test-download-candidate-dry-run-does-not-create-files",
#       "kind": "function"
#     },
#     {
#       "id": "test-process-one-work-logs-manifest-in-dry-run",
#       "name": "test_process_one_work_logs_manifest_in_dry_run",
#       "anchor": "function-test-process-one-work-logs-manifest-in-dry-run",
#       "kind": "function"
#     },
#     {
#       "id": "nooppipeline",
#       "name": "_NoopPipeline",
#       "anchor": "class-nooppipeline",
#       "kind": "class"
#     },
#     {
#       "id": "test-resume-skips-completed-work",
#       "name": "test_resume_skips_completed_work",
#       "anchor": "function-test-resume-skips-completed-work",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-resume-from-partial-metadata",
#       "name": "test_cli_resume_from_partial_metadata",
#       "anchor": "function-test-cli-resume-from-partial-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-workers-apply-domain-jitter",
#       "name": "test_cli_workers_apply_domain_jitter",
#       "anchor": "function-test-cli-workers-apply-domain-jitter",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-head-precheck-handles-head-hostile",
#       "name": "test_cli_head_precheck_handles_head_hostile",
#       "anchor": "function-test-cli-head-precheck-handles-head-hostile",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-attempt-records-cover-all-resolvers",
#       "name": "test_cli_attempt_records_cover_all_resolvers",
#       "anchor": "function-test-cli-attempt-records-cover-all-resolvers",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-dry-run-metrics-align",
#       "name": "test_cli_dry_run_metrics_align",
#       "anchor": "function-test-cli-dry-run-metrics-align",
#       "kind": "function"
#     },
#     {
#       "id": "test-envrc-configures-virtualenv-and-pythonpath",
#       "name": "test_envrc_configures_virtualenv_and_pythonpath",
#       "anchor": "function-test-envrc-configures-virtualenv-and-pythonpath",
#       "kind": "function"
#     },
#     {
#       "id": "test-bootstrap-script-installs-project",
#       "name": "test_bootstrap_script_installs_project",
#       "anchor": "function-test-bootstrap-script-installs-project",
#       "kind": "function"
#     },
#     {
#       "id": "test-documentation-mentions-bootstrap-and-direnv",
#       "name": "test_documentation_mentions_bootstrap_and_direnv",
#       "anchor": "function-test-documentation-mentions-bootstrap-and-direnv",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""CLI, dry-run, resume, and environment validation for OpenAlex downloads."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from DocsToKG.ContentDownload import cli as downloader
from DocsToKG.ContentDownload import pipeline as resolvers
from DocsToKG.ContentDownload.core import Classification, DownloadContext
from DocsToKG.ContentDownload.download import DownloadConfig
from DocsToKG.ContentDownload.telemetry import MANIFEST_SCHEMA_VERSION, build_manifest_entry
from tools.manifest_to_index import convert_manifest_to_index

print("[MODULE IMPORT] tests.cli.test_cli_flows loaded", flush=True)

# --- Globals ---

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVRC = REPO_ROOT / ".envrc"
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_env.sh"
README = REPO_ROOT / "README.md"
DOCS_SETUP = REPO_ROOT / "docs" / "02-setup" / "index.md"
AGENTS = REPO_ROOT / "openspec" / "AGENTS.md"


@pytest.fixture(autouse=True)
def _print_test_boundary(request):
    """Emit markers so hangs surface in CI output."""

    test_name = request.node.name
    print(f"[TEST BEGIN] {test_name}", flush=True)
    yield
    print(f"[TEST END] {test_name}", flush=True)


@pytest.fixture
# --- Test Fixtures ---


def download_modules():
    """Provide downloader/resolver modules guarded by optional dependencies."""

    print("[FIXTURE BEGIN] download_modules", flush=True)
    print("  importing pyalex", flush=True)
    pytest.importorskip("pyalex")
    print("  importing requests", flush=True)
    requests = pytest.importorskip("requests")
    print("  fixture ready", flush=True)
    print("[FIXTURE END] download_modules", flush=True)
    return SimpleNamespace(downloader=downloader, resolvers=resolvers, requests=requests)


# --- Test Cases ---


def test_read_resolver_config_yaml_requires_pyyaml(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    config_path = tmp_path / "config.yaml"
    config_path.write_text("resolver_order: ['a']", encoding="utf-8")

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return original_import(name, *args, **kwargs)

    patcher.setattr("builtins.__import__", fake_import)
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


def test_main_writes_manifest_and_sets_mailto(download_modules, patcher, tmp_path):
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

    patcher.setattr(downloader, "iterate_openalex", fake_iterate)
    patcher.setattr("DocsToKG.ContentDownload.runner.iterate_openalex", fake_iterate)
    calls = []

    def fake_resolve(topic):
        calls.append(topic)
        return "https://openalex.org/T1"

    patcher.setattr(downloader, "resolve_topic_id_if_needed", fake_resolve)
    patcher.setattr("DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", fake_resolve)
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])

    class StubPipeline:
        def __init__(
            self,
            resolvers=None,
            config=None,
            download_func=None,
            logger=None,
            metrics=None,
            **kwargs,
        ):
            self.logger = logger
            self.metrics = metrics
            self.run_id = kwargs.get("run_id")

        def run(self, session, artifact, context=None, session_factory=None):
            self.logger.log_attempt(
                resolvers.AttemptRecord(
                    run_id=self.run_id,
                    work_id=artifact.work_id,
                    resolver_name="openalex",
                    resolver_order=1,
                    url="https://oa.example/direct.pdf",
                    status=outcome.classification.value,
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

    patcher.setattr(downloader, "ResolverPipeline", StubPipeline)
    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", StubPipeline)

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
    patcher.setattr("sys.argv", value=argv)

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
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert downloader.oa_config.email == "team@example.org"
    assert calls == ["machine learning"]

    index_path = manifest_path.with_suffix(".index.json")
    assert index_path.exists()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert manifest["work_id"] in index_payload
    assert index_payload[manifest["work_id"]]["classification"] == "pdf"
    assert index_payload[manifest["work_id"]]["pdf_path"].endswith("out.pdf")


def _run_cli_with_csv_logging(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers

    out_dir = tmp_path / "out"
    manifest_path = out_dir / "manifest.jsonl"
    csv_path = tmp_path / "attempts.csv"
    out_dir.mkdir()

    work = {
        "id": "https://openalex.org/WCSV",
        "title": "CSV Work",
        "publication_year": 2021,
        "ids": {"doi": "10.1000/csv"},
        "open_access": {"oa_url": None},
        "best_oa_location": {"pdf_url": "https://oa.example/direct.pdf"},
        "primary_location": {},
        "locations": [],
    }

    patcher.setattr(downloader, "iterate_openalex", lambda *args, **kwargs: iter([work]))
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.iterate_openalex", lambda *args, **kwargs: iter([work])
    )
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])

    outcome = resolvers.DownloadOutcome(
        classification="pdf",
        path=str(out_dir / "pdfs" / "out.pdf"),
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=5.0,
        error=None,
    )
    (out_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (out_dir / "pdfs" / "out.pdf").write_bytes(b"%PDF-1.4\n%%EOF")

    class StubPipeline:
        def __init__(self, *_, logger=None, metrics=None, **kwargs):
            self.logger = logger
            self.metrics = metrics
            self.run_id = kwargs.get("run_id")
            self.run_id = kwargs.get("run_id")

        def run(self, session, artifact, context=None, session_factory=None):
            self.logger.log_attempt(
                resolvers.AttemptRecord(
                    run_id=self.run_id,
                    work_id=artifact.work_id,
                    resolver_name="openalex",
                    resolver_order=1,
                    url="https://oa.example/direct.pdf",
                    status=outcome.classification.value,
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

    patcher.setattr(downloader, "ResolverPipeline", StubPipeline)
    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", StubPipeline)

    argv = [
        "download_pyalex_pdfs.py",
        "--topic",
        "csv testing",
        "--year-start",
        "2021",
        "--year-end",
        "2021",
        "--out",
        str(out_dir),
        "--manifest",
        str(manifest_path),
        "--log-format",
        "csv",
        "--log-csv",
        str(csv_path),
    ]
    patcher.setattr("sys.argv", value=argv)

    downloader.main()

    return out_dir, manifest_path, csv_path


def test_main_with_csv_writes_last_attempt_csv(download_modules, patcher, tmp_path):
    out_dir, manifest_path, csv_path = _run_cli_with_csv_logging(
        download_modules, patcher, tmp_path
    )

    assert not manifest_path.exists()
    assert csv_path.exists()

    last_csv = manifest_path.with_suffix(".last.csv")
    assert last_csv.exists()

    with last_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    captured_row = rows[0]
    assert captured_row.pop("run_id")
    assert captured_row == {
        "work_id": "WCSV",
        "title": "CSV Work",
        "publication_year": "2021",
        "resolver": "openalex",
        "url": "https://oa.example/direct.pdf",
        "classification": "pdf",
        "path": str(out_dir / "pdfs" / "out.pdf"),
        "sha256": "",
        "content_length": "",
        "etag": "",
        "last_modified": "",
    }


def test_main_with_csv_creates_expected_non_json_outputs(download_modules, patcher, tmp_path):
    out_dir, manifest_path, csv_path = _run_cli_with_csv_logging(
        download_modules, patcher, tmp_path
    )

    jsonl_outputs = list(out_dir.glob("manifest.jsonl*"))
    assert jsonl_outputs == []

    sqlite_path = manifest_path.with_suffix(".sqlite")
    last_attempt_path = manifest_path.with_suffix(".last.csv")
    summary_path = manifest_path.with_suffix(".summary.json")
    index_path = manifest_path.with_suffix(".index.json")

    assert csv_path.exists()
    assert sqlite_path.exists()
    assert last_attempt_path.exists()
    assert summary_path.exists()
    assert index_path.exists()

def test_main_with_staging_creates_timestamped_directories(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers

    base_out = tmp_path / "runs"
    manifest_override = tmp_path / "should_not_be_used.jsonl"
    work = {
        "id": "https://openalex.org/WSTAGING",
        "title": "Staging Work",
        "publication_year": 2022,
        "ids": {"doi": "10.1000/staging"},
        "open_access": {"oa_url": None},
        "best_oa_location": {"pdf_url": "https://oa.example/staging.pdf"},
        "primary_location": {},
        "locations": [],
    }

    fixed_timestamp = datetime(2024, 5, 6, 7, 8, tzinfo=UTC)

    class _FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            return fixed_timestamp

    patcher.setattr("DocsToKG.ContentDownload.args.datetime", _FixedDateTime)
    patcher.setattr(downloader, "iterate_openalex", lambda *args, **kwargs: iter([work]))
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.iterate_openalex", lambda *args, **kwargs: iter([work])
    )
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])

    run_dir = base_out / fixed_timestamp.strftime("%Y%m%d_%H%M%S")
    pdf_path = run_dir / "PDF" / "out.pdf"

    outcome = resolvers.DownloadOutcome(
        classification="pdf",
        path=str(pdf_path),
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=7.0,
        error=None,
    )

    class StubPipeline:
        def __init__(self, *_, logger=None, metrics=None, **kwargs):
            self.logger = logger
            self.metrics = metrics
            self.run_id = kwargs.get("run_id")

        def run(self, session, artifact, context=None, session_factory=None):
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
            self.logger.log_attempt(
                resolvers.AttemptRecord(
                    run_id=self.run_id,
                    work_id=artifact.work_id,
                    resolver_name="openalex",
                    resolver_order=1,
                    url="https://oa.example/staging.pdf",
                    status=outcome.classification.value,
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
                url="https://oa.example/staging.pdf",
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    patcher.setattr(downloader, "ResolverPipeline", StubPipeline)

    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", StubPipeline)

    argv = [
        "download_pyalex_pdfs.py",
        "--topic-id",
        "T123",
        "--year-start",
        "2022",
        "--year-end",
        "2022",
        "--out",
        str(base_out),
        "--manifest",
        str(manifest_override),
        "--staging",
    ]
    patcher.setattr("sys.argv", value=argv)

    downloader.main()

    assert run_dir.is_dir()
    assert (run_dir / "PDF").is_dir()
    assert (run_dir / "HTML").is_dir()
    assert (run_dir / "XML").is_dir()

    manifest_path = run_dir / "manifest.jsonl"
    assert manifest_path.exists()
    assert not manifest_override.exists()

    index_path = manifest_path.with_suffix(".index.json")
    convert_manifest_to_index(manifest_path, index_path)
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["WSTAGING"]["pdf_path"].endswith("out.pdf")


def test_main_dry_run_skips_writing_files(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers

    out_dir = tmp_path / "out"
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir()

    works = [
        {
            "id": f"https://openalex.org/WDry{i}",
            "title": f"Dry Run Work {i}",
            "publication_year": 2023,
            "ids": {"doi": f"10.1000/dry{i}"},
            "open_access": {"oa_url": None},
            "best_oa_location": {"pdf_url": f"https://oa.example/dry{i}.pdf"},
            "primary_location": {},
            "locations": [],
        }
        for i in range(3)
    ]

    patcher.setattr(downloader, "iterate_openalex", lambda *args, **kwargs: iter(works))
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.iterate_openalex", lambda *args, **kwargs: iter(works)
    )
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])

    outcome = resolvers.DownloadOutcome(
        classification="pdf",
        path=None,
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=3.0,
        error=None,
    )

    class StubPipeline:
        def __init__(self, *_, logger=None, metrics=None, **kwargs):
            self.logger = logger
            self.metrics = metrics

        def run(self, session, artifact, context=None, session_factory=None):
            assert isinstance(context, DownloadContext)
            assert context.dry_run is True
            self.logger.log_attempt(
                resolvers.AttemptRecord(
                    work_id=artifact.work_id,
                    resolver_name="stub",
                    resolver_order=1,
                    url=f"https://oa.example/{artifact.work_id}.pdf",
                    status=outcome.classification.value,
                    http_status=outcome.http_status,
                    content_type=outcome.content_type,
                    elapsed_ms=outcome.elapsed_ms,
                    dry_run=True,
                )
            )
            self.metrics.record_attempt("stub", outcome)
            return resolvers.PipelineResult(
                success=True,
                resolver_name="stub",
                url=f"https://oa.example/{artifact.work_id}.pdf",
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    patcher.setattr(downloader, "ResolverPipeline", StubPipeline)

    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", StubPipeline)

    argv = [
        "download_pyalex_pdfs.py",
        "--topic",
        "dry run",
        "--year-start",
        "2023",
        "--year-end",
        "2023",
        "--out",
        str(out_dir),
        "--manifest",
        str(manifest_path),
        "--dry-run",
    ]
    patcher.setattr("sys.argv", value=argv)

    downloader.main()

    assert not any(out_dir.glob("*.pdf"))
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    manifest_rows = [entry for entry in entries if entry.get("record_type") == "manifest"]
    assert len(manifest_rows) == 3
    assert all(row["dry_run"] is True for row in manifest_rows)


def test_main_with_max_zero_skips_artifacts(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader

    out_dir = tmp_path / "max-zero"
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir()

    works = [
        {
            "id": f"https://openalex.org/WZero{i}",
            "title": f"Zero Work {i}",
            "publication_year": 2024,
        }
        for i in range(5)
    ]

    def fake_iterate(*_, **__):
        return iter(works)

    patcher.setattr(downloader, "iterate_openalex", fake_iterate)
    patcher.setattr("DocsToKG.ContentDownload.runner.iterate_openalex", fake_iterate)
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])

    call_count = 0

    def fake_process_one_work(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {}

    patcher.setattr("DocsToKG.ContentDownload.runner.process_one_work", fake_process_one_work)

    argv = [
        "download_pyalex_pdfs.py",
        "--topic",
        "max zero",
        "--year-start",
        "2024",
        "--year-end",
        "2024",
        "--out",
        str(out_dir),
        "--manifest",
        str(manifest_path),
        "--max",
        "0",
    ]
    patcher.setattr("sys.argv", value=argv)

    result = downloader.main()

    assert result.processed == 0
    assert result.saved == 0
    assert result.skipped == 0
    assert call_count == 0


def test_main_requires_topic_or_topic_id(download_modules, patcher):
    downloader = download_modules.downloader
    patcher.setattr(
        "sys.argv", value=["download_pyalex_pdfs.py", "--year-start", "2020", "--year-end", "2020"]
    )
    with pytest.raises(SystemExit):
        downloader.main()


def test_cli_flag_propagation_and_metrics_export(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader

    manifest_path = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    patcher.setattr(downloader, "iterate_openalex", lambda *args, **kwargs: iter(()))
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.iterate_openalex", lambda *args, **kwargs: iter(())
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])

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

    patcher.setattr(downloader, "ResolverPipeline", _StubPipeline)

    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", _StubPipeline)

    patcher.setattr(
        "sys.argv",
        value=[
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
    expected_keys = {"run_id", "processed", "saved", "html_only", "skipped", "resolvers"}
    assert expected_keys.issubset(metrics_doc.keys())
    assert metrics_doc["run_id"]
    assert metrics_doc["processed"] == 0
    resolver_summary = metrics_doc["resolvers"]
    expected_resolver_keys = {
        "attempts",
        "successes",
        "html",
        "xml",
        "skips",
        "failures",
        "latency_ms",
        "status_counts",
        "error_reasons",
        "classification_totals",
        "reason_totals",
    }
    assert expected_resolver_keys.issubset(resolver_summary.keys())
    for key in expected_resolver_keys:
        assert resolver_summary[key] == {}


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
        xml_dir=tmp_path / "xml",
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

    assert outcome.classification is Classification.PDF
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
        xml_dir=tmp_path / "xml",
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
    sink = downloader.JsonlSink(logger_path)
    logger = downloader.RunTelemetry(sink)
    metrics = resolvers.ResolverMetrics()

    class _StubPipeline:
        def run(
            self, session, artifact, context=None, session_factory=None
        ):  # pragma: no cover - interface shim
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

    options = DownloadConfig(
        dry_run=True,
        list_only=False,
        extract_html_text=False,
        run_id="test-run",
        previous_lookup={},
        resume_completed=set(),
        sniff_bytes=downloader.DEFAULT_SNIFF_BYTES,
        min_pdf_bytes=downloader.DEFAULT_MIN_PDF_BYTES,
        tail_check_bytes=downloader.DEFAULT_TAIL_CHECK_BYTES,
    )
    result = downloader.process_one_work(
        work,
        session,
        artifact.pdf_dir,
        artifact.html_dir,
        artifact.xml_dir,
        pipeline=_StubPipeline(),
        logger=logger,
        metrics=metrics,
        options=options,
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
    def run(
        self, session, artifact, context=None, session_factory=None
    ):  # pragma: no cover - should not run
        raise AssertionError("Pipeline should not execute when resume skips work")


def test_resume_skips_completed_work(download_modules, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers
    requests = download_modules.requests

    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    xml_dir = tmp_path / "xml"
    session = requests.Session()
    logger_path = tmp_path / "attempts.jsonl"
    sink = downloader.JsonlSink(logger_path)
    logger = downloader.RunTelemetry(sink)
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

    options = DownloadConfig(
        dry_run=False,
        list_only=False,
        extract_html_text=False,
        run_id="test-run",
        previous_lookup={},
        resume_completed={"W-RESUME"},
        sniff_bytes=downloader.DEFAULT_SNIFF_BYTES,
        min_pdf_bytes=downloader.DEFAULT_MIN_PDF_BYTES,
        tail_check_bytes=downloader.DEFAULT_TAIL_CHECK_BYTES,
    )
    result = downloader.process_one_work(
        work,
        session,
        pdf_dir,
        html_dir,
        xml_dir,
        pipeline=_NoopPipeline(),
        logger=logger,
        metrics=metrics,
        options=options,
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


def test_cli_resume_from_partial_metadata(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers

    previous_manifest = tmp_path / "resume_manifest.jsonl"
    previous_entry = {
        "record_type": "manifest",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "timestamp": "2024-05-01T00:00:00Z",
        "work_id": "WPARTIAL",
        "title": "Partial Record",
        "publication_year": 2024,
        "resolver": "stub",
        "url": "https://oa.example/partial.pdf",
        "classification": "miss",
        "path": None,
        "sha256": None,
        "content_length": None,
        "etag": 'W/"resume-metadata"',
        "last_modified": None,
        "dry_run": False,
    }
    previous_manifest.write_text(json.dumps(previous_entry) + "\n", encoding="utf-8")

    work = {
        "id": "https://openalex.org/WPARTIAL",
        "title": previous_entry["title"],
        "publication_year": previous_entry["publication_year"],
        "ids": {"doi": "10.1000/resume"},
        "open_access": {"oa_url": None},
        "best_oa_location": {"pdf_url": previous_entry["url"]},
        "primary_location": {},
        "locations": [],
    }

    contexts: List[Dict[str, Any]] = []

    patcher.setattr(downloader, "iterate_openalex", lambda *args, **kwargs: iter([work]))
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.iterate_openalex", lambda *args, **kwargs: iter([work])
    )
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])

    class RecordingPipeline:
        def __init__(self, *_, logger=None, metrics=None, **kwargs):
            self.logger = logger
            self.metrics = metrics
            self.run_id = kwargs.get("run_id")

        def run(self, session, artifact, context=None, session_factory=None):
            assert context is not None
            payload = context.to_dict() if hasattr(context, "to_dict") else context
            contexts.append(payload)
            pdf_path = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
            outcome = resolvers.DownloadOutcome(
                classification="pdf",
                path=str(pdf_path),
                http_status=200,
                content_type="application/pdf",
                elapsed_ms=5.0,
                error=None,
            )
            if self.logger is not None:
                self.logger.log_attempt(
                    resolvers.AttemptRecord(
                        run_id=self.run_id,
                        work_id=artifact.work_id,
                        resolver_name="stub",
                        resolver_order=1,
                        url=previous_entry["url"],
                        status=outcome.classification.value,
                        http_status=outcome.http_status,
                        content_type=outcome.content_type,
                        elapsed_ms=outcome.elapsed_ms,
                        dry_run=False,
                    )
                )
            if self.metrics is not None:
                self.metrics.record_attempt("stub", outcome)
            return resolvers.PipelineResult(
                success=True,
                resolver_name="stub",
                url=previous_entry["url"],
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    patcher.setattr(downloader, "ResolverPipeline", RecordingPipeline)

    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", RecordingPipeline)

    manifest_path = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    patcher.setattr(
        "sys.argv",
        value=[
            "download_pyalex_pdfs.py",
            "--topic",
            "resume test",
            "--year-start",
            "2024",
            "--year-end",
            "2024",
            "--out",
            str(out_dir),
            "--manifest",
            str(manifest_path),
            "--resume-from",
            str(previous_manifest),
        ],
    )

    downloader.main()

    assert contexts, "expected pipeline context to be captured"
    context_payload = contexts[0]
    assert context_payload["sniff_bytes"] == downloader.DEFAULT_SNIFF_BYTES
    assert context_payload["min_pdf_bytes"] == downloader.DEFAULT_MIN_PDF_BYTES
    assert context_payload["tail_check_bytes"] == downloader.DEFAULT_TAIL_CHECK_BYTES
    previous_map = context_payload["previous"]
    assert previous_entry["url"] not in previous_map

    new_entries = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    manifest_records = [entry for entry in new_entries if entry.get("record_type") == "manifest"]
    assert any(record.get("resolver") == "stub" for record in manifest_records)


def test_cli_resume_from_sqlite_when_manifest_missing(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers

    manifest_path = tmp_path / "resume_missing.jsonl"
    sqlite_path = manifest_path.with_suffix(".sqlite3")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "pdfs").mkdir(parents=True, exist_ok=True)

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
                "WSQLITE",
                "SQLite Resume",
                2023,
                "stub",
                "https://oa.example/sqlite.pdf",
                "https://oa.example/sqlite.pdf",
                str(out_dir / "pdfs" / "existing.pdf"),
                None,
                "pdf",
                "application/pdf",
                None,
                None,
                None,
                "feedface",
                2048,
                None,
                None,
                None,
                0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    work = {
        "id": "https://openalex.org/WSQLITE",
        "title": "SQLite Resume",
        "publication_year": 2023,
        "ids": {"doi": "10.1000/sqlite"},
        "open_access": {"oa_url": None},
        "best_oa_location": {"pdf_url": "https://oa.example/sqlite.pdf"},
        "primary_location": {},
        "locations": [],
    }

    contexts: List[Dict[str, Any]] = []

    patcher.setattr(downloader, "iterate_openalex", lambda *args, **kwargs: iter([work]))
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.iterate_openalex", lambda *args, **kwargs: iter([work])
    )
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])

    class InspectingPipeline:
        def __init__(self, *_, logger=None, metrics=None, **kwargs):
            self.logger = logger
            self.metrics = metrics
            self.run_id = kwargs.get("run_id")

        def run(self, session, artifact, context=None, session_factory=None):
            assert context is not None
            payload = context.to_dict() if hasattr(context, "to_dict") else context
            contexts.append(payload)
            outcome = resolvers.DownloadOutcome(
                classification="pdf",
                path=str(out_dir / "pdfs" / "downloaded.pdf"),
                http_status=200,
                content_type="application/pdf",
                elapsed_ms=1.0,
                error=None,
            )
            if self.logger is not None:
                self.logger.log_attempt(
                    resolvers.AttemptRecord(
                        run_id=self.run_id,
                        work_id=artifact.work_id,
                        resolver_name="stub",
                        resolver_order=1,
                        url="https://oa.example/sqlite.pdf",
                        status=outcome.classification.value,
                        http_status=outcome.http_status,
                        content_type=outcome.content_type,
                        elapsed_ms=outcome.elapsed_ms,
                        dry_run=False,
                    )
                )
            if self.metrics is not None:
                self.metrics.record_attempt("stub", outcome)
            return resolvers.PipelineResult(
                success=True,
                resolver_name="stub",
                url="https://oa.example/sqlite.pdf",
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    patcher.setattr(downloader, "ResolverPipeline", InspectingPipeline)
    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", InspectingPipeline)

    patcher.setattr(
        "sys.argv",
        value=[
            "download_pyalex_pdfs.py",
            "--topic",
            "sqlite resume",
            "--year-start",
            "2023",
            "--year-end",
            "2023",
            "--out",
            str(out_dir),
            "--manifest",
            str(manifest_path),
            "--resume-from",
            str(manifest_path),
        ],
    )

    downloader.main()

    assert contexts, "expected pipeline to receive resume context"
    payload = contexts[0]
    previous_map = payload["previous"]
    assert previous_map, "expected resume metadata from SQLite fallback"
    sqlite_entry = next(iter(previous_map.values()))
    assert sqlite_entry["path"].endswith("existing.pdf")


def test_cli_workers_apply_domain_jitter(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers

    works = [
        {
            "id": f"https://openalex.org/WJITTER{i}",
            "title": f"Jitter Work {i}",
            "publication_year": 2024,
            "ids": {"doi": f"10.1000/jitter{i}"},
            "open_access": {"oa_url": None},
            "best_oa_location": {"pdf_url": f"https://example.org/work{i}.pdf"},
            "primary_location": {},
            "locations": [],
        }
        for i in range(2)
    ]

    patcher.setattr(downloader, "iterate_openalex", lambda *a, **k: iter(works))
    patcher.setattr("DocsToKG.ContentDownload.runner.iterate_openalex", lambda *a, **k: iter(works))
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [SimpleNamespace(name="stub")])
    patcher.setattr(
        "DocsToKG.ContentDownload.resolvers.default_resolvers",
        lambda: [SimpleNamespace(name="stub")],
    )
    patcher.setattr(
        "DocsToKG.ContentDownload.args.default_resolvers",
        lambda: [SimpleNamespace(name="stub")],
    )

    sleep_calls: List[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monotonic_values = iter([0.0, 0.02, 0.02])

    def fake_monotonic() -> float:
        nonlocal monotonic_values
        try:
            return next(monotonic_values)
        except StopIteration:
            return 0.02

    patcher.setattr(resolvers._time, "sleep", fake_sleep)
    patcher.setattr(resolvers._time, "monotonic", fake_monotonic)
    patcher.setattr(resolvers.random, "random", lambda: 0.5)

    executor_meta: Dict[str, Any] = {}

    class RecordingExecutor:
        def __init__(self, max_workers: int):
            executor_meta["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        class _RecordingFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        def submit(self, fn, *args, **kwargs):
            result = fn(*args, **kwargs)
            return self._RecordingFuture(result)

    patcher.setattr("DocsToKG.ContentDownload.runner.ThreadPoolExecutor", RecordingExecutor)
    patcher.setattr("DocsToKG.ContentDownload.runner.as_completed", lambda futures: futures)

    class DummyThrottle:
        def __init__(self, config):
            self.config = config
            self._host_lock = resolvers.threading.Lock()
            self._last_host_hit = defaultdict(float)
            self._host_buckets = {}
            self._host_bucket_lock = resolvers.threading.Lock()

        def _ensure_host_bucket(self, host: str):  # pragma: no cover - minimal shim
            spec = self.config.domain_token_buckets.get(host)
            if not spec:
                return None
            with self._host_bucket_lock:
                bucket = self._host_buckets.get(host)
                if bucket is None:
                    bucket = resolvers.TokenBucket(
                        rate_per_second=float(spec.get("rate_per_second", 1.0)),
                        capacity=float(spec.get("capacity", 1.0)),
                    )
                    self._host_buckets[host] = bucket
                return bucket

    class RecordingPipeline:
        def __init__(
            self,
            *,
            resolvers=None,
            config=None,
            download_func=None,
            logger=None,
            metrics=None,
            **kwargs,
        ):
            self.config = config
            self.logger = logger
            self.metrics = metrics
            self.run_id = kwargs.get("run_id")
            self._throttle = DummyThrottle(config)

        def run(self, session, artifact, context=None, session_factory=None):
            resolvers.ResolverPipeline._respect_domain_limit(
                self._throttle, "https://example.org/resource.pdf"
            )
            pdf_path = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
            outcome = resolvers.DownloadOutcome(
                classification="pdf",
                path=str(pdf_path),
                http_status=200,
                content_type="application/pdf",
                elapsed_ms=5.0,
                error=None,
            )
            if self.logger is not None:
                self.logger.log_attempt(
                    resolvers.AttemptRecord(
                        run_id=self.run_id,
                        work_id=artifact.work_id,
                        resolver_name="stub",
                        resolver_order=1,
                        url="https://example.org/resource.pdf",
                        status="pdf",
                        http_status=200,
                        content_type="application/pdf",
                        elapsed_ms=5.0,
                        dry_run=False,
                    )
                )
            if self.metrics is not None:
                self.metrics.record_attempt("stub", outcome)
            return resolvers.PipelineResult(
                success=True,
                resolver_name="stub",
                url="https://example.org/resource.pdf",
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    patcher.setattr(downloader, "ResolverPipeline", RecordingPipeline)

    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", RecordingPipeline)

    manifest_path = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    patcher.setattr(
        "sys.argv",
        value=[
            "download_pyalex_pdfs.py",
            "--topic",
            "jitter",
            "--year-start",
            "2024",
            "--year-end",
            "2024",
            "--out",
            str(out_dir),
            "--manifest",
            str(manifest_path),
            "--workers",
            "3",
            "--domain-min-interval",
            "example.org=0.1",
        ],
    )

    downloader.main()

    assert executor_meta["max_workers"] == 3
    assert len(sleep_calls) == 1
    expected_wait = 0.1 - 0.02 + 0.5 * 0.05
    assert sleep_calls[0] == pytest.approx(expected_wait)


def test_cli_head_precheck_handles_head_hostile(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers
    from DocsToKG.ContentDownload import networking as network_module

    work = {
        "id": "https://openalex.org/WHEAD",
        "title": "Head Hostile",
        "publication_year": 2024,
        "ids": {"doi": "10.1000/head"},
        "open_access": {"oa_url": None},
        "best_oa_location": {"pdf_url": "https://example.org/hostile.pdf"},
        "primary_location": {},
        "locations": [],
    }

    patcher.setattr(downloader, "iterate_openalex", lambda *a, **k: iter([work]))
    patcher.setattr(
        "DocsToKG.ContentDownload.runner.iterate_openalex", lambda *a, **k: iter([work])
    )
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])

    head_calls: List[str] = []

    class _HeadResponse:
        status_code = 405
        headers: Dict[str, str] = {}

        def close(self) -> None:
            return None

    class _StreamResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "application/pdf"}
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def iter_content(self, chunk_size: int = 1024):
            yield b"%PDF"

        def close(self) -> None:
            self.closed = True

    responses = [_HeadResponse(), _StreamResponse()]

    def fake_request(session, method, url, **kwargs):
        head_calls.append(method)
        return responses.pop(0)

    patcher.setattr(network_module, "request_with_retries", fake_request)

    class RecordingPipeline:
        def __init__(self, *_, logger=None, metrics=None, **kwargs):
            self.logger = logger
            self.metrics = metrics
            self.run_id = kwargs.get("run_id")

        def run(self, session, artifact, context=None, session_factory=None):
            passed = network_module.head_precheck(session, artifact.pdf_urls[0], timeout=3.0)
            assert passed is True
            pdf_path = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
            outcome = resolvers.DownloadOutcome(
                classification="pdf",
                path=str(pdf_path),
                http_status=200,
                content_type="application/pdf",
                elapsed_ms=4.0,
                error=None,
            )
            if self.logger is not None:
                self.logger.log_attempt(
                    resolvers.AttemptRecord(
                        run_id=self.run_id,
                        work_id=artifact.work_id,
                        resolver_name="stub",
                        resolver_order=1,
                        url=artifact.pdf_urls[0],
                        status="pdf",
                        http_status=200,
                        content_type="application/pdf",
                        elapsed_ms=4.0,
                        dry_run=False,
                    )
                )
            if self.metrics is not None:
                self.metrics.record_attempt("stub", outcome)
            return resolvers.PipelineResult(
                success=True,
                resolver_name="stub",
                url=artifact.pdf_urls[0],
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    patcher.setattr(downloader, "ResolverPipeline", RecordingPipeline)

    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", RecordingPipeline)

    manifest_path = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    patcher.setattr(
        "sys.argv",
        value=[
            "download_pyalex_pdfs.py",
            "--topic",
            "head hostile",
            "--year-start",
            "2024",
            "--year-end",
            "2024",
            "--out",
            str(out_dir),
            "--manifest",
            str(manifest_path),
        ],
    )

    downloader.main()

    assert head_calls == ["HEAD", "GET"]


def test_cli_attempt_records_cover_all_resolvers(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers

    works = [
        {
            "id": "https://openalex.org/WATTEMPT",
            "title": "Attempt Coverage",
            "publication_year": 2024,
            "ids": {"doi": "10.1000/attempt"},
            "open_access": {"oa_url": None},
            "best_oa_location": {"pdf_url": "https://example.org/coverage.pdf"},
            "primary_location": {},
            "locations": [],
        }
    ]

    resolver_order = ["alpha", "beta", "gamma"]

    class StubResolver:
        def __init__(self, name: str) -> None:
            self.name = name

    patcher.setattr(downloader, "iterate_openalex", lambda *a, **k: iter(works))
    patcher.setattr("DocsToKG.ContentDownload.runner.iterate_openalex", lambda *a, **k: iter(works))
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(
        downloader, "default_resolvers", lambda: [StubResolver(name) for name in resolver_order]
    )
    patcher.setattr(
        "DocsToKG.ContentDownload.args.default_resolvers",
        lambda: [StubResolver(name) for name in resolver_order],
    )

    class RecordingPipeline:
        def __init__(
            self,
            *,
            resolvers=None,
            config=None,
            download_func=None,
            logger=None,
            metrics=None,
            **kwargs,
        ):
            self.resolvers = resolvers or []
            self.logger = logger
            self.metrics = metrics
            self.run_id = kwargs.get("run_id")

        def run(self, session, artifact, context=None, session_factory=None):
            for order, resolver in enumerate(self.resolvers, start=1):
                if self.logger is not None:
                    self.logger.log_attempt(
                        resolvers.AttemptRecord(
                            run_id=self.run_id,
                            work_id=artifact.work_id,
                            resolver_name=resolver.name,
                            resolver_order=order,
                            url=f"https://example.org/{resolver.name}.pdf",
                            status="skipped" if order < len(self.resolvers) else "pdf",
                            http_status=200,
                            content_type="application/pdf",
                            elapsed_ms=order * 1.0,
                            dry_run=False,
                        )
                    )
            outcome = resolvers.DownloadOutcome(
                classification="pdf",
                path=str(artifact.pdf_dir / f"{artifact.base_stem}.pdf"),
                http_status=200,
                content_type="application/pdf",
                elapsed_ms=10.0,
                error=None,
            )
            if self.logger is not None:
                self.logger.log_manifest(
                    build_manifest_entry(
                        artifact,
                        resolver="gamma",
                        url="https://example.org/gamma.pdf",
                        outcome=outcome,
                        html_paths=[],
                        dry_run=False,
                        run_id=self.run_id,
                    )
                )
            if self.metrics is not None:
                self.metrics.record_attempt("gamma", outcome)
            return resolvers.PipelineResult(
                success=True,
                resolver_name="gamma",
                url="https://example.org/gamma.pdf",
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    patcher.setattr(downloader, "ResolverPipeline", RecordingPipeline)

    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", RecordingPipeline)

    manifest_path = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    patcher.setattr(
        "sys.argv",
        value=[
            "download_pyalex_pdfs.py",
            "--topic",
            "attempt records",
            "--year-start",
            "2024",
            "--year-end",
            "2024",
            "--out",
            str(out_dir),
            "--manifest",
            str(manifest_path),
        ],
    )

    downloader.main()

    entries = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    attempt_entries = [entry for entry in entries if entry.get("record_type") == "attempt"]
    assert [entry["resolver_name"] for entry in attempt_entries] == resolver_order
    assert [entry["resolver_order"] for entry in attempt_entries] == [1, 2, 3]


def test_cli_dry_run_metrics_align(download_modules, patcher, tmp_path):
    downloader = download_modules.downloader
    resolvers = download_modules.resolvers

    works = [
        {
            "id": f"https://openalex.org/WDRY{i}",
            "title": f"Dry Run {i}",
            "publication_year": 2024,
            "ids": {"doi": f"10.1000/dry{i}"},
            "open_access": {"oa_url": None},
            "best_oa_location": {"pdf_url": f"https://example.org/dry{i}.pdf"},
            "primary_location": {},
            "locations": [],
        }
        for i in range(2)
    ]

    patcher.setattr(downloader, "iterate_openalex", lambda *a, **k: iter(works))
    patcher.setattr("DocsToKG.ContentDownload.runner.iterate_openalex", lambda *a, **k: iter(works))
    patcher.setattr(downloader, "resolve_topic_id_if_needed", lambda value, *_: value)
    patcher.setattr(
        "DocsToKG.ContentDownload.args.resolve_topic_id_if_needed", lambda value, *_: value
    )
    patcher.setattr(downloader, "default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.resolvers.default_resolvers", lambda: [])
    patcher.setattr("DocsToKG.ContentDownload.args.default_resolvers", lambda: [])

    class RecordingPipeline:
        def __init__(self, *_, logger=None, metrics=None, **kwargs):
            self.logger = logger
            self.metrics = metrics
            self.run_id = kwargs.get("run_id")

        def run(self, session, artifact, context=None, session_factory=None):
            assert isinstance(context, DownloadContext)
            assert context.dry_run is True
            outcome = resolvers.DownloadOutcome(
                classification="pdf",
                path=None,
                http_status=200,
                content_type="application/pdf",
                elapsed_ms=2.0,
                error=None,
            )
            if self.metrics is not None:
                self.metrics.record_attempt("stub", outcome)
            return resolvers.PipelineResult(
                success=True,
                resolver_name="stub",
                url="https://example.org/dry.pdf",
                outcome=outcome,
                html_paths=[],
                failed_urls=[],
            )

    patcher.setattr(downloader, "ResolverPipeline", RecordingPipeline)

    patcher.setattr("DocsToKG.ContentDownload.runner.ResolverPipeline", RecordingPipeline)

    manifest_path = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    patcher.setattr(
        "sys.argv",
        value=[
            "download_pyalex_pdfs.py",
            "--topic",
            "dry metrics",
            "--year-start",
            "2024",
            "--year-end",
            "2024",
            "--out",
            str(out_dir),
            "--manifest",
            str(manifest_path),
            "--dry-run",
        ],
    )

    downloader.main()

    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    manifest_rows = [record for record in records if record.get("record_type") == "manifest"]
    assert len(manifest_rows) == len(works)
    assert all(row["dry_run"] is True for row in manifest_rows)

    metrics_path = manifest_path.with_suffix(".metrics.json")
    metrics_doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_doc["processed"] == len(works)
    assert metrics_doc["saved"] == len(works)
    assert metrics_doc["resolvers"]["attempts"]["stub"] == len(works)


def test_envrc_configures_virtualenv_and_pythonpath() -> None:
    text = ENVRC.read_text(encoding="utf-8")
    assert 'export VIRTUAL_ENV="$VENVP"' in text
    assert 'PATH_add "$VIRTUAL_ENV/bin"' in text
    assert 'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"' in text


def test_bootstrap_script_installs_project() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert ' -m venv "$VENV_PATH"' in text
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
