# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_regression_compatibility",
#   "purpose": "Pytest coverage for content download regression compatibility scenarios",
#   "sections": [
#     {
#       "id": "test-default-resolver-order-remains-stable",
#       "name": "test_default_resolver_order_remains_stable",
#       "anchor": "function-test-default-resolver-order-remains-stable",
#       "kind": "function"
#     },
#     {
#       "id": "test-manifest-entry-schema-backward-compatible",
#       "name": "test_manifest_entry_schema_backward_compatible",
#       "anchor": "function-test-manifest-entry-schema-backward-compatible",
#       "kind": "function"
#     },
#     {
#       "id": "test-load-previous-manifest-rejects-legacy-entries",
#       "name": "test_load_previous_manifest_rejects_legacy_entries",
#       "anchor": "function-test-load-previous-manifest-rejects-legacy-entries",
#       "kind": "function"
#     },
#     {
#       "id": "test-load-resolver-config-rejects-legacy-rate-limits",
#       "name": "test_load_resolver_config_rejects_legacy_rate_limits",
#       "anchor": "function-test-load-resolver-config-rejects-legacy-rate-limits",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Regression compatibility tests for the ContentDownload refactor."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload import resolvers

# --- Test Cases ---


def test_default_resolver_order_remains_stable():
    pytest.importorskip("pyalex")
    names = [resolver.name for resolver in resolvers.default_resolvers()]
    assert names == resolvers.DEFAULT_RESOLVER_ORDER


def test_manifest_entry_schema_backward_compatible(tmp_path: Path):
    work = {
        "id": "https://openalex.org/WSCHEMA",
        "title": "Manifest Schema",
        "publication_year": 2024,
        "ids": {"doi": "10.1000/schema"},
        "best_oa_location": {"pdf_url": "https://example.org/schema.pdf"},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }
    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    artifact = downloader.create_artifact(work, pdf_dir=pdf_dir, html_dir=html_dir)

    outcome = resolvers.DownloadOutcome(
        classification="pdf",
        path=str(pdf_dir / "schema.pdf"),
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=12.0,
        sha256="deadbeef",
        content_length=1024,
        etag='"etag-value"',
        last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
    )

    entry = downloader.build_manifest_entry(
        artifact,
        resolver="stub",
        url="https://example.org/schema.pdf",
        outcome=outcome,
        html_paths=[],
        dry_run=False,
    )
    payload = asdict(entry)
    expected_keys = {
        "timestamp",
        "schema_version",
        "work_id",
        "title",
        "publication_year",
        "resolver",
        "url",
        "path",
        "classification",
        "content_type",
        "reason",
        "reason_detail",
        "html_paths",
        "sha256",
        "content_length",
        "etag",
        "last_modified",
        "extracted_text_path",
        "dry_run",
    }
    assert set(payload.keys()) == expected_keys


def test_load_previous_manifest_rejects_legacy_entries(tmp_path: Path) -> None:
    manifest_path = tmp_path / "legacy.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "work_id": "https://openalex.org/WLEGACY",
                "url": "https://example.org/legacy.pdf",
                "classification": "pdf",
                "path": "/tmp/legacy.pdf",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy manifest entries"):
        downloader.load_previous_manifest(manifest_path)


def test_load_previous_manifest_requires_schema_version(tmp_path: Path) -> None:
    manifest_path = tmp_path / "no_schema.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "record_type": "manifest",
                "timestamp": "2024-05-01T00:00:00Z",
                "work_id": "WMISSING",
                "url": "https://example.org/missing.pdf",
                "classification": "pdf",
                "path": "/tmp/missing.pdf",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        downloader.load_previous_manifest(manifest_path)


def test_load_previous_manifest_rejects_mismatched_schema_version(tmp_path: Path) -> None:
    manifest_path = tmp_path / "old_schema.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "record_type": "manifest",
                "timestamp": "2024-05-01T00:00:00Z",
                "schema_version": 1,
                "work_id": "WLEGACYSCHEMA",
                "url": "https://example.org/legacy-schema.pdf",
                "classification": "pdf",
                "path": "/tmp/legacy-schema.pdf",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported manifest schema_version"):
        downloader.load_previous_manifest(manifest_path)


def test_load_resolver_config_rejects_legacy_rate_limits(tmp_path: Path):
    config_payload: Dict[str, object] = {
        "resolver_rate_limits": {"example.org": 0.75},
        "resolver_order": ["alpha", "beta"],
    }
    config_path = tmp_path / "resolvers.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    args = SimpleNamespace(
        resolver_config=str(config_path),
        unpaywall_email=None,
        core_api_key=None,
        semantic_scholar_api_key=None,
        doaj_api_key=None,
        max_resolver_attempts=None,
        resolver_timeout=None,
        disable_resolver=[],
        enable_resolver=[],
        resolver_order=None,
        concurrent_resolvers=None,
        head_precheck=None,
        accept=None,
        mailto=None,
        global_url_dedup=None,
        domain_min_interval=[],
    )

    resolver_names: List[str] = ["alpha", "beta", "gamma"]
    with pytest.raises(ValueError, match="resolver_rate_limits.*no longer supported"):
        downloader.load_resolver_config(args, resolver_names)
