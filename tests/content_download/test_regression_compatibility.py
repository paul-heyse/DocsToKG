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
#     },
#     {
#       "id": "test-load-previous-manifest-uses-sqlite-fallback",
#       "name": "test_load_previous_manifest_uses_sqlite_fallback",
#       "anchor": "function-test-load-previous-manifest-uses-sqlite-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "test-load-previous-manifest-truncated-jsonl-uses-sqlite-fallback",
#       "name": "test_load_previous_manifest_truncated_jsonl_uses_sqlite_fallback",
#       "anchor": "function-test-load-previous-manifest-truncated-jsonl-uses-sqlite-fallback",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Compatibility regression tests for ContentDownload public contracts.

The suite locks down resolver ordering, manifest schema invariants, resume
behaviour when legacy JSONL segments are missing, and configuration migrations.
It acts as an early warning when schema bumps or default toggles risk breaking
downstream tooling that depends on stable manifests and config semantics. The
tests guard the refactor against regressions that would violate public
contracts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

from DocsToKG.ContentDownload import cli as downloader
from DocsToKG.ContentDownload import pipeline as resolvers
from DocsToKG.ContentDownload.telemetry import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest_entry,
)


def _seed_sqlite_resume(sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

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
                "2025-01-02T00:00:00Z",
                "resume-run",
                MANIFEST_SCHEMA_VERSION,
                "W-SQLITE",
                "SQLite Resume",
                2024,
                "openalex",
                "https://example.org/W-SQLITE.pdf",
                "https://example.org/w-sqlite.pdf",
                "/data/stored.pdf",
                None,
                "pdf",
                "application/pdf",
                None,
                None,
                None,
                "deadbeef",
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
    xml_dir = tmp_path / "xml"
    artifact = downloader.create_artifact(work, pdf_dir=pdf_dir, html_dir=html_dir, xml_dir=xml_dir)

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

    entry = build_manifest_entry(
        artifact,
        resolver="stub",
        url="https://example.org/schema.pdf",
        outcome=outcome,
        html_paths=[],
        dry_run=False,
        run_id="test-run",
    )
    payload = asdict(entry)
    assert payload["run_id"] == "test-run"
    expected_keys = {
        "timestamp",
        "run_id",
        "schema_version",
        "work_id",
        "title",
        "publication_year",
        "resolver",
        "url",
        "path",
        "path_mtime_ns",
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


def test_load_previous_manifest_missing_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "resume.jsonl"

    with pytest.raises(ValueError) as excinfo:
        downloader.load_previous_manifest(manifest_path)

    message = str(excinfo.value)
    assert str(manifest_path) in message
    assert "--resume-from" in message


def test_load_previous_manifest_uses_sqlite_fallback(tmp_path: Path) -> None:
    manifest_path = tmp_path / "resume.jsonl"
    sqlite_path = tmp_path / "resume.sqlite3"
    _seed_sqlite_resume(sqlite_path)

    per_work, completed = downloader.load_previous_manifest(
        manifest_path,
        sqlite_path=sqlite_path,
        allow_sqlite_fallback=True,
    )

    assert "W-SQLITE" in completed
    resume_entry = per_work.get("W-SQLITE")
    assert resume_entry is not None and resume_entry
    first_entry = next(iter(resume_entry.values()))
    assert first_entry["classification"] == "pdf"
    assert first_entry["path"] == "/data/stored.pdf"


def test_load_previous_manifest_truncated_jsonl_uses_sqlite_fallback(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "resume.jsonl"
    sqlite_path = tmp_path / "resume.sqlite3"
    manifest_path.write_text(
        '{"record_type": "manifest", "schema_version": 3, "work_id": "W-BAD"',
        encoding="utf-8",
    )
    _seed_sqlite_resume(sqlite_path)

    per_work, completed = downloader.load_previous_manifest(
        manifest_path,
        sqlite_path=sqlite_path,
        allow_sqlite_fallback=True,
    )

    assert "W-SQLITE" in completed
    resume_entry = per_work.get("W-SQLITE")
    assert resume_entry is not None and resume_entry
    first_entry = next(iter(resume_entry.values()))
    assert first_entry["classification"] == "pdf"
    assert first_entry["path"] == "/data/stored.pdf"


def test_load_previous_manifest_sqlite_path_has_no_warning(tmp_path: Path, caplog) -> None:
    sqlite_path = tmp_path / "resume.sqlite3"
    _seed_sqlite_resume(sqlite_path)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="DocsToKG.ContentDownload.telemetry"):
        per_work, completed = downloader.load_previous_manifest(
            sqlite_path,
            sqlite_path=sqlite_path,
            allow_sqlite_fallback=True,
        )

    assert not caplog.records
    assert "W-SQLITE" in completed
    resume_entry = per_work.get("W-SQLITE")
    assert resume_entry is not None and resume_entry
    first_entry = next(iter(resume_entry.values()))
    assert first_entry["classification"] == "pdf"
    assert first_entry["path"] == "/data/stored.pdf"


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


def test_load_resolver_config_applies_concurrency_and_dedup_overrides(tmp_path: Path):
    config_payload: Dict[str, object] = {
        "max_concurrent_resolvers": 5,
        "enable_global_url_dedup": False,
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
        retry_after_cap=None,
        concurrent_resolvers=None,
        max_concurrent_per_host=None,
        disable_resolver=[],
        enable_resolver=[],
        resolver_order=None,
        head_precheck=None,
        accept=None,
        mailto=None,
        global_url_dedup=None,
        domain_min_interval=[],
        domain_token_bucket=[],
    )

    config = downloader.load_resolver_config(args, ["alpha", "beta"])

    assert config.max_concurrent_resolvers == 5
    assert config.enable_global_url_dedup is False


def test_load_resolver_config_rejects_invalid_concurrency_override(tmp_path: Path):
    config_payload: Dict[str, object] = {"max_concurrent_resolvers": 0}
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
        retry_after_cap=None,
        concurrent_resolvers=None,
        max_concurrent_per_host=None,
        disable_resolver=[],
        enable_resolver=[],
        resolver_order=None,
        head_precheck=None,
        accept=None,
        mailto=None,
        global_url_dedup=None,
        domain_min_interval=[],
        domain_token_bucket=[],
    )

    with pytest.raises(ValueError, match="max_concurrent_resolvers"):
        downloader.load_resolver_config(args, ["alpha", "beta"])
