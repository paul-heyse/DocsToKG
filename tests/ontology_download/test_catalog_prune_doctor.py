from __future__ import annotations

import json
from pathlib import Path

import duckdb

from DocsToKG.OntologyDownload.catalog.doctor import generate_doctor_report
from DocsToKG.OntologyDownload.catalog.prune import prune_with_staging
from DocsToKG.OntologyDownload.database import _MIGRATIONS as DB_MIGRATIONS


def _setup_connection() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect("::memory::")
    for _, sql in DB_MIGRATIONS:
        conn.execute(sql)
    return conn


def test_prune_with_staging(tmp_path) -> None:
    conn = _setup_connection()

    # Seed minimal catalog state
    conn.execute(
        "INSERT INTO versions (version_id, service, created_at, plan_hash) VALUES (?, ?, now(), ?)",
        ["2025-01-01", "svc", None],
    )
    conn.execute(
        """
        INSERT INTO artifacts (artifact_id, version_id, service, source_url, size_bytes, fs_relpath, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "artifact_a",
            "2025-01-01",
            "svc",
            "https://example.com/onto.zip",
            12,
            "svc/2025-01-01/src/onto.zip",
            "fresh",
        ],
    )
    conn.execute(
        """
        INSERT INTO extracted_files (file_id, artifact_id, version_id, relpath_in_version, format, size_bytes, mtime, cas_relpath)
        VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        [
            "file_a",
            "artifact_a",
            "2025-01-01",
            "data/ontology.ttl",
            "ttl",
            5,
        ],
    )

    artifacts_root = tmp_path / "artifacts"
    extracted_root = tmp_path / "extracted"
    (artifacts_root / "svc/2025-01-01/src").mkdir(parents=True, exist_ok=True)
    (extracted_root / "svc/2025-01-01/data").mkdir(parents=True, exist_ok=True)

    # Create files for catalog entries
    (artifacts_root / "svc/2025-01-01/src/onto.zip").write_bytes(b"zip-bytes")
    (extracted_root / "svc/2025-01-01/data/ontology.ttl").write_text("ttl", encoding="utf-8")

    # Add orphan file
    orphan_path = artifacts_root / "svc/2025-01-01/orphan.tmp"
    orphan_path.parent.mkdir(parents=True, exist_ok=True)
    orphan_path.write_bytes(b"orphan")

    stats_dry = prune_with_staging(conn, artifacts_root, dry_run=True)
    assert stats_dry.orphan_count == 1
    assert stats_dry.deleted_count == 0
    assert stats_dry.total_bytes_freed >= len(b"orphan")
    assert orphan_path.exists()

    stats_apply = prune_with_staging(conn, artifacts_root, dry_run=False)
    assert stats_apply.deleted_count == 1
    assert stats_apply.total_bytes_freed >= len(b"orphan")
    assert not orphan_path.exists()


def test_generate_doctor_report(tmp_path) -> None:
    conn = _setup_connection()

    conn.execute(
        "INSERT INTO versions (version_id, service, created_at, plan_hash) VALUES (?, ?, now(), ?)",
        ["2025-01-01", "svc", None],
    )
    conn.execute(
        """
        INSERT INTO artifacts (artifact_id, version_id, service, source_url, size_bytes, fs_relpath, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "artifact_a",
            "2025-01-01",
            "svc",
            "https://example.com/onto.zip",
            12,
            "svc/2025-01-01/src/onto.zip",
            "fresh",
        ],
    )
    conn.execute(
        """
        INSERT INTO extracted_files (file_id, artifact_id, version_id, relpath_in_version, format, size_bytes, mtime, cas_relpath)
        VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        [
            "file_a",
            "artifact_a",
            "2025-01-01",
            "data/ontology.ttl",
            "ttl",
            5,
        ],
    )

    artifacts_root = tmp_path / "artifacts"
    extracted_root = tmp_path / "extracted"
    (artifacts_root / "svc/2025-01-01/src").mkdir(parents=True, exist_ok=True)
    (extracted_root / "svc/2025-01-01/data").mkdir(parents=True, exist_ok=True)

    # Only create extracted file to simulate missing artifact; also create orphan extracted file
    (extracted_root / "svc/2025-01-01/data/ontology.ttl").write_text("ttl", encoding="utf-8")
    orphan_extracted = extracted_root / "svc/2025-01-01/data/orphan.txt"
    orphan_extracted.write_text("orphan", encoding="utf-8")

    # Missing LATEST.json triggers mismatch
    report = generate_doctor_report(conn, artifacts_root, extracted_root)

    assert report.issues_found >= 2
    severities = {issue.severity for issue in report.issues}
    assert "error" in severities or "warning" in severities

    issue_types = {issue.issue_type for issue in report.issues}
    assert "missing_fs_artifact" in issue_types
    assert "orphan_extracted_file" in issue_types

    # Ensure issues include paths for context
    assert any(issue.fs_path is not None for issue in report.issues)

