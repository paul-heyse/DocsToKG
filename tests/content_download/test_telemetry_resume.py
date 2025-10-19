"""Tests covering resume helpers in the telemetry module."""

from __future__ import annotations

import tracemalloc
from pathlib import Path
from typing import Optional

from DocsToKG.ContentDownload.download import DownloadConfig
from DocsToKG.ContentDownload.telemetry import JsonlResumeLookup, MANIFEST_SCHEMA_VERSION

MEMORY_CAP_BYTES = 32 * 1024 * 1024


def _write_manifest(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                (
                    f'{{"record_type":"manifest","schema_version":{MANIFEST_SCHEMA_VERSION},'
                    f'"timestamp":"2024-01-01T00:00:00Z","run_id":"resume-run",'
                    f'"work_id":"W-{index}","title":"Work {index}",'
                    f'"url":"https://example.org/W-{index}.pdf","classification":"pdf"}}\n'
                )
            )


def test_jsonl_resume_lookup_handles_large_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, 100_000)

    tracemalloc.start()
    lookup: Optional[JsonlResumeLookup] = None
    try:
        lookup = JsonlResumeLookup(manifest_path)
        _current, peak = tracemalloc.get_traced_memory()
        assert peak < MEMORY_CAP_BYTES

        assert len(lookup) == 100_000
        completed_ids = lookup.completed_work_ids
        assert len(completed_ids) == 100_000
        assert "W-50000" in completed_ids

        entries = lookup["W-50000"]
        assert entries
        sample_entry = next(iter(entries.values()))
        assert sample_entry["classification"] == "pdf"
        assert sample_entry["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert sample_entry.get("normalized_url") is not None
    finally:
        tracemalloc.stop()
        if lookup is not None:
            lookup.close()


def test_download_config_accepts_lazy_resume_lookup(tmp_path) -> None:
    manifest_path = tmp_path / "resume.jsonl"
    _write_manifest(manifest_path, 3)

    lookup = JsonlResumeLookup(manifest_path)
    try:
        config = DownloadConfig(
            dry_run=True,
            previous_lookup=lookup,
            resume_completed=lookup.completed_work_ids,
        )
        assert config.previous_lookup is lookup
        assert config.resume_completed == set(lookup.completed_work_ids)
    finally:
        lookup.close()
