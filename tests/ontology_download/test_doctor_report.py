"""Tests for the ontology download doctor diagnostics."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import DocsToKG.OntologyDownload.cli as cli
from DocsToKG.OntologyDownload.testing import TestingEnvironment


FakeUsage = namedtuple("FakeUsage", "total used free")


def _doctor_report_with_disk(total_bytes: int, free_bytes: int):
    """Execute ``_doctor_report`` within a controlled testing environment."""

    usage = FakeUsage(total=total_bytes, used=total_bytes - free_bytes, free=free_bytes)
    dummy_response = SimpleNamespace(status_code=200, ok=True, reason="OK")
    dummy_config = SimpleNamespace(defaults=SimpleNamespace(http=SimpleNamespace(rate_limits={})))

    with TestingEnvironment():
        with patch.object(cli.shutil, "disk_usage", return_value=usage), patch.object(
            cli.shutil, "which", return_value=None
        ), patch.object(cli.requests, "head", return_value=dummy_response), patch.object(
            cli.requests, "get", return_value=dummy_response
        ), patch.object(
            cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: dummy_config)
        ):
            return cli._doctor_report()


def test_doctor_report_threshold_capped_by_total():
    """Threshold should never exceed the reported total capacity."""

    total = 5 * 1_000_000_000

    report = _doctor_report_with_disk(total, total)
    disk = report["disk"]

    assert disk["threshold_bytes"] == total
    assert disk["warning"] is False

    report = _doctor_report_with_disk(total, total - 1)

    assert report["disk"]["warning"] is True


def test_doctor_report_warning_respects_adjusted_threshold():
    """Warning toggles only when free space drops below the adjusted threshold."""

    total = 200 * 1_000_000_000
    threshold = 20 * 1_000_000_000

    report = _doctor_report_with_disk(total, 25 * 1_000_000_000)
    disk = report["disk"]

    assert disk["threshold_bytes"] == threshold
    assert disk["warning"] is False

    report = _doctor_report_with_disk(total, 19 * 1_000_000_000)

    assert report["disk"]["warning"] is True


def test_doctor_report_falls_back_to_existing_parent_when_missing_dir():
    """Disk diagnostics should fall back to the closest existing parent directory."""

    usage = FakeUsage(total=123_000_000_000, used=23_000_000_000, free=100_000_000_000)
    dummy_response = SimpleNamespace(status_code=200, ok=True, reason="OK")
    dummy_config = SimpleNamespace(defaults=SimpleNamespace(http=SimpleNamespace(rate_limits={})))

    with TestingEnvironment() as env:
        missing_dir = env.root / "shadow" / "ontologies"
        fallback_dir = env.root

        original_mkdir = cli.Path.mkdir

        def fake_mkdir(self, *args, **kwargs):
            if self == missing_dir:
                raise OSError("synthetic permissions failure")
            return original_mkdir(self, *args, **kwargs)

        recorded_paths = []

        def fake_disk_usage(target):
            recorded_paths.append(Path(target))
            return usage

        with patch.object(cli, "LOCAL_ONTOLOGY_DIR", missing_dir):
            with patch.object(cli.Path, "mkdir", fake_mkdir):
                with patch.object(cli.shutil, "disk_usage", side_effect=fake_disk_usage):
                    with patch.object(cli.shutil, "which", return_value=None):
                        with patch.object(cli.requests, "head", return_value=dummy_response):
                            with patch.object(cli.requests, "get", return_value=dummy_response):
                                with patch.object(
                                    cli.ResolvedConfig,
                                    "from_defaults",
                                    classmethod(lambda cls: dummy_config),
                                ):
                                    report = cli._doctor_report()

    assert recorded_paths == [fallback_dir]

    disk = report["disk"]
    assert Path(disk["path"]) == fallback_dir
    assert disk["total_bytes"] == usage.total
    assert disk["free_bytes"] == usage.free

    ontologies_entry = report["directories"]["ontologies"]
    assert ontologies_entry["path"] == str(missing_dir)
    assert ontologies_entry["exists"] is False
