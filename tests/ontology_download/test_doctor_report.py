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


def test_doctor_report_handles_unreadable_bioportal_key():
    """Unreadable BioPortal API key file is reported instead of crashing."""

    usage = FakeUsage(total=100 * 1_000_000_000, used=0, free=100 * 1_000_000_000)
    dummy_response = SimpleNamespace(status_code=200, ok=True, reason="OK")
    dummy_config = SimpleNamespace(defaults=SimpleNamespace(http=SimpleNamespace(rate_limits={})))

    with TestingEnvironment():
        api_key_path = cli.CONFIG_DIR / "bioportal_api_key.txt"
        api_key_path.write_text("secret")

        original_read_text = Path.read_text

        def fake_read_text(self, *args, **kwargs):
            if self == api_key_path:
                raise PermissionError("mock permission denied")
            return original_read_text(self, *args, **kwargs)

        with patch.object(cli.shutil, "disk_usage", return_value=usage), patch.object(
            cli.shutil, "which", return_value=None
        ), patch.object(cli.requests, "head", return_value=dummy_response), patch.object(
            cli.requests, "get", return_value=dummy_response
        ), patch.object(
            cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: dummy_config)
        ), patch.object(Path, "read_text", fake_read_text):
            report = cli._doctor_report()

    bioportal = report["bioportal_api_key"]

    assert bioportal["configured"] is False
    assert "error" in bioportal
    assert "mock permission denied" in bioportal["error"]


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
