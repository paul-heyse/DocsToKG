"""Tests for the ontology download doctor diagnostics."""

from __future__ import annotations

from collections import namedtuple
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
