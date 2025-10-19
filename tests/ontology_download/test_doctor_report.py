"""Unit-level coverage for doctor diagnostics assembly.

Exercises the helper routines that gather disk usage, optional dependency
status, resolver/plugin metadata, and structured suggestions so that the doctor
CLI can render actionable reports.
"""

from __future__ import annotations

from collections import namedtuple
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Callable, Optional
from unittest.mock import patch

import DocsToKG.OntologyDownload.cli as cli
from DocsToKG.OntologyDownload.testing import TestingEnvironment

FakeUsage = namedtuple("FakeUsage", "total used free")


def _doctor_report_with_disk(
    total_bytes: int,
    free_bytes: int,
    *,
    before_run: Optional[Callable[[TestingEnvironment], None]] = None,
):
    """Execute ``_doctor_report`` within a controlled testing environment."""

    usage = FakeUsage(total=total_bytes, used=total_bytes - free_bytes, free=free_bytes)
    dummy_response = SimpleNamespace(status_code=200, ok=True, reason="OK")
    dummy_config = SimpleNamespace(defaults=SimpleNamespace(http=SimpleNamespace(rate_limits={})))

    with TestingEnvironment() as env:
        if before_run is not None:
            before_run(env)

        with ExitStack() as stack:
            stack.enter_context(patch.object(cli, "CONFIG_DIR", env.config_dir))
            stack.enter_context(patch.object(cli, "CACHE_DIR", env.cache_dir))
            stack.enter_context(patch.object(cli, "LOG_DIR", env.log_dir))
            stack.enter_context(patch.object(cli, "LOCAL_ONTOLOGY_DIR", env.ontology_dir))
            stack.enter_context(patch.object(cli, "STORAGE", env._storage))
            stack.enter_context(patch.object(cli.shutil, "disk_usage", return_value=usage))
            stack.enter_context(patch.object(cli.shutil, "which", return_value=None))
            stack.enter_context(patch.object(cli.requests, "head", return_value=dummy_response))
            stack.enter_context(patch.object(cli.requests, "get", return_value=dummy_response))
            stack.enter_context(
                patch.object(
                    cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: dummy_config)
                )
            )

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


def test_doctor_report_handles_unreadable_bioportal_key():
    """The doctor report should capture errors encountered reading the API key file."""

    key_path_holder = {}

    def before_run(env: TestingEnvironment) -> None:
        key_path = env.config_dir / "bioportal_api_key.txt"
        key_path.write_text("secret")
        key_path_holder["path"] = key_path

    original_read_text = cli.Path.read_text

    def read_text_side_effect(self, *args, **kwargs):
        key_path = key_path_holder.get("path")
        if key_path is not None and self == key_path:
            raise PermissionError("mocked permission denied")
        return original_read_text(self, *args, **kwargs)

    with patch.object(cli.Path, "read_text", autospec=True, side_effect=read_text_side_effect):
        report = _doctor_report_with_disk(
            200 * 1_000_000_000,
            200 * 1_000_000_000,
            before_run=before_run,
        )

    bioportal = report["bioportal_api_key"]

    assert bioportal["configured"] is False
    assert "error" in bioportal
    assert "PermissionError" in bioportal["error"]
