from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name", [
        "DocsToKG.ContentDownload.telemetry_wayback",
        "DocsToKG.ContentDownload.telemetry_wayback_legacy",
    ],
)
def test_module_has_single_future_import(module_name: str) -> None:
    module = importlib.import_module(module_name)
    path = Path(module.__file__ or "")
    source = path.read_text()

    assert source.count("from __future__ import annotations") == 1


def test_runtime_and_legacy_interfaces_are_split() -> None:
    runtime = importlib.import_module("DocsToKG.ContentDownload.telemetry_wayback")

    assert hasattr(runtime, "TelemetryWayback")
    assert not hasattr(runtime, "WaybackTelemetry")
