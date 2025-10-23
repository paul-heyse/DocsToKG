"""Smoke tests for the observability CLI commands."""

from __future__ import annotations

import json
from typing import Iterable

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from DocsToKG.OntologyDownload.cli.obs_cmd import app
from DocsToKG.OntologyDownload.observability import (
    clear_context,
    emit_event,
    flush_events,
    initialize_events,
    set_context,
)
from DocsToKG.OntologyDownload.observability import events as events_module
from DocsToKG.OntologyDownload.settings import (
    get_default_config,
    invalidate_default_config_cache,
)


def _close_sinks(sinks: Iterable[object]) -> None:
    for sink in sinks:
        close = getattr(sink, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


def test_obs_stats_reads_shared_duckdb(tmp_path, monkeypatch):
    """Events emitted via DuckDB are visible to ``obs stats`` queries."""

    runner = CliRunner()

    # Point configuration-derived paths at the temporary directory.
    monkeypatch.setenv("HOME", str(tmp_path))
    invalidate_default_config_cache()
    config = get_default_config(copy=True)
    db_path = config.defaults.db.path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure a clean sink registry so we can flush deterministically.
    original_sinks = list(events_module._sinks)
    events_module._sinks.clear()

    try:
        initialize_events(enable_stdout=False, enable_duckdb=True, db_path=db_path)
        set_context(run_id="test-run", config_hash="hash", service="service-A")
        emit_event(
            type="net.request",
            level="INFO",
            payload={"elapsed_ms": 123, "cache": "hit"},
        )
        flush_events()
        _close_sinks(list(events_module._sinks))

        result = runner.invoke(
            app, ["stats", "net_latency_distribution", "--json"]
        )
        assert result.exit_code == 0, result.stdout

        payload = result.stdout.strip()
        assert payload, "expected JSON output"
        rows = json.loads(payload)
        assert any(row["service"] == "service-A" for row in rows)
    finally:
        clear_context()
        _close_sinks(list(events_module._sinks))
        events_module._sinks[:] = original_sinks
        invalidate_default_config_cache()
