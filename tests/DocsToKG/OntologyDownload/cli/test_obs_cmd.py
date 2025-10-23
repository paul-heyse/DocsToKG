from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip(
    "duckdb", reason="duckdb is required for lock-file regression coverage"
)
if getattr(duckdb, "_DOCSTOKG_TEST_STUB", False):
    pytest.skip("duckdb optional dependency not installed")

typer = pytest.importorskip(
    "typer", reason="typer is required for CLI regression coverage"
)
if getattr(typer, "_DOCSTOKG_TEST_STUB", False):
    pytest.skip("typer optional dependency not installed")

from DocsToKG.OntologyDownload.cli import obs_cmd


def _lock_path(database_path: Path) -> Path:
    """Return the DuckDB lock-file path for the given database file."""

    return database_path.with_suffix(database_path.suffix + ".lock")


def test_obs_tail_releases_duckdb_lock_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "events.duckdb"
    lock_path = _lock_path(database_path)

    probe = duckdb.connect(str(database_path))
    try:
        if not lock_path.exists():
            pytest.skip("DuckDB version under test does not emit lock files")
    finally:
        probe.close()

    lock_observed = False

    def _connect() -> duckdb.DuckDBPyConnection:  # type: ignore[attr-defined]
        nonlocal lock_observed
        connection = duckdb.connect(str(database_path))
        lock_observed = lock_path.exists()
        return connection

    monkeypatch.setattr(obs_cmd, "_get_duckdb_connection", _connect)

    with pytest.raises(typer.Exit):
        obs_cmd.obs_tail(count=1)

    assert lock_observed, "Regression test requires observing an on-disk lock"
    assert not lock_path.exists(), "DuckDB lock file should be cleared after CLI exit"

    with duckdb.connect(str(database_path)) as connection:
        connection.execute("PRAGMA database_list")
