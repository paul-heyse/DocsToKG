"""Tests for DocsToKG.OntologyDownload.cli.obs_cmd."""

from pathlib import Path

import pytest

typer = pytest.importorskip("typer")

duckdb = pytest.importorskip("duckdb")

from DocsToKG.OntologyDownload.cli import obs_cmd


def test_obs_stats_closes_connection_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid queries should not leave DuckDB lock files behind."""

    db_path = tmp_path / "catalog.duckdb"
    # Create the database file and ensure no lingering lock.
    duckdb.connect(str(db_path)).close()
    lock_path = Path(f"{db_path}.lock")
    if lock_path.exists():
        lock_path.unlink()
    assert not lock_path.exists()

    def _connect():
        return duckdb.connect(str(db_path))

    monkeypatch.setattr(obs_cmd, "_get_duckdb_connection", _connect)
    monkeypatch.setattr(obs_cmd, "get_query", lambda _name: "SELECT * FROM missing_table")

    with pytest.raises(typer.Exit):
        obs_cmd.obs_stats("broken-query", False, False)

    assert not lock_path.exists(), "DuckDB connection leak left a lock file behind"
    # A fresh connection should succeed without lock contention.
    duckdb.connect(str(db_path)).close()
    assert not lock_path.exists()
