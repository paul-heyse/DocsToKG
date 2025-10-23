from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import pytest

if "typer" not in sys.modules:
    typer_stub = ModuleType("typer")

    class _Exit(Exception):
        def __init__(self, code: int = 0) -> None:  # pragma: no cover - tiny shim
            self.exit_code = code

    class _Typer:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - tiny shim
            pass

        def command(self, *args, **kwargs):  # pragma: no cover - tiny shim
            def decorator(func):
                return func

            return decorator

    def _echo(message: str, err: bool = False) -> None:  # pragma: no cover - shim
        stream = sys.stderr if err else sys.stdout
        print(message, file=stream)

    def _secho(message: str, **kwargs) -> None:  # pragma: no cover - shim
        _echo(message, err=kwargs.get("err", False))

    def _option(default=None, *args, **kwargs):  # pragma: no cover - shim
        return default

    def _argument(default=None, *args, **kwargs):  # pragma: no cover - shim
        return default

    typer_stub.Typer = _Typer
    typer_stub.Option = _option
    typer_stub.Argument = _argument
    typer_stub.echo = _echo
    typer_stub.secho = _secho
    typer_stub.Exit = _Exit
    sys.modules["typer"] = typer_stub

if "duckdb" not in sys.modules:
    duckdb_stub = ModuleType("duckdb")

    def _connect(*args, **kwargs):  # pragma: no cover - shim
        raise RuntimeError("duckdb stub should not be used directly")

    duckdb_stub.connect = _connect
    sys.modules["duckdb"] = duckdb_stub

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

OBS_CMD_PATH = SRC_PATH / "DocsToKG" / "OntologyDownload" / "cli" / "obs_cmd.py"
OBS_CMD_SPEC = importlib.util.spec_from_file_location("tests.obs_cmd", OBS_CMD_PATH)
obs_cmd = importlib.util.module_from_spec(OBS_CMD_SPEC)
assert OBS_CMD_SPEC and OBS_CMD_SPEC.loader  # sanity check for loading
OBS_CMD_SPEC.loader.exec_module(obs_cmd)


class RecordingCursor:
    """Minimal cursor that exposes DuckDB-like fetch behaviour for tests."""

    def __init__(self, rows: list[tuple[Any, ...]], headers: list[str]) -> None:
        self._rows = rows
        self.description = [(header,) for header in headers]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class RecordingConnection:
    """Connection double that records SQL and parameters for assertions."""

    def __init__(self, rows: list[tuple[Any, ...]], headers: list[str]) -> None:
        self._rows = rows
        self._headers = headers
        self.last_query: str | None = None
        self.last_params: list[Any] | None = None
        self.closed = False

    def execute(
        self, query: str, params: Iterable[Any] | None = None
    ) -> RecordingCursor:  # pragma: no cover - exercised via tests
        params_list = list(params or [])
        self.last_query = query
        self.last_params = params_list

        filtered = list(self._rows)
        param_iter = iter(params_list)

        if "level = ?" in query:
            level = next(param_iter)
            filtered = [row for row in filtered if row[2] == level]
        if "type LIKE ?" in query:
            pattern = next(param_iter)
            prefix = pattern.rstrip("%")
            filtered = [row for row in filtered if row[1].startswith(prefix)]
        if "service = ?" in query:
            service = next(param_iter)
            filtered = [row for row in filtered if row[3] == service]
        if "ts >= ?" in query:
            since = next(param_iter)
            filtered = [row for row in filtered if row[0] >= since]

        if "LIMIT ?" in query and params_list:
            limit = params_list[-1]
            if isinstance(limit, int):
                filtered = filtered[:limit]

        return RecordingCursor(filtered, self._headers)

    def close(self) -> None:
        self.closed = True


class FakeDataFrame:
    """Lightweight stand-in for pandas.DataFrame used in CLI output paths."""

    def __init__(self, rows: list[tuple[Any, ...]], headers: list[str]) -> None:
        self.rows = rows
        self.headers = headers

    def to_json(
        self,
        path_or_buf=None,
        orient: str = "records",
        date_format: str = "iso",
        lines: bool = False,
    ):
        records = [dict(zip(self.headers, row)) for row in self.rows]
        if lines:
            text = "\n".join(json.dumps(record) for record in records)
        else:
            text = json.dumps(records)

        if path_or_buf is None:
            return text

        path = Path(path_or_buf)
        if lines and records:
            path.write_text(text + "\n")
        else:
            path.write_text(text)
        return None

    def to_parquet(self, path: Path) -> None:
        Path(path).write_bytes(b"PARQUET")

    def to_csv(self, path: Path, index: bool = False) -> None:
        header_line = ",".join(self.headers)
        body = "\n".join(
            ",".join(str(value) for value in row) for row in self.rows
        )
        text = header_line + ("\n" + body if body else "\n")
        Path(path).write_text(text)

    def __len__(self) -> int:
        return len(self.rows)


@pytest.fixture
def sample_rows() -> list[tuple[Any, ...]]:
    return [
        ("2024-01-01T00:00:00", "net.request", "INFO", "ontology-service", "run-1"),
    ]


@pytest.fixture(autouse=True)
def patch_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        obs_cmd,
        "_rows_to_dataframe",
        lambda rows, headers: FakeDataFrame(list(rows), list(headers)),
    )


def test_obs_tail_handles_literal_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_rows: list[tuple[Any, ...]],
) -> None:
    headers = ["ts", "type", "level", "service", "run_id"]
    connection = RecordingConnection(sample_rows, headers)
    monkeypatch.setattr(obs_cmd, "_get_duckdb_connection", lambda: connection)

    injection_value = "INFO'; DROP TABLE events; --"

    obs_cmd.obs_tail(level=injection_value, json_output=True)

    captured = capsys.readouterr()
    assert captured.out.strip() == "[]"
    assert connection.last_query is not None
    assert injection_value not in connection.last_query
    assert connection.last_params == [injection_value, 20]
    assert connection.closed


def test_obs_export_handles_literal_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_rows: list[tuple[Any, ...]],
) -> None:
    headers = ["ts", "type", "level", "service", "run_id"]
    connection = RecordingConnection(sample_rows, headers)
    monkeypatch.setattr(obs_cmd, "_get_duckdb_connection", lambda: connection)

    injection_value = "net'; DROP TABLE events; --"
    output_path = tmp_path / "export.json"

    obs_cmd.obs_export(output_path=output_path, event_type=injection_value)

    assert output_path.exists()
    assert json.loads(output_path.read_text() or "[]") == []

    assert connection.last_query is not None
    assert injection_value not in connection.last_query
    assert connection.last_params == [f"{injection_value}%"]
    assert connection.closed
