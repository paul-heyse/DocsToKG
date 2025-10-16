"""Integration tests for the ontology downloader CLI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import (
    DefaultsConfig,
    FetchResult,
    FetchSpec,
    PlannedFetch,
    ResolverCandidate,
    ResolvedConfig,
)
from DocsToKG.OntologyDownload import cli
from DocsToKG.OntologyDownload.resolvers import FetchPlan


@pytest.fixture()
def stub_logger():
    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    return _Logger()


def _default_config() -> ResolvedConfig:
    defaults = DefaultsConfig()
    defaults.http.allowed_hosts = ["example.org"]
    return ResolvedConfig(defaults=defaults, specs=())


def _planned_fetch(ontology_id: str, url: str = "https://example.org/doc.owl") -> PlannedFetch:
    plan = FetchPlan(
        url=url,
        headers={},
        filename_hint=None,
        version="2024-01-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )
    return PlannedFetch(
        spec=FetchSpec(id=ontology_id, resolver="obo", extras={}, target_formats=["owl"]),
        resolver="obo",
        plan=plan,
        candidates=(ResolverCandidate(resolver="obo", plan=plan),),
        last_modified="2024-01-01T00:00:00Z",
        size=128,
    )


def test_cli_pull_json_output(monkeypatch, stub_logger, tmp_path, capsys):
    result = FetchResult(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        local_path=tmp_path / "hp.owl",
        status="fresh",
        sha256="abc123",
        manifest_path=tmp_path / "manifest.json",
        artifacts=[str(tmp_path / "hp.owl")],
    )
    monkeypatch.setattr(cli, "fetch_all", lambda specs, config, force: [result])
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.main(["pull", "hp", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["id"] == "hp"
    assert payload[0]["status"] == "fresh"


def test_cli_pull_table_output(monkeypatch, stub_logger, tmp_path, capsys):
    result = FetchResult(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        local_path=tmp_path / "hp.owl",
        status="fresh",
        sha256="abc123",
        manifest_path=tmp_path / "manifest.json",
        artifacts=[str(tmp_path / "hp.owl")],
    )
    monkeypatch.setattr(cli, "fetch_all", lambda specs, config, force: [result])
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.main(["pull", "hp"])
    assert exit_code == 0
    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith("id")
    assert any("hp | obo" in line for line in output)


def test_cli_pull_dry_run(monkeypatch, stub_logger, capsys):
    plans = [_planned_fetch("hp")]
    monkeypatch.setattr(cli, "plan_all", lambda specs, *, config=None, since=None: list(plans))
    monkeypatch.setattr(
        cli, "fetch_all", lambda *_, **__: pytest.fail("fetch_all should not run in dry-run")
    )
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.main(["pull", "hp", "--dry-run"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "hp" in output
    assert "application/rdf+xml" in output


def test_cli_pull_concurrency_and_hosts(monkeypatch, stub_logger):
    captured: Dict[str, List[str] | int] = {}

    def _fake_fetch(specs, config, force):
        captured["downloads"] = config.defaults.http.concurrent_downloads
        captured["hosts"] = list(config.defaults.http.allowed_hosts)
        return []

    monkeypatch.setattr(cli, "plan_all", lambda *_, **__: [])
    monkeypatch.setattr(cli, "fetch_all", _fake_fetch)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.main(
        [
            "pull",
            "hp",
            "--concurrent-downloads",
            "4",
            "--allowed-hosts",
            "mirror.example.org,cdn.example.com",
        ]
    )
    assert exit_code == 0
    assert captured["downloads"] == 4
    assert set(captured["hosts"]) == {"example.org", "mirror.example.org", "cdn.example.com"}


def test_cli_plan_json_output(monkeypatch, stub_logger, capsys):
    plan = _planned_fetch("hp")
    monkeypatch.setattr(cli, "plan_all", lambda specs, *, config=None, since=None: [plan])
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.main(["plan", "hp", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["id"] == "hp"
    assert payload[0]["url"].startswith("https://example.org/")


def test_cli_plan_since_passed_to_plan_all(monkeypatch, stub_logger):
    captured_since = None

    def _plan_all(specs, *, config=None, since=None):
        nonlocal captured_since
        captured_since = since
        return [_planned_fetch("hp")]

    monkeypatch.setattr(cli, "plan_all", _plan_all)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.main(["plan", "hp", "--since", "2024-05-01"])
    assert exit_code == 0
    assert isinstance(captured_since, datetime)
    assert captured_since.tzinfo is timezone.utc
    assert captured_since.date().isoformat() == "2024-05-01"


def test_cli_plan_concurrency_override(monkeypatch, stub_logger):
    captured = {}

    def _plan_all(specs, *, config=None, since=None):
        captured["plans"] = config.defaults.http.concurrent_plans
        return [_planned_fetch("hp")]

    monkeypatch.setattr(cli, "plan_all", _plan_all)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.main(["plan", "hp", "--concurrent-plans", "5"])
    assert exit_code == 0
    assert captured["plans"] == 5


def test_cli_plan_diff_outputs(monkeypatch, stub_logger, tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            [
                {
                    "id": "hp",
                    "url": "https://example.org/old.owl",
                    "version": "2023",
                    "license": "CC-BY",
                    "media_type": "application/rdf+xml",
                    "service": "obo",
                }
            ]
        )
    )

    new_plan = _planned_fetch("hp", url="https://example.org/new.owl")
    monkeypatch.setattr(cli, "plan_all", lambda specs, *, config=None, since=None: [new_plan])
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.main(["plan-diff", "hp", "--baseline", str(baseline_path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["added"] == []
    assert payload["removed"] == []
    assert payload["modified"][0]["id"] == "hp"

    capsys.readouterr()  # clear
    exit_code = cli.main(["plan-diff", "hp", "--baseline", str(baseline_path)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "+ hp" in output or "~ hp" in output


def test_cli_prune_dry_run_json(monkeypatch, stub_logger, capsys):
    metadata = [
        {
            "version": "2023",
            "path": Path("/tmp/hp/2023"),
            "size": 1024,
            "timestamp": datetime(2023, 1, 1, tzinfo=timezone.utc),
        },
        {
            "version": "2024",
            "path": Path("/tmp/hp/2024"),
            "size": 2048,
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        },
    ]

    monkeypatch.setattr(cli, "_collect_version_metadata", lambda oid: list(metadata))
    monkeypatch.setattr(cli, "_update_latest_symlink", lambda *_, **__: None)
    monkeypatch.setattr(
        cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    monkeypatch.setattr(cli.STORAGE, "available_ontologies", lambda: ["hp"])
    monkeypatch.setattr(cli.STORAGE, "available_versions", lambda oid: ["2023", "2024"])
    monkeypatch.setattr(cli.STORAGE, "delete_version", lambda *_, **__: 0)

    exit_code = cli.main(["prune", "--keep", "1", "--dry-run", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["total_deleted"] == 1
    assert payload["total_reclaimed_bytes"] == 1024
