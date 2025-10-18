# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli",
#   "purpose": "Pytest coverage for ontology download cli scenarios",
#   "sections": [
#     {
#       "id": "stub-logger",
#       "name": "stub_logger",
#       "anchor": "function-stub-logger",
#       "kind": "function"
#     },
#     {
#       "id": "default-config",
#       "name": "_default_config",
#       "anchor": "function-default-config",
#       "kind": "function"
#     },
#     {
#       "id": "planned-fetch",
#       "name": "_planned_fetch",
#       "anchor": "function-planned-fetch",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-pull-json-output",
#       "name": "test_cli_pull_json_output",
#       "anchor": "function-test-cli-pull-json-output",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-pull-table-output",
#       "name": "test_cli_pull_table_output",
#       "anchor": "function-test-cli-pull-table-output",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-pull-dry-run",
#       "name": "test_cli_pull_dry_run",
#       "anchor": "function-test-cli-pull-dry-run",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-pull-concurrency-and-hosts",
#       "name": "test_cli_pull_concurrency_and_hosts",
#       "anchor": "function-test-cli-pull-concurrency-and-hosts",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-plan-json-output",
#       "name": "test_cli_plan_json_output",
#       "anchor": "function-test-cli-plan-json-output",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-plan-since-passed-to-plan-all",
#       "name": "test_cli_plan_since_passed_to_plan_all",
#       "anchor": "function-test-cli-plan-since-passed-to-plan-all",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-plan-concurrency-override",
#       "name": "test_cli_plan_concurrency_override",
#       "anchor": "function-test-cli-plan-concurrency-override",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-plan-diff-outputs",
#       "name": "test_cli_plan_diff_outputs",
#       "anchor": "function-test-cli-plan-diff-outputs",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-prune-dry-run-json",
#       "name": "test_cli_prune_dry_run_json",
#       "anchor": "function-test-cli-prune-dry-run-json",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-prune-updates-latest-marker",
#       "name": "test_cli_prune_updates_latest_marker",
#       "anchor": "function-test-cli-prune-updates-latest-marker",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-plan-serializes-enriched-metadata",
#       "name": "test_cli_plan_serializes_enriched_metadata",
#       "anchor": "function-test-cli-plan-serializes-enriched-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-doctor-reports-diagnostics",
#       "name": "test_cli_doctor_reports_diagnostics",
#       "anchor": "function-test-cli-doctor-reports-diagnostics",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Integration tests for the ontology downloader CLI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

import DocsToKG.OntologyDownload.cli as cli_module
from DocsToKG.OntologyDownload import FetchResult, FetchSpec, PlannedFetch
from DocsToKG.OntologyDownload import api as cli
from DocsToKG.OntologyDownload.planning import FetchPlan
from DocsToKG.OntologyDownload.resolvers import ResolverCandidate
from DocsToKG.OntologyDownload.settings import DefaultsConfig, ResolvedConfig


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


# --- Helper Functions ---


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


def _patch_cli_attr(monkeypatch: pytest.MonkeyPatch, name: str, value) -> None:
    """Patch CLI-facing attributes on both API and CLI modules when available."""

    monkeypatch.setattr(cli, name, value, raising=False)
    if hasattr(cli_module, name):
        monkeypatch.setattr(cli_module, name, value, raising=False)


# --- Test Cases ---


def test_cli_pull_json_output(monkeypatch, stub_logger, tmp_path, capsys):
    result = FetchResult(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        local_path=tmp_path / "hp.owl",
        status="fresh",
        sha256="abc123",
        manifest_path=tmp_path / "manifest.json",
        artifacts=[str(tmp_path / "hp.owl")],
    )
    _patch_cli_attr(monkeypatch, "fetch_all", lambda specs, config, force: [result])
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.cli_main(["pull", "hp", "--json"])
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
    _patch_cli_attr(monkeypatch, "fetch_all", lambda specs, config, force: [result])
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.cli_main(["pull", "hp"])
    assert exit_code == 0
    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith("id")
    assert any("hp | obo" in line for line in output)


def test_cli_pull_dry_run(monkeypatch, stub_logger, capsys):
    plans = [_planned_fetch("hp")]
    _patch_cli_attr(
        monkeypatch, "plan_all", lambda specs, *, config=None, since=None: list(plans)
    )
    _patch_cli_attr(
        monkeypatch,
        "fetch_all",
        lambda *_, **__: pytest.fail("fetch_all should not run in dry-run"),
    )
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.cli_main(["pull", "hp", "--dry-run"])
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

    _patch_cli_attr(monkeypatch, "plan_all", lambda *_, **__: [])
    _patch_cli_attr(monkeypatch, "fetch_all", _fake_fetch)
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.cli_main(
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


def test_cli_plan_disables_planner_probes():
    config = _default_config()
    args = SimpleNamespace(
        concurrent_downloads=None,
        concurrent_plans=None,
        allowed_hosts=None,
        planner_probes=False,
    )
    cli_module._apply_cli_overrides(config, args)
    assert config.defaults.planner.probing_enabled is False


def test_cli_plan_enables_planner_probes():
    config = _default_config()
    config.defaults.planner.probing_enabled = False
    args = SimpleNamespace(
        concurrent_downloads=None,
        concurrent_plans=None,
        allowed_hosts=None,
        planner_probes=True,
    )
    cli_module._apply_cli_overrides(config, args)
    assert config.defaults.planner.probing_enabled is True


def test_cli_plan_json_output(monkeypatch, stub_logger, capsys):
    plan = _planned_fetch("hp")
    _patch_cli_attr(monkeypatch, "plan_all", lambda specs, *, config=None, since=None: [plan])
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )
    captured_lock: Dict[str, str] = {}

    def _fake_lockfile(plans, path):
        captured_lock["path"] = str(path)
        return path

    monkeypatch.setattr(cli_module, "write_lockfile", _fake_lockfile, raising=False)

    exit_code = cli.cli_main(["plan", "hp", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["id"] == "hp"
    assert payload[0]["url"].startswith("https://example.org/")
    assert captured_lock["path"].endswith("ontologies.lock.json")


def test_cli_plan_since_passed_to_plan_all(monkeypatch, stub_logger):
    captured_since = None
    captured_lock: Dict[str, str] = {}

    def _plan_all(specs, *, config=None, since=None):
        nonlocal captured_since
        captured_since = since
        return [_planned_fetch("hp")]

    _patch_cli_attr(monkeypatch, "plan_all", _plan_all)
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )
    monkeypatch.setattr(
        cli_module,
        "write_lockfile",
        lambda plans, path: captured_lock.setdefault("path", str(path)) or path,
        raising=False,
    )

    exit_code = cli.cli_main(["plan", "hp", "--since", "2024-05-01"])
    assert exit_code == 0
    assert isinstance(captured_since, datetime)
    assert captured_since.tzinfo is timezone.utc
    assert captured_since.date().isoformat() == "2024-05-01"


def test_cli_plan_concurrency_override(monkeypatch, stub_logger):
    captured = {}
    monkeypatch.setattr(cli_module, "write_lockfile", lambda plans, path: path, raising=False)
    monkeypatch.setattr(cli_module, "write_lockfile", lambda plans, path: path, raising=False)

    def _plan_all(specs, *, config=None, since=None):
        captured["plans"] = config.defaults.http.concurrent_plans
        return [_planned_fetch("hp")]

    _patch_cli_attr(monkeypatch, "plan_all", _plan_all)
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.cli_main(["plan", "hp", "--concurrent-plans", "5"])
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
    _patch_cli_attr(monkeypatch, "plan_all", lambda specs, *, config=None, since=None: [new_plan])
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )
    captured_lock: Dict[str, str] = {}
    monkeypatch.setattr(
        cli_module,
        "write_lockfile",
        lambda plans, path: captured_lock.setdefault("path", str(path)) or path,
        raising=False,
    )

    exit_code = cli.cli_main(["plan-diff", "hp", "--baseline", str(baseline_path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["added"] == []
    assert payload["removed"] == []
    assert payload["modified"][0]["id"] == "hp"
    assert payload["lockfile"].endswith("ontologies.lock.json")

    capsys.readouterr()  # clear
    exit_code = cli.cli_main(["plan-diff", "hp", "--baseline", str(baseline_path)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "+ hp" in output or "~ hp" in output


def test_resolve_specs_from_args_lockfile(monkeypatch, tmp_path):
    lock_path = tmp_path / "ontologies.lock.json"
    expected_digest = "a" * 64
    lock_payload = {
        "schema_version": "1.0",
        "entries": [
            {
                "id": "hp",
                "resolver": "obo",
                "url": "https://example.org/hp.owl",
                "version": "2024-03-01",
                "license": "CC-BY",
                "media_type": "application/rdf+xml",
                "service": "obo",
                "headers": {"Accept": "application/rdf+xml"},
                "expected_checksum": {"algorithm": "sha256", "value": expected_digest},
                "target_formats": ["owl"],
            }
        ],
    }
    lock_path.write_text(json.dumps(lock_payload))

    args = SimpleNamespace(
        target_formats=None,
        spec=None,
        ids=[],
        lock=lock_path,
        log_level="INFO",
        resolver=None,
        concurrent_plans=None,
        concurrent_downloads=None,
        allowed_hosts=None,
    )
    config, specs = cli_module._resolve_specs_from_args(args, _default_config())
    assert config.defaults.resolver_fallback_enabled is False
    assert config.defaults.prefer_source == ["direct"]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.resolver == "direct"
    assert spec.extras["url"] == "https://example.org/hp.owl"
    checksum = spec.extras["checksum"]
    assert checksum["algorithm"] == "sha256"
    assert checksum["value"] == expected_digest
    assert spec.target_formats == ("owl",)


def test_cli_plan_diff_updates_baseline(monkeypatch, stub_logger, tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    plan = _planned_fetch("hp", url="https://example.org/new.owl")

    _patch_cli_attr(monkeypatch, "plan_all", lambda specs, *, config=None, since=None: [plan])
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )
    monkeypatch.setattr(cli_module, "write_lockfile", lambda plans, path: path, raising=False)

    exit_code = cli.cli_main(
        ["plan-diff", "hp", "--baseline", str(baseline_path), "--update-baseline"]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Updated baseline" in output
    assert baseline_path.exists()
    baseline_payload = json.loads(baseline_path.read_text())
    assert baseline_payload[0]["url"] == "https://example.org/new.owl"


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

    monkeypatch.setattr(
        cli_module,
        "collect_version_metadata",
        lambda oid: list(metadata),
        raising=False,
    )
    monkeypatch.setattr(
        cli_module,
        "collect_version_metadata",
        lambda oid: list(metadata),
        raising=False,
    )
    monkeypatch.setattr(
        cli_module,
        "collect_version_metadata",
        lambda oid: list(metadata),
        raising=False,
    )
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)

    monkeypatch.setattr(cli.STORAGE, "available_ontologies", lambda: ["hp"])
    monkeypatch.setattr(cli.STORAGE, "available_versions", lambda oid: ["2023", "2024"])
    monkeypatch.setattr(cli.STORAGE, "delete_version", lambda *_, **__: 0)
    monkeypatch.setattr(cli.STORAGE, "set_latest_version", lambda *_, **__: None)

    exit_code = cli.cli_main(["prune", "--keep", "1", "--dry-run", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["total_deleted"] == 1
    assert payload["total_reclaimed_bytes"] == 1024


def test_cli_prune_older_than(monkeypatch, stub_logger, capsys):
    metadata = [
        {
            "version": "2024",
            "path": Path("/tmp/hp/2024"),
            "size": 2048,
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        },
        {
            "version": "2023",
            "path": Path("/tmp/hp/2023"),
            "size": 1024,
            "timestamp": datetime(2023, 1, 1, tzinfo=timezone.utc),
        },
        {
            "version": "2022",
            "path": Path("/tmp/hp/2022"),
            "size": 512,
            "timestamp": datetime(2022, 1, 1, tzinfo=timezone.utc),
        },
    ]

    monkeypatch.setattr(
        cli_module,
        "collect_version_metadata",
        lambda oid: list(metadata),
        raising=False,
    )
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(cli.STORAGE, "available_ontologies", lambda: ["hp"])
    monkeypatch.setattr(cli.STORAGE, "delete_version", lambda *_, **__: 0)
    monkeypatch.setattr(cli.STORAGE, "set_latest_version", lambda *_, **__: None)

    exit_code = cli.cli_main(
        ["prune", "--keep", "1", "--older-than", "2023-06-01", "--dry-run", "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_deleted"] == 2
    assert payload["total_reclaimed_bytes"] == 1024 + 512


def test_cli_prune_updates_latest_marker(monkeypatch, stub_logger):
    metadata = [
        {
            "version": "2024",
            "path": Path("/tmp/hp/2024"),
            "size": 2048,
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        },
        {
            "version": "2023",
            "path": Path("/tmp/hp/2023"),
            "size": 1024,
            "timestamp": datetime(2023, 1, 1, tzinfo=timezone.utc),
        },
    ]

    monkeypatch.setattr(
        cli_module,
        "collect_version_metadata",
        lambda oid: list(metadata),
        raising=False,
    )
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)

    monkeypatch.setattr(cli.STORAGE, "available_ontologies", lambda: ["hp"])
    monkeypatch.setattr(cli.STORAGE, "available_versions", lambda oid: ["2023", "2024"])

    monkeypatch.setattr(cli.STORAGE, "delete_version", lambda *_: 0)

    latest_calls: list[tuple[str, str]] = []

    def _record_latest(ontology_id: str, version: str) -> None:
        latest_calls.append((ontology_id, version))

    monkeypatch.setattr(cli.STORAGE, "set_latest_version", _record_latest)

    exit_code = cli.cli_main(["prune", "--keep", "1"])
    assert exit_code == 0
    assert latest_calls == [("hp", "2024")]


def test_cli_plan_serializes_enriched_metadata(monkeypatch, stub_logger, capsys):
    plan = _planned_fetch("hp")
    plan.metadata = {
        "last_modified": "Tue, 01 Aug 2023 00:00:00 GMT",
        "content_length": 4096,
        "etag": '"abc123"',
    }
    _patch_cli_attr(monkeypatch, "plan_all", lambda specs, *, config=None, since=None: [plan])
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )

    exit_code = cli.cli_main(["plan", "hp", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["last_modified"] == "Tue, 01 Aug 2023 00:00:00 GMT"
    assert payload[0]["content_length"] == 4096
    assert payload[0]["etag"] == '"abc123"'


def test_cli_doctor_reports_diagnostics(monkeypatch, stub_logger, tmp_path, capsys):
    config_dir = tmp_path / "configs"
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "logs"
    ontology_dir = tmp_path / "ontologies"
    for path in (config_dir, cache_dir, log_dir, ontology_dir):
        path.mkdir()

    _patch_cli_attr(monkeypatch, "CONFIG_DIR", config_dir)
    _patch_cli_attr(monkeypatch, "CACHE_DIR", cache_dir)
    _patch_cli_attr(monkeypatch, "LOG_DIR", log_dir)
    _patch_cli_attr(monkeypatch, "LOCAL_ONTOLOGY_DIR", ontology_dir)
    _patch_cli_attr(monkeypatch, "ONTOLOGY_DIR", ontology_dir)

    manifest_dir = ontology_dir / "hp" / "2024-01-01"
    manifest_dir.mkdir(parents=True)
    manifest_payload = {
        "schema_version": "1.0",
        "id": "hp",
        "resolver": "obo",
        "url": "https://example.org/hp.owl",
        "filename": "hp.owl",
        "version": "2024-01-01",
        "license": "CC-BY",
        "status": "fresh",
        "sha256": "abc",
        "downloaded_at": "2024-01-01T00:00:00Z",
        "target_formats": ["owl"],
        "validation": {},
        "artifacts": [],
        "resolver_attempts": [],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest_payload))

    config_yaml = config_dir / "sources.yaml"
    config_yaml.write_text(
        "defaults:\n  http:\n    rate_limits:\n      ols: '10/minute'\n      invalid: bogus\n"
    )

    default_config = _default_config()
    default_config.defaults.http.rate_limits = {"ols": "10/minute"}

    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: default_config)
    )
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)
    _patch_cli_attr(monkeypatch, "validate_manifest_dict", lambda payload, source=None: None)
    _patch_cli_attr(monkeypatch, "get_manifest_schema", lambda: {"type": "object"})
    monkeypatch.setattr(
        cli_module.Draft202012Validator,
        "check_schema",
        lambda schema: None,
        raising=False,
    )

    original_find_spec = cli_module.importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name in {"rdflib", "pronto", "owlready2"}:
            return object()
        if name == "arelle":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(
        cli_module.importlib.util,
        "find_spec",
        _fake_find_spec,
        raising=False,
    )

    monkeypatch.setattr(
        cli_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=10_000_000_000, used=4_000_000_000, free=6_000_000_000),
    )
    monkeypatch.setattr(
        cli_module.shutil,
        "which",
        lambda name: "/usr/bin/robot" if name == "robot" else None,
    )
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(stdout="ROBOT version 1.9", stderr="", returncode=0),
    )

    class _Response:
        def __init__(self, status: int, ok: bool, reason: str = "OK") -> None:
            self.status_code = status
            self.ok = ok
            self.reason = reason

    def _fake_head(url: str, **_kwargs):
        if "ols4" in url:
            return _Response(200, True)
        if "bioontology" in url:
            return _Response(405, False, "Method Not Allowed")
        raise cli_module.requests.RequestException("timeout")

    def _fake_get(url: str, **_kwargs):
        return _Response(200, True)

    monkeypatch.setattr(cli_module.requests, "head", _fake_head, raising=False)
    monkeypatch.setattr(cli_module.requests, "get", _fake_get, raising=False)

    exit_code = cli.cli_main(["doctor", "--json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["directories"]["configs"]["exists"] is True
    assert payload["disk"]["free_gb"] == 6.0
    assert payload["dependencies"]["rdflib"] is True
    assert payload["dependencies"]["arelle"] is False
    assert payload["robot"]["available"] is True
    assert payload["robot"]["version"] == "1.9"
    assert payload["network"]["ols"]["ok"] is True
    assert payload["network"]["bioportal"]["status"] == 200
    assert payload["network"]["bioregistry"]["ok"] is False
    assert payload["rate_limits"]["configured"]["ols"]["value"] == "10/minute"
    assert "invalid" in payload["rate_limits"]["invalid"]
    assert payload["manifest_schema"]["sample"]["valid"] is True


def test_cli_doctor_fix_applies_actions(monkeypatch, stub_logger, tmp_path, capsys):
    config_dir = tmp_path / "configs"
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "logs"
    ontology_dir = tmp_path / "ontologies"

    _patch_cli_attr(monkeypatch, "CONFIG_DIR", config_dir)
    _patch_cli_attr(monkeypatch, "CACHE_DIR", cache_dir)
    _patch_cli_attr(monkeypatch, "LOG_DIR", log_dir)
    _patch_cli_attr(monkeypatch, "LOCAL_ONTOLOGY_DIR", ontology_dir)
    _patch_cli_attr(monkeypatch, "ONTOLOGY_DIR", ontology_dir)

    log_dir.mkdir(parents=True)
    (log_dir / "ontofetch.log").write_text("log")

    monkeypatch.setattr(
        cli_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=10_000_000_000, used=4_000_000_000, free=6_000_000_000),
    )
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        cli_module.requests,
        "head",
        lambda *_, **__: SimpleNamespace(status_code=200, ok=True, reason="OK"),
        raising=False,
    )
    monkeypatch.setattr(
        cli_module.requests,
        "get",
        lambda *_, **__: SimpleNamespace(status_code=200, ok=True, reason="OK"),
        raising=False,
    )
    monkeypatch.setattr(
        cli_module.importlib.util,
        "find_spec",
        lambda name: object(),
        raising=False,
    )
    _patch_cli_attr(monkeypatch, "validate_manifest_dict", lambda payload, source=None: None)
    _patch_cli_attr(monkeypatch, "get_manifest_schema", lambda: {"type": "object"})
    monkeypatch.setattr(
        cli_module.Draft202012Validator,
        "check_schema",
        lambda schema: None,
        raising=False,
    )
    _patch_cli_attr(monkeypatch, "setup_logging", lambda *_, **__: stub_logger)

    exit_code = cli.cli_main(["doctor", "--fix"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Applied fixes" in output
    assert config_dir.exists()
    assert cache_dir.exists()
    assert ontology_dir.exists()
    assert (config_dir / "bioportal_api_key.txt").exists()
    assert (config_dir / "ols_api_token.txt").exists()
    assert (log_dir / "ontofetch.log.1").exists()


def test_cli_plugins_json(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "list_plugins",
        lambda kind: {"demo": f"{kind}.Demo"},
    )
    monkeypatch.setattr(
        ResolvedConfig, "from_defaults", classmethod(lambda cls: _default_config())
    )
    exit_code = cli.cli_main(["plugins", "--kind", "all", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolver"]["demo"]["qualified"] == "resolver.Demo"
    assert payload["validator"]["demo"]["qualified"] == "validator.Demo"


def test_handle_plan_diff_uses_manifest_baseline(monkeypatch):
    args = SimpleNamespace(
        use_manifest=True,
        since=None,
        baseline=Path("ignored.json"),
        spec=None,
        ids=["hp"],
        resolver=None,
        target_formats=None,
        concurrent_plans=None,
        concurrent_downloads=None,
        allowed_hosts=None,
    )
    monkeypatch.setattr(cli_module, "write_lockfile", lambda plans, path: path, raising=False)

    spec = FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])
    monkeypatch.setattr(
        cli_module,
        "_resolve_specs_from_args",
        lambda *_args, **_kwargs: (_default_config(), [spec]),
        raising=False,
    )
    _patch_cli_attr(
        monkeypatch,
        "plan_all",
        lambda specs, *, config=None, since=None: [_planned_fetch("hp")],
    )
    monkeypatch.setattr(
        cli_module,
        "load_latest_manifest",
        lambda oid: {
            "id": oid,
            "resolver": "obo",
            "url": "https://example.org/old.owl",
            "version": "2023-12-31",
            "license": "CC-BY",
            "content_type": "application/rdf+xml",
            "last_modified": "2023-12-31T00:00:00Z",
            "content_length": 42,
        },
        raising=False,
    )
    diff = cli_module._handle_plan_diff(args, _default_config())
    assert diff["baseline"] == "manifests"
    assert diff["modified"], "expected modified entries when manifests differ"
    assert diff["lockfile"].endswith("ontologies.lock.json")
