"""
Ontology Download CLI Tests

This module exercises the ontology download command line interface to
ensure pull, validate, and configuration commands emit structured output
for both human-readable and JSON modes.

Key Scenarios:
- Fetches ontologies via `pull` with table and JSON output formatting
- Validates manifests using optional backends such as RDFLib
- Checks configuration validation including missing file error handling

Dependencies:
- pytest: Monkeypatching and fixtures
- DocsToKG.OntologyDownload.cli: CLI entry point under test

Usage:
    pytest tests/ontology_download/test_cli.py
"""

import json
import shutil
from datetime import datetime, timezone
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import cli
from DocsToKG.OntologyDownload.config import DefaultsConfig, ResolvedConfig
from DocsToKG.OntologyDownload.core import (
    FetchResult,
    FetchSpec,
    MANIFEST_SCHEMA_VERSION,
    PlannedFetch,
    ResolverCandidate,
)
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


class _StubStorage:
    def __init__(self, root: Path):
        self.root = root
        self.latest: Dict[str, str] = {}

    def available_versions(self, ontology_id: str):
        target = self.root / ontology_id
        if not target.exists():
            return []
        return sorted([entry.name for entry in target.iterdir() if entry.is_dir()])

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        path = self.root / ontology_id / version
        path.mkdir(parents=True, exist_ok=True)
        return path

    def available_ontologies(self):
        if not self.root.exists():
            return []
        return sorted([entry.name for entry in self.root.iterdir() if entry.is_dir()])

    def version_path(self, ontology_id: str, version: str) -> Path:
        return self.ensure_local_version(ontology_id, version)

    def delete_version(self, ontology_id: str, version: str) -> int:
        path = self.root / ontology_id / version
        if not path.exists():
            return 0
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        for entry in sorted(path.rglob("*"), reverse=True):
            if entry.is_file():
                entry.unlink()
        for entry in sorted(path.rglob("*"), reverse=True):
            if entry.is_dir():
                entry.rmdir()
        path.rmdir()
        return size

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        self.latest[ontology_id] = version


class _FakeResponse:
    def __init__(self, headers: Dict[str, str], status_code: int = 200):
        self.headers = headers
        self.status_code = status_code
        self.ok = status_code < 400

    def close(self):
        pass


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
    def _defaults_factory():
        defaults = DefaultsConfig()
        defaults.http.allowed_hosts = ["example.org"]
        return ResolvedConfig(defaults=defaults, specs=())

    monkeypatch.setattr(cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: _defaults_factory()))
    assert cli.main(["pull", "hp", "--json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload[0]["id"] == "hp"


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
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    assert cli.main(["pull", "hp"]) == 0
    output = capsys.readouterr().out
    lines = output.splitlines()
    assert "resolver" in lines[0]
    assert any("hp | obo" in line for line in lines)


def test_cli_pull_outputs_include_expected_fields(monkeypatch, stub_logger, tmp_path, capsys):
    def _make_result(ontology_id: str, resolver: str, status: str) -> FetchResult:
        local_path = tmp_path / f"{ontology_id}.owl"
        local_path.write_text("data")
        manifest_path = tmp_path / f"{ontology_id}-manifest.json"
        manifest_path.write_text("{}")
        return FetchResult(
            spec=FetchSpec(id=ontology_id, resolver=resolver, extras={}, target_formats=["owl"]),
            local_path=local_path,
            status=status,
            sha256=f"hash-{ontology_id}",
            manifest_path=manifest_path,
            artifacts=[str(local_path)],
        )

    results = [
        _make_result("hp", "obo", "fresh"),
        _make_result("chebi", "lov", "cached"),
    ]

    monkeypatch.setattr(cli, "fetch_all", lambda specs, config, force: list(results))
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    assert cli.main(["pull", "hp", "chebi", "--json"]) == 0
    json_output = json.loads(capsys.readouterr().out)
    assert json_output[0]["id"] == "hp"
    assert json_output[0]["manifest"].endswith("hp-manifest.json")
    assert json_output[1]["resolver"] == "lov"
    assert json_output[1]["status"] == "cached"
    assert json_output[1]["artifacts"] == [str(results[1].local_path)]

    assert cli.main(["pull", "hp", "chebi"]) == 0
    table_output = capsys.readouterr().out
    lines = table_output.splitlines()
    assert "resolver" in lines[0]
    assert "status" in lines[0]
    assert "hp | obo | fresh" in table_output
    assert "chebi | lov | cached" in table_output


def test_cli_pull_dry_run(monkeypatch, stub_logger, capsys):
    plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint=None,
        version="2024",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )
    planned = PlannedFetch(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        resolver="obo",
        plan=plan,
        candidates=(ResolverCandidate(resolver="obo", plan=plan),),
    )

    monkeypatch.setattr(
        cli,
        "plan_all",
        lambda specs, *, config=None, since=None: [planned],
    )
    monkeypatch.setattr(
        cli, "fetch_all", lambda *_, **__: pytest.fail("dry-run should not download")
    )
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    assert cli.main(["pull", "hp", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "hp" in output
    assert "application/rdf+xml" in output


def test_cli_pull_applies_concurrency_and_hosts(monkeypatch, stub_logger):
    captured = {}

    def _fake_fetch_all(specs, config, force):
        captured["downloads"] = config.defaults.http.concurrent_downloads
        captured["hosts"] = list(config.defaults.http.allowed_hosts or [])
        return []

    monkeypatch.setattr(cli, "fetch_all", _fake_fetch_all)
    monkeypatch.setattr(
        cli,
        "plan_all",
        lambda specs, *, config=None, since=None: [],
    )
    monkeypatch.setattr(cli, "plan_all", lambda specs, config: [])
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    exit_code = cli.main(
        [
            "pull",
            "hp",
            "--concurrent-downloads",
            "3",
            "--allowed-hosts",
            "example.org,*.example.com",
        ]
    )

    assert exit_code == 0
    assert captured["downloads"] == 3
    assert set(captured["hosts"]) == {"example.org", "*.example.com"}


def test_cli_plan_json_output(monkeypatch, stub_logger, capsys):
    plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint=None,
        version="2024",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )
    planned = PlannedFetch(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        resolver="obo",
        plan=plan,
        candidates=(ResolverCandidate(resolver="obo", plan=plan),),
    )

    monkeypatch.setattr(
        cli,
        "plan_all",
        lambda specs, *, config=None, since=None: [planned],
    )
    monkeypatch.setattr(cli, "plan_all", lambda specs, config: [planned])
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.requests,
        "head",
        lambda url, headers=None, timeout=None, allow_redirects=None: _FakeResponse(
            {"Last-Modified": "Wed, 01 May 2024 00:00:00 GMT", "Content-Length": "1234"}
        ),
        raising=False,
    )

    assert cli.main(["plan", "hp", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["id"] == "hp"
    assert payload[0]["url"] == "https://example.org/hp.owl"
    assert payload[0]["candidates"][0]["resolver"] == "obo"


def test_cli_plan_applies_concurrency_overrides(monkeypatch, stub_logger):
    captured = {}

    def _fake_plan_all(specs, config):
        captured["plans"] = config.defaults.http.concurrent_plans
        captured["downloads"] = config.defaults.http.concurrent_downloads
        captured["hosts"] = list(config.defaults.http.allowed_hosts or [])
        return []

    monkeypatch.setattr(cli, "plan_all", _fake_plan_all)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    exit_code = cli.main(
        [
            "plan",
            "hp",
            "--concurrent-plans",
            "4",
            "--concurrent-downloads",
            "2",
            "--allowed-hosts",
            "mirror.example.org",
        ]
    )

    assert exit_code == 0
    assert captured["plans"] == 4
    assert captured["downloads"] == 2
    assert captured["hosts"] == ["mirror.example.org"]


def test_cli_plan_since_filters(monkeypatch, stub_logger, capsys):
    recent_plan = FetchPlan(
        url="https://example.org/new.owl",
        headers={},
        filename_hint=None,
        version="2024",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )
    old_plan = FetchPlan(
        url="https://example.org/old.owl",
        headers={},
        filename_hint=None,
        version="2023",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )
    plans = [
        PlannedFetch(
            spec=FetchSpec(id="new", resolver="obo", extras={}, target_formats=["owl"]),
            resolver="obo",
            plan=recent_plan,
            candidates=(ResolverCandidate(resolver="obo", plan=recent_plan),),
            metadata={"last_modified": "2024-05-02T00:00:00Z"},
        ),
        PlannedFetch(
            spec=FetchSpec(id="old", resolver="obo", extras={}, target_formats=["owl"]),
            resolver="obo",
            plan=old_plan,
            candidates=(ResolverCandidate(resolver="obo", plan=old_plan),),
            metadata={"last_modified": "2023-01-01T00:00:00Z"},
        ),
    ]

    monkeypatch.setattr(cli, "plan_all", lambda specs, config: list(plans))
    monkeypatch.setattr(cli, "_collect_plan_metadata", lambda plans, config: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    assert cli.main(["plan", "new", "old", "--since", "2024-05-01", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["id"] == "new"


def test_cli_plan_diff_reports_changes_json(monkeypatch, stub_logger, tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            [
                {
                    "id": "hp",
                    "resolver": "obo",
                    "url": "https://example.org/old.owl",
                    "version": "2023",
                    "license": "CC-BY",
                    "media_type": "application/rdf+xml",
                    "service": "obo",
                    "last_modified": "2023-01-01T00:00:00Z",
                    "content_length": 128,
                }
            ]
        )
    )

    current_plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint=None,
        version="2024",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )
    planned = PlannedFetch(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        resolver="obo",
        plan=current_plan,
        candidates=(ResolverCandidate(resolver="obo", plan=current_plan),),
        metadata={"last_modified": "2024-01-01T00:00:00Z", "content_length": 256},
    )

    monkeypatch.setattr(cli, "plan_all", lambda specs, config, since=None: [planned])
    monkeypatch.setattr(cli, "_collect_plan_metadata", lambda plans, config: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    exit_code = cli.main(
        ["plan", "diff", "hp", "--baseline", str(baseline_path), "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["added"] == []
    assert payload["removed"] == []
    assert payload["modified"][0]["id"] == "hp"
    assert "version" in payload["modified"][0]["changes"]


def test_cli_plan_diff_text_summary(monkeypatch, stub_logger, tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            [
                {
                    "id": "hp",
                    "resolver": "obo",
                    "url": "https://example.org/old.owl",
                    "version": "2023-01-01",
                    "license": "CC-BY",
                    "media_type": "application/rdf+xml",
                    "service": "obo",
                    "content_length": 1024,
                }
            ]
        )
    )

    current_plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint=None,
        version="2024-05-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )
    planned = PlannedFetch(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        resolver="obo",
        plan=current_plan,
        candidates=(ResolverCandidate(resolver="obo", plan=current_plan),),
        metadata={"last_modified": "2024-05-01T00:00:00Z", "content_length": 2048},
    )

    monkeypatch.setattr(cli, "plan_all", lambda specs, config, since=None: [planned])
    monkeypatch.setattr(cli, "_collect_plan_metadata", lambda plans, config: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    assert (
        cli.main(
            ["plan", "diff", "hp", "--baseline", str(baseline_path)]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "~ hp" in output
    assert "https://example.org/hp.owl" in output

    assert (
        cli.main(["plan", "diff", "--baseline", str(baseline_path), "--json"]) == 0
    )
    diff = json.loads(capsys.readouterr().out)
    assert len(diff["modified"]) == 1
    assert diff["modified"][0]["id"] == "hp"
    assert "version" in diff["modified"][0]["changes"]
    exit_code = cli.main(
        [
            "plan",
            "hp",
            "--concurrent-plans",
            "4",
            "--concurrent-downloads",
            "2",
            "--allowed-hosts",
            "mirror.example.org",
        ]
    )

    assert exit_code == 0
    assert captured["plans"] == 4
    assert captured["downloads"] == 2
    assert captured["hosts"] == ["mirror.example.org"]


def test_cli_plan_since_argument(monkeypatch, stub_logger):
    captured = {}

    def _fake_plan_all(specs, config, since=None):
        captured["since"] = since
        return []

    monkeypatch.setattr(cli, "plan_all", _fake_plan_all)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    assert cli.main(["plan", "hp", "--since", "2024-02-01"]) == 0
    assert captured["since"].isoformat().startswith("2024-02-01")

    monkeypatch.setattr(
        cli,
        "plan_all",
        lambda specs, *, config=None, since=None: [planned],
    )
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    exit_code = cli.main(["plan", "diff", "--baseline", str(baseline_path)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "~ hp" in output


def test_cli_plan_applies_concurrency_overrides(monkeypatch, stub_logger):
    captured = {}

    def _fake_plan_all(specs, *, config=None, since=None):
        captured["plans"] = config.defaults.http.concurrent_plans
        captured["downloads"] = config.defaults.http.concurrent_downloads
        captured["hosts"] = list(config.defaults.http.allowed_hosts or [])
        captured["since"] = since
        return []

    monkeypatch.setattr(cli, "plan_all", _fake_plan_all)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    exit_code = cli.main(
        [
            "plan",
            "hp",
            "--concurrent-plans",
            "4",
            "--concurrent-downloads",
            "2",
            "--allowed-hosts",
            "mirror.example.org",
            "--since",
            "2024-02-01",
        ]
    )

    assert exit_code == 0
    assert captured["plans"] == 4
    assert captured["downloads"] == 2
    assert captured["hosts"] == ["mirror.example.org"]
    assert captured["since"] == datetime(2024, 2, 1, tzinfo=timezone.utc)


def test_cli_prune_dry_run(monkeypatch, stub_logger, tmp_path, capsys):
    root = tmp_path / "ontologies"

    def _create_version(version: str, size: int) -> None:
        safe_id = cli.sanitize_filename("hp")
        safe_version = cli.sanitize_filename(version)
        version_dir = root / safe_id / safe_version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "manifest.json").write_text(
            json.dumps({"downloaded_at": f"{version}-01-01T00:00:00Z"})
        )
        (version_dir / "ontology.owl").write_bytes(b"x" * size)

    _create_version("2023", 100)
    _create_version("2024", 200)

    class _StubStorage:
        def __init__(self, base: Path):
            self.base = base

        def available_ontologies(self):
            if not self.base.exists():
                return []
            return [entry.name for entry in self.base.iterdir() if entry.is_dir()]

        def available_versions(self, ontology_id: str):
            directory = self.base / cli.sanitize_filename(ontology_id)
            if not directory.exists():
                return []
            return sorted([entry.name for entry in directory.iterdir() if entry.is_dir()])

        def delete_version(self, ontology_id: str, version: str) -> int:
            directory = (
                self.base
                / cli.sanitize_filename(ontology_id)
                / cli.sanitize_filename(version)
            )
            total = 0
            if directory.exists():
                for path in directory.rglob("*"):
                    if path.is_file():
                        total += path.stat().st_size
            shutil.rmtree(directory, ignore_errors=True)
            return total

    monkeypatch.setattr(cli, "LOCAL_ONTOLOGY_DIR", root)
    monkeypatch.setattr(cli, "STORAGE", _StubStorage(root))
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    exit_code = cli.main(["prune", "--keep", "1", "--ids", "hp", "--dry-run"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "[DRY-RUN] hp version 2023" in output
    assert "Reclaimed" in output
    assert (root / "hp" / "2023").exists()


def test_cli_prune_executes(monkeypatch, stub_logger, tmp_path, capsys):
    root = tmp_path / "ontologies"

    def _create_version(version: str, size: int) -> None:
        safe_id = cli.sanitize_filename("hp")
        safe_version = cli.sanitize_filename(version)
        version_dir = root / safe_id / safe_version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "manifest.json").write_text(
            json.dumps({"downloaded_at": f"{version}-01-01T00:00:00Z"})
        )
        (version_dir / "ontology.owl").write_bytes(b"x" * size)

    _create_version("2023", 50)
    _create_version("2024", 75)
    latest_link = root / cli.sanitize_filename("hp") / "latest"
    latest_link.parent.mkdir(parents=True, exist_ok=True)
    latest_link.symlink_to(
        root / cli.sanitize_filename("hp") / cli.sanitize_filename("2023"),
        target_is_directory=True,
    )

    class _StubStorage:
        def __init__(self, base: Path):
            self.base = base

        def available_ontologies(self):
            return [entry.name for entry in self.base.iterdir() if entry.is_dir()]

        def available_versions(self, ontology_id: str):
            directory = self.base / cli.sanitize_filename(ontology_id)
            return sorted([entry.name for entry in directory.iterdir() if entry.is_dir()])

        def delete_version(self, ontology_id: str, version: str) -> int:
            directory = (
                self.base
                / cli.sanitize_filename(ontology_id)
                / cli.sanitize_filename(version)
            )
            total = 0
            for path in directory.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
            shutil.rmtree(directory, ignore_errors=True)
            return total

    monkeypatch.setattr(cli, "LOCAL_ONTOLOGY_DIR", root)
    monkeypatch.setattr(cli, "STORAGE", _StubStorage(root))
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    exit_code = cli.main(["prune", "--keep", "1", "--ids", "hp"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Deleted 1 versions for hp" in output
    assert not (root / "hp" / "2023").exists()
    retained_dir = root / "hp" / "2024"
    assert retained_dir.exists()
    if latest_link.is_symlink():
        assert latest_link.resolve() == retained_dir.resolve()
    else:
        assert latest_link.read_text().strip() == str(retained_dir)

def test_cli_validate_json_output(monkeypatch, stub_logger, tmp_path, capsys):
    manifest_dir = tmp_path / "hp" / "2024"
    original_dir = manifest_dir / "original"
    validation_dir = manifest_dir / "validation"
    normalized_dir = manifest_dir / "normalized"
    original_dir.mkdir(parents=True)
    validation_dir.mkdir()
    normalized_dir.mkdir()
    original_file = original_dir / "hp.owl"
    original_file.write_text("content")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "id": "hp",
        "resolver": "obo",
        "url": "https://example.org/hp.owl",
        "filename": "hp.owl",
        "version": "2024",
        "license": "CC-BY",
        "status": "fresh",
        "sha256": "abc",
        "normalized_sha256": "def",
        "fingerprint": "f" * 64,
        "etag": "tag",
        "last_modified": "today",
        "downloaded_at": "now",
        "target_formats": ["owl"],
        "validation": {},
        "artifacts": [str(original_file)],
        "resolver_attempts": [],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli,
        "run_validators",
        lambda requests, logger: {
            req.name: type("R", (), {"to_dict": lambda self: {"ok": True}})() for req in requests
        },
    )
    monkeypatch.setattr(cli, "STORAGE", _StubStorage(tmp_path))
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    assert cli.main(["validate", "hp", "2024", "--json", "--rdflib"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert "rdflib" in payload


def test_cli_validate_table_output(monkeypatch, stub_logger, tmp_path, capsys):
    manifest_dir = tmp_path / "hp" / "2024"
    original_dir = manifest_dir / "original"
    validation_dir = manifest_dir / "validation"
    normalized_dir = manifest_dir / "normalized"
    original_dir.mkdir(parents=True)
    validation_dir.mkdir()
    normalized_dir.mkdir()
    original_file = original_dir / "hp.owl"
    original_file.write_text("content")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "id": "hp",
        "resolver": "obo",
        "url": "https://example.org/hp.owl",
        "filename": "hp.owl",
        "version": "2024",
        "license": "CC-BY",
        "status": "fresh",
        "sha256": "abc",
        "normalized_sha256": "def",
        "fingerprint": "f" * 64,
        "etag": "tag",
        "last_modified": "today",
        "downloaded_at": "now",
        "target_formats": ["owl"],
        "validation": {},
        "artifacts": [str(original_file)],
        "resolver_attempts": [],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    class _Result:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return self._payload

    monkeypatch.setattr(
        cli,
        "run_validators",
        lambda requests, logger: {
            "rdflib": _Result({"ok": True, "details": {"triples": 1234}}),
        },
    )
    monkeypatch.setattr(cli, "STORAGE", _StubStorage(tmp_path))
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    assert cli.main(["validate", "hp", "2024", "--rdflib"]) == 0
    output = capsys.readouterr().out
    lines = output.splitlines()
    assert "validator" in lines[0]
    assert any("rdflib" in line and "triples=1234" in line for line in lines)


def test_cli_config_validate(monkeypatch, stub_logger, tmp_path, capsys):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text("ontologies:\n  - id: hp\n    resolver: obo\n")
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    assert cli.main(["config", "validate", "--spec", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Configuration passed" in output


def test_cli_config_validate_json(monkeypatch, stub_logger, tmp_path, capsys):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text("ontologies:\n  - id: hp\n    resolver: obo\n")
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    assert cli.main(["config", "validate", "--spec", str(config_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"]


def test_cli_config_validate_missing_file(monkeypatch, stub_logger, tmp_path):
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["config", "validate", "--spec", str(missing)])
    assert exc_info.value.code == 2


def _create_version(root: Path, ontology: str, version: str, downloaded_at: str, size: int) -> None:
    path = root / ontology / version
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({"downloaded_at": downloaded_at}))
    (path / "data.bin").write_bytes(b"x" * size)


def test_cli_prune_deletes_old_versions(monkeypatch, stub_logger, tmp_path, capsys):
    storage_root = tmp_path / "store"
    _create_version(storage_root, "hp", "2024-05-01", "2024-05-01T00:00:00Z", 10)
    _create_version(storage_root, "hp", "2024-04-01", "2024-04-01T00:00:00Z", 5)
    _create_version(storage_root, "hp", "2023-12-01", "2023-12-01T00:00:00Z", 2)

    storage = _StubStorage(storage_root)
    monkeypatch.setattr(cli, "STORAGE", storage)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    assert (
        cli.main(["prune", "--keep", "1", "--ids", "hp", "--json"]) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["deleted_versions"] == 2
    assert storage.available_versions("hp") == ["2024-05-01"]
    assert storage.latest["hp"] == "2024-05-01"


def test_cli_prune_dry_run_keeps_versions(monkeypatch, stub_logger, tmp_path, capsys):
    storage_root = tmp_path / "store"
    _create_version(storage_root, "hp", "2024-05-01", "2024-05-01T00:00:00Z", 10)
    _create_version(storage_root, "hp", "2024-04-01", "2024-04-01T00:00:00Z", 5)

    storage = _StubStorage(storage_root)
    monkeypatch.setattr(cli, "STORAGE", storage)
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    assert (
        cli.main(["prune", "--keep", "1", "--ids", "hp", "--dry-run"]) == 0
    )
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert set(storage.available_versions("hp")) == {"2024-05-01", "2024-04-01"}


def test_cli_init_writes_example(monkeypatch, stub_logger, tmp_path, capsys):
    target = tmp_path / "sources.yaml"
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    exit_code = cli.main(["init", str(target)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote example configuration" in output
    contents = target.read_text()
    assert "defaults:" in contents
    assert "ontologies:" in contents


def test_cli_init_refuses_overwrite(monkeypatch, stub_logger, tmp_path, capsys):
    target = tmp_path / "sources.yaml"
    target.write_text("existing")
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    exit_code = cli.main(["init", str(target)])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Refusing to overwrite" in stderr


def test_cli_doctor_json(monkeypatch, stub_logger, tmp_path, capsys):
    base = tmp_path
    config_dir = base / "config"
    cache_dir = base / "cache"
    log_dir = base / "logs"
    onto_dir = base / "ontologies"
    for directory in (config_dir, cache_dir, log_dir, onto_dir):
        directory.mkdir(parents=True)

    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(cli, "LOG_DIR", log_dir)
    monkeypatch.setattr(cli, "LOCAL_ONTOLOGY_DIR", onto_dir)

    monkeypatch.setattr(
        cli.requests,
        "head",
        lambda url, timeout, allow_redirects=True: SimpleNamespace(
            ok=True, status_code=200, reason="OK"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli.requests,
        "get",
        lambda url, timeout, allow_redirects=True: SimpleNamespace(
            ok=True, status_code=200, reason="OK"
        ),
        raising=False,
    )

    original_find_spec = cli.importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name in {"rdflib", "pronto", "owlready2", "arelle"}:
            return ModuleSpec(name, loader=None)
        return original_find_spec(name)

    monkeypatch.setattr(cli.importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/robot" if name == "robot" else None)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="ROBOT version 1.9.0", stderr="", returncode=0),
    )
    monkeypatch.setattr(
        cli.requests,
        "head",
        lambda url, timeout, allow_redirects=True: SimpleNamespace(
            ok=True, status_code=200, reason="OK"
        ),
    )
    monkeypatch.setattr(
        cli.requests,
        "get",
        lambda url, timeout, allow_redirects=True: SimpleNamespace(
            ok=True, status_code=200, reason="OK"
        ),
    )

    original_find_spec = cli.importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name in {"rdflib", "pronto", "owlready2", "arelle"}:
            return ModuleSpec(name, loader=None)
        return original_find_spec(name)

    monkeypatch.setattr(cli.importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/robot" if name == "robot" else None)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="ROBOT version 1.9.0", stderr="", returncode=0),
    )
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)

    assert cli.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "directories" in payload
    assert "dependencies" in payload
    assert "robot" in payload
    assert "network" in payload
    assert "rate_limits" in payload


def test_doctor_reports_invalid_rate_limit(monkeypatch, tmp_path):
    base = tmp_path
    config_dir = base / "config"
    cache_dir = base / "cache"
    log_dir = base / "logs"
    onto_dir = base / "ontologies"
    for directory in (config_dir, cache_dir, log_dir, onto_dir):
        directory.mkdir(parents=True)

    sources = config_dir / "sources.yaml"
    sources.write_text(
        "defaults:\n  http:\n    rate_limits:\n      ols: invalid\n"
    )

    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(cli, "LOG_DIR", log_dir)
    monkeypatch.setattr(cli, "LOCAL_ONTOLOGY_DIR", onto_dir)
    monkeypatch.setattr(
        cli.requests,
        "head",
        lambda url, timeout, allow_redirects=True: SimpleNamespace(
            ok=True, status_code=200, reason="OK"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli.requests,
        "get",
        lambda url, timeout, allow_redirects=True: SimpleNamespace(
            ok=True, status_code=200, reason="OK"
        ),
        raising=False,
    )
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        cli.importlib.util,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None),
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="", returncode=1),
    )

    report = cli._doctor_report()
    assert "invalid" in report["rate_limits"]
    assert report["rate_limits"]["invalid"]["ols"] == "invalid"


def test_cli_plan_diff_output(monkeypatch, stub_logger, tmp_path, capsys):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir(parents=True)
    baseline = plans_dir / "latest.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "id": "hp",
                    "url": "https://old.example.org/hp.owl",
                    "version": "2023-01-01",
                    "license": "CC-BY",
                    "media_type": "application/rdf+xml",
                }
            ]
        )
    )

    plan = FetchPlan(
        url="https://new.example.org/hp.owl",
        headers={},
        filename_hint=None,
        version="2024-02-01",
        license="CC0",
        media_type="application/rdf+xml",
        service="obo",
    )
    planned = PlannedFetch(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        resolver="obo",
        plan=plan,
        candidates=(ResolverCandidate(resolver="obo", plan=plan),),
        last_modified="Wed, 01 Feb 2024 12:00:00 GMT",
    )

    monkeypatch.setattr(cli, "plan_all", lambda specs, config, since=None: [planned])
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    assert cli.main(["plan", "diff", "hp"]) == 0
    output = capsys.readouterr().out
    assert "~ hp" in output
    assert "https://new.example.org" in output


def test_cli_plan_diff_json(monkeypatch, stub_logger, tmp_path, capsys):
    baseline = tmp_path / "plans" / "latest.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps([]))

    plan = FetchPlan(
        url="https://example.org/efo.owl",
        headers={},
        filename_hint=None,
        version="2024-03-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="ols",
    )
    planned = PlannedFetch(
        spec=FetchSpec(id="efo", resolver="ols", extras={}, target_formats=["owl"]),
        resolver="ols",
        plan=plan,
        candidates=(ResolverCandidate(resolver="ols", plan=plan),),
        last_modified=None,
    )

    monkeypatch.setattr(cli, "plan_all", lambda specs, config, since=None: [planned])
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: stub_logger)
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    assert cli.main(["plan", "diff", "efo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["added"][0]["id"] == "efo"


def _write_version(base: Path, ontology: str, version: str, timestamp: str, size: int = 1024) -> None:
    version_dir = base / ontology / version
    (version_dir / "original").mkdir(parents=True, exist_ok=True)
    payload = {"downloaded_at": timestamp, "id": ontology, "version": version}
    (version_dir / "manifest.json").write_text(json.dumps(payload))
    data_file = version_dir / "original" / f"{ontology}.owl"
    data_file.write_bytes(b"x" * size)


def test_cli_prune_dry_run(monkeypatch, tmp_path, capsys):
    storage = _StubStorage(tmp_path)
    _write_version(tmp_path, "hp", "2024-01-01", "2024-01-01T00:00:00Z")
    _write_version(tmp_path, "hp", "2023-01-01", "2023-01-01T00:00:00Z")

    monkeypatch.setattr(cli, "STORAGE", storage)
    monkeypatch.setattr(cli, "LOCAL_ONTOLOGY_DIR", tmp_path)

    assert cli.main(["prune", "--keep", "1", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Dry-run" in output
    assert (tmp_path / "hp" / "2023-01-01").exists()


def test_cli_prune_executes(monkeypatch, tmp_path, capsys):
    storage = _StubStorage(tmp_path)
    _write_version(tmp_path, "hp", "2024-01-01", "2024-01-01T00:00:00Z", size=2048)
    _write_version(tmp_path, "hp", "2023-01-01", "2023-01-01T00:00:00Z", size=1024)
    latest_link = tmp_path / "hp" / "latest"
    latest_link.parent.mkdir(parents=True, exist_ok=True)
    latest_link.symlink_to(tmp_path / "hp" / "2023-01-01")

    monkeypatch.setattr(cli, "STORAGE", storage)
    monkeypatch.setattr(cli, "LOCAL_ONTOLOGY_DIR", tmp_path)

    assert cli.main(["prune", "--keep", "1"]) == 0
    output = capsys.readouterr().out
    assert "Pruned" in output
    assert not (tmp_path / "hp" / "2023-01-01").exists()
    assert latest_link.readlink() == tmp_path / "hp" / "2024-01-01"


def test_cli_prune_ids_filter(monkeypatch, tmp_path, capsys):
    storage = _StubStorage(tmp_path)
    _write_version(tmp_path, "hp", "2024-01-01", "2024-01-01T00:00:00Z")
    _write_version(tmp_path, "hp", "2023-01-01", "2023-01-01T00:00:00Z")
    _write_version(tmp_path, "chebi", "2024-01-01", "2024-01-01T00:00:00Z")
    _write_version(tmp_path, "chebi", "2023-01-01", "2023-01-01T00:00:00Z")

    monkeypatch.setattr(cli, "STORAGE", storage)
    monkeypatch.setattr(cli, "LOCAL_ONTOLOGY_DIR", tmp_path)

    assert cli.main(["prune", "--keep", "1", "--ids", "hp"]) == 0
    capsys.readouterr()
    assert not (tmp_path / "hp" / "2023-01-01").exists()
    assert (tmp_path / "chebi" / "2023-01-01").exists()
