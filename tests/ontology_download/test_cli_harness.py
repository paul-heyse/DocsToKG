# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli_harness",
#   "purpose": "Happy-path CLI smoke tests executed against the harness sandbox.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Happy-path CLI smoke tests executed against the harness sandbox.

Exercises ``ontofetch`` commands (pull, plan, plugins, config) using the
TestingEnvironment harness that spins up loopback resolvers. Validates content
writing, manifest generation, plugin enumeration, and JSON mode to guard the
golden path for automation and documentation snippets."""

from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.testing import TestingEnvironment, temporary_resolver


def _allowed_hosts_arg(ontology_env) -> str:
    allowed = ontology_env.build_download_config().allowed_hosts or []
    return ",".join(allowed)


def _static_resolver_for(env, *, name: str, filename: str):
    fixture_url = env.register_fixture(
        filename,
        (f"{name} fixture\n").encode("utf-8"),
        media_type="application/rdf+xml",
        repeats=2,
    )
    resolver_name = f"{name}-cli"
    resolver = env.static_resolver(
        name=resolver_name,
        fixture_url=fixture_url,
        filename=filename,
        media_type="application/rdf+xml",
        service="obo",
    )
    return resolver_name, resolver


def test_cli_pull_json_uses_harness(ontology_env, capsys):
    """`ontofetch pull` should download via the temporary resolver and emit JSON."""

    resolver_name, resolver = _static_resolver_for(ontology_env, name="hp", filename="hp.owl")
    allowed_hosts = _allowed_hosts_arg(ontology_env)
    with temporary_resolver(resolver_name, resolver):
        exit_code = cli_module.cli_main(
            [
                "pull",
                "hp",
                "--resolver",
                resolver_name,
                "--allowed-hosts",
                allowed_hosts,
                "--json",
            ]
        )

        assert exit_code == 0
        first_payload = json.loads(capsys.readouterr().out)
        first_record = first_payload[0]
        assert first_record["id"] == "hp"
        assert first_record["status"] in {"fresh", "updated", "cached"}
        assert first_record["content_type"] == "application/rdf+xml"
        assert first_record["content_length"] == len("hp fixture\n")
        assert isinstance(first_record.get("cache_status"), dict)

        # run again to validate cached metadata is exposed
        exit_code = cli_module.cli_main(
            [
                "pull",
                "hp",
                "--resolver",
                resolver_name,
                "--allowed-hosts",
                allowed_hosts,
                "--json",
            ]
        )

    assert exit_code == 0
    second_payload = json.loads(capsys.readouterr().out)
    cached_record = second_payload[0]
    cache_meta = cached_record.get("cache_status") or {}
    assert isinstance(cache_meta, dict)
    assert "from_cache" in cache_meta


def test_cli_defaults_to_pull_when_subcommand_missing(ontology_env, capsys):
    """Calling the CLI without a subcommand should behave like ``pull``."""

    resolver_name, resolver = _static_resolver_for(
        ontology_env,
        name="hp-default",
        filename="hp-default.owl",
    )
    with temporary_resolver(resolver_name, resolver):
        exit_code = cli_module.cli_main(
            [
                "hp",  # positional ontology id without explicit subcommand
                "--resolver",
                resolver_name,
                "--allowed-hosts",
                _allowed_hosts_arg(ontology_env),
                "--json",
            ]
        )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    record = output[0]
    assert record["id"] == "hp"
    assert record["status"] in {"fresh", "updated", "cached"}
    for key in ("content_type", "content_length", "etag", "cache_status"):
        assert key in record


def test_cli_pull_dry_run_json(ontology_env, capsys):
    """`ontofetch pull --dry-run` should emit plan metadata as JSON."""

    resolver_name, resolver = _static_resolver_for(ontology_env, name="hp-dry-run", filename="hp.owl")
    with temporary_resolver(resolver_name, resolver):
        exit_code = cli_module.cli_main(
            [
                "pull",
                "hp",
                "--resolver",
                resolver_name,
                "--allowed-hosts",
                _allowed_hosts_arg(ontology_env),
                "--dry-run",
                "--json",
            ]
        )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["spec"]["id"] == "hp"
    assert payload[0]["plan"]["url"].startswith("http://")


def test_cli_plan_json_respects_resolver(ontology_env, capsys, tmp_path):
    """`ontofetch plan` should surface resolver details using the harness environment."""

    resolver_name, resolver = _static_resolver_for(ontology_env, name="go", filename="go.owl")
    lock_path = tmp_path / "ontologies.lock.json"

    with temporary_resolver(resolver_name, resolver):
        exit_code = cli_module.cli_main(
            [
                "plan",
                "go",
                "--resolver",
                resolver_name,
                "--allowed-hosts",
                _allowed_hosts_arg(ontology_env),
                "--json",
                "--lock-output",
                str(lock_path),
            ]
        )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["spec"]["id"] == "go"
    assert payload[0]["plan"]["url"].startswith("http://127.0.0.1")
    assert lock_path.exists()


def test_cli_pull_accepts_tilde_spec_path(ontology_env, tmp_path, capsys):
    """`ontofetch pull` should expand ``~`` when resolving ``--spec`` configuration paths."""

    resolver_name, resolver = _static_resolver_for(ontology_env, name="hp-tilde", filename="hp.owl")
    home_dir = tmp_path / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    config_path = home_dir / "sources.yaml"
    config_payload = textwrap.dedent(
        f"""
        defaults:
          normalize_to: [ttl]
        ontologies:
          - id: hp
            resolver: {resolver_name}
            extras:
              acronym: HP
            normalize_to: [ttl]
        """
    ).strip()
    config_path.write_text(config_payload, encoding="utf-8")

    with mock.patch.dict(os.environ, {"HOME": str(home_dir)}, clear=False):
        tilde_spec = Path("~") / "sources.yaml"
        with temporary_resolver(resolver_name, resolver):
            exit_code = cli_module.cli_main(
                [
                    "pull",
                    "hp",
                    "--spec",
                    str(tilde_spec),
                    "--allowed-hosts",
                    _allowed_hosts_arg(ontology_env),
                    "--json",
                ]
            )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload[0]["id"] == "hp"


def test_cli_config_validate_expands_tilde_path(tmp_path, capsys):
    """`ontofetch config validate` should resolve ``~`` before loading configuration files."""

    home_dir = tmp_path / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    config_path = home_dir / "custom.yaml"
    config_payload = textwrap.dedent(
        """
        defaults:
          normalize_to: [ttl]
        ontologies:
          - id: hp
            resolver: obo
            normalize_to: [ttl]
        """
    ).strip()
    config_path.write_text(config_payload, encoding="utf-8")

    with mock.patch.dict(os.environ, {"HOME": str(home_dir)}, clear=False):
        tilde_spec = Path("~") / "custom.yaml"
        with TestingEnvironment():
            exit_code = cli_module.cli_main(
                [
                    "config",
                    "validate",
                    "--spec",
                    str(tilde_spec),
                ]
            )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Configuration passed" in stdout
    assert str(config_path) in stdout


def test_handle_config_validate_invokes_validate_once(tmp_path):
    """`_handle_config_validate` should normalize once and call `validate_config` a single time."""

    config_path = tmp_path / "custom.yaml"
    config_path.write_text("ontologies: []\n", encoding="utf-8")

    with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=False):
        raw_path = Path("~") / config_path.name
        expected_path = cli_module.normalize_config_path(raw_path)
        fake_config = SimpleNamespace(specs=["hp", "mp"])

        with mock.patch.object(
            cli_module,
            "validate_config",
            autospec=True,
            return_value=fake_config,
        ) as validate_mock:
            result = cli_module._handle_config_validate(raw_path)

    validate_mock.assert_called_once_with(expected_path)
    assert result == {
        "ok": True,
        "ontologies": len(fake_config.specs),
        "path": str(expected_path),
    }


def test_cli_init_expands_tilde_destination(tmp_path, capsys):
    """`ontofetch init` should expand ``~`` destinations before writing files."""

    home_dir = tmp_path / "home"
    home_dir.mkdir(parents=True, exist_ok=True)

    literal_parent = Path.cwd() / "~"
    literal_destination = literal_parent / "temp.yaml"
    if literal_parent.exists():
        shutil.rmtree(literal_parent)

    with mock.patch.dict(os.environ, {"HOME": str(home_dir)}, clear=False):
        tilde_argument = Path("~") / "temp.yaml"

        exit_code = cli_module.cli_main(["init", str(tilde_argument)])
        assert exit_code == 0
        first_output = capsys.readouterr()
        created_path = home_dir / "temp.yaml"
        assert created_path.exists()
        assert "Wrote example configuration" in first_output.out
        assert str(created_path) in first_output.out
        assert not literal_destination.exists()

        second_exit = cli_module.cli_main(["init", str(tilde_argument)])
        assert second_exit == 1
        second_output = capsys.readouterr()
    assert "Error: Refusing to overwrite existing file" in second_output.err
    assert str(created_path) in second_output.err

    if literal_parent.exists():
        shutil.rmtree(literal_parent)
