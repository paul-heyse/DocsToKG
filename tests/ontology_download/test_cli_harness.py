"""Happy-path CLI smoke tests executed against the harness sandbox.

Exercises ``ontofetch`` commands (pull, plan, plugins, config) using the
TestingEnvironment harness that spins up loopback resolvers. Validates content
writing, manifest generation, plugin enumeration, and JSON mode to guard the
golden path for automation and documentation snippets.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.testing import temporary_resolver


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
    with temporary_resolver(resolver_name, resolver):
        exit_code = cli_module.cli_main(
            [
                "pull",
                "hp",
                "--resolver",
                resolver_name,
                "--allowed-hosts",
                _allowed_hosts_arg(ontology_env),
                "--json",
            ]
        )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output[0]["id"] == "hp"
    assert output[0]["status"] in {"fresh", "updated", "cached"}


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
    assert output[0]["id"] == "hp"
    assert output[0]["status"] in {"fresh", "updated", "cached"}


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


def test_cli_pull_accepts_tilde_spec_path(monkeypatch, ontology_env, tmp_path, capsys):
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
    monkeypatch.setenv("HOME", str(home_dir))
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
