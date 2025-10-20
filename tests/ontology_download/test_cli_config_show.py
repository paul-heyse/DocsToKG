# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli_config_show",
#   "purpose": "CLI coverage for the ``ontofetch config show`` subcommand.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""CLI coverage for the ``ontofetch config show`` subcommand."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.testing import TestingEnvironment


def _write_config(path: Path, *, secret: str = "hunter2") -> None:
    payload = textwrap.dedent(
        f"""
        defaults:
          logging:
            level: DEBUG
        ontologies:
          - id: hp
            resolver: obo
            target_formats: [ttl]
            extras:
              api_key: {secret}
        """
    ).strip()
    path.write_text(payload, encoding="utf-8")


def test_config_show_json_redacts_credentials(tmp_path, capsys) -> None:
    """JSON mode should mask credential-like keys by default."""

    with TestingEnvironment():
        config_path = tmp_path / "sources.yaml"
        _write_config(config_path, secret="supersecret")

        exit_code = cli_module.cli_main(
            ["config", "show", "--spec", str(config_path), "--json"]
        )

    assert exit_code == 0
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert payload["path"] == str(config_path.resolve())
    assert payload["source"] == "file"
    assert payload["config"]["defaults"]["logging"]["level"] == "DEBUG"
    extras = payload["config"]["specs"][0]["extras"]
    assert extras["api_key"] == "***"
    assert payload["redacted"] is True


def test_config_show_plaintext_allows_disabling_redaction(tmp_path, capsys) -> None:
    """Operators should be able to inspect secrets when explicitly requested."""

    with TestingEnvironment():
        config_path = tmp_path / "sources.yaml"
        _write_config(config_path, secret="letmein")

        exit_code = cli_module.cli_main(
            [
                "config",
                "show",
                "--spec",
                str(config_path),
                "--no-redact-secrets",
            ]
        )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Configuration source" in captured.out
    assert "api_key: letmein" in captured.out


def test_config_show_defaults_snapshot(capsys) -> None:
    """The ``--defaults`` flag should surface the baked-in configuration."""

    with TestingEnvironment():
        exit_code = cli_module.cli_main(["config", "show", "--defaults", "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["source"] == "defaults"
    assert output["path"] is None
    assert output["spec_count"] == 0


def test_config_show_missing_file(tmp_path, capsys) -> None:
    """Referencing a missing configuration file should yield a non-zero exit."""

    missing = tmp_path / "missing.yaml"
    exit_code = cli_module.cli_main(["config", "show", "--spec", str(missing), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Configuration file not found" in captured.err
