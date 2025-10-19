# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli_spec_resolution",
#   "purpose": "Regression coverage for CLI spec resolution overrides",
#   "sections": [
#     {
#       "id": "test-pull-spec-retains-configured-fetch-spec",
#       "name": "test_pull_spec_retains_configured_fetch_spec",
#       "anchor": "function-test-pull-spec-retains-configured-fetch-spec",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Regression tests for CLI fetch specification resolution."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import cli as cli_module


def _write_sources_yaml(path: Path) -> Path:
    """Create a minimal ``sources.yaml`` tailored for spec resolution tests."""

    payload = textwrap.dedent(
        """
        defaults:
          prefer_source: [ols]
          normalize_to: [ttl]
        ontologies:
          - id: hp
            resolver: obo
            normalize_to: [owl, obo]
            extras:
              acronym: HP
        """
    ).strip()
    path.write_text(payload, encoding="utf-8")
    return path


def test_pull_spec_retains_configured_fetch_spec(tmp_path: Path) -> None:
    """`ontofetch pull --spec <file> hp` should reuse resolver and extras from the config."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    args = argparse.Namespace(
        command="pull",
        ids=["hp"],
        spec=config_path,
        resolver="custom-resolver",
        target_formats="ttl",
        log_level="INFO",
        lock=None,
        json=False,
        dry_run=False,
        force=False,
        concurrent_downloads=None,
        concurrent_plans=None,
        allowed_hosts=None,
        planner_probes=None,
    )

    _, specs = cli_module._resolve_specs_from_args(args, base_config=None)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.id == "hp"
    assert spec.resolver == "obo"
    assert spec.extras == {"acronym": "HP"}
    assert tuple(spec.target_formats) == ("owl", "obo")
