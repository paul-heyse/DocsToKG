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
import json
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.errors import ConfigError


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


def test_pull_spec_applies_resolver_override(tmp_path: Path) -> None:
    """`ontofetch pull --spec <file> hp --resolver direct` should override the resolver."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    args = argparse.Namespace(
        command="pull",
        ids=["hp"],
        spec=config_path,
        resolver="direct",
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
    assert spec.resolver == "direct"
    assert spec.extras == {"acronym": "HP"}
    assert tuple(spec.target_formats) == ("owl", "obo")


def test_resolver_override_validated(tmp_path: Path) -> None:
    """Unknown resolver overrides should raise configuration errors."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    args = argparse.Namespace(
        command="pull",
        ids=["hp"],
        spec=config_path,
        resolver="not-a-resolver",
        target_formats=None,
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

    with pytest.raises(ConfigError, match=r"Unknown resolver\(s\) specified"):
        cli_module._resolve_specs_from_args(args, base_config=None)


def test_lockfile_resolver_override(tmp_path: Path) -> None:
    """`--resolver` should replace resolvers sourced from lockfiles."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    lock_path = tmp_path / "ontologies.lock.json"
    lock_payload = {
        "entries": [
            {
                "id": "hp",
                "url": "https://example.org/hp.owl",
                "resolver": "direct",
                "target_formats": ["owl"],
            }
        ]
    }
    lock_path.write_text(json.dumps(lock_payload), encoding="utf-8")
    args = argparse.Namespace(
        command="pull",
        ids=["hp"],
        spec=config_path,
        resolver="obo",
        target_formats=None,
        log_level="INFO",
        lock=lock_path,
        json=False,
        dry_run=False,
        force=False,
        concurrent_downloads=None,
        concurrent_plans=None,
        allowed_hosts=None,
        planner_probes=None,
    )

    config, specs = cli_module._resolve_specs_from_args(args, base_config=None)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.id == "hp"
    assert spec.resolver == "obo"
    assert spec.extras["url"] == "https://example.org/hp.owl"
    assert config.specs[0].resolver == "obo"


def test_plan_resolver_override_resolves_specs(tmp_path: Path) -> None:
    """`_resolve_specs_from_args` should honor overrides for plan workflows."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    args = argparse.Namespace(
        command="plan",
        ids=["hp"],
        spec=config_path,
        resolver="direct",
        target_formats=None,
        log_level="INFO",
        since=None,
        concurrent_plans=None,
        concurrent_downloads=None,
        allowed_hosts=None,
        planner_probes=None,
        json=False,
        lock_output=tmp_path / "ontologies.lock.json",
        no_lock=True,
    )

    _, specs = cli_module._resolve_specs_from_args(args, base_config=None)

    assert len(specs) == 1
    assert specs[0].resolver == "direct"
    assert specs[0].extras == {"acronym": "HP"}


def test_plan_diff_resolver_override_resolves_specs(tmp_path: Path) -> None:
    """`_resolve_specs_from_args` should honor overrides during plan-diff."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    args = argparse.Namespace(
        command="plan-diff",
        ids=["hp"],
        spec=config_path,
        resolver="direct",
        target_formats=None,
        log_level="INFO",
        since=None,
        baseline=tmp_path / "baseline.json",
        update_baseline=False,
        use_manifest=False,
        concurrent_plans=None,
        concurrent_downloads=None,
        allowed_hosts=None,
        planner_probes=None,
        json=False,
        lock_output=tmp_path / "ontologies.lock.json",
        no_lock=True,
    )

    _, specs = cli_module._resolve_specs_from_args(args, base_config=None, allow_empty=True)

    assert len(specs) == 1
    assert specs[0].resolver == "direct"
