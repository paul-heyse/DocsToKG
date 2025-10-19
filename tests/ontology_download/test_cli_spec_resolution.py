# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli_spec_resolution",
#   "purpose": "Regression coverage for CLI spec resolution overrides",
#   "sections": [
#     {
#       "id": "test-pull-spec-cli-target-formats-override",
#       "name": "test_pull_spec_cli_target_formats_override",
#       "anchor": "function-test-pull-spec-cli-target-formats-override",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-ad-hoc-ids-use-cli-target-formats",
#       "name": "test_cli_ad_hoc_ids_use_cli_target_formats",
#       "anchor": "function-test-cli-ad-hoc-ids-use-cli-target-formats",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Comprehensive coverage for CLI fetch-spec override semantics.

Verifies that CLI-supplied target formats, resolver choices, and ad-hoc
ontology IDs correctly override YAML configuration defaults without mutating
the on-disk config. Ensures validation errors surface for incompatible
combinations and that plan/pull commands honour the resolved specification.
"""

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


def _make_args(**kwargs) -> argparse.Namespace:
    """Create argparse.Namespace with default values."""
    defaults = {
        "command": "pull",
        "ids": [],
        "spec": None,
        "resolver": None,
        "target_formats": None,
        "log_level": "INFO",
        "lock": None,
        "json": False,
        "dry_run": False,
        "force": False,
        "concurrent_downloads": None,
        "concurrent_plans": None,
        "allowed_hosts": None,
        "planner_probes": None,
        "since": None,
        "no_lock": False,
        "lock_output": None,
        "baseline": Path("baseline.json"),
        "use_manifest": True,
        "update_baseline": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_pull_spec_applies_resolver_override(tmp_path: Path) -> None:
    """`ontofetch pull --spec <file> hp --resolver direct` should override the resolver."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    args = _make_args(
        command="pull",
        ids=["hp"],
        spec=config_path,
        resolver="direct",
        target_formats="ttl",
    )

    _, specs = cli_module._resolve_specs_from_args(args, base_config=None)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.id == "hp"
    assert spec.resolver == "direct"
    assert spec.extras == {"acronym": "HP"}
    assert tuple(spec.target_formats) == ("ttl",)


def test_pull_spec_cli_target_formats_override(tmp_path: Path) -> None:
    """CLI formats override configured formats while preserving resolver/extras."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    args = _make_args(
        command="pull",
        ids=["hp"],
        spec=config_path,
        resolver="direct",
        target_formats="ttl, owl",
    )

    _, specs = cli_module._resolve_specs_from_args(args, base_config=None)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.id == "hp"
    assert spec.resolver == "direct"
    assert spec.extras == {"acronym": "HP"}
    assert tuple(spec.target_formats) == ("ttl", "owl")


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
