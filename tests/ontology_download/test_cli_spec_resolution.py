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

"""Regression tests for CLI fetch specification resolution."""

from __future__ import annotations

import argparse
import textwrap
from typing import Iterable
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


def _make_args(
    *,
    command: str,
    ids: Iterable[str],
    spec: Path,
    target_formats: str,
    resolver: str | None = None,
    use_manifest: bool = False,
):
    """Construct a CLI namespace mimicking parsed arguments for the downloader."""

    return argparse.Namespace(
        command=command,
        ids=list(ids),
        spec=spec,
        resolver=resolver,
        target_formats=target_formats,
        log_level="INFO",
        lock=None,
        json=False,
        dry_run=False,
        force=False,
        concurrent_downloads=None,
        concurrent_plans=None,
        allowed_hosts=None,
        planner_probes=None,
        since=None,
        no_lock=False,
        lock_output=None,
        baseline=Path("baseline.json"),
        use_manifest=use_manifest,
        update_baseline=False,
    )


def test_pull_spec_cli_target_formats_override(tmp_path: Path) -> None:
    """CLI formats override configured formats while preserving resolver/extras."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    args = _make_args(
        command="pull",
        ids=["hp"],
        spec=config_path,
        resolver="custom-resolver",
        target_formats="ttl, owl",
    )

    _, specs = cli_module._resolve_specs_from_args(args, base_config=None)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.id == "hp"
    assert spec.resolver == "obo"
    assert spec.extras == {"acronym": "HP"}
    assert tuple(spec.target_formats) == ("ttl", "owl")


@pytest.mark.parametrize("command", ["pull", "plan", "plan-diff"])
def test_cli_ad_hoc_ids_use_cli_target_formats(command: str, tmp_path: Path) -> None:
    """Ad-hoc ontology IDs respect CLI formats regardless of the command."""

    config_path = _write_sources_yaml(tmp_path / "sources.yaml")
    args = _make_args(
        command=command,
        ids=["new-id"],
        spec=config_path,
        target_formats="json, ttl",
        use_manifest=(command == "plan-diff"),
    )

    _, specs = cli_module._resolve_specs_from_args(args, base_config=None)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.id == "new-id"
    assert spec.resolver == "ols"
    assert spec.extras == {}
    assert tuple(spec.target_formats) == ("json", "ttl")
