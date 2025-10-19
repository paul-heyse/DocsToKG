"""Regression tests for the ``ontofetch prune`` command."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload import manifests as manifests_module
from DocsToKG.OntologyDownload.testing import TestingEnvironment


def _seed_versions(storage, ontology_id: str, versions: list[str]) -> None:
    """Create version directories and manifests for ``ontology_id``."""

    for index, version in enumerate(versions, start=1):
        base = storage.prepare_version(ontology_id, version)
        manifest_path = base / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "id": ontology_id,
                    "version": version,
                    "downloaded_at": datetime(2024, 1, index, tzinfo=timezone.utc).isoformat(),
                }
            )
        )
        original_dir = base / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        (original_dir / f"{version}.txt").write_text(f"{ontology_id}-{version}")


def test_prune_deduplicates_ids_preserving_order(capsys) -> None:
    """Duplicate ``--ids`` entries should be ignored while keeping the first order."""

    with TestingEnvironment():
        storage_shim = cli_module.STORAGE
        backend = storage_shim._delegate  # type: ignore[attr-defined]
        _seed_versions(backend, "alpha", ["v1", "v2"])
        _seed_versions(backend, "beta", ["v1", "v2"])
        _seed_versions(backend, "gamma", ["v1", "v2"])

        cli_module.STORAGE = backend
        manifests_module.STORAGE = backend
        try:
            exit_code = cli_module.cli_main(
                [
                    "prune",
                    "--keep",
                    "1",
                    "--dry-run",
                    "--json",
                    "--ids",
                    "beta",
                    "alpha",
                    "beta",
                    "gamma",
                    "alpha",
                ]
            )
        finally:
            cli_module.STORAGE = storage_shim
            manifests_module.STORAGE = storage_shim

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is True
    assert [entry["id"] for entry in payload["ontologies"]] == ["beta", "alpha", "gamma"]

    messages = payload["messages"]
    assert messages[0] == (
        "[DRY-RUN] Requested ontologies (duplicates ignored, order preserved): "
        "beta, alpha, gamma"
    )
    assert messages[1].startswith("[DRY-RUN] beta:")
    assert messages[2].startswith("[DRY-RUN] alpha:")
    assert messages[3].startswith("[DRY-RUN] gamma:")


def test_prune_preserves_custom_id_order(capsys) -> None:
    """Custom ``--ids`` ordering should be reflected in the dry-run output."""

    with TestingEnvironment():
        storage_shim = cli_module.STORAGE
        backend = storage_shim._delegate  # type: ignore[attr-defined]
        _seed_versions(backend, "omega", ["v1", "v2"])
        _seed_versions(backend, "delta", ["v1", "v2"])
        _seed_versions(backend, "sigma", ["v1", "v2"])

        cli_module.STORAGE = backend
        manifests_module.STORAGE = backend
        try:
            exit_code = cli_module.cli_main(
                [
                    "prune",
                    "--keep",
                    "1",
                    "--dry-run",
                    "--json",
                    "--ids",
                    "sigma",
                    "omega",
                    "delta",
                ]
            )
        finally:
            cli_module.STORAGE = storage_shim
            manifests_module.STORAGE = storage_shim

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is True
    assert [entry["id"] for entry in payload["ontologies"]] == ["sigma", "omega", "delta"]

    messages = payload["messages"]
    assert messages[0] == (
        "[DRY-RUN] Requested ontologies (order preserved): "
        "sigma, omega, delta"
    )
    assert messages[1].startswith("[DRY-RUN] sigma:")
    assert messages[2].startswith("[DRY-RUN] omega:")
    assert messages[3].startswith("[DRY-RUN] delta:")
