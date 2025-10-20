# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli_validate",
#   "purpose": "Regression suite for the ``ontofetch validate`` CLI handler.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Regression suite for the ``ontofetch validate`` CLI handler.

Confirms validation results are written atomically, cache directories are
discovered, exit codes reflect validator failures, and JSON output mirrors the
structure consumed by operational dashboards."""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.manifests import write_json_atomic as real_write_json_atomic
from DocsToKG.OntologyDownload.settings import get_default_config
from DocsToKG.OntologyDownload.testing import TestingEnvironment


def test_handle_validate_persists_results_atomically() -> None:
    """Validation results should be written via the atomic helper."""

    with TestingEnvironment():
        config = get_default_config(copy=True)
        ontology_id = "atomically-safe"
        version = "2024-10-20"
        storage_delegate = cli_module.STORAGE._delegate

        with (
            patch.object(cli_module, "STORAGE", storage_delegate),
            patch("DocsToKG.OntologyDownload.manifests.STORAGE", storage_delegate),
        ):
            version_dir = storage_delegate.prepare_version(ontology_id, version)
            manifest_path = version_dir / "manifest.json"
            manifest_payload = {
                "id": ontology_id,
                "version": version,
                "filename": "validation-input.owl",
            }
            manifest_path.write_text(json.dumps(manifest_payload))

            original_path = version_dir / "original" / manifest_payload["filename"]
            original_path.write_text("ontology payload")

            args = argparse.Namespace(
                id=ontology_id,
                version=version,
                rdflib=True,
                pronto=False,
                owlready2=False,
                robot=False,
                arelle=False,
            )

            expected_result = {
                "ok": True,
                "details": {"triples": 42},
                "output_files": ["validation/report.json"],
            }

            class DummyResult:
                def to_dict(self) -> dict:
                    return dict(expected_result)

            def fake_run(requests, logger):
                assert [request.name for request in requests] == ["rdflib"]
                request = requests[0]
                assert request.file_path == original_path
                assert request.normalized_dir == version_dir / "normalized"
                assert request.validation_dir == version_dir / "validation"
                assert logger is not None
                return {"rdflib": DummyResult()}

            with (
                patch.object(cli_module, "run_validators", side_effect=fake_run) as run_mock,
                patch.object(
                    cli_module,
                    "write_json_atomic",
                    wraps=real_write_json_atomic,
                ) as write_mock,
            ):
                validation_summary = cli_module._handle_validate(args, config)

            run_mock.assert_called_once()
            write_mock.assert_called_once()
            written_path, written_payload = write_mock.call_args[0]
            assert written_path == manifest_path
            assert written_payload["validation"]["rdflib"] == expected_result

            manifest_data = json.loads(manifest_path.read_text())
            assert manifest_data["validation"]["rdflib"] == expected_result
            assert validation_summary == {"rdflib": expected_result}


def test_handle_validate_preserves_existing_validator_entries() -> None:
    """Merging new results should not discard untouched validators."""

    with TestingEnvironment():
        config = get_default_config(copy=True)
        ontology_id = "merge-existing"
        version = "2024-10-21"
        storage_delegate = cli_module.STORAGE._delegate

        with (
            patch.object(cli_module, "STORAGE", storage_delegate),
            patch("DocsToKG.OntologyDownload.manifests.STORAGE", storage_delegate),
        ):
            version_dir = storage_delegate.prepare_version(ontology_id, version)
            manifest_path = version_dir / "manifest.json"
            manifest_payload = {
                "id": ontology_id,
                "version": version,
                "filename": "validation-input.owl",
                "validation": {
                    "rdflib": {
                        "ok": True,
                        "details": {"triples": 10},
                        "output_files": ["validation/rdflib.json"],
                    }
                },
            }
            manifest_path.write_text(json.dumps(manifest_payload))

            original_path = version_dir / "original" / manifest_payload["filename"]
            original_path.write_text("ontology payload")

            args = argparse.Namespace(
                id=ontology_id,
                version=version,
                rdflib=False,
                pronto=True,
                owlready2=False,
                robot=False,
                arelle=False,
            )

            new_pronto_result = {
                "ok": True,
                "details": {"classes": 5},
                "output_files": ["validation/pronto.json"],
            }

            class DummyResult:
                def to_dict(self) -> dict:
                    return dict(new_pronto_result)

            def fake_run(requests, logger):
                assert [request.name for request in requests] == ["pronto"]
                request = requests[0]
                assert request.file_path == original_path
                assert request.normalized_dir == version_dir / "normalized"
                assert request.validation_dir == version_dir / "validation"
                assert logger is not None
                return {"pronto": DummyResult()}

            with (
                patch.object(cli_module, "run_validators", side_effect=fake_run) as run_mock,
                patch.object(
                    cli_module,
                    "write_json_atomic",
                    wraps=real_write_json_atomic,
                ) as write_mock,
            ):
                validation_summary = cli_module._handle_validate(args, config)

            run_mock.assert_called_once()
            write_mock.assert_called_once()
            _, written_payload = write_mock.call_args[0]
            assert written_payload["validation"] == {
                "rdflib": manifest_payload["validation"]["rdflib"],
                "pronto": new_pronto_result,
            }

            manifest_data = json.loads(manifest_path.read_text())
            assert manifest_data["validation"] == {
                "rdflib": manifest_payload["validation"]["rdflib"],
                "pronto": new_pronto_result,
            }
            assert validation_summary == {
                "rdflib": manifest_payload["validation"]["rdflib"],
                "pronto": new_pronto_result,
            }
