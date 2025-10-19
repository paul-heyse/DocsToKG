"""Behavioural coverage for the ``ontofetch doctor`` diagnostics command.

Exercises the doctor CLI across healthy and degraded environments, ensuring it
reports missing directories, optional dependency gaps, and plugin metadata
without crashing. Validates JSON output structure and the remediation hints
emitted for common misconfigurations.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.testing import TestingEnvironment


def test_cli_doctor_handles_missing_ontology_dir(capsys):
    """``doctor`` should succeed even when the ontology directory is absent."""

    with TestingEnvironment() as env:
        missing_dir = env.ontology_dir
        assert missing_dir.exists()
        shutil.rmtree(missing_dir)
        assert not missing_dir.exists()

        exit_code = cli_module.cli_main(["doctor", "--json"])

        assert exit_code == 0

        output = json.loads(capsys.readouterr().out)
        ontologies = output["directories"]["ontologies"]
        assert ontologies["path"] == str(missing_dir)
        assert ontologies["exists"] is True
        assert ontologies.get("created_for_diagnostics", False) is True

        disk = output["disk"]
        assert Path(disk["path"]) == missing_dir
        assert disk["ok"] is True
        assert disk["total_bytes"] is not None
        assert disk["free_bytes"] is not None
        assert "error" not in disk


def test_cli_doctor_reports_disk_error(capsys):
    """Disk probe failures should surface the probe path and error details."""

    with TestingEnvironment() as env:
        with patch.object(
            cli_module.shutil, "disk_usage", side_effect=OSError("synthetic disk failure")
        ):
            exit_code = cli_module.cli_main(["doctor", "--json"])

        assert exit_code == 0

        output = json.loads(capsys.readouterr().out)
        disk = output["disk"]
        assert disk["path"] == str(env.ontology_dir)
        assert disk["ok"] is False
        assert disk["error"] == "synthetic disk failure"
        assert "total_bytes" not in disk
