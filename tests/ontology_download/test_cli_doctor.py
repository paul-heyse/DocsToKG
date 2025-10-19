"""Regression coverage for the ``ontofetch doctor`` command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest import mock

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


def test_cli_doctor_handles_permission_denied_ontology_dir(capsys):
    """``doctor`` should still succeed if the ontology dir cannot be created."""

    with TestingEnvironment() as env:
        missing_dir = env.ontology_dir
        shutil.rmtree(missing_dir)
        assert not missing_dir.exists()

        original_mkdir = Path.mkdir

        def fake_mkdir(self, *args, **kwargs):
            if self == missing_dir:
                raise PermissionError("test permission denied")
            return original_mkdir(self, *args, **kwargs)

        probed_paths = []

        def fake_disk_usage(path):
            probed_paths.append(Path(path))
            raise PermissionError("disk usage denied")

        with mock.patch.object(cli_module.Path, "mkdir", new=fake_mkdir):
            with mock.patch.object(cli_module.shutil, "disk_usage", new=fake_disk_usage):
                exit_code = cli_module.cli_main(["doctor", "--json"])

        assert exit_code == 0

        output = json.loads(capsys.readouterr().out)
        ontologies = output["directories"]["ontologies"]
        assert ontologies["path"] == str(missing_dir)
        assert ontologies["exists"] is False

        assert probed_paths == [env.root]

        disk = output["disk"]
        assert disk["path"] == str(env.root)
        assert disk["ok"] is False
        assert "disk usage denied" in disk["error"]
