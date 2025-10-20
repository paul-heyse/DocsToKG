"""Regression coverage for the ontology download CLI entrypoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.testing import TestingEnvironment


def test_doctor_fix_reports_directory_creation_failure(capsys):
    """``doctor --fix`` should surface directory creation failures."""

    with TestingEnvironment() as env:
        failing_dir = env.cache_dir
        shutil.rmtree(failing_dir)
        assert not failing_dir.exists()

        original_mkdir = Path.mkdir

        def fake_mkdir(self, *args, **kwargs):
            if self == failing_dir:
                raise OSError("synthetic permission denied")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", new=fake_mkdir):
            exit_code = cli_module.cli_main(["doctor", "--fix", "--json"])

    assert exit_code == 0

    output = json.loads(capsys.readouterr().out)
    cache_entry = output["directories"]["cache"]
    assert cache_entry["path"] == str(failing_dir)
    assert cache_entry["exists"] is False

    fixes = output.get("fixes", [])
    assert any(
        message.startswith(f"Failed to create directory {failing_dir}")
        and "synthetic permission denied" in message
        for message in fixes
    )
