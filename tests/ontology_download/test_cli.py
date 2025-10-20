# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli",
#   "purpose": "Regression coverage for the ontology download CLI entrypoints.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Regression coverage for the ontology download CLI entrypoints."""

from __future__ import annotations

import difflib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.testing import TestingEnvironment


def _normalized_lines(text: str) -> list[str]:
    """Normalize whitespace for comparison while preserving line order."""

    return [line.rstrip() for line in text.strip().splitlines()]


def test_example_sources_yaml_matches_docs_examples_sources_yaml() -> None:
    """``cli.EXAMPLE_SOURCES_YAML`` should stay in sync with the docs template."""

    cli_lines = _normalized_lines(cli_module.EXAMPLE_SOURCES_YAML)
    docs_lines = _normalized_lines(
        Path("docs/examples/sources.yaml").read_text(encoding="utf-8")
    )

    if cli_lines != docs_lines:
        diff = "\n".join(
            difflib.unified_diff(
                docs_lines,
                cli_lines,
                fromfile="docs/examples/sources.yaml",
                tofile="cli.EXAMPLE_SOURCES_YAML",
                lineterm="",
            )
        )
        pytest.fail(
            "Example sources YAML drift detected between CLI constant and docs." "\n" + diff
        )


class _DoctorResponse:
    def __init__(self, status: int = 200, reason: str = "OK") -> None:
        self.status_code = status
        self.reason_phrase = reason

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 400


class _DoctorHttpClient:
    def __init__(self) -> None:
        self._response = _DoctorResponse()

    def head(self, *_args, **_kwargs):
        return self._response

    def get(self, *_args, **_kwargs):
        return self._response


def test_doctor_fix_reports_directory_creation_failure(capsys):
    """``doctor --fix`` should surface directory creation failures."""

    with TestingEnvironment() as env, patch.object(
        cli_module.net, "get_http_client", return_value=_DoctorHttpClient()
    ):
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
