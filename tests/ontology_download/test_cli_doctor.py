# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli_doctor",
#   "purpose": "Behavioural coverage for the ``ontofetch doctor`` diagnostics command.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Behavioural coverage for the ``ontofetch doctor`` diagnostics command.

Exercises the doctor CLI across healthy and degraded environments, ensuring it
reports missing directories, optional dependency gaps, and plugin metadata
without crashing. Validates JSON output structure and the remediation hints
emitted for common misconfigurations."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from DocsToKG.OntologyDownload import cli as cli_module
from DocsToKG.OntologyDownload.testing import TestingEnvironment
from tests.conftest import PatchManager


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


def test_cli_doctor_handles_missing_ontology_dir(capsys):
    """``doctor`` should succeed even when the ontology directory is absent."""

    with TestingEnvironment() as env, patch.object(
        cli_module.net, "get_http_client", return_value=_DoctorHttpClient()
    ):
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

    with TestingEnvironment() as env, patch.object(
        cli_module.net, "get_http_client", return_value=_DoctorHttpClient()
    ):
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


def test_cli_doctor_reports_invalid_rate_limit_override_json(capsys):
    """Invalid rate-limit overrides should surface as doctor JSON errors."""

    patcher = PatchManager()
    try:
        with TestingEnvironment(), patch.object(
            cli_module.net, "get_http_client", return_value=_DoctorHttpClient()
        ):
            patcher.setenv("ONTOFETCH_PER_HOST_RATE_LIMIT", "not-a-limit")
            exit_code = cli_module.cli_main(["doctor", "--json"])
    finally:
        patcher.close()

    assert exit_code == 0

    output = json.loads(capsys.readouterr().out)
    rate_limits = output["rate_limits"]
    error_message = rate_limits.get("error", "")
    assert "Failed to load default rate limits" in error_message
    assert "not-a-limit" in error_message


def test_cli_doctor_reports_invalid_rate_limit_override_tty(capsys):
    """TTY rendering should include invalid rate-limit override errors."""

    patcher = PatchManager()
    try:
        with TestingEnvironment(), patch.object(
            cli_module.net, "get_http_client", return_value=_DoctorHttpClient()
        ):
            patcher.setenv("ONTOFETCH_PER_HOST_RATE_LIMIT", "not-a-limit")
            exit_code = cli_module.cli_main(["doctor"])
    finally:
        patcher.close()

    assert exit_code == 0

    stdout = capsys.readouterr().out
    assert "Rate limit check error" in stdout
    assert "not-a-limit" in stdout
@pytest.mark.skipif(os.name == "nt", reason="POSIX-style permissions required")
def test_cli_doctor_reports_missing_execute_permission_in_json(capsys):
    """The JSON report should surface directories lacking execute permissions."""

    with TestingEnvironment() as env, patch.object(
        cli_module.net, "get_http_client", return_value=_DoctorHttpClient()
    ):
        restricted_dir = env.cache_dir
        original_access = cli_module.os.access

        def fake_access(path, mode, *, dir_fd=None, effective_ids=False, follow_symlinks=True):
            path_obj = Path(path)
            if path_obj == restricted_dir:
                if mode == cli_module.os.W_OK:
                    return True
                if mode == cli_module.os.X_OK:
                    return False
                if mode == (cli_module.os.W_OK | cli_module.os.X_OK):
                    return False
            return original_access(
                path,
                mode,
                dir_fd=dir_fd,
                effective_ids=effective_ids,
                follow_symlinks=follow_symlinks,
            )

        with patch.object(cli_module.os, "access", side_effect=fake_access):
            exit_code = cli_module.cli_main(["doctor", "--json"])

    assert exit_code == 0

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    cache_entry = output["directories"]["cache"]
    assert cache_entry["exists"] is True
    assert cache_entry["directory"] is True
    assert cache_entry["write_permission"] is True
    assert cache_entry["execute_permission"] is False
    assert cache_entry["writable"] is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX-style permissions required")
def test_cli_doctor_prints_missing_execute_permission(capsys):
    """Human-readable doctor output should note missing directory execute permissions."""

    with TestingEnvironment() as env, patch.object(
        cli_module.net, "get_http_client", return_value=_DoctorHttpClient()
    ):
        restricted_dir = env.cache_dir
        original_access = cli_module.os.access

        def fake_access(path, mode, *, dir_fd=None, effective_ids=False, follow_symlinks=True):
            path_obj = Path(path)
            if path_obj == restricted_dir:
                if mode == cli_module.os.W_OK:
                    return True
                if mode == cli_module.os.X_OK:
                    return False
                if mode == (cli_module.os.W_OK | cli_module.os.X_OK):
                    return False
            return original_access(
                path,
                mode,
                dir_fd=dir_fd,
                effective_ids=effective_ids,
                follow_symlinks=follow_symlinks,
            )

        with patch.object(cli_module.os, "access", side_effect=fake_access):
            exit_code = cli_module.cli_main(["doctor"])

    assert exit_code == 0

    captured = capsys.readouterr()
    cache_line = next(
        (line for line in captured.out.splitlines() if "cache:" in line),
        "",
    )
    assert cache_line
    assert "no execute permission" in cache_line
    assert "read-only" not in cache_line.split(":", 1)[-1]
