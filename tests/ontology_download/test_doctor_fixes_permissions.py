"""Tests covering permission hardening in `_apply_doctor_fixes`."""

from __future__ import annotations

import os
import stat

import pytest

from DocsToKG.OntologyDownload import cli as cli_module


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics are not enforced on Windows")
def test_apply_doctor_fixes_sets_placeholder_permissions(tmp_path, monkeypatch):
    """Doctor fixes should restrict placeholder files to owner read/write."""

    monkeypatch.setattr(cli_module, "CONFIG_DIR", tmp_path)

    actions = cli_module._apply_doctor_fixes({})

    placeholder = tmp_path / "bioportal_api_key.txt"
    assert placeholder.exists(), "Expected doctor fixes to create BioPortal placeholder"
    assert stat.S_IMODE(placeholder.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR
    assert f"Ensured placeholder {placeholder.name}" in actions


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics are not enforced on Windows")
def test_apply_doctor_fixes_restricts_existing_empty_placeholders(tmp_path, monkeypatch):
    """Existing placeholders with lax permissions should be hardened when updated."""

    monkeypatch.setattr(cli_module, "CONFIG_DIR", tmp_path)

    placeholder = tmp_path / "ols_api_token.txt"
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text("")
    os.chmod(placeholder, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)

    cli_module._apply_doctor_fixes({})

    assert placeholder.read_text().strip(), "Placeholder content should be written"
    assert stat.S_IMODE(placeholder.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics are not enforced on Windows")
def test_apply_doctor_fixes_logs_permission_failures(tmp_path, monkeypatch):
    """Permission update failures should be surfaced in the remediation actions."""

    monkeypatch.setattr(cli_module, "CONFIG_DIR", tmp_path)

    def _failing_chmod(path, mode):
        raise PermissionError("denied")

    monkeypatch.setattr(cli_module.os, "chmod", _failing_chmod)

    actions = cli_module._apply_doctor_fixes({})

    assert any("Failed to update permissions" in action for action in actions)
