# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_strict_imports",
#   "purpose": "Validates strict import toggles for optional shims",
#   "sections": []
# }
# === /NAVMAP ===

"""Tests for ONTOFETCH_STRICT_IMPORTS behavior on compatibility shims."""

from __future__ import annotations

import subprocess
import sys


def _run_strict_import(module: str) -> subprocess.CompletedProcess[str]:
    script = "import os; " "os.environ['ONTOFETCH_STRICT_IMPORTS']='1'; " f"__import__('{module}')"
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )


def test_settings_strict_import_allowed() -> None:
    result = _run_strict_import("DocsToKG.OntologyDownload.settings")
    assert result.returncode == 0, result.stderr or result.stdout


def test_io_strict_import_allowed() -> None:
    result = _run_strict_import("DocsToKG.OntologyDownload.io")
    assert result.returncode == 0, result.stderr or result.stdout
