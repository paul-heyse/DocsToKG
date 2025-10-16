# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_storage",
#   "purpose": "Pytest coverage for ontology download storage scenarios",
#   "sections": [
#     {
#       "id": "test-fsspec-storage-roundtrip",
#       "name": "test_fsspec_storage_roundtrip",
#       "anchor": "function-test-fsspec-storage-roundtrip",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Storage backend tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import settings as storage


@pytest.mark.skipif(importlib.util.find_spec("fsspec") is None, reason="fsspec not installed")
# --- Test Cases ---


def test_fsspec_storage_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Remote storage should mirror uploads and allow subsequent retrievals."""

    monkeypatch.setenv("PYSTOW_HOME", str(tmp_path))
    monkeypatch.setenv("ONTOFETCH_STORAGE_URL", "memory://ontologies")

    mod = importlib.reload(storage)
    backend = mod.STORAGE

    local_dir = backend.prepare_version("hp", "2024")
    manifest = local_dir / "manifest.json"
    manifest.write_text("{}")

    backend.finalize_version("hp", "2024", local_dir)

    remote_dir = backend._remote_version_path("hp", "2024")  # type: ignore[attr-defined]
    remote_manifest = remote_dir / "manifest.json"
    assert backend.fs.exists(str(remote_manifest))

    manifest.unlink()
    assert not manifest.exists()

    backend.ensure_local_version("hp", "2024")
    assert not manifest.exists()

    backend.fs.get_file(str(remote_manifest), str(manifest))
    assert manifest.exists()

    monkeypatch.delenv("ONTOFETCH_STORAGE_URL", raising=False)
    importlib.reload(storage)
