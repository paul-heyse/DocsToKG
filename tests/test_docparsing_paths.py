"""Integration checks for DocParsing path utilities."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

COMMANDS = [
    (Path("src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py"), ["--help"]),
    (Path("src/DocsToKG/DocParsing/pipelines.py"), ["--pdf", "--help"]),
    (Path("src/DocsToKG/DocParsing/pipelines.py"), ["--html", "--help"]),
    (Path("src/DocsToKG/DocParsing/EmbeddingV2.py"), ["--help"]),
]


def test_scripts_respect_data_root(tmp_path: Path) -> None:
    """Scripts should honor DOCSTOKG_DATA_ROOT when resolving defaults."""

    data_root = tmp_path / "DataRoot"
    data_root.mkdir()

    env = os.environ.copy()
    env["DOCSTOKG_DATA_ROOT"] = str(data_root)
    project_root = Path(__file__).resolve().parents[1] / "src"
    existing_path = env.get("PYTHONPATH")
    path_entries = [str(project_root)]
    if existing_path:
        path_entries.append(existing_path)
    env["PYTHONPATH"] = os.pathsep.join(path_entries)

    for script, args in COMMANDS:
        result = subprocess.run(
            [sys.executable, str(script), *args],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    expected = {
        "DocTagsFiles",
        "ChunkedDocTagFiles",
        "Vectors",
        "HTML",
        "PDFs",
    }
    existing = {p.name for p in data_root.iterdir() if p.is_dir()}
    assert expected <= existing
