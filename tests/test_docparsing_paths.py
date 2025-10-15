"""Integration checks for DocParsing path utilities."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = [
    Path("src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py"),
    Path("src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py"),
    Path("src/DocsToKG/DocParsing/run_docling_parallel_with_vllm_debug.py"),
    Path("src/DocsToKG/DocParsing/EmbeddingV2.py"),
]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("docling_core") is None,
    reason="docling_core is not available in the test environment",
)


def test_scripts_respect_data_root(tmp_path: Path) -> None:
    """Scripts should honor DOCSTOKG_DATA_ROOT when resolving defaults."""

    data_root = tmp_path / "DataRoot"
    data_root.mkdir()

    env = os.environ.copy()
    env["DOCSTOKG_DATA_ROOT"] = str(data_root)

    for script in SCRIPTS:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
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
