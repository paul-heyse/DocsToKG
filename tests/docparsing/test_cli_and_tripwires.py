"""DocParsing CLI, path resolution, and trip-wire regression tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("transformers")

from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (  # noqa: E402
    Rec,
    coalesce_small_runs,
)
from tests.docparsing.stubs import dependency_stubs  # noqa: E402

# Optional dependency used for property-based checks.
hypothesis = pytest.importorskip("hypothesis")  # noqa: E402
given = hypothesis.given
st = hypothesis.strategies

COMMANDS = [
    (Path("src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py"), ["--help"]),
    (Path("src/DocsToKG/DocParsing/pipelines.py"), ["--pdf", "--help"]),
    (Path("src/DocsToKG/DocParsing/pipelines.py"), ["--html", "--help"]),
    (Path("src/DocsToKG/DocParsing/EmbeddingV2.py"), ["--help"]),
]

GOLDEN_DIR = Path("tests/data/docparsing/golden")
GOLDEN_CHUNKS = GOLDEN_DIR / "sample.chunks.jsonl"
GOLDEN_VECTORS = GOLDEN_DIR / "sample.vectors.jsonl"


# ---------------------------------------------------------------------------
# CLI entry points


def _reload_cli_modules():
    """Reload CLI modules so newly installed stubs are honoured."""

    chunk_module = importlib.import_module("DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin")
    embed_module = importlib.import_module("DocsToKG.DocParsing.EmbeddingV2")
    cli_module = importlib.import_module("DocsToKG.DocParsing.cli")
    importlib.reload(chunk_module)
    importlib.reload(embed_module)
    importlib.reload(cli_module)
    return cli_module


def test_chunk_and_embed_cli_with_dependency_stubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the chunking and embedding CLIs end-to-end with synthetic stubs."""

    data_root = tmp_path / "data"
    doc_dir = data_root / "DocTagsFiles"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doctags_path = doc_dir / "sample.doctags"
    doctags_path.write_text(
        json.dumps({"uuid": "chunk-1", "text": "Example text", "doc_id": "doc"}) + "\n",
        encoding="utf-8",
    )

    dependency_stubs()
    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))
    monkeypatch.setenv("DOCSTOKG_SPLADE_DIR", str(tmp_path / "splade"))
    monkeypatch.setenv("DOCSTOKG_QWEN_DIR", str(tmp_path / "qwen"))
    monkeypatch.setenv("DOCSTOKG_MODEL_ROOT", str(tmp_path / "models"))
    (tmp_path / "splade").mkdir()
    (tmp_path / "qwen").mkdir()
    (tmp_path / "models").mkdir()

    module = _reload_cli_modules()
    result = module.main(
        [
            "docparse",
            "chunk",
            "--input",
            str(doctags_path),
            "--output",
            str(data_root / "ChunkedDocTagFiles"),
        ]
    )
    assert result == 0

    result = module.main(
        [
            "docparse",
            "embed",
            "--chunks-dir",
            str(data_root / "ChunkedDocTagFiles"),
            "--out-dir",
            str(data_root / "Vectors"),
        ]
    )
    assert result == 0


# ---------------------------------------------------------------------------
# CLI path smoke tests


def test_scripts_respect_data_root(tmp_path: Path) -> None:
    """Scripts should honor DOCSTOKG_DATA_ROOT when resolving defaults."""

    data_root = tmp_path / "DataRoot"
    data_root.mkdir()

    env = os.environ.copy()
    env["DOCSTOKG_DATA_ROOT"] = str(data_root)
    project_root = Path(__file__).resolve().parents[2] / "src"
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


# ---------------------------------------------------------------------------
# Trip-wire regression checks


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def test_golden_chunk_count_and_hash():
    """Golden chunk fixtures should maintain deterministic order and count."""

    rows = list(_load_jsonl(GOLDEN_CHUNKS))
    assert len(rows) == 2
    hashes = [hashlib.sha1(row["text"].encode("utf-8")).hexdigest() for row in rows]
    assert hashes == [
        "f0b795882ec58d82ddbf4dc3fd830814d0048360",
        "b2fcd8346677a9b04ba75dc7716f8fd88d4bbd75",
    ]


def test_golden_vectors_hashes():
    rows = list(_load_jsonl(GOLDEN_VECTORS))
    assert len(rows) == 2
    hashes = [hashlib.sha1(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest() for row in rows]
    assert hashes == [
        "cd31b0d5bafbbe48a4aab1baad4dc72c91236748",
        "23938490f35c7c6b5bff50ccaa0435915eb82542",
    ]


def test_coalesce_small_runs_idempotent():
    rows = [
        Rec(start=0, end=5, text="hello"),
        Rec(start=5, end=10, text="world"),
        Rec(start=10, end=21, text="another chunk"),
    ]
    merged = list(coalesce_small_runs(rows, min_tokens=1))
    assert merged == rows


@given(st.text())
def test_coalesce_small_runs_handles_unicode(payload: str):
    rows = [Rec(start=0, end=len(payload), text=payload)]
    merged = list(coalesce_small_runs(rows, min_tokens=1))
    assert merged[0].text == payload
