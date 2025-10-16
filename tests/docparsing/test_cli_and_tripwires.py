# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_cli_and_tripwires",
#   "purpose": "Pytest coverage for docparsing cli and tripwires scenarios",
#   "sections": [
#     {
#       "id": "tokencountingstub",
#       "name": "_TokenCountingStub",
#       "anchor": "class-tokencountingstub",
#       "kind": "class"
#     },
#     {
#       "id": "make-rec",
#       "name": "_make_rec",
#       "anchor": "function-make-rec",
#       "kind": "function"
#     },
#     {
#       "id": "reload-cli-modules",
#       "name": "_reload_cli_modules",
#       "anchor": "function-reload-cli-modules",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunk-and-embed-cli-with-dependency-stubs",
#       "name": "test_chunk_and_embed_cli_with_dependency_stubs",
#       "anchor": "function-test-chunk-and-embed-cli-with-dependency-stubs",
#       "kind": "function"
#     },
#     {
#       "id": "test-scripts-respect-data-root",
#       "name": "test_scripts_respect_data_root",
#       "anchor": "function-test-scripts-respect-data-root",
#       "kind": "function"
#     },
#     {
#       "id": "load-jsonl",
#       "name": "_load_jsonl",
#       "anchor": "function-load-jsonl",
#       "kind": "function"
#     },
#     {
#       "id": "test-golden-chunk-count-and-hash",
#       "name": "test_golden_chunk_count_and_hash",
#       "anchor": "function-test-golden-chunk-count-and-hash",
#       "kind": "function"
#     },
#     {
#       "id": "test-golden-vectors-hashes",
#       "name": "test_golden_vectors_hashes",
#       "anchor": "function-test-golden-vectors-hashes",
#       "kind": "function"
#     },
#     {
#       "id": "test-coalesce-small-runs-idempotent",
#       "name": "test_coalesce_small_runs_idempotent",
#       "anchor": "function-test-coalesce-small-runs-idempotent",
#       "kind": "function"
#     },
#     {
#       "id": "test-coalesce-small-runs-handles-unicode",
#       "name": "test_coalesce_small_runs_handles_unicode",
#       "anchor": "function-test-coalesce-small-runs-handles-unicode",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""DocParsing CLI, path resolution, and trip-wire regression tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings(
    "ignore",
    message=".*SwigPy.*__module__ attribute",
    category=DeprecationWarning,
)

pytest.importorskip("transformers")

pytestmark = pytest.mark.filterwarnings("ignore:.*SwigPy.*__module__ attribute:DeprecationWarning")

from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (  # noqa: E402
    Rec,
    coalesce_small_runs,
)
from tests.docparsing.stubs import dependency_stubs  # noqa: E402

# Optional dependency used for property-based checks.
hypothesis = pytest.importorskip("hypothesis")  # noqa: E402
given = hypothesis.given
st = hypothesis.strategies


class _TokenCountingStub:
    """Lightweight tokenizer stub for coalescence tests."""

    def count_tokens(self, *, text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return len(stripped.split())


def _make_rec(
    text: str,
    *,
    n_tok: int | None = None,
    src_idxs: list[int] | None = None,
    refs: list[str] | None = None,
    pages: list[int] | None = None,
) -> Rec:
    """Create a Rec with sensible defaults for unit tests."""

    return Rec(
        text=text,
        n_tok=len(text.split()) if n_tok is None else n_tok,
        src_idxs=src_idxs or [],
        refs=refs or [],
        pages=pages or [1],
    )


COMMANDS = [
    (Path("src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py"), ["--help"]),
    (Path("src/DocsToKG/DocParsing/pipelines.py"), ["--pdf", "--help"]),
    (Path("src/DocsToKG/DocParsing/pipelines.py"), ["--html", "--help"]),
    (Path("src/DocsToKG/DocParsing/EmbeddingV2.py"), ["--help"]),
]

GOLDEN_DIR = Path("tests/data/docparsing/golden")
GOLDEN_CHUNKS = GOLDEN_DIR / "sample.chunks.jsonl"
GOLDEN_VECTORS = GOLDEN_DIR / "sample.vectors.jsonl"


# --- CLI entry points ---

def _reload_cli_modules():
    """Reload CLI modules so newly installed stubs are honoured."""

    chunk_module = importlib.import_module(
        "DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin"
    )
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
            "chunk",
            "--in-dir",
            str(doc_dir),
            "--out-dir",
            str(data_root / "ChunkedDocTagFiles"),
        ]
    )
    assert result == 0

    result = module.main(
        [
            "embed",
            "--chunks-dir",
            str(data_root / "ChunkedDocTagFiles"),
            "--out-dir",
            str(data_root / "Vectors"),
        ]
    )
    assert result == 0


# --- CLI path smoke tests ---

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
        "Embeddings",
        "HTML",
        "PDFs",
    }
    existing = {p.name for p in data_root.iterdir() if p.is_dir()}
    assert expected <= existing


# --- Trip-wire regression checks ---

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
        "9d40a282aefb81ec15147275d8d490b40c334694",
        "4baf2d39299bd11c643b7c398248aae3b80765ae",
    ]


def test_golden_vectors_hashes():
    rows = list(_load_jsonl(GOLDEN_VECTORS))
    assert len(rows) == 2
    hashes = [
        hashlib.sha1(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest() for row in rows
    ]
    assert hashes == [
        "c86f87cefb058167082f3aa0522b7a5076e7eb32",
        "25265361eca16c886ea307ad6406610343b1f8de",
    ]


def test_coalesce_small_runs_idempotent():
    rows = [
        _make_rec("hello", src_idxs=[0]),
        _make_rec("world", src_idxs=[1]),
        _make_rec("another chunk", src_idxs=[2]),
    ]
    tokenizer = _TokenCountingStub()
    merged = coalesce_small_runs(rows, tokenizer=tokenizer, min_tokens=1)
    assert merged == rows


@given(st.text())
def test_coalesce_small_runs_handles_unicode(payload: str):
    rows = [_make_rec(payload, src_idxs=[0])]
    tokenizer = _TokenCountingStub()
    merged = coalesce_small_runs(rows, tokenizer=tokenizer, min_tokens=1)
    assert merged[0].text == payload
