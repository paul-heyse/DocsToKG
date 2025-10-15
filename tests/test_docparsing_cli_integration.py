"""Integration tests for DocParsing CLIs using optional dependency stubs."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

import DocsToKG.DocParsing.schemas as schemas
from DocsToKG.DocParsing.testing import dependency_stubs


def _reload_cli_modules():
    """Reload CLI modules so newly installed stubs are honoured."""

    chunk_module = importlib.import_module(
        "DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin"
    )
    embed_module = importlib.import_module("DocsToKG.DocParsing.EmbeddingV2")
    chunk_cli = importlib.import_module("DocsToKG.DocParsing.cli.chunk_and_coalesce")
    embed_cli = importlib.import_module("DocsToKG.DocParsing.cli.embed_vectors")
    importlib.reload(chunk_module)
    importlib.reload(embed_module)
    importlib.reload(chunk_cli)
    importlib.reload(embed_cli)
    return chunk_cli, embed_cli


def test_chunk_and_embed_cli_with_dependency_stubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the chunking and embedding CLIs end-to-end with synthetic stubs."""

    data_root = tmp_path / "data"
    doc_dir = data_root / "DocTagsFiles"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doctags_path = doc_dir / "sample.doctags"
    doctags_path.write_text(
        "Paragraph one with figures.\n\nParagraph two with tables.",
        encoding="utf-8",
    )

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    with dependency_stubs():
        chunk_cli, embed_cli = _reload_cli_modules()
        importlib.reload(schemas)
        import DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin as chunk_module
        import DocsToKG.DocParsing.EmbeddingV2 as embed_module

        importlib.reload(embed_module)
        chunk_module.ProvenanceMetadata = schemas.ProvenanceMetadata
        chunk_module.ChunkRow = schemas.ChunkRow
        embed_module.VectorRow = schemas.VectorRow
        embed_module.BM25Vector = schemas.BM25Vector
        embed_module.SPLADEVector = schemas.SPLADEVector
        embed_module.DenseVector = schemas.DenseVector

        chunk_exit = chunk_cli.main(["--data-root", str(data_root)])
        assert chunk_exit == 0

        chunk_dir = data_root / "ChunkedDocTagFiles"
        chunk_files = sorted(chunk_dir.glob("*.chunks.jsonl"))
        assert chunk_files, "Chunk CLI did not produce any chunk outputs"

        chunk_rows = []
        for path in chunk_files:
            with path.open("r", encoding="utf-8") as handle:
                chunk_rows.extend(json.loads(line) for line in handle)

        assert chunk_rows, "Chunk outputs were empty"
        for row in chunk_rows:
            schemas.validate_chunk_row(row)

        embed_exit = embed_cli.main(["--data-root", str(data_root)])
        assert embed_exit == 0

        vector_dir = data_root / "Vectors"
        vector_files = sorted(vector_dir.glob("*.vectors.jsonl"))
        assert vector_files, "Embedding CLI did not produce vector outputs"

        vector_rows = []
        for path in vector_files:
            with path.open("r", encoding="utf-8") as handle:
                vector_rows.extend(json.loads(line) for line in handle)

        assert vector_rows, "Vector outputs were empty"
        for row in vector_rows:
            schemas.validate_vector_row(row)

    manifest_dir = data_root / "Manifests"
    chunk_manifest = manifest_dir / "docparse.chunks.manifest.jsonl"
    embed_manifest = manifest_dir / "docparse.embeddings.manifest.jsonl"
    assert chunk_manifest.exists(), "Chunk stage manifest missing"
    assert embed_manifest.exists(), "Embedding stage manifest missing"
    with chunk_manifest.open("r", encoding="utf-8") as handle:
        chunk_rows = [json.loads(line) for line in handle]
    with embed_manifest.open("r", encoding="utf-8") as handle:
        embed_rows = [json.loads(line) for line in handle]

    assert any(row["stage"] == "chunks" and row["status"] == "success" for row in chunk_rows)
    assert any(row["stage"] == "embeddings" and row["status"] == "success" for row in embed_rows)


def test_embedding_dependency_guard_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure missing SPLADE dependencies raise a clear error."""

    import DocsToKG.DocParsing.EmbeddingV2 as embed_module

    tqdm_stub = ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **_: iterable if iterable is not None else []
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_stub)

    with monkeypatch.context() as m:
        placeholder = ModuleType("sentence_transformers")
        m.setitem(sys.modules, "sentence_transformers", placeholder)
        importlib.reload(embed_module)
        with pytest.raises(ImportError) as exc:
            embed_module._ensure_splade_dependencies()
        assert "sentence-transformers" in str(exc.value)

    importlib.reload(embed_module)


def test_validate_chunk_row_without_pydantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """`validate_chunk_row` should raise a helpful error when pydantic is missing."""

    import DocsToKG.DocParsing.schemas as schemas

    with monkeypatch.context() as m:
        placeholder = ModuleType("pydantic")
        m.setitem(sys.modules, "pydantic", placeholder)
        importlib.reload(schemas)
        with pytest.raises(RuntimeError) as exc:
            schemas.validate_chunk_row({})
        assert "pydantic" in str(exc.value)

    importlib.reload(schemas)
