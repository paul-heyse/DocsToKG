"""Regression tests for SPLADE attention metadata and documentation."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest


def _find_action(parser, option: str):
    for action in parser._actions:  # pragma: no cover - exercised in tests
        if option in action.option_strings:
            return action
    raise AssertionError(f"Option {option} not found in parser")


def test_splade_attn_help_text_describes_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI help should explain attention fallbacks and explicit modes."""

    tqdm_stub = ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **_: iterable if iterable is not None else []
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_stub)

    st_stub = ModuleType("sentence_transformers")

    class _SparseEncoder:  # pragma: no cover - simple placeholder
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    st_stub.SparseEncoder = _SparseEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_stub)

    vllm_stub = ModuleType("vllm")

    class _LLM:  # pragma: no cover - simple placeholder
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def embed(self, _batch, pooling_params=None):
            return []

    class _PoolingParams:  # pragma: no cover - simple placeholder
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    vllm_stub.LLM = _LLM
    vllm_stub.PoolingParams = _PoolingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)

    import DocsToKG.DocParsing.EmbeddingV2 as embed_module
    embed_module = importlib.reload(embed_module)

    parser = embed_module.build_parser()
    action = _find_action(parser, "--splade-attn")
    help_text = action.help
    assert "FlashAttention 2" in help_text
    assert "scaled dot-product" in help_text
    assert "standard attention" in help_text


def test_summary_manifest_includes_splade_backend_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Summary manifest entries should record SPLADE backend metadata."""

    tqdm_stub = ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **_: iterable if iterable is not None else []
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_stub)

    import DocsToKG.DocParsing.EmbeddingV2 as embed_module

    manifests: List[Dict[str, Any]] = []

    def record_manifest(**entry: Any) -> None:
        manifests.append(entry)

    chunk_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    manifests_dir = tmp_path / "manifests"
    chunk_dir.mkdir()
    vectors_dir.mkdir()
    manifests_dir.mkdir()

    chunk_file = chunk_dir / "sample.chunks.jsonl"
    chunk_file.write_text(
        '{"uuid": "chunk-1", "text": "Example", "doc_id": "doc"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(embed_module, "manifest_append", record_manifest)
    monkeypatch.setattr(embed_module, "iter_chunk_files", lambda _: [chunk_file])
    monkeypatch.setattr(
        embed_module,
        "process_pass_a",
        lambda files, logger: (
            {"chunk-1": embed_module.Chunk(uuid="chunk-1", text="Example", doc_id="doc")},
            embed_module.BM25Stats(N=1, avgdl=1.0, df={}),
        ),
    )

    def fake_process_chunk_file_vectors(
        chunk_file, uuid_to_chunk, stats, args, validator, logger
    ) -> tuple[int, list[int], list[float]]:
        validator.validate("chunk-1", ["tok"], [1.0])
        return 1, [1], [1.0]

    monkeypatch.setattr(
        embed_module,
        "process_chunk_file_vectors",
        fake_process_chunk_file_vectors,
    )
    monkeypatch.setattr(embed_module, "load_manifest_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(embed_module, "compute_content_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(embed_module, "data_chunks", lambda _root: chunk_dir)
    monkeypatch.setattr(embed_module, "data_vectors", lambda _root: vectors_dir)
    monkeypatch.setattr(embed_module, "data_manifests", lambda _root: manifests_dir)
    monkeypatch.setattr(
        embed_module,
        "detect_data_root",
        lambda override=None: Path(override) if override else tmp_path,
    )

    embed_module._SPLADE_ENCODER_BACKENDS.clear()
    cfg = embed_module.SpladeCfg()
    key = (str(cfg.model_dir), cfg.device, cfg.attn_impl, cfg.max_active_dims)
    embed_module._SPLADE_ENCODER_BACKENDS[key] = "sdpa"

    args = embed_module.parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--chunks-dir",
            str(chunk_dir),
            "--out-dir",
            str(vectors_dir),
        ]
    )

    exit_code = embed_module.main(args)
    assert exit_code == 0

    summary_entries = [row for row in manifests if row.get("doc_id") == "__corpus__"]
    assert summary_entries, "Corpus summary manifest entry missing"
    summary = summary_entries[-1]
    assert summary["splade_attn_backend_used"] == "sdpa"
    assert summary["sparsity_warn_threshold_pct"] == pytest.approx(1.0)
