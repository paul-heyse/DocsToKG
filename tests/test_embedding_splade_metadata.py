"""Regression tests for SPLADE attention metadata and documentation."""

from __future__ import annotations

import sys
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
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

    embed_module = _reload_embedding_module(monkeypatch)
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

    embed_module = _reload_embedding_module(monkeypatch)
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
        '{"uuid": "chunk-1", "text": "Example", "doc_id": "doc", "schema_version": "docparse/1.1.0"}\n',
        '{"uuid": "chunk-1", "text": "Example", "doc_id": "doc"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(embed_module, "manifest_append", record_manifest)
    monkeypatch.setattr(embed_module, "iter_chunk_files", lambda _: [chunk_file])
    monkeypatch.setattr(
        embed_module,
        "process_pass_a",
        lambda files, logger: embed_module.BM25Stats(N=1, avgdl=1.0, df={}),
    )

    def fake_process_chunk_file_vectors(
        chunk_file, stats, args, validator, logger
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


def test_model_dirs_follow_environment_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variables should steer model directory defaults."""

    env_splade = tmp_path / "env_splade"
    env_qwen = tmp_path / "env_qwen"
    env_splade.mkdir()
    env_qwen.mkdir()
    monkeypatch.setenv("DOCSTOKG_SPLADE_DIR", str(env_splade))
    monkeypatch.setenv("DOCSTOKG_QWEN_DIR", str(env_qwen))

    embed_module = _reload_embedding_module(monkeypatch)

    manifests: List[Dict[str, Any]] = []

    def record_manifest(stage, doc_id, status, **metadata):
        manifests.append({"stage": stage, "doc_id": doc_id, "status": status, **metadata})

    monkeypatch.setattr(embed_module, "manifest_append", record_manifest)
    monkeypatch.setattr(embed_module, "_ensure_splade_dependencies", lambda: None)
    monkeypatch.setattr(embed_module, "_ensure_qwen_dependencies", lambda: None)
    monkeypatch.setattr(embed_module, "load_manifest_index", lambda *_, **__: {})
    monkeypatch.setattr(embed_module, "compute_content_hash", lambda *_: "hash")

    chunk_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    manifests_dir = tmp_path / "manifests"
    for directory in (chunk_dir, vectors_dir, manifests_dir):
        directory.mkdir()

    chunk_file = chunk_dir / "doc.chunks.jsonl"
    chunk_file.write_text(
        '{"uuid": "chunk-1", "text": "Example", "doc_id": "doc", "schema_version": "docparse/1.1.0"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(embed_module, "iter_chunk_files", lambda *_: [chunk_file])
    monkeypatch.setattr(embed_module, "data_chunks", lambda _root: chunk_dir)
    monkeypatch.setattr(embed_module, "data_vectors", lambda _root: vectors_dir)
    monkeypatch.setattr(embed_module, "data_manifests", lambda _root: manifests_dir)
    monkeypatch.setattr(
        embed_module,
        "detect_data_root",
        lambda override=None: Path(override) if override else tmp_path,
    )

    monkeypatch.setattr(embed_module, "BM25Vector", _DummyBM25)
    monkeypatch.setattr(embed_module, "SPLADEVector", _DummySPLADE)
    monkeypatch.setattr(embed_module, "DenseVector", _DummyDense)
    monkeypatch.setattr(embed_module, "VectorRow", _DummyVectorRow)

    recorded: Dict[str, Path] = {}

    def fake_splade(cfg, texts, batch_size=None):
        recorded["splade"] = Path(cfg.model_dir)
        return [[f"tok-{i}"] for i, _ in enumerate(texts)], [[0.5] for _ in texts]

    def fake_qwen(cfg, texts, batch_size=None):
        recorded["qwen"] = Path(cfg.model_dir)
        return [[0.1] * 2560 for _ in texts]

    monkeypatch.setattr(embed_module, "splade_encode", fake_splade)
    monkeypatch.setattr(embed_module, "qwen_embed", fake_qwen)

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

    assert embed_module.main(args) == 0
    assert recorded["splade"] == env_splade.resolve()
    assert recorded["qwen"] == env_qwen.resolve()


def test_cli_model_dirs_override_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit CLI arguments must override environment-provided paths."""

    env_splade = tmp_path / "env_splade"
    env_qwen = tmp_path / "env_qwen"
    env_splade.mkdir()
    env_qwen.mkdir()
    monkeypatch.setenv("DOCSTOKG_SPLADE_DIR", str(env_splade))
    monkeypatch.setenv("DOCSTOKG_QWEN_DIR", str(env_qwen))

    cli_splade = tmp_path / "cli_splade"
    cli_qwen = tmp_path / "cli_qwen"
    cli_splade.mkdir()
    cli_qwen.mkdir()

    embed_module = _reload_embedding_module(monkeypatch)

    manifests: List[Dict[str, Any]] = []

    def record_manifest(stage, doc_id, status, **metadata):
        manifests.append({"stage": stage, "doc_id": doc_id, "status": status, **metadata})

    monkeypatch.setattr(embed_module, "manifest_append", record_manifest)
    monkeypatch.setattr(embed_module, "_ensure_splade_dependencies", lambda: None)
    monkeypatch.setattr(embed_module, "_ensure_qwen_dependencies", lambda: None)
    monkeypatch.setattr(embed_module, "load_manifest_index", lambda *_, **__: {})
    monkeypatch.setattr(embed_module, "compute_content_hash", lambda *_: "hash")

    chunk_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    manifests_dir = tmp_path / "manifests"
    for directory in (chunk_dir, vectors_dir, manifests_dir):
        directory.mkdir()

    chunk_file = chunk_dir / "doc.chunks.jsonl"
    chunk_file.write_text(
        '{"uuid": "chunk-1", "text": "Example", "doc_id": "doc", "schema_version": "docparse/1.1.0"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(embed_module, "iter_chunk_files", lambda *_: [chunk_file])
    monkeypatch.setattr(embed_module, "data_chunks", lambda _root: chunk_dir)
    monkeypatch.setattr(embed_module, "data_vectors", lambda _root: vectors_dir)
    monkeypatch.setattr(embed_module, "data_manifests", lambda _root: manifests_dir)
    monkeypatch.setattr(
        embed_module,
        "detect_data_root",
        lambda override=None: Path(override) if override else tmp_path,
    )

    monkeypatch.setattr(embed_module, "BM25Vector", _DummyBM25)
    monkeypatch.setattr(embed_module, "SPLADEVector", _DummySPLADE)
    monkeypatch.setattr(embed_module, "DenseVector", _DummyDense)
    monkeypatch.setattr(embed_module, "VectorRow", _DummyVectorRow)

    recorded: Dict[str, Path] = {}

    def splade_stub(cfg, texts, batch_size=None):
        recorded["splade"] = Path(cfg.model_dir)
        return [[f"tok-{i}"] for i, _ in enumerate(texts)], [[0.5] for _ in texts]

    def qwen_stub(cfg, texts, batch_size=None):
        recorded["qwen"] = Path(cfg.model_dir)
        return [[0.1] * 2560 for _ in texts]

    monkeypatch.setattr(embed_module, "splade_encode", splade_stub)
    monkeypatch.setattr(embed_module, "qwen_embed", qwen_stub)

    args = embed_module.parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--chunks-dir",
            str(chunk_dir),
            "--out-dir",
            str(vectors_dir),
            "--splade-model-dir",
            str(cli_splade),
            "--qwen-model-dir",
            str(cli_qwen),
        ]
    )

    assert embed_module.main(args) == 0
    assert recorded["splade"] == cli_splade.resolve()
    assert recorded["qwen"] == cli_qwen.resolve()


def test_offline_requires_local_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline mode should fail fast when local models are absent."""

    embed_module = _reload_embedding_module(monkeypatch)

    chunk_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    chunk_dir.mkdir()
    vectors_dir.mkdir()

    missing_splade = tmp_path / "missing_splade"
    missing_qwen = tmp_path / "missing_qwen"

    args = embed_module.parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--chunks-dir",
            str(chunk_dir),
            "--out-dir",
            str(vectors_dir),
            "--offline",
            "--splade-model-dir",
            str(missing_splade),
            "--qwen-model-dir",
            str(missing_qwen),
        ]
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        embed_module.main(args)
    message = str(excinfo.value)
    assert "Offline mode requires local model directories" in message
    assert str(missing_splade) in message
    assert str(missing_qwen) in message


def test_pass_a_rejects_incompatible_chunk_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass A should fail fast when chunk schema versions are unsupported."""

    embed_module = _reload_embedding_module(monkeypatch)

    chunk_file = tmp_path / "bad.chunks.jsonl"
    row = {
        "doc_id": "doc",
        "source_path": "doc.doctags",
        "chunk_id": 0,
        "source_chunk_idxs": [0],
        "num_tokens": 1,
        "text": "bad",
        "uuid": "chunk-1",
        "schema_version": "docparse/0.9.0",
    }
    chunk_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    logger = SimpleNamespace(info=lambda *_, **__: None)

    with pytest.raises(ValueError, match="Unsupported chunk schema_version"):
        embed_module.process_pass_a([chunk_file], logger)


def _reload_embedding_module(monkeypatch: pytest.MonkeyPatch):
    """Reload EmbeddingV2 with lightweight optional dependency stubs."""

    tqdm_stub = ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **_: iterable if iterable is not None else []
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_stub)

    st_stub = ModuleType("sentence_transformers")

    class _SparseEncoder:  # pragma: no cover - placeholder to satisfy imports
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def encode(self, batch):  # pragma: no cover - unused stub
            return batch

        def decode(self, *_args, **_kwargs):  # pragma: no cover - unused stub
            return []

    st_stub.SparseEncoder = _SparseEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_stub)

    vllm_stub = ModuleType("vllm")

    class _LLM:  # pragma: no cover - placeholder to satisfy imports
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def embed(self, _batch, pooling_params=None):
            return []

    class _PoolingParams:  # pragma: no cover - placeholder to satisfy imports
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    vllm_stub.LLM = _LLM
    vllm_stub.PoolingParams = _PoolingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)

    import DocsToKG.DocParsing.EmbeddingV2 as embed_module

    return importlib.reload(embed_module)


class _DummyBM25:
    def __init__(self, **kwargs):
        self.data = kwargs


class _DummySPLADE:
    def __init__(self, **kwargs):
        self.data = kwargs


class _DummyDense:
    def __init__(self, **kwargs):
        self.data = kwargs


class _DummyVectorRow:
    def __init__(self, **kwargs):
        payload = dict(kwargs)
        for key in ("BM25", "SPLADEv3", "Qwen3_4B"):
            value = payload.get(key)
            if hasattr(value, "data"):
                payload[key] = value.data
        self.data = payload

    def model_dump(self, by_alias: bool = True):
        return self.data
