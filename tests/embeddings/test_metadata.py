# === NAVMAP v1 ===
# {
#   "module": "tests.embeddings.test_metadata",
#   "purpose": "Pytest coverage for embeddings metadata scenarios",
#   "sections": [
#     {
#       "id": "install-minimal-stubs",
#       "name": "_install_minimal_stubs",
#       "anchor": "function-install-minimal-stubs",
#       "kind": "function"
#     },
#     {
#       "id": "test-process-pass-a-returns-stats-only",
#       "name": "test_process_pass_a_returns_stats_only",
#       "anchor": "function-test-process-pass-a-returns-stats-only",
#       "kind": "function"
#     },
#     {
#       "id": "test-process-chunk-file-vectors-reads-texts",
#       "name": "test_process_chunk_file_vectors_reads_texts",
#       "anchor": "function-test-process-chunk-file-vectors-reads-texts",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-path-overrides-take-precedence",
#       "name": "test_cli_path_overrides_take_precedence",
#       "anchor": "function-test-cli-path-overrides-take-precedence",
#       "kind": "function"
#     },
#     {
#       "id": "test-offline-mode-requires-local-models",
#       "name": "test_offline_mode_requires_local_models",
#       "anchor": "function-test-offline-mode-requires-local-models",
#       "kind": "function"
#     },
#     {
#       "id": "find-action",
#       "name": "_find_action",
#       "anchor": "function-find-action",
#       "kind": "function"
#     },
#     {
#       "id": "test-splade-attn-help-text-describes-fallbacks",
#       "name": "test_splade_attn_help_text_describes_fallbacks",
#       "anchor": "function-test-splade-attn-help-text-describes-fallbacks",
#       "kind": "function"
#     },
#     {
#       "id": "test-summary-manifest-includes-splade-backend-metadata",
#       "name": "test_summary_manifest_includes_splade_backend_metadata",
#       "anchor": "function-test-summary-manifest-includes-splade-backend-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-model-dirs-follow-environment-defaults",
#       "name": "test_model_dirs_follow_environment_defaults",
#       "anchor": "function-test-model-dirs-follow-environment-defaults",
#       "kind": "function"
#     },
#     {
#       "id": "test-cli-model-dirs-override-environment",
#       "name": "test_cli_model_dirs_override_environment",
#       "anchor": "function-test-cli-model-dirs-override-environment",
#       "kind": "function"
#     },
#     {
#       "id": "test-offline-requires-local-models",
#       "name": "test_offline_requires_local_models",
#       "anchor": "function-test-offline-requires-local-models",
#       "kind": "function"
#     },
#     {
#       "id": "test-pass-a-rejects-incompatible-chunk-schema",
#       "name": "test_pass_a_rejects_incompatible_chunk_schema",
#       "anchor": "function-test-pass-a-rejects-incompatible-chunk-schema",
#       "kind": "function"
#     },
#     {
#       "id": "reload-embedding-module",
#       "name": "_reload_embedding_module",
#       "anchor": "function-reload-embedding-module",
#       "kind": "function"
#     },
#     {
#       "id": "dummybm25",
#       "name": "_DummyBM25",
#       "anchor": "class-dummybm25",
#       "kind": "class"
#     },
#     {
#       "id": "dummysplade",
#       "name": "_DummySPLADE",
#       "anchor": "class-dummysplade",
#       "kind": "class"
#     },
#     {
#       "id": "dummydense",
#       "name": "_DummyDense",
#       "anchor": "class-dummydense",
#       "kind": "class"
#     },
#     {
#       "id": "dummyvectorrow",
#       "name": "_DummyVectorRow",
#       "anchor": "class-dummyvectorrow",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Embedding pipeline metadata and configuration regression tests."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List

import pytest

from DocsToKG.DocParsing.io import iter_jsonl

from tests._stubs import dependency_stubs

# --- Helper Functions ---


def _install_minimal_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install lightweight dependency stubs for embedding module tests."""

    tqdm_stub = ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **_: iterable if iterable is not None else []
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_stub)

    st_stub = ModuleType("sentence_transformers")

    class _SparseEncoder:  # pragma: no cover - trivial stub
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def encode(self, batch):
            class _Result:
                def __init__(self, size: int):
                    self._size = size

                def __getitem__(self, index):
                    return self

                def coalesce(self):
                    return self

                def values(self):
                    class _Values:
                        def __init__(self, size: int):
                            self._size = size

                        def numel(self):
                            return self._size

                    return _Values(len(batch[0]))

                def shape(self):  # pragma: no cover - defensive fallback
                    return (len(batch), 1)

            return _Result(len(batch))

        def decode(self, *_args, **_kwargs):
            return []

    st_stub.SparseEncoder = _SparseEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_stub)

    vllm_stub = ModuleType("vllm")

    class _LLM:  # pragma: no cover - trivial stub
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def embed(self, batch, pooling_params=None):
            class _Output:
                def __init__(self, dim: int):
                    self.outputs = type("E", (), {"embedding": [0.0] * dim})

            return [_Output(2560) for _ in batch]

    class _PoolingParams:  # pragma: no cover - trivial stub
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    vllm_stub.LLM = _LLM
    vllm_stub.PoolingParams = _PoolingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)

    original_write_text = Path.write_text

    def _write_text(path_self: Path, *args, **kwargs):
        data = "".join(args) if args else ""
        return original_write_text(path_self, data, **kwargs)

    monkeypatch.setattr(Path, "write_text", _write_text)


# --- Test Cases ---


def test_process_pass_a_returns_stats_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`process_pass_a` should return BM25 statistics without chunk caches."""

    _install_minimal_stubs(monkeypatch)
    import DocsToKG.DocParsing.embedding as embed_module

    chunk_file = tmp_path / "sample.chunks.jsonl"
    chunk_file.write_text('{"text": "Example text"}\n', encoding="utf-8")

    stats = embed_module.process_pass_a([chunk_file], embed_module.get_logger(__name__))
    assert isinstance(stats, embed_module.BM25Stats)
    assert stats.N == 1
    rows = list(iter_jsonl(chunk_file))
    assert rows[0]["uuid"], "UUIDs should be assigned in-place"


def test_process_chunk_file_vectors_reads_texts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunk texts should be sourced directly from file rows when encoding."""

    _install_minimal_stubs(monkeypatch)
    import DocsToKG.DocParsing.embedding as embed_module

    chunk_file = tmp_path / "doc.chunks.jsonl"
    chunk_file.write_text(
        json.dumps({"uuid": "u1", "text": "Hello world", "doc_id": "doc"}) + "\n",
        encoding="utf-8",
    )

    captured_texts: List[str] = []

    monkeypatch.setattr(
        embed_module,
        "splade_encode",
        lambda cfg, texts, batch_size=None: ([["tok"] for _ in texts], [[1.0] for _ in texts]),
    )
    monkeypatch.setattr(
        embed_module,
        "qwen_embed",
        lambda cfg, texts, batch_size=None: captured_texts.extend(texts) or [[1.0] + [0.0] * 2559],
    )
    captured_write_texts: List[str] = []

    def _write_vectors(
        path,
        uuids,
        texts,
        splade_results,
        qwen_results,
        stats,
        args,
        rows,
        validator,
        logger,
        **_kwargs,
    ) -> tuple[int, List[int], List[float]]:
        captured_write_texts.extend(texts)
        return len(uuids), [1] * len(uuids), [1.0] * len(uuids)

    monkeypatch.setattr(embed_module, "write_vectors", _write_vectors)

    args = embed_module.build_parser().parse_args(
        [
            "--chunks-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
            "--splade-model-dir",
            str(tmp_path),
            "--qwen-model-dir",
            str(tmp_path),
        ]
    )
    args.splade_cfg = embed_module.SpladeCfg(model_dir=tmp_path, cache_folder=tmp_path)
    args.qwen_cfg = embed_module.QwenCfg(model_dir=tmp_path)
    args.batch_size_splade = 1
    args.batch_size_qwen = 1
    args.bm25_k1 = 1.5
    args.bm25_b = 0.75
    args.out_dir = tmp_path

    stats = embed_module.BM25Stats(N=1, avgdl=1.0, df={})
    out_path = tmp_path / "doc.vectors.jsonl"
    validator = embed_module.SPLADEValidator()

    embed_module.process_chunk_file_vectors(
        chunk_file,
        out_path,
        stats,
        args,
        validator,
        embed_module.get_logger(__name__),
    )

    assert captured_texts == ["Hello world"]
    assert captured_write_texts == ["Hello world"]


def test_cli_path_overrides_take_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI supplied model directories should override environment variables."""

    _install_minimal_stubs(monkeypatch)
    dependency_stubs()
    sys.modules.pop("DocsToKG.DocParsing.embedding", None)
    import DocsToKG.DocParsing.embedding as embed_module

    env_splade = tmp_path / "env-splade"
    env_qwen = tmp_path / "env-qwen"
    cli_splade = tmp_path / "cli-splade"
    cli_qwen = tmp_path / "cli-qwen"
    cli_splade.mkdir()
    cli_qwen.mkdir()
    env_splade.mkdir()
    env_qwen.mkdir()

    monkeypatch.setenv("DOCSTOKG_SPLADE_DIR", str(env_splade))
    monkeypatch.setenv("DOCSTOKG_QWEN_DIR", str(env_qwen))

    captured: Dict[str, Path] = {}

    def _capture(
        chunk_file, out_path, stats, args, validator, logger
    ) -> tuple[int, List[int], List[float]]:
        captured["splade"] = args.splade_cfg.model_dir
        captured["qwen"] = args.qwen_cfg.model_dir
        return 0, [], []

    chunk_file = tmp_path / "example.chunks.jsonl"
    chunk_file.write_text(
        json.dumps({"uuid": "u1", "text": "text", "doc_id": "doc"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(embed_module, "iter_chunk_files", lambda _: [chunk_file])
    monkeypatch.setattr(
        embed_module,
        "process_pass_a",
        lambda files, logger: embed_module.BM25Stats(N=1, avgdl=1.0, df={}),
    )
    monkeypatch.setattr(embed_module, "process_chunk_file_vectors", _capture)
    monkeypatch.setattr(embed_module, "load_manifest_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(embed_module, "compute_content_hash", lambda *_: "hash")

    args = embed_module.parse_args(
        [
            "--chunks-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
            "--splade-model-dir",
            str(cli_splade),
            "--qwen-model-dir",
            str(cli_qwen),
        ]
    )

    exit_code = embed_module.main(args)

    assert exit_code == 0
    assert captured["splade"] == cli_splade.resolve()
    assert captured["qwen"] == cli_qwen.resolve()


def test_offline_mode_requires_local_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline mode should raise when required models are absent."""

    _install_minimal_stubs(monkeypatch)
    import DocsToKG.DocParsing.embedding as embed_module

    missing = tmp_path / "missing"

    args = embed_module.parse_args(
        [
            "--chunks-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
            "--splade-model-dir",
            str(missing / "splade"),
            "--qwen-model-dir",
            str(missing / "qwen"),
            "--offline",
        ]
    )

    with pytest.raises(FileNotFoundError) as exc:
        embed_module.main(args)

    message = str(exc.value)
    assert "SPLADE model directory missing" in message
    assert "Qwen model directory not found" in message


def test_validate_only_mode_checks_vectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation-only flag should scan vector files without invoking models."""

    embed_module = _reload_embedding_module(monkeypatch)
    from DocsToKG.DocParsing import schemas as _schemas

    vector_version = _schemas.VECTOR_SCHEMA_VERSION

    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    chunks_dir.mkdir()
    vectors_dir.mkdir()

    chunk_file = chunks_dir / "doc.chunks.jsonl"
    chunk_file.write_text(
        json.dumps({"uuid": "chunk-1", "doc_id": "doc", "text": "sample text"}) + "\n",
        encoding="utf-8",
    )

    vector_file = vectors_dir / "doc.vectors.jsonl"
    vector_file.write_text(
        json.dumps(
            {
                "UUID": "chunk-1",
                "BM25": {"terms": ["doc"], "weights": [1.0], "avgdl": 1.0, "N": 1},
                "SPLADEv3": {"tokens": ["doc"], "weights": [0.1]},
                "Qwen3-4B": {"model_id": "encoder", "vector": [0.1], "dimension": 1},
                "schema_version": vector_version,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))
    args = embed_module.parse_args(
        ["--chunks-dir", str(chunks_dir), "--out-dir", str(vectors_dir), "--validate-only"]
    )

    exit_code = embed_module.main(args)
    assert exit_code == 0


def _find_action(parser, option: str):
    for action in parser._actions:  # pragma: no cover - exercised in tests
        if option in action.option_strings:
            return action
    raise AssertionError(f"Option {option} not found in parser")


def test_splade_attn_help_text_describes_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
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

    manifests: List[Dict[str, Any]] = []

    def record_manifest(stage: str, doc_id: str, status: str, **metadata: Any) -> None:
        manifests.append({"stage": stage, "doc_id": doc_id, "status": status, **metadata})

    def capture_success(
        *,
        stage: str,
        doc_id: str,
        duration_s: float,
        schema_version: str,
        input_path: str | Path,
        input_hash: str,
        output_path: str | Path,
        hash_alg: str | None = None,
        **extra: Any,
    ) -> None:
        entry = {
            "stage": stage,
            "doc_id": doc_id,
            "status": "success",
            "duration_s": duration_s,
            "schema_version": schema_version,
            "input_path": input_path,
            "input_hash": input_hash,
            "output_path": output_path,
            "hash_alg": hash_alg,
        }
        entry.update(extra)
        manifests.append(entry)

    chunk_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    manifests_dir = tmp_path / "manifests"
    chunk_dir.mkdir()
    vectors_dir.mkdir()
    manifests_dir.mkdir()

    chunk_file = chunk_dir / "sample.chunks.jsonl"
    chunk_file.write_text(
        (
            '{"uuid": "chunk-1", "text": "Example", "doc_id": "doc", "schema_version": "docparse/1.1.0"}\n'
            '{"uuid": "chunk-1", "text": "Example", "doc_id": "doc"}\n'
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(embed_module, "manifest_append", record_manifest)
    monkeypatch.setattr(embed_module, "manifest_log_success", capture_success)
    monkeypatch.setattr(embed_module, "iter_chunk_files", lambda _: [chunk_file])
    monkeypatch.setattr(
        embed_module,
        "process_pass_a",
        lambda files, logger: embed_module.BM25Stats(N=1, avgdl=1.0, df={}),
    )

    def fake_process_chunk_file_vectors(
        chunk_file, out_path, stats, args, validator, logger
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
    monkeypatch.setattr(
        embed_module,
        "detect_data_root",
        lambda override=None: Path(override) if override else tmp_path,
    )
    embed_module._SPLADE_ENCODER_BACKENDS.clear()
    cfg = embed_module.SpladeCfg(model_dir=chunk_dir, device="cpu")
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

    def capture_success(
        *,
        stage: str,
        doc_id: str,
        duration_s: float,
        schema_version: str,
        input_path: str | Path,
        input_hash: str,
        output_path: str | Path,
        hash_alg: str | None = None,
        **extra: Any,
    ) -> None:
        entry = {
            "stage": stage,
            "doc_id": doc_id,
            "status": "success",
            "duration_s": duration_s,
            "schema_version": schema_version,
            "input_path": input_path,
            "input_hash": input_hash,
            "output_path": output_path,
            "hash_alg": hash_alg,
        }
        entry.update(extra)
        manifests.append(entry)

    def capture_success(
        *,
        stage: str,
        doc_id: str,
        duration_s: float,
        schema_version: str,
        input_path: str | Path,
        input_hash: str,
        output_path: str | Path,
        hash_alg: str | None = None,
        **extra: Any,
    ) -> None:
        entry = {
            "stage": stage,
            "doc_id": doc_id,
            "status": "success",
            "duration_s": duration_s,
            "schema_version": schema_version,
            "input_path": input_path,
            "input_hash": input_hash,
            "output_path": output_path,
            "hash_alg": hash_alg,
        }
        entry.update(extra)
        manifests.append(entry)

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(embed_module, "manifest_append", record_manifest)
    monkeypatch.setattr(embed_module, "manifest_log_success", capture_success)
    monkeypatch.setattr(embed_module, "manifest_log_success", capture_success)
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

    with pytest.raises(ValueError, match="Unsupported chunk schema version"):
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

    original_write_text = Path.write_text

    def _write_text(path_self: Path, *args, **kwargs):
        data = "".join(args) if args else ""
        return original_write_text(path_self, data, **kwargs)

    monkeypatch.setattr(Path, "write_text", _write_text)

    import DocsToKG.DocParsing.embedding as embed_module

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
