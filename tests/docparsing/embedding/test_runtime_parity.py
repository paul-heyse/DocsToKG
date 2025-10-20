from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from DocsToKG.DocParsing.core import cli as core_cli
from DocsToKG.DocParsing.embedding.backends import (
    ProviderBundle,
    ProviderContext,
    ProviderFactory,
    ProviderIdentity,
)


class _StubDenseProvider:
    identity = ProviderIdentity(name="dense.stub", version="1.0.0")

    def open(self, context: ProviderContext) -> None:
        self._ctx = context

    def close(self) -> None:
        return None

    def embed(self, texts, *, batch_hint=None):
        return [[float(len(text))] for text in texts]


class _StubSparseProvider:
    identity = ProviderIdentity(name="sparse.stub", version="1.0.0")

    def open(self, context: ProviderContext) -> None:
        self._ctx = context

    def close(self) -> None:
        return None

    def encode(self, texts):
        return [[(f"tok-{len(text)}", 1.0)] for text in texts]


class _StubLexicalProvider:
    identity = ProviderIdentity(name="lexical.stub", version="1.0.0")

    def open(self, context: ProviderContext) -> None:
        self._ctx = context

    def close(self) -> None:
        return None

    def accumulate_stats(self, chunks):  # pragma: no cover - not used
        return None

    def vector(self, text, stats):
        from DocsToKG.DocParsing.embedding.runtime import bm25_vector

        return bm25_vector(text, stats)


def _provider_bundle_from_cfg(cfg, telemetry_emitter=None) -> ProviderBundle:
    settings = cfg.provider_settings()["embedding"]
    context = ProviderContext(
        device=settings["device"],
        dtype=settings["dtype"],
        batch_hint=settings["batch_size"],
        max_concurrency=settings["max_concurrency"],
        normalize_l2=settings["normalize_l2"],
        offline=settings["offline"],
        cache_dir=settings["cache_dir"],
        telemetry_tags=settings["telemetry_tags"],
        telemetry_emitter=telemetry_emitter,
    )
    return ProviderBundle(
        dense=_StubDenseProvider(),
        sparse=_StubSparseProvider(),
        lexical=_StubLexicalProvider(),
        context=context,
    )


def test_embedding_stub_providers(tmp_path, patcher):
    runtime = importlib.import_module("DocsToKG.DocParsing.embedding.runtime")
    importlib.reload(runtime)

    data_root = tmp_path / "Data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    model_root = tmp_path / "models"
    qwen_dir = model_root / "qwen"
    splade_dir = model_root / "splade"
    chunks_dir.mkdir(parents=True)
    vectors_dir.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)
    splade_dir.mkdir(parents=True)

    chunk_path = chunks_dir / "doc-1.chunks.jsonl"
    chunk_path.write_text(
        json.dumps(
            {
                "uuid": "chunk-1",
                "doc_id": "doc-1",
                "text": "Example text",
                "schema_version": "docparse/1.0.0",
                "num_tokens": 2,
            }
        )
        + "\n"
    )

    patcher.setattr(ProviderFactory, "create", staticmethod(_provider_bundle_from_cfg))
    patcher.setattr(runtime, "ensure_model_environment", lambda: (model_root, model_root))
    patcher.setattr(runtime, "expand_path", lambda candidate: Path(candidate or model_root).resolve())
    patcher.setattr(runtime, "_resolve_qwen_dir", lambda _root: qwen_dir)
    patcher.setattr(runtime, "_resolve_splade_dir", lambda _root: splade_dir)
    patcher.setattr(runtime, "ensure_splade_environment", lambda **_: {"device": "cpu"})
    patcher.setattr(
        runtime, "ensure_qwen_environment", lambda **_: {"device": "cpu", "dtype": "float32"}
    )
    patcher.setattr(runtime, "_ensure_pyarrow_vectors", lambda: (object(), object()))
    patcher.setattr(runtime, "prepare_data_root", lambda override, detected: Path(override or detected))
    patcher.setattr(runtime, "detect_data_root", lambda: data_root)
    patcher.setattr(runtime, "data_chunks", lambda root, ensure=False: root / "ChunkedDocTagFiles")
    patcher.setattr(runtime, "data_vectors", lambda root, ensure=False: root / "Embeddings")

    runner = CliRunner()
    result = runner.invoke(
        core_cli.app,
        [
            "embed",
            "--data-root",
            str(data_root),
            "--chunks-dir",
            str(chunks_dir),
            "--out-dir",
            str(vectors_dir),
            "--offline",
        ],
    )
    assert result.exit_code == 0

    output_path = vectors_dir / "doc-1.vectors.jsonl"
    payload = json.loads(output_path.read_text().strip())
    assert payload["Qwen3-4B"]["vector"] == [12.0]
    assert payload["SPLADEv3"]["tokens"] == ["tok-12"]
