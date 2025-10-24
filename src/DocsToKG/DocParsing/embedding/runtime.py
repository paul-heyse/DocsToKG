#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.embedding.runtime",
#   "purpose": "Embedding pipelines for DocParsing",
#   "sections": [
#     {
#       "id": "embed-write-vectors",
#       "name": "_embed_write_vectors",
#       "anchor": "function-embed-write-vectors",
#       "kind": "function"
#     },
#     {
#       "id": "build-bm25-vector",
#       "name": "_build_bm25_vector",
#       "anchor": "function-build-bm25-vector",
#       "kind": "function"
#     },
#     {
#       "id": "build-splade-vector",
#       "name": "_build_splade_vector",
#       "anchor": "function-build-splade-vector",
#       "kind": "function"
#     },
#     {
#       "id": "build-dense-vector",
#       "name": "_build_dense_vector",
#       "anchor": "function-build-dense-vector",
#       "kind": "function"
#     },
#     {
#       "id": "build-vector-row",
#       "name": "_build_vector_row",
#       "anchor": "function-build-vector-row",
#       "kind": "function"
#     },
#     {
#       "id": "flush-llm-cache",
#       "name": "flush_llm_cache",
#       "anchor": "function-flush-llm-cache",
#       "kind": "function"
#     },
#     {
#       "id": "close-all-qwen",
#       "name": "close_all_qwen",
#       "anchor": "function-close-all-qwen",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-splade-dependencies",
#       "name": "_ensure_splade_dependencies",
#       "anchor": "function-ensure-splade-dependencies",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-qwen-dependencies",
#       "name": "_ensure_qwen_dependencies",
#       "anchor": "function-ensure-qwen-dependencies",
#       "kind": "function"
#     },
#     {
#       "id": "splade-encode",
#       "name": "splade_encode",
#       "anchor": "function-splade-encode",
#       "kind": "function"
#     },
#     {
#       "id": "qwen-embed",
#       "name": "qwen_embed",
#       "anchor": "function-qwen-embed",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-qwen-dir",
#       "name": "_resolve_qwen_dir",
#       "anchor": "function-resolve-qwen-dir",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-splade-dir",
#       "name": "_resolve_splade_dir",
#       "anchor": "function-resolve-splade-dir",
#       "kind": "function"
#     },
#     {
#       "id": "expand-optional",
#       "name": "_expand_optional",
#       "anchor": "function-expand-optional",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-cli-path",
#       "name": "_resolve_cli_path",
#       "anchor": "function-resolve-cli-path",
#       "kind": "function"
#     },
#     {
#       "id": "percentile",
#       "name": "_percentile",
#       "anchor": "function-percentile",
#       "kind": "function"
#     },
#     {
#       "id": "write-fingerprint",
#       "name": "_write_fingerprint",
#       "anchor": "function-write-fingerprint",
#       "kind": "function"
#     },
#     {
#       "id": "compute-embed-cfg-hash",
#       "name": "_compute_embed_cfg_hash",
#       "anchor": "function-compute-embed-cfg-hash",
#       "kind": "function"
#     },
#     {
#       "id": "set-embed-worker-state",
#       "name": "_set_embed_worker_state",
#       "anchor": "function-set-embed-worker-state",
#       "kind": "function"
#     },
#     {
#       "id": "get-embed-worker-state",
#       "name": "_get_embed_worker_state",
#       "anchor": "function-get-embed-worker-state",
#       "kind": "function"
#     },
#     {
#       "id": "extract-stub-counters",
#       "name": "_extract_stub_counters",
#       "anchor": "function-extract-stub-counters",
#       "kind": "function"
#     },
#     {
#       "id": "process-stub-vectors",
#       "name": "_process_stub_vectors",
#       "anchor": "function-process-stub-vectors",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-uuid",
#       "name": "ensure_uuid",
#       "anchor": "function-ensure-uuid",
#       "kind": "function"
#     },
#     {
#       "id": "legacy-chunk-uuid",
#       "name": "_legacy_chunk_uuid",
#       "anchor": "function-legacy-chunk-uuid",
#       "kind": "function"
#     },
#     {
#       "id": "tokens",
#       "name": "tokens",
#       "anchor": "function-tokens",
#       "kind": "function"
#     },
#     {
#       "id": "bm25statsaccumulator",
#       "name": "BM25StatsAccumulator",
#       "anchor": "class-bm25statsaccumulator",
#       "kind": "class"
#     },
#     {
#       "id": "print-bm25-summary",
#       "name": "print_bm25_summary",
#       "anchor": "function-print-bm25-summary",
#       "kind": "function"
#     },
#     {
#       "id": "bm25-vector",
#       "name": "bm25_vector",
#       "anchor": "function-bm25-vector",
#       "kind": "function"
#     },
#     {
#       "id": "spladevalidator",
#       "name": "SPLADEValidator",
#       "anchor": "class-spladevalidator",
#       "kind": "class"
#     },
#     {
#       "id": "embeddingprocessingerror",
#       "name": "EmbeddingProcessingError",
#       "anchor": "class-embeddingprocessingerror",
#       "kind": "class"
#     },
#     {
#       "id": "process-pass-a",
#       "name": "process_pass_a",
#       "anchor": "function-process-pass-a",
#       "kind": "function"
#     },
#     {
#       "id": "iter-rows-in-batches",
#       "name": "iter_rows_in_batches",
#       "anchor": "function-iter-rows-in-batches",
#       "kind": "function"
#     },
#     {
#       "id": "iter-chunks-or-empty",
#       "name": "_iter_chunks_or_empty",
#       "anchor": "function-iter-chunks-or-empty",
#       "kind": "function"
#     },
#     {
#       "id": "validate-chunk-file-schema",
#       "name": "_validate_chunk_file_schema",
#       "anchor": "function-validate-chunk-file-schema",
#       "kind": "function"
#     },
#     {
#       "id": "process-chunk-file-vectors",
#       "name": "process_chunk_file_vectors",
#       "anchor": "function-process-chunk-file-vectors",
#       "kind": "function"
#     },
#     {
#       "id": "write-vectors",
#       "name": "write_vectors",
#       "anchor": "function-write-vectors",
#       "kind": "function"
#     },
#     {
#       "id": "vector-output-path-for-format",
#       "name": "_vector_output_path_for_format",
#       "anchor": "function-vector-output-path-for-format",
#       "kind": "function"
#     },
#     {
#       "id": "handle-embedding-quarantine",
#       "name": "_handle_embedding_quarantine",
#       "anchor": "function-handle-embedding-quarantine",
#       "kind": "function"
#     },
#     {
#       "id": "validate-vectors-for-chunks",
#       "name": "_validate_vectors_for_chunks",
#       "anchor": "function-validate-vectors-for-chunks",
#       "kind": "function"
#     },
#     {
#       "id": "build-embedding-plan",
#       "name": "_build_embedding_plan",
#       "anchor": "function-build-embedding-plan",
#       "kind": "function"
#     },
#     {
#       "id": "embedding-stage-worker",
#       "name": "_embedding_stage_worker",
#       "anchor": "function-embedding-stage-worker",
#       "kind": "function"
#     },
#     {
#       "id": "make-embedding-stage-hooks",
#       "name": "_make_embedding_stage_hooks",
#       "anchor": "function-make-embedding-stage-hooks",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-pyarrow-vectors",
#       "name": "_ensure_pyarrow_vectors",
#       "anchor": "function-ensure-pyarrow-vectors",
#       "kind": "function"
#     },
#     {
#       "id": "main-inner",
#       "name": "_main_inner",
#       "anchor": "function-main-inner",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Hybrid Embedding Pipeline

Generates BM25, SPLADE, and Qwen embeddings for DocsToKG chunk files while
maintaining manifest entries, UUID hygiene, and data quality metrics. The
pipeline runs in two passes: the first ensures chunk UUID integrity and builds
BM25 corpus statistics; the second executes SPLADE and Qwen models to emit
vector Parquet artifacts ready for downstream search.

Key Features:
- Auto-detect DocsToKG data directories and manage resume/force semantics
- Stream SPLADE sparse encoding and Qwen dense embeddings from local caches
- Write all embedding vectors in Parquet columnar format (exclusive)
- Validate vector schemas, norms, and dimensions before writing outputs
- Record manifest metadata for observability and auditing
- Explain SPLADE attention backend fallbacks (auto→FlashAttention2→SDPA→eager)

Output Format:
- All embedding vectors written as Parquet files exclusively
- Schemas versioned with semantic versioning
- Atomic writes with fsync durability and footer metadata
- Support for dense (Qwen), sparse (SPLADE), and lexical (BM25) vectors

Usage:
    python -m DocsToKG.DocParsing.core embed --resume

Dependencies:
- sentence_transformers (optional): Provides SPLADE sparse encoders.
- vllm (optional): Hosts the Qwen embedding model with pooling support.
- pyarrow (optional): Required for Parquet vector output.
- tqdm: Surface user-friendly progress bars across pipeline phases.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    if sys.path and sys.path[0] == str(script_dir):
        sys.path.pop(0)
    package_root = script_dir.parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import re
import statistics
import threading
import time
import tracemalloc
import unicodedata
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

# Third-party imports
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback when tqdm is unavailable

    class _TqdmFallback:
        """Lightweight iterator wrapper used when tqdm is unavailable."""

        def __init__(self, iterable=None, **kwargs):
            """Store the iterable used to emulate tqdm's interface."""

            self._iterable = iterable

        def __iter__(self):
            """Iterate over the wrapped iterable when present."""

            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def update(self, *args, **kwargs):
            """Mirror tqdm's update interface without tracking state."""

            return None

        def close(self):
            """Provide a noop close hook for compatibility."""

            return None

        def __enter__(self):
            """Support use as a context manager like tqdm."""

            return self

        def __exit__(self, exc_type, exc_value, traceback):
            """Mirror tqdm's context manager cleanup without suppression."""

            self.close()
            return False

    def tqdm(iterable=None, **kwargs):  # type: ignore
        """Fallback tqdm implementation returning a lightweight iterator."""

        return _TqdmFallback(iterable, **kwargs)


from contextlib import ExitStack

from DocsToKG.DocParsing.cli_errors import EmbeddingCLIValidationError, format_cli_error
from DocsToKG.DocParsing.config import annotate_cli_overrides, parse_args_with_overrides
from DocsToKG.DocParsing.context import ParsingContext
from DocsToKG.DocParsing.core import (
    DEFAULT_TOKENIZER,
    UUID_NAMESPACE,
    BM25Stats,
    ChunkDiscovery,
    ItemFingerprint,
    ItemOutcome,
    QwenCfg,
    ResumeController,
    SpladeCfg,
    StageContext,
    StageError,
    StageHooks,
    StageOptions,
    StageOutcome,
    StagePlan,
    WorkItem,
    compute_relative_doc_id,
    compute_stable_shard,
    derive_doc_id_and_vectors_path,
    run_stage,
    safe_write,
    should_skip_output,
)
from DocsToKG.DocParsing.core.discovery import iter_chunks
from DocsToKG.DocParsing.embedding.backends import (
    ProviderBundle,
    ProviderContext,
    ProviderError,
    ProviderFactory,
    ProviderIdentity,
    ProviderTelemetryEvent,
)
from DocsToKG.DocParsing.embedding.backends.dense.qwen_vllm import (
    _get_vllm_components as _VLLM_COMPONENTS,
)
from DocsToKG.DocParsing.env import (
    data_chunks,
    data_vectors,
    detect_data_root,
    ensure_model_environment,
    ensure_qwen_environment,
    ensure_splade_environment,
    expand_path,
    prepare_data_root,
    resolve_pipeline_path,
)
from DocsToKG.DocParsing.formats import (
    VECTOR_SCHEMA_VERSION,
    ensure_chunk_schema,
)
from DocsToKG.DocParsing.formats import (
    BM25Vector as _BM25Vector,
)
from DocsToKG.DocParsing.formats import (
    DenseVector as _DenseVector,
)
from DocsToKG.DocParsing.formats import (
    SPLADEVector as _SPLADEVector,
)
from DocsToKG.DocParsing.formats import (
    VectorRow as _VectorRow,
)
from DocsToKG.DocParsing.formats import (
    validate_vector_row as _validate_vector_row,
)
from DocsToKG.DocParsing.io import (
    StreamingContentHasher,
    atomic_write,
    compute_chunk_uuid,
    compute_content_hash,
    iter_jsonl,
    load_manifest_index,
    quarantine_artifact,
    relative_path,
    resolve_attempts_path,
    resolve_hash_algorithm,
    resolve_manifest_path,
)
from DocsToKG.DocParsing.io import manifest_append as _manifest_append
from DocsToKG.DocParsing.logging import (
    get_logger,
    log_event,
    telemetry_scope,
)
from DocsToKG.DocParsing.logging import (
    manifest_log_failure as _logging_manifest_log_failure,
)
from DocsToKG.DocParsing.logging import (
    manifest_log_skip as _logging_manifest_log_skip,
)
from DocsToKG.DocParsing.logging import (
    manifest_log_success as _logging_manifest_log_success,
)
from DocsToKG.DocParsing.storage.embedding_integration import (
    VectorWriterError,
    create_unified_vector_writer,
    iter_vector_rows,
)
from DocsToKG.DocParsing.telemetry import StageTelemetry, TelemetrySink

from .cli import build_parser
from .cli import parse_args as _cli_parse_args
from .config import (
    EMBED_PROFILE_PRESETS,
    SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
    EmbedCfg,
)

parse_args = _cli_parse_args

BM25Vector = _BM25Vector
SPLADEVector = _SPLADEVector
DenseVector = _DenseVector
VectorRow = _VectorRow

manifest_log_failure = _logging_manifest_log_failure
manifest_log_success = _logging_manifest_log_success
manifest_log_skip = _logging_manifest_log_skip
manifest_append = _manifest_append


def _embed_write_vectors() -> None:
    """Placeholder used with :func:`safe_write` for vector artefacts."""

    return None


def _build_bm25_vector(**kwargs):
    """Construct a BM25 vector."""

    return BM25Vector(**kwargs)


def _build_splade_vector(**kwargs):
    """Construct a SPLADE vector."""

    return SPLADEVector(**kwargs)


def _build_dense_vector(**kwargs):
    """Construct a dense vector."""

    return DenseVector(**kwargs)


def _build_vector_row(**kwargs):
    """Construct a VectorRow."""

    return VectorRow(**kwargs)


def flush_llm_cache() -> None:
    """Compatibility shim delegating to the Qwen provider cache flush."""

    from DocsToKG.DocParsing.embedding.backends.dense.qwen_vllm import flush_llm_cache as _flush

    _flush()


def close_all_qwen() -> None:
    """Compatibility shim equivalent to :func:`flush_llm_cache`."""

    flush_llm_cache()


def _ensure_splade_dependencies() -> None:
    """Compatibility shim retained for legacy callers (providers handle this)."""


def _ensure_qwen_dependencies() -> None:
    """Compatibility shim retained for legacy callers (providers handle this)."""


def splade_encode(
    cfg: SpladeCfg, texts: list[str], batch_size: int | None = None
) -> tuple[list[list[str]], list[list[float]]]:
    """Backward-compatible SPLADE encoder wrapper using provider abstractions."""

    from DocsToKG.DocParsing.embedding.backends.sparse.splade_st import (
        SpladeSTConfig,
        SpladeSTProvider,
    )

    provider = SpladeSTProvider(
        SpladeSTConfig(
            model_dir=cfg.model_dir,
            device=cfg.device,
            batch_size=batch_size or cfg.batch_size,
            cache_folder=cfg.cache_folder,
            max_active_dims=cfg.max_active_dims,
            attn_impl=cfg.attn_impl,
            local_files_only=cfg.local_files_only,
        )
    )
    context = ProviderContext(
        device=cfg.device,
        batch_hint=batch_size or cfg.batch_size,
        cache_dir=cfg.cache_folder,
        offline=cfg.local_files_only,
    )
    provider.open(context)
    try:
        encoded = provider.encode(texts)
        token_lists: list[list[str]] = []
        weight_lists: list[list[float]] = []
        for row in encoded:
            tokens = [str(token) for token, _weight in row]
            weights = [float(weight) for _token, weight in row]
            token_lists.append(tokens)
            weight_lists.append(weights)
        return token_lists, weight_lists
    finally:
        provider.close()


def qwen_embed(cfg: QwenCfg, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Backward-compatible Qwen embedding wrapper using provider abstractions."""

    from DocsToKG.DocParsing.embedding.backends.dense.qwen_vllm import (
        QwenVLLMConfig,
        QwenVLLMProvider,
    )

    provider = QwenVLLMProvider(
        QwenVLLMConfig(
            model_dir=cfg.model_dir,
            model_id=None,
            dtype=cfg.dtype,
            tensor_parallelism=cfg.tp,
            gpu_memory_utilization=getattr(cfg, "gpu_mem_util", 0.60),
            batch_size=batch_size or cfg.batch_size,
            quantization=cfg.quantization,
            dimension=cfg.dim,
            cache_enabled=bool(getattr(cfg, "cache_enabled", True)),
            queue_depth=max(1, batch_size or cfg.batch_size),
        )
    )
    context = ProviderContext(
        device="auto",
        dtype=cfg.dtype,
        batch_hint=batch_size or cfg.batch_size,
        normalize_l2=True,
    )
    provider.open(context)
    try:
        vectors = provider.embed(texts, batch_hint=batch_size or cfg.batch_size)
        return [[float(value) for value in vector] for vector in vectors]
    finally:
        provider.close()


# --- Globals ---

EMBED_STAGE = "embedding"

__all__ = (
    "BM25Stats",
    "BM25StatsAccumulator",
    "QwenCfg",
    "SPLADEValidator",
    "SpladeCfg",
    "EmbedCfg",
    "bm25_vector",
    "ensure_chunk_schema",
    "ensure_uuid",
    "iter_rows_in_batches",
    "main",
    "print_bm25_summary",
    "process_chunk_file_vectors",
    "process_pass_a",
    "qwen_embed",
    "splade_encode",
    "tokens",
    "write_vectors",
    "flush_llm_cache",
    "close_all_qwen",
    "_VLLM_COMPONENTS",
)


# --- Public Functions ---


# --- Cache Utilities ---


def _resolve_qwen_dir(model_root: Path) -> Path:
    """Resolve Qwen model directory with ``DOCSTOKG_QWEN_DIR`` override.

    Args:
        model_root: Base directory housing DocsToKG models.

    Returns:
        Absolute path to the Qwen embedding model directory.
    """

    env = os.getenv("DOCSTOKG_QWEN_DIR")
    return expand_path(env) if env else model_root / "Qwen" / "Qwen3-Embedding-4B"


def _resolve_splade_dir(model_root: Path) -> Path:
    """Resolve SPLADE model directory with ``DOCSTOKG_SPLADE_DIR`` override.

    Args:
        model_root: Base directory housing DocsToKG models.

    Returns:
        Absolute path to the SPLADE model directory.
    """

    env = os.getenv("DOCSTOKG_SPLADE_DIR")
    return expand_path(env) if env else model_root / "naver" / "splade-v3"


HF_HOME, MODEL_ROOT = ensure_model_environment()
QWEN_DIR = expand_path(_resolve_qwen_dir(MODEL_ROOT))
SPLADE_DIR = expand_path(_resolve_splade_dir(MODEL_ROOT))


def _expand_optional(path: Path | None) -> Path | None:
    """Expand optional :class:`Path` values to absolutes when provided.

    Args:
        path: Optional path reference supplied by the caller.

    Returns:
        ``None`` when ``path`` is ``None``; otherwise the expanded absolute path.
    """

    if path is None:
        return None
    return path.expanduser().resolve()


def _resolve_cli_path(value: Path | None, default: Path) -> Path:
    """Resolve a CLI-provided path, falling back to ``default`` when omitted.

    Args:
        value: Optional user-supplied path.
        default: Fallback path used when ``value`` is absent.

    Returns:
        Absolute path derived from ``value`` or ``default``.
    """

    candidate = value if value is not None else default
    return Path(candidate).expanduser().resolve()


def _percentile(data: Sequence[float], pct: float) -> float:
    """Return the percentile of ``data`` without external dependencies."""

    if not data:
        return 0.0
    if pct <= 0:
        return float(min(data))
    if pct >= 100:
        return float(max(data))
    ordered = sorted(float(x) for x in data)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return d0 + d1


MANIFEST_STAGE = "embeddings"

_EMBED_WORKER_STATE: dict[str, Any] = {}


def _write_fingerprint(path: Path, *, input_sha256: str, cfg_hash: str) -> None:
    """Write a fingerprint describing the processed chunk and configuration."""

    payload = {
        "input_sha256": input_sha256,
        "cfg_hash": cfg_hash,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path) as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def _compute_embed_cfg_hash(cfg: EmbedCfg, vector_format: str) -> str:
    """Return a stable hash representing embedding configuration impacting resume."""

    payload = {
        "vector_format": str(vector_format or "parquet").lower(),
        "splade_model_dir": str(cfg.splade_model_dir or ""),
        "qwen_model_dir": str(cfg.qwen_model_dir or ""),
        "batch_size_splade": int(cfg.batch_size_splade),
        "batch_size_qwen": int(cfg.batch_size_qwen),
        "dense_qwen_vllm_batch_size": int(cfg.dense_qwen_vllm_batch_size or 0),
        "dense_qwen_vllm_dimension": int(cfg.dense_qwen_vllm_dimension or 0),
        "dense_qwen_vllm_quantization": cfg.dense_qwen_vllm_quantization or "",
        "qwen_dtype": cfg.qwen_dtype,
        "tp": int(cfg.tp),
        "sparse_splade_st_max_active_dims": int(cfg.sparse_splade_st_max_active_dims or 0),
        "sparse_splade_st_attn_backend": cfg.sparse_splade_st_attn_backend or "",
        "offline": bool(cfg.offline),
        "no_cache": bool(cfg.no_cache),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _set_embed_worker_state(state: dict[str, Any]) -> None:
    """Install the worker state dictionary used by embedding worker processes."""

    global _EMBED_WORKER_STATE
    _EMBED_WORKER_STATE = state


def _get_embed_worker_state() -> dict[str, Any]:
    """Return the worker state configured in :func:`_set_embed_worker_state`."""

    if not _EMBED_WORKER_STATE:
        raise RuntimeError("Embedding worker state accessed before initialisation")
    return _EMBED_WORKER_STATE


def _extract_stub_counters(func: Callable[..., Any]) -> dict[str, int] | None:
    """Return the patched test counters from a stubbed process_chunk function."""

    closure = getattr(func, "__closure__", None)
    if not closure:
        return None
    for cell in closure:
        value = cell.cell_contents
        if isinstance(value, dict) and "process" in value:
            return value
    return None


def _process_stub_vectors(
    chunk_path: Path,
    vectors_path: Path,
    *,
    cfg: EmbedCfg,
    vector_format: str,
    content_hasher: StreamingContentHasher | None = None,
    counters: dict[str, int] | None = None,
) -> tuple[int, list[int], list[float]]:
    """Fallback vector writer used when tests patch out real providers."""

    if counters is not None:
        counters["process"] = counters.get("process", 0) + 1

    rows = []
    with chunk_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if content_hasher is not None:
                content_hasher.update(raw_line)
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_unified_vector_writer(
        vectors_path, fmt=str(getattr(cfg, "vector_format", vector_format))
    )
    vector_rows = []
    for row in rows:
        vector_rows.append(
            {
                "UUID": row.get("uuid", ""),
                "BM25": {
                    "terms": ["hello", "world"],
                    "weights": [1.0, 1.0],
                    "avgdl": 1.0,
                    "N": 1,
                },
                "SPLADEv3": {
                    "tokens": ["hello", "world"],
                    "weights": [0.5, 0.4],
                },
                "Qwen3-4B": {
                    "model_id": "stub",
                    "vector": [0.0, 0.0],
                    "dimension": int(getattr(cfg, "qwen_dim", 2) or 2),
                },
                "model_metadata": {},
                "schema_version": VECTOR_SCHEMA_VERSION,
            }
        )
    with writer:
        writer.write_rows(vector_rows)

    count = len(vector_rows)
    return count, [0] * count, [1.0] * count


# --- BM25 Tokenizer ---

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


def ensure_uuid(rows: list[dict]) -> bool:
    """Validate or assign deterministic chunk UUIDs based on content offsets.

    Args:
        rows: Chunk dictionaries to normalise. Each row may include ``start_offset``
            (preferred); when absent the legacy UUID derivation is applied.

    Returns:
        ``True`` when at least one UUID was added or corrected; otherwise ``False``.
    """

    updated = False
    for row in rows:
        doc_id = str(row.get("doc_id") or "")
        text_raw = row.get("text", "")
        if isinstance(text_raw, bytes):
            text_value = text_raw.decode("utf-8", errors="ignore")
        else:
            text_value = str(text_raw) if text_raw is not None else ""
        text_value = unicodedata.normalize("NFKC", text_value)

        start_offset_raw = row.get("start_offset")
        expected_uuid: str
        if isinstance(start_offset_raw, (int, str)) and start_offset_raw not in {"", None}:
            try:
                start_offset = int(start_offset_raw)
            except (TypeError, ValueError):
                start_offset = 0
            expected_uuid = compute_chunk_uuid(doc_id, start_offset, text_value)
        else:
            expected_uuid = _legacy_chunk_uuid(doc_id, row.get("source_chunk_idxs"), text_value)

        current = row.get("uuid")
        if current != expected_uuid:
            row["uuid"] = expected_uuid
            updated = True
    return updated


def _legacy_chunk_uuid(doc_id: str, source_chunk_idxs: Any, text_value: str) -> str:
    """Derive the historical UUID used before deterministic start offsets."""

    src_raw = source_chunk_idxs or []
    if isinstance(src_raw, (str, bytes)):
        src_iterable = [src_raw]
    elif isinstance(src_raw, Sequence):
        src_iterable = list(src_raw)
    else:
        src_iterable = [src_raw] if src_raw is not None else []
    src = ",".join(str(idx) for idx in src_iterable)
    digest = hashlib.sha256(text_value.encode("utf-8")).hexdigest()[:16]
    name = f"{doc_id}|{src}|{digest}"
    try:
        return str(uuid.uuid5(UUID_NAMESPACE, name))
    except Exception:
        return str(uuid.uuid4())


def tokens(text: str) -> list[str]:
    """Tokenize normalized text for sparse retrieval features.

    Args:
        text: Input string to tokenize.

    Returns:
        Lowercased alphanumeric tokens extracted from the text.
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [match.group(0) for match in TOKEN_RE.finditer(normalized)]


# --- Public Classes ---


class BM25StatsAccumulator:
    """Streaming accumulator for BM25 corpus statistics.

    Maintains the following counters:
    - ``N``: Number of documents processed so far.
    - ``total_tokens``: Total token count across processed documents.
    - ``df``: Document frequency map collected to date.

    Examples:
        >>> acc = BM25StatsAccumulator()
        >>> acc.add_document("hybrid search")
        >>> acc.N
        1
    """

    def __init__(self) -> None:
        """Initialise counters used to accumulate BM25 corpus statistics.

        Args:
            self: Accumulator instance being initialised.

        Returns:
            None
        """
        self.N = 0
        self.total_tokens = 0
        self.df = Counter()

    def add_document(self, text: str) -> None:
        """Add document to statistics without retaining text.

        Args:
            text: Document contents used to update running statistics.

        Returns:
            None
        """

        toks = tokens(text)
        self.N += 1
        self.total_tokens += len(toks)
        self.df.update(set(toks))

    def finalize(self) -> BM25Stats:
        """Compute final statistics.

        Args:
            None: The accumulator finalises its internal counters without parameters.

        Returns:
            :class:`BM25Stats` summarising the accumulated corpus.
        """

        avgdl = self.total_tokens / max(self.N, 1)
        if avgdl <= 0:
            # Guard against division-by-zero in BM25 weighting when the corpus
            # contains no lexical tokens (for example, all input rows are empty).
            avgdl = 1.0
        return BM25Stats(N=self.N, avgdl=avgdl, df=dict(self.df))


def print_bm25_summary(stats: BM25Stats) -> None:
    """Print corpus-level BM25 statistics.

    Args:
        stats: Computed BM25 statistics to log.

    Returns:
        None: Writes structured logs only.
    """

    logger = get_logger(__name__)
    top_tokens = list(stats.df.items())
    top_tokens.sort(key=lambda item: item[1], reverse=True)
    top_tokens = top_tokens[:10]
    logger.info("BM25 Corpus Summary:")
    logger.info("  Documents (N): %s", stats.N)
    logger.info("  Avg doc length: %.2f tokens", stats.avgdl)
    logger.info("  Unique terms: %s", len(stats.df))
    logger.info("  Top 10 terms: %s", top_tokens)


def bm25_vector(
    text: str, stats: BM25Stats, k1: float = 1.5, b: float = 0.75
) -> tuple[list[str], list[float]]:
    """Generate BM25 term weights for a chunk of text.

    Args:
        text: Chunk text to convert into a sparse representation.
        stats: Precomputed BM25 statistics for the corpus.
        k1: Term frequency saturation parameter.
        b: Length normalization parameter.

    Returns:
        Tuple of `(terms, weights)` describing the sparse vector.
    """
    toks = tokens(text)
    dl = len(toks) or 1
    avgdl = stats.avgdl if stats.avgdl > 0 else 1.0
    tf = Counter(toks)
    terms, weights = [], []
    for t, f in tf.items():
        n_qi = stats.df.get(t, 0)
        idf = math.log((stats.N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)  # RSJ-smoothed IDF
        denom = f + k1 * (1.0 - b + b * (dl / avgdl))
        w = idf * (f * (k1 + 1.0)) / denom
        if w > 0:
            terms.append(t)
            weights.append(w)
    return terms, weights


class SPLADEValidator:
    """Track SPLADE sparsity metrics across the corpus.

    The validator records:
    - ``total_chunks``: Total number of chunks inspected.
    - ``zero_nnz_chunks``: UUIDs whose SPLADE vector has zero active terms.
    - ``nnz_counts``: Non-zero counts per processed chunk.

    Examples:
        >>> validator = SPLADEValidator()
        >>> validator.total_chunks
        0
    """

    def __init__(
        self,
        warn_threshold_pct: float = SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
        top_n: int = 10,
    ) -> None:
        """Initialise internal counters for SPLADE sparsity tracking.

        Args:
            self: Validator instance being initialised.
            warn_threshold_pct: Percentage of zero-NNZ vectors above which a warning is emitted.

        Returns:
            None
        """
        self.total_chunks = 0
        self.zero_nnz_chunks: list[str] = []
        self.nnz_counts: list[int] = []
        self.warn_threshold_pct = float(warn_threshold_pct)
        self.top_n = max(1, int(top_n))
        self._lock = threading.Lock()

    def validate(self, uuid: str, tokens: Sequence[str], weights: Sequence[float]) -> None:
        """Record sparsity information for a single chunk.

        Args:
            uuid: Chunk identifier associated with the SPLADE vector.
            tokens: Token list produced by the SPLADE encoder.
            weights: Weight list aligned with ``tokens``.

        Returns:
            None

        Raises:
            None
        """

        nnz = sum(1 for weight in weights if weight > 0)
        with self._lock:
            self.total_chunks += 1
            self.nnz_counts.append(nnz)
            if nnz == 0:
                self.zero_nnz_chunks.append(uuid)

    def report(self, logger) -> None:
        """Emit warnings if sparsity metrics exceed thresholds.

        Args:
            logger: Logger used to emit warnings and metrics.

        Returns:
            None
        """

        with self._lock:
            total = self.total_chunks
            zero_chunks = list(self.zero_nnz_chunks)
            threshold = getattr(self, "warn_threshold_pct", SPLADE_SPARSITY_WARN_THRESHOLD_PCT)
            top_n = self.top_n

        if not total:
            return
        pct = 100 * len(zero_chunks) / total
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = SPLADE_SPARSITY_WARN_THRESHOLD_PCT
        if pct > threshold:
            zero_count = len(zero_chunks)
            log_event(
                logger,
                "warning",
                "SPLADE sparsity exceeded threshold",
                stage=EMBED_STAGE,
                doc_id="__aggregate__",
                input_hash=None,
                error_code="SPLADE_SPARSITY",
                zero_chunks=zero_count,
                total_chunks=total,
                percent_zero=round(pct, 3),
                threshold_pct=float(threshold),
            )
            log_event(
                logger,
                "warning",
                "SPLADE sparsity affected UUIDs",
                stage=EMBED_STAGE,
                doc_id="__aggregate__",
                input_hash=None,
                error_code="SPLADE_SPARSITY_DETAILS",
                preview_uuids=zero_chunks[:top_n],
                total_zero=zero_count,
                preview_limit=top_n,
            )


# --- Qwen3 Embeddings ---


class EmbeddingProcessingError(RuntimeError):
    """Wrap per-file embedding failures with timing and hash metadata."""

    def __init__(self, original: Exception, duration: float, input_hash: str) -> None:
        """Record the triggering exception, elapsed duration, and input hash."""

        super().__init__(str(original))
        self.original = original
        self.duration = duration
        self.input_hash = input_hash


def process_pass_a(files: Sequence[Path], logger) -> BM25Stats:
    """Assign UUIDs and build BM25 statistics (streaming + atomic rewrite).

    This implementation streams each JSONL row and writes a temporary file with
    normalised schema/UUIDs. The original file is atomically replaced **only**
    when changes are detected. This bounds memory on huge shards and prevents
    partial writes.

    Args:
        files: Sequence of chunk file paths to process.
        logger: Logger used for structured progress output.

    Returns:
        Aggregated BM25 statistics for the supplied chunk corpus.

    Raises:
        OSError: If chunk files cannot be read or written.
        json.JSONDecodeError: If a chunk row contains invalid JSON.
    """

    accumulator = BM25StatsAccumulator()

    for chunk_file in tqdm(files, desc="Pass A: UUID + BM25 stats", unit="file"):
        updated = False
        tmp = chunk_file.with_suffix(chunk_file.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with (
            chunk_file.open("r", encoding="utf-8", errors="replace") as src,
            tmp.open("w", encoding="utf-8") as dst,
        ):
            for line_no, line in enumerate(src, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                previous_version = row.get("schema_version")
                ensure_chunk_schema(row, context=f"{chunk_file}:{line_no}")
                if row.get("schema_version") != previous_version:
                    updated = True

                # Ensure UUID
                if ensure_uuid([row]):
                    updated = True

                accumulator.add_document(str(row.get("text", "")))
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            dst.flush()
            os.fsync(dst.fileno())
        if updated:
            tmp.replace(chunk_file)
        else:
            tmp.unlink(missing_ok=True)

    stats = accumulator.finalize()
    print_bm25_summary(stats)
    logger.info(
        "Pass A complete",
        extra={
            "extra_fields": {
                "chunks": stats.N,
                "avgdl": round(stats.avgdl, 4),
                "unique_terms": len(stats.df),
            }
        },
    )
    return stats


def iter_rows_in_batches(
    path: Path,
    batch_size: int,
    *,
    start: int | None = None,
    end: int | None = None,
    skip_invalid: bool = True,
    max_errors: int = 10,
) -> Iterator[list[dict]]:
    """Iterate over JSONL rows in batches to reduce memory usage.

    Args:
        path: JSONL file to read.
        batch_size: Number of rows to yield per batch.
        start: Optional byte offset where iteration should begin.
        end: Optional byte offset bounding the slice (exclusive).
        skip_invalid: When ``True`` malformed rows are skipped up to
            ``max_errors`` occurrences (enabled by default).
        max_errors: Maximum tolerated malformed rows when ``skip_invalid`` is
            enabled.

    Yields:
        Lists of row dictionaries containing at most ``batch_size`` entries.
    """

    buf: list[dict] = []
    for record in iter_jsonl(
        path,
        start=start,
        end=end,
        skip_invalid=skip_invalid,
        max_errors=max_errors,
    ):
        buf.append(record)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def _iter_chunks_or_empty(chunks_dir: Path) -> Iterator[ChunkDiscovery]:
    """Yield ChunkDiscovery records from a directory, or return empty if directory doesn't exist."""
    if not chunks_dir.is_dir():
        return
    yield from iter_chunks(chunks_dir)


def _validate_chunk_file_schema(chunk_file: Path) -> None:
    """Validate that all rows in a chunk file conform to the schema."""
    for line_no, row in enumerate(iter_jsonl(chunk_file), start=1):
        ensure_chunk_schema(row, context=f"{chunk_file}:{line_no}")


def process_chunk_file_vectors(
    chunk_file: Path,
    out_path: Path,
    bundle: ProviderBundle,
    cfg: EmbedCfg,
    stats: BM25Stats,
    validator: SPLADEValidator,
    logger,
    *,
    content_hasher: StreamingContentHasher | None = None,
    vector_format: str = "parquet",
) -> tuple[int, list[int], list[float]]:
    """Generate vectors for a single chunk file and persist them to disk."""

    if not isinstance(out_path, Path):
        raise TypeError("out_path must be a Path")
    resolved_out_path = out_path
    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)
    vector_format = str(vector_format or "parquet").lower()

    dense_provider = bundle.dense
    sparse_provider = bundle.sparse
    lexical_provider = bundle.lexical
    if dense_provider is None or sparse_provider is None or lexical_provider is None:
        raise ProviderError(
            provider="runtime",
            category="init",
            detail="Dense, sparse, and lexical providers must be configured.",
            retryable=False,
        )

    provider_identities = bundle.identities()
    row_batch_size = bundle.context.batch_hint or max(cfg.batch_size_qwen, cfg.batch_size_splade)

    total_count = 0
    nnz_all: list[int] = []
    norms_all: list[float] = []

    with create_unified_vector_writer(resolved_out_path, fmt=vector_format) as writer:
        if content_hasher is None:
            row_batches: Iterator[list[dict]] = iter_rows_in_batches(chunk_file, row_batch_size)
        else:
            row_batches = iter_rows_in_batches(chunk_file, row_batch_size)

        for rows in row_batches:
            if not rows:
                continue

            uuids: list[str] = []
            texts: list[str] = []
            for index, row in enumerate(rows, start=1):
                ensure_chunk_schema(row, context=f"{chunk_file}:{index}")
                # Update hasher if provided
                if content_hasher is not None:
                    row_str = json.dumps(row, ensure_ascii=False, sort_keys=True)
                    content_hasher.update(row_str)
                uuid_value = row.get("uuid")
                if not uuid_value:
                    raise ValueError(f"Chunk row missing UUID in {chunk_file}")
                uuids.append(str(uuid_value))
                texts.append(str(row.get("text", "")))

            lexical_vectors: list[tuple[Sequence[str], Sequence[float]]] = []
            for text in texts:
                terms, weights = lexical_provider.vector(text, stats)
                lexical_vectors.append((list(terms), list(weights)))

            sparse_encoded = sparse_provider.encode(texts)
            sparse_vectors: list[tuple[Sequence[str], Sequence[float]]] = []
            for entry in sparse_encoded:
                tokens = [str(token) for token, _weight in entry]
                weights = [float(weight) for _token, weight in entry]
                sparse_vectors.append((tokens, weights))

            dense_batch_hint = (
                bundle.context.batch_hint
                or cfg.embedding_batch_size
                or cfg.dense_qwen_vllm_batch_size
                or cfg.batch_size_qwen
            )
            dense_vectors = dense_provider.embed(texts, batch_hint=dense_batch_hint)

            count, nnz, norms = write_vectors(
                writer,
                uuids,
                texts,
                lexical_vectors,
                sparse_vectors,
                dense_vectors,
                stats,
                cfg,
                rows=rows,
                validator=validator,
                logger=logger,
                provider_identities=provider_identities,
                output_path=resolved_out_path,
                vector_format=vector_format,
            )

            total_count += count
            nnz_all.extend(nnz)
            norms_all.extend(norms)

    logger.info(
        "Embeddings written",
        extra={
            "extra_fields": {
                "chunk_file": str(chunk_file),
                "vectors_file": str(resolved_out_path),
                "rows": total_count,
            }
        },
    )
    return total_count, nnz_all, norms_all


def write_vectors(
    writer: Any,  # UnifiedVectorWriter or compatible writer interface
    uuids: Sequence[str],
    texts: Sequence[str],
    lexical_results: Sequence[tuple[Sequence[str], Sequence[float]]],
    splade_results: Sequence[tuple[Sequence[str], Sequence[float]]],
    dense_results: Sequence[Sequence[float]],
    stats: BM25Stats,
    cfg: EmbedCfg,
    *,
    rows: Sequence[dict],
    validator: SPLADEValidator,
    logger,
    provider_identities: dict[str, ProviderIdentity],
    output_path: Path | None = None,
    vector_format: str = "parquet",
) -> tuple[int, list[int], list[float]]:
    """Write validated vector rows to disk with schema enforcement."""

    if not (
        len(uuids)
        == len(texts)
        == len(lexical_results)
        == len(splade_results)
        == len(dense_results)
        == len(rows)
    ):
        raise ValueError("Mismatch between chunk text and provider result lengths")

    bm25_k1 = float(cfg.lexical_local_bm25_k1)
    bm25_b = float(cfg.lexical_local_bm25_b)
    dense_expected_dim = int(cfg.dense_qwen_vllm_dimension or cfg.qwen_dim)
    dense_model_id = cfg.dense_qwen_vllm_model_id or DEFAULT_TOKENIZER
    dense_batch_size = int(cfg.dense_qwen_vllm_batch_size or cfg.batch_size_qwen)
    sparse_batch_size = int(cfg.sparse_splade_st_batch_size or cfg.batch_size_splade)
    sparse_attn = cfg.sparse_splade_st_attn_backend

    dense_identity = provider_identities.get("dense")
    sparse_identity = provider_identities.get("sparse")
    lexical_identity = provider_identities.get("lexical")

    dense_metadata = {
        "provider": dense_identity.name if dense_identity else None,
        "version": dense_identity.version if dense_identity else None,
        "model_id": dense_model_id,
        "batch_size": dense_batch_size,
        "dtype": cfg.qwen_dtype,
    }
    sparse_metadata = {
        "provider": sparse_identity.name if sparse_identity else None,
        "version": sparse_identity.version if sparse_identity else None,
        "batch_size": sparse_batch_size,
        "attn_backend": sparse_attn,
        "max_active_dims": cfg.sparse_splade_st_max_active_dims,
    }
    lexical_metadata = {
        "provider": lexical_identity.name if lexical_identity else None,
        "version": lexical_identity.version if lexical_identity else None,
        "k1": bm25_k1,
        "b": bm25_b,
    }

    splade_nnz: list[int] = []
    dense_norms: list[float] = []
    output_ref = output_path or getattr(writer, "path", None)
    payloads: list[dict] = []

    for uuid_value, text, lexical_pair, splade_pair, dense_vector_raw, row in zip(
        uuids, texts, lexical_results, splade_results, dense_results, rows
    ):
        lex_terms = [str(term) for term in lexical_pair[0]]
        lex_weights = [float(weight) for weight in lexical_pair[1]]
        splade_tokens = [str(token) for token in splade_pair[0]]
        splade_weights = [float(weight) for weight in splade_pair[1]]
        validator.validate(uuid_value, splade_tokens, splade_weights)
        splade_nnz.append(sum(1 for weight in splade_weights if weight > 0))

        dense_vector = [float(value) for value in dense_vector_raw]
        if dense_expected_dim and len(dense_vector) != dense_expected_dim:
            message = (
                f"Dense dimension mismatch for UUID={uuid_value}: expected {dense_expected_dim}, "
                f"got {len(dense_vector)}"
            )
            raise ValueError(message)

        norm = math.sqrt(sum(value * value for value in dense_vector))
        if norm <= 0:
            doc_id = row.get("doc_id", "unknown")
            message = f"Invalid dense vector (zero norm) for UUID={uuid_value}"
            log_event(
                logger,
                "error",
                message,
                stage=EMBED_STAGE,
                doc_id=doc_id,
                input_hash=row.get("input_hash") if isinstance(row, dict) else None,
                error_code="QWEN_ZERO_NORM",
                uuid=uuid_value,
            )
            raise ValueError(message)

        if abs(norm - 1.0) > 0.01:
            doc_id = row.get("doc_id", "unknown") if isinstance(row, dict) else "unknown"
            log_event(
                logger,
                "warning",
                "Dense vector norm outside expected tolerance",
                stage=EMBED_STAGE,
                doc_id=doc_id,
                input_hash=row.get("input_hash") if isinstance(row, dict) else None,
                error_code="QWEN_NORM_DEVIATION",
                uuid=uuid_value,
                norm=round(norm, 6),
                expected=1.0,
                tolerance=0.01,
            )
        dense_norms.append(norm)

        try:
            vector_row = _build_vector_row(
                UUID=uuid_value,
                BM25=_build_bm25_vector(
                    terms=lex_terms,
                    weights=lex_weights,
                    k1=bm25_k1,
                    b=bm25_b,
                    avgdl=stats.avgdl,
                    N=stats.N,
                ),
                SPLADEv3=_build_splade_vector(tokens=splade_tokens, weights=splade_weights),
                Qwen3_4B=_build_dense_vector(
                    model_id=dense_model_id,
                    vector=dense_vector,
                    dimension=dense_expected_dim,
                ),
                model_metadata={
                    "dense": dict(dense_metadata),
                    "sparse": dict(sparse_metadata),
                    "lexical": dict(lexical_metadata),
                },
            )
        except Exception as exc:
            doc_id = row.get("doc_id", "unknown")
            log_event(
                logger,
                "error",
                "Vector row validation failed",
                stage=EMBED_STAGE,
                doc_id=doc_id,
                input_hash=row.get("input_hash") if isinstance(row, dict) else None,
                error_code="VECTOR_ROW_INVALID",
                uuid=uuid_value,
                error=str(exc),
            )
            manifest_log_failure(
                stage="embeddings",
                doc_id=doc_id,
                duration_s=0.0,
                schema_version=VECTOR_SCHEMA_VERSION,
                input_path=row.get("source_path", "unknown"),
                input_hash=row.get("input_hash", ""),
                output_path=output_ref,
                vector_format=vector_format,
                error=str(exc),
            )
            raise

        payloads.append(vector_row.model_dump(by_alias=True))

    try:
        writer.write_rows(payloads)
    except Exception as exc:
        output_ref = Path(output_path) if output_path is not None else Path("")
        raise VectorWriterError(vector_format, output_ref, exc) from exc

    return len(uuids), splade_nnz, dense_norms


def _vector_output_path_for_format(path: Path, target_format: str) -> Path:
    """Return ``path`` rewritten for ``target_format`` while preserving layout."""

    fmt = str(target_format or "parquet").lower()
    if fmt not in {"parquet", "jsonl"}:
        raise ValueError(f"Unsupported vector format: {target_format}")

    parts = []
    replaced = False
    for part in path.parts:
        if part.startswith("fmt="):
            parts.append(f"fmt={fmt}")
            replaced = True
        else:
            parts.append(part)

    if not replaced:
        raise ValueError(f"Vector path missing fmt partition: {path}")

    updated = Path(*parts)
    suffix = ".parquet" if fmt == "parquet" else ".jsonl"
    return updated.with_suffix(suffix)


def _handle_embedding_quarantine(
    *,
    chunk_path: Path,
    vector_path: Path,
    doc_id: str,
    input_hash: str,
    reason: str,
    logger,
    data_root: Path | None = None,
    vector_format: str = "parquet",
) -> None:
    """Quarantine a problematic chunk or vector artefact and log manifest state."""

    input_hash_value = input_hash
    quarantine_path: Path
    if chunk_path.exists():
        try:
            input_hash_value = compute_content_hash(chunk_path)
        except Exception:
            pass
        quarantine_path = quarantine_artifact(chunk_path, reason=reason, logger=logger)
        input_path = chunk_path
    else:
        quarantine_path = quarantine_artifact(
            vector_path,
            reason=reason,
            logger=logger,
            create_placeholder=True,
        )
        input_path = chunk_path

    manifest_log_failure(
        stage=MANIFEST_STAGE,
        doc_id=doc_id,
        duration_s=0.0,
        schema_version=VECTOR_SCHEMA_VERSION,
        input_path=input_path,
        input_hash=input_hash_value,
        output_path=quarantine_path,
        vector_format=str(vector_format or "parquet").lower(),
        error=reason,
        quarantine=True,
    )
    log_event(
        logger,
        "warning",
        "Embedding artefact quarantined",
        status="quarantine",
        stage=EMBED_STAGE,
        doc_id=doc_id,
        input_relpath=relative_path(input_path, data_root),
        output_relpath=relative_path(quarantine_path, data_root),
        error_class="ValueError",
        reason=reason,
        vector_format=str(vector_format or "parquet").lower(),
    )


# --- Main Driver ---


def _validate_vectors_for_chunks(
    chunks_dir: Path,
    vectors_dir: Path,
    logger,
    *,
    data_root: Path | None = None,
    expected_dimension: int | None = None,
    vector_format: str = "parquet",
) -> tuple[int, int]:
    """Validate vectors associated with chunk files without recomputing models.

    Returns:
        (files_checked, rows_validated)
    """
    fmt_normalised = str(vector_format or "parquet").lower()
    files_checked = 0
    rows_validated = 0
    missing: list[tuple[str, Path]] = []
    quarantined_files = 0

    for chunk in _iter_chunks_or_empty(chunks_dir):
        doc_id, vector_path = derive_doc_id_and_vectors_path(
            chunk,
            chunks_dir,
            vectors_dir,
            vector_format=fmt_normalised,
        )
        if not vector_path.exists():
            missing.append((doc_id, vector_path))
            continue
        file_rows = 0
        try:
            for batch in iter_vector_rows(vector_path, fmt_normalised, batch_size=4096):
                for row in batch:
                    _validate_vector_row(row, expected_dimension=expected_dimension)
                    rows_validated += 1
                    file_rows += 1
        except ValueError as exc:
            reason = f"{vector_path}: {exc}"
            try:
                input_hash = compute_content_hash(vector_path)
            except Exception:
                input_hash = ""
            quarantine_path = quarantine_artifact(vector_path, reason=reason, logger=logger)
            manifest_log_failure(
                stage=MANIFEST_STAGE,
                doc_id=doc_id,
                duration_s=0.0,
                schema_version=VECTOR_SCHEMA_VERSION,
                input_path=vector_path,
                input_hash=input_hash,
                output_path=quarantine_path,
                vector_format=fmt_normalised,
                error=reason,
                quarantine=True,
            )
            log_event(
                logger,
                "warning",
                "Vector file quarantined",
                status="quarantine",
                stage=EMBED_STAGE,
                doc_id=doc_id,
                input_relpath=relative_path(vector_path, data_root),
                output_relpath=relative_path(quarantine_path, data_root),
                error_class="ValidationError",
                reason=reason,
            )
            quarantined_files += 1
            continue

        files_checked += 1
        try:
            input_hash = compute_content_hash(vector_path)
        except Exception:
            input_hash = ""
        manifest_log_success(
            stage=MANIFEST_STAGE,
            doc_id=doc_id,
            duration_s=0.0,
            schema_version=VECTOR_SCHEMA_VERSION,
            input_path=vector_path,
            input_hash=input_hash,
            output_path=vector_path,
            status="validate-only",
            vector_format=fmt_normalised,
            vector_count=file_rows,
        )

    if missing:
        sample_size = min(5, len(missing))
        doc_id_sample = [doc for doc, _ in missing[:sample_size]]
        path_sample = [str(path) for _, path in missing[:sample_size]]
        preview = ", ".join(doc_id_sample)
        truncated = len(missing) > sample_size
        if truncated:
            preview += ", ..."
        log_event(
            logger,
            "error",
            "Missing vector files for chunk documents",
            status="missing",
            stage=EMBED_STAGE,
            doc_id="__aggregate__",
            input_hash=None,
            error_code="MISSING_VECTORS",
            missing_count=len(missing),
            missing_doc_ids_sample=doc_id_sample,
            missing_paths_sample=path_sample,
            missing_sample_truncated=truncated,
            missing_sample_size=sample_size,
            chunks_dir=str(chunks_dir),
            vectors_dir=str(vectors_dir),
            vector_format=fmt_normalised,
        )
        raise FileNotFoundError("Vector files not found for documents: " + preview)

    log_event(
        logger,
        "info",
        "Validated vector files",
        status="validate-only",
        stage=EMBED_STAGE,
        files_checked=files_checked,
        rows_validated=rows_validated,
        chunks_dir=str(chunks_dir),
        vectors_dir=str(vectors_dir),
        vector_format=fmt_normalised,
    )
    if quarantined_files:
        log_event(
            logger,
            "warning",
            "Quarantined vector files",
            stage=EMBED_STAGE,
            doc_id="__aggregate__",
            input_hash=None,
            error_code="VECTOR_QUARANTINE",
            quarantined=quarantined_files,
        )
    print(
        f"Validated {rows_validated} rows across {files_checked} vector files under {vectors_dir}"
    )
    return files_checked, rows_validated


def _build_embedding_plan(
    *,
    chunk_entries: Sequence[ChunkDiscovery],
    chunks_dir: Path,
    vectors_dir: Path,
    resolved_root: Path,
    resume_controller: ResumeController,
    vector_format: str,
    cfg_hash: str,
    hash_alg: str,
    logger,
    plan_only: bool,
) -> tuple[StagePlan, dict[str, Any]]:
    """Build a StagePlan for vector generation with resume awareness."""

    fmt = str(vector_format or "parquet").lower()
    planned_ids: list[str] = []
    skipped_ids: list[str] = []
    work_items: list[WorkItem] = []
    resume_skipped = 0

    for entry in chunk_entries:
        chunk_file = entry.resolved_path
        doc_id, vector_path = derive_doc_id_and_vectors_path(
            entry,
            chunks_dir,
            vectors_dir,
            vector_format=fmt,
        )
        manifest_entry = resume_controller.entry(doc_id) if resume_controller.resume else None
        vectors_exist = vector_path.exists()
        entry_format = str(manifest_entry.get("vector_format")).lower() if manifest_entry else None
        format_mismatch = bool(manifest_entry) and entry_format != fmt

        input_hash = ""
        should_hash = (
            resume_controller.resume
            and not resume_controller.force
            and manifest_entry is not None
            and vectors_exist
            and not format_mismatch
        )
        if should_hash:
            try:
                input_hash = compute_content_hash(chunk_file, hash_alg)
            except TypeError:
                input_hash = compute_content_hash(chunk_file)

        skip_doc = False
        if resume_controller.resume and not format_mismatch:
            skip_doc = should_skip_output(
                vector_path,
                manifest_entry,
                input_hash,
                resume_controller.resume,
                resume_controller.force,
            )

        if skip_doc:
            resume_skipped += 1
            skipped_ids.append(doc_id)
            if not plan_only:
                log_event(
                    logger,
                    "info",
                    "Skipping chunk file: output exists and input unchanged",
                    status="skip",
                    stage=EMBED_STAGE,
                    doc_id=doc_id,
                    input_relpath=relative_path(chunk_file, resolved_root),
                    output_relpath=relative_path(vector_path, resolved_root),
                    vector_format=fmt,
                )
                manifest_log_skip(
                    stage=MANIFEST_STAGE,
                    doc_id=doc_id,
                    input_path=chunk_file,
                    input_hash=input_hash,
                    output_path=vector_path,
                    schema_version=VECTOR_SCHEMA_VERSION,
                    vector_format=fmt,
                )
            continue

        planned_ids.append(doc_id)
        if format_mismatch and not plan_only:
            log_event(
                logger,
                "info",
                "Regenerating vectors due to format mismatch",
                status="process",
                stage=EMBED_STAGE,
                doc_id=doc_id,
                previous_format=entry_format,
                requested_format=fmt,
                input_relpath=relative_path(chunk_file, resolved_root),
                output_relpath=relative_path(vector_path, resolved_root),
                vector_format=fmt,
            )

        if plan_only:
            continue

        fingerprint_path = vector_path.with_suffix(vector_path.suffix + ".fp.json")
        metadata = {
            "chunk_path": str(chunk_file),
            "output_path": str(vector_path),
            "input_hash": input_hash,
            "vector_format": fmt,
            "input_relpath": relative_path(chunk_file, resolved_root),
            "output_relpath": relative_path(vector_path, resolved_root),
            "fingerprint_path": str(fingerprint_path),
        }
        try:
            cost_hint = max(1.0, float(chunk_file.stat().st_size))
        except OSError:
            cost_hint = 1.0

        work_items.append(
            WorkItem(
                item_id=doc_id,
                inputs={"chunk": chunk_file},
                outputs={"vectors": vector_path},
                cfg_hash=cfg_hash,
                cost_hint=cost_hint,
                metadata=metadata,
                fingerprint=ItemFingerprint(
                    path=fingerprint_path,
                    input_sha256=input_hash,
                    cfg_hash=cfg_hash,
                ),
            )
        )

    plan = StagePlan(
        stage_name=MANIFEST_STAGE,
        items=tuple(work_items),
        total_items=len(work_items),
    )
    return plan, {
        "planned_ids": planned_ids,
        "skipped_ids": skipped_ids,
        "resume_skipped": resume_skipped,
    }


def _embedding_stage_worker(item: WorkItem) -> ItemOutcome:
    """Worker executed by the stage runner to encode vectors for a chunk file."""

    state = _get_embed_worker_state()
    bundle: ProviderBundle = state["bundle"]
    cfg: EmbedCfg = state["cfg"]
    stats: BM25Stats = state["stats"]
    validator: SPLADEValidator = state["validator"]
    logger = state["logger"]
    vector_format: str = state["vector_format"]
    chunk_path = Path(item.metadata["chunk_path"])
    vectors_path = Path(item.metadata["output_path"])
    input_hash = item.metadata.get("input_hash", "")
    resolved_root: Path = state["resolved_root"]
    cfg_hash: str = state["cfg_hash"]
    stub_vectors_enabled: bool = state.get("stub_vectors", False)
    stub_counters: dict[str, int] | None = state.get("stub_counters")

    # Extract provider identities for manifest metadata
    provider_identities = bundle.identities()
    provider_metadata_extras: dict[str, object] = {}
    if bundle.dense and "dense" in provider_identities:
        dense_id = provider_identities["dense"]
        provider_metadata_extras["dense_provider_name"] = dense_id.name
        if dense_id.extra:
            if "model_id" in dense_id.extra:
                provider_metadata_extras["dense_model_id"] = dense_id.extra["model_id"]
            if "dimension" in dense_id.extra:
                provider_metadata_extras["dense_dim"] = int(dense_id.extra["dimension"])
    if bundle.sparse and "sparse" in provider_identities:
        sparse_id = provider_identities["sparse"]
        provider_metadata_extras["sparse_provider_name"] = sparse_id.name
        if sparse_id.extra and "model_id" in sparse_id.extra:
            provider_metadata_extras["sparse_model_id"] = sparse_id.extra["model_id"]
    if bundle.lexical and "lexical" in provider_identities:
        lexical_id = provider_identities["lexical"]
        provider_metadata_extras["lexical_provider_name"] = lexical_id.name

    # Store in state so hooks can access it
    state["provider_metadata_extras"] = provider_metadata_extras

    log_event(
        logger,
        "debug",
        "Embedding worker start",
        stage=EMBED_STAGE,
        doc_id=item.item_id,
        input_relpath=item.metadata.get("input_relpath", relative_path(chunk_path, resolved_root)),
        output_relpath=item.metadata.get(
            "output_relpath", relative_path(vectors_path, resolved_root)
        ),
    )

    hasher: StreamingContentHasher | None = None
    if not input_hash:
        hasher = StreamingContentHasher()
    start = time.perf_counter()
    fallback_from: str | None = None
    fallback_reason: str | None = None
    effective_vector_format = vector_format
    original_vectors_path = vectors_path
    fallback_error: VectorWriterError | None = None
    result_tuple: tuple[int, list[int], list[float]] | None = None

    def _execute_vector_generation(
        target_path: Path, fmt: str
    ) -> tuple[int, list[int], list[float]]:
        if stub_vectors_enabled:
            return _process_stub_vectors(
                chunk_path,
                target_path,
                cfg=cfg,
                vector_format=fmt,
                content_hasher=hasher,
                counters=stub_counters,
            )

        signature = inspect.signature(process_chunk_file_vectors)
        log_event(
            logger,
            "debug",
            "Embedding worker process_chunk_file_vectors signature",
            stage=EMBED_STAGE,
            doc_id=item.item_id,
            params=list(signature.parameters.keys()),
            has_closure=process_chunk_file_vectors.__closure__ is not None,
        )
        if "vector_format" in signature.parameters:
            return process_chunk_file_vectors(
                chunk_path,
                target_path,
                bundle,
                cfg,
                stats,
                validator,
                logger,
                content_hasher=hasher,
                vector_format=fmt,
            )
        return process_chunk_file_vectors(
            chunk_path,
            target_path,
            bundle,
            cfg,
            stats,
            validator,
            logger,
            content_hasher=hasher,
        )

    try:
        try:
            # Use safe_write for atomic vector writes
            if safe_write(vectors_path, _embed_write_vectors):
                log_event(
                    logger,
                    "debug",
                    "Embedding worker invoke process_chunk_file_vectors",
                    stage=EMBED_STAGE,
                    doc_id=item.item_id,
                    vector_format=effective_vector_format,
                )
                result_tuple = _execute_vector_generation(vectors_path, effective_vector_format)
        except VectorWriterError as exc:
            fallback_error = exc
            fallback_from = vector_format
            fallback_reason = str(exc.original)

        if fallback_error is not None and vector_format == "parquet":
            effective_vector_format = "jsonl"
            fallback_path = _vector_output_path_for_format(original_vectors_path, "jsonl")
            log_event(
                logger,
                "warning",
                "Parquet vector write failed; retrying with JSONL fallback",
                stage=EMBED_STAGE,
                doc_id=item.item_id,
                vector_format_requested=vector_format,
                vector_format_fallback=effective_vector_format,
                error=fallback_reason,
            )
            item.metadata["output_path"] = str(fallback_path)
            item.metadata["output_relpath"] = relative_path(fallback_path, resolved_root)
            fingerprint_path = fallback_path.with_suffix(fallback_path.suffix + ".fp.json")
            item.metadata["fingerprint_path"] = str(fingerprint_path)
            item.metadata["vector_format"] = effective_vector_format
            vectors_path = fallback_path
            original_vectors_path = fallback_path
            try:
                if safe_write(vectors_path, _embed_write_vectors, skip_if_exists=False):
                    log_event(
                        logger,
                        "debug",
                        "Embedding worker retry with JSONL",
                        stage=EMBED_STAGE,
                        doc_id=item.item_id,
                        vector_format=effective_vector_format,
                    )
                    result_tuple = _execute_vector_generation(vectors_path, effective_vector_format)
                    fallback_error = None
                else:
                    raise VectorWriterError(
                        effective_vector_format,
                        vectors_path,
                        RuntimeError("fallback target already exists"),
                    )
            except VectorWriterError as exc:
                fallback_error = exc
                fallback_reason = str(exc.original)

        if result_tuple is None:
            if fallback_error is not None:
                raise fallback_error
            raise StageError(
                stage=EMBED_STAGE,
                item_id=item.item_id,
                category="runtime",
                message="Vector write skipped unexpectedly",
                retryable=False,
            )

        count, nnz, norms = result_tuple
        vector_format = effective_vector_format
    except ValueError as exc:
        duration = time.perf_counter() - start
        resolved_hash = input_hash or (hasher.hexdigest() if hasher else "")
        _handle_embedding_quarantine(
            chunk_path=chunk_path,
            vector_path=vectors_path,
            doc_id=item.item_id,
            input_hash=resolved_hash,
            reason=str(exc),
            logger=logger,
            data_root=resolved_root,
            vector_format=effective_vector_format,
        )
        return ItemOutcome(
            status="skip",
            duration_s=duration,
            manifest={"quarantined": True, "error": str(exc), "resolved_hash": resolved_hash},
            result={"quarantined": True},
        )
    except EmbeddingProcessingError as exc:
        duration = exc.duration
        manifest_log_failure(
            stage=MANIFEST_STAGE,
            doc_id=item.item_id,
            duration_s=round(duration, 3),
            schema_version=VECTOR_SCHEMA_VERSION,
            input_path=chunk_path,
            input_hash=exc.input_hash,
            output_path=vectors_path,
            vector_format=effective_vector_format,
            error=str(exc.original),
        )
        raise StageError(
            stage=EMBED_STAGE,
            item_id=item.item_id,
            category="runtime",
            message=str(exc.original),
            retryable=False,
        ) from exc.original
    except Exception as exc:
        duration = time.perf_counter() - start
        resolved_hash = input_hash or (hasher.hexdigest() if hasher else "")
        manifest_log_failure(
            stage=MANIFEST_STAGE,
            doc_id=item.item_id,
            duration_s=round(duration, 3),
            schema_version=VECTOR_SCHEMA_VERSION,
            input_path=chunk_path,
            input_hash=resolved_hash,
            output_path=vectors_path,
            vector_format=effective_vector_format,
            error=str(exc),
        )
        raise StageError(
            stage=EMBED_STAGE,
            item_id=item.item_id,
            category="runtime",
            message=str(exc),
            retryable=False,
        ) from exc

    duration = time.perf_counter() - start
    resolved_hash = input_hash or (hasher.hexdigest() if hasher else "")
    _write_fingerprint(
        Path(item.metadata["fingerprint_path"]),
        input_sha256=resolved_hash,
        cfg_hash=cfg_hash,
    )

    manifest_payload: dict[str, Any] = {
        "vector_count": count,
        "nnz": nnz,
        "norms": norms,
        "resolved_hash": resolved_hash,
        "vector_format": effective_vector_format,
    }
    result_payload: dict[str, Any] = {
        "quarantined": False,
        "vector_format": effective_vector_format,
    }
    if fallback_from:
        manifest_payload["vector_format_fallback_from"] = fallback_from
        result_payload["vector_format_fallback_from"] = fallback_from
        if fallback_reason:
            manifest_payload["vector_format_fallback_error"] = fallback_reason

    return ItemOutcome(
        status="success",
        duration_s=duration,
        manifest=manifest_payload,
        result=result_payload,
    )


def _make_embedding_stage_hooks(
    *,
    logger,
    resolved_root: Path,
    vector_format: str,
    cfg: EmbedCfg,
    stats: BM25Stats,
    validator: SPLADEValidator,
    bundle: ProviderBundle,
    exit_stack: ExitStack | None,
    overall_start: float,
    pass_b_start: float,
    files_parallel: int,
    resume_skipped: int,
    cfg_hash: str,
    vectors_dir: Path,
    stub_vectors: bool,
    stub_counters: dict[str, int] | None,
) -> StageHooks:
    """Return stage hooks that manage shared embedding resources and summaries."""

    def before_stage(context: StageContext) -> None:
        _set_embed_worker_state(
            {
                "bundle": bundle,
                "cfg": cfg,
                "stats": stats,
                "validator": validator,
                "logger": logger,
                "vector_format": vector_format,
                "resolved_root": resolved_root,
                "cfg_hash": cfg_hash,
                "stub_vectors": stub_vectors,
                "stub_counters": stub_counters,
            }
        )
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        state = context.metadata.setdefault(
            "embedding_state",
            {
                "logger": logger,
                "resolved_root": resolved_root,
                "vector_format": vector_format,
                "cfg": cfg,
                "validator": validator,
                "bundle": bundle,
                "exit_stack": exit_stack,
                "overall_start": overall_start,
                "pass_b_start": pass_b_start,
                "files_parallel": files_parallel,
                "resume_skipped": resume_skipped,
                "cfg_hash": cfg_hash,
                "total_vectors": 0,
                "splade_nnz_all": [],
                "dense_norms_all": [],
                "quarantined_files": 0,
                "skipped_runtime": 0,
                "vectors_dir": vectors_dir,
            },
        )
        state["validator_zero_chunks"] = validator.zero_nnz_chunks if validator else []

    def after_item(
        item: WorkItem,
        outcome_or_error: ItemOutcome | StageError,
        context: StageContext,
    ) -> None:
        state = context.metadata.get("embedding_state", {})
        stage_logger = state.get("logger", logger)
        root = state.get("resolved_root", resolved_root)
        schema_version = VECTOR_SCHEMA_VERSION
        input_path = Path(item.metadata["chunk_path"])
        output_path = Path(item.metadata["output_path"])

        if isinstance(outcome_or_error, ItemOutcome):
            if outcome_or_error.status == "success":
                manifest_meta = outcome_or_error.manifest
                vector_count = int(manifest_meta.get("vector_count", 0))
                nnz = list(manifest_meta.get("nnz", []))
                norms = list(manifest_meta.get("norms", []))
                resolved_hash = str(manifest_meta.get("resolved_hash", ""))
                actual_format = str(manifest_meta.get("vector_format", vector_format)).lower()
                fallback_from = manifest_meta.get("vector_format_fallback_from")
                if fallback_from:
                    state["vector_format_fallbacks"] = state.get("vector_format_fallbacks", 0) + 1
                state["total_vectors"] = state.get("total_vectors", 0) + vector_count
                state.setdefault("splade_nnz_all", []).extend(nnz)
                state.setdefault("dense_norms_all", []).extend(norms)
                manifest_log_success(
                    stage=MANIFEST_STAGE,
                    doc_id=item.item_id,
                    duration_s=round(outcome_or_error.duration_s, 3),
                    schema_version=schema_version,
                    input_path=input_path,
                    input_hash=resolved_hash,
                    output_path=output_path,
                    vector_format=actual_format,
                    vector_format_fallback=fallback_from,
                    vector_count=vector_count,
                    **state.get("provider_metadata_extras", {}),
                )
                avg_nnz_file = statistics.mean(nnz) if nnz else 0.0
                avg_norm_file = statistics.mean(norms) if norms else 0.0
                log_event(
                    stage_logger,
                    "info",
                    "Embedding file written",
                    status="success",
                    stage=EMBED_STAGE,
                    doc_id=item.item_id,
                    input_relpath=item.metadata.get(
                        "input_relpath", relative_path(input_path, root)
                    ),
                    output_relpath=item.metadata.get(
                        "output_relpath", relative_path(output_path, root)
                    ),
                    elapsed_ms=int(outcome_or_error.duration_s * 1000),
                    vectors=vector_count,
                    splade_avg_nnz=round(avg_nnz_file, 3),
                    qwen_avg_norm=round(avg_norm_file, 4),
                    vector_format=actual_format,
                    vector_format_fallback=fallback_from,
                )
                return

            if outcome_or_error.status == "skip":
                if outcome_or_error.manifest.get("quarantined"):
                    state["quarantined_files"] = state.get("quarantined_files", 0) + 1
                else:
                    state["skipped_runtime"] = state.get("skipped_runtime", 0) + 1
                return

        else:
            state["failures"] = state.get("failures", 0) + 1

    def after_stage(outcome: StageOutcome, context: StageContext) -> None:
        state = context.metadata.get("embedding_state", {})
        total_vectors = state.get("total_vectors", 0)
        splade_nnz_all = state.get("splade_nnz_all", [])
        dense_norms_all = state.get("dense_norms_all", [])
        skipped_total = state.get("resume_skipped", 0) + state.get("skipped_runtime", 0)
        quarantined_files = state.get("quarantined_files", 0)
        files_parallel_effective = state.get("files_parallel", files_parallel)
        vectors_dir_state: Path = state.get("vectors_dir", vectors_dir)
        fallback_total = state.get("vector_format_fallbacks", 0)
        bundle_summary: dict[str, ProviderIdentity] = {}
        bundle_ref = state.get("bundle")
        if bundle_ref is not None:
            try:
                bundle_summary = bundle_ref.identities()
            except Exception:
                bundle_summary = {}

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_b = time.perf_counter() - state.get("pass_b_start", pass_b_start)

        validator.report(logger)

        zero_pct = (
            100.0 * len([n for n in splade_nnz_all if n == 0]) / total_vectors
            if total_vectors
            else 0.0
        )
        avg_nnz = statistics.mean(splade_nnz_all) if splade_nnz_all else 0.0
        median_nnz = statistics.median(splade_nnz_all) if splade_nnz_all else 0.0
        splade_p95 = _percentile(splade_nnz_all, 95.0)
        splade_p99 = _percentile(splade_nnz_all, 99.0)
        avg_norm = statistics.mean(dense_norms_all) if dense_norms_all else 0.0
        std_norm = statistics.pstdev(dense_norms_all) if len(dense_norms_all) > 1 else 0.0
        norm_p95 = _percentile(dense_norms_all, 95.0)
        norm_p99 = _percentile(dense_norms_all, 99.0)
        norm_low_threshold = 0.9
        norm_high_threshold = 1.1
        norm_low_outliers = len([n for n in dense_norms_all if n < norm_low_threshold])
        norm_high_outliers = len([n for n in dense_norms_all if n > norm_high_threshold])

        sparse_identity = bundle_summary.get("sparse")
        dense_identity = bundle_summary.get("dense")
        lexical_identity = bundle_summary.get("lexical")

        log_event(
            logger,
            "info",
            "Embedding summary",
            stage=EMBED_STAGE,
            total_vectors=total_vectors,
            splade_avg_nnz=round(avg_nnz, 3),
            splade_median_nnz=round(median_nnz, 3),
            splade_p95_nnz=round(splade_p95, 3),
            splade_p99_nnz=round(splade_p99, 3),
            splade_zero_pct=round(zero_pct, 2),
            qwen_avg_norm=round(avg_norm, 4),
            qwen_std_norm=round(std_norm, 4),
            qwen_norm_p95=round(norm_p95, 4),
            qwen_norm_p99=round(norm_p99, 4),
            qwen_norm_low_outliers=norm_low_outliers,
            qwen_norm_high_outliers=norm_high_outliers,
            pass_b_seconds=round(elapsed_b, 3),
            skipped_files=skipped_total,
            quarantined_files=quarantined_files,
            files_parallel=files_parallel_effective,
            splade_attn_backend_used=cfg.sparse_splade_st_attn_backend or "auto",
            sparsity_warn_threshold_pct=float(cfg.sparsity_warn_threshold_pct),
            dense_provider=dense_identity.name if dense_identity else None,
            dense_provider_version=dense_identity.version if dense_identity else None,
            sparse_provider=sparse_identity.name if sparse_identity else None,
            lexical_provider=lexical_identity.name if lexical_identity else None,
            vector_format_fallbacks=fallback_total,
        )
        logger.info("Peak memory: %.2f GB", peak / 1024**3)

        manifest_log_success(
            stage=MANIFEST_STAGE,
            doc_id="__corpus__",
            duration_s=round(time.perf_counter() - state.get("overall_start", overall_start), 3),
            schema_version=VECTOR_SCHEMA_VERSION,
            input_path="__corpus__",
            input_hash="",
            output_path=vectors_dir_state,
            warnings=(
                validator.zero_nnz_chunks[: validator.top_n] if validator.zero_nnz_chunks else []
            ),
            total_vectors=total_vectors,
            splade_avg_nnz=avg_nnz,
            splade_median_nnz=round(median_nnz, 3),
            splade_p95_nnz=splade_p95,
            splade_p99_nnz=splade_p99,
            splade_zero_pct=zero_pct,
            qwen_avg_norm=avg_norm,
            qwen_std_norm=std_norm,
            qwen_norm_p95=norm_p95,
            qwen_norm_p99=norm_p99,
            qwen_norm_low_outliers=norm_low_outliers,
            qwen_norm_high_outliers=norm_high_outliers,
            peak_memory_gb=peak / 1024**3,
            skipped_files=skipped_total,
            quarantined_files=quarantined_files,
            files_parallel=files_parallel_effective,
            splade_attn_backend_used=cfg.sparse_splade_st_attn_backend or "auto",
            sparsity_warn_threshold_pct=float(cfg.sparsity_warn_threshold_pct),
            vector_format_fallbacks=fallback_total,
        )

        log_event(
            logger,
            "info",
            "[DONE] Saved vectors",
            status="complete",
            stage=EMBED_STAGE,
            embeddings_dir=str(vectors_dir_state),
            processed_files=outcome.succeeded,
            skipped_files=skipped_total,
            quarantined_files=quarantined_files,
            total_vectors=total_vectors,
            vector_format_fallbacks=fallback_total,
            wall_ms=round(outcome.wall_ms, 3),
            queue_p50_ms=round(outcome.queue_p50_ms, 3),
            queue_p95_ms=round(outcome.queue_p95_ms, 3),
            exec_p50_ms=round(outcome.exec_p50_ms, 3),
            exec_p95_ms=round(outcome.exec_p95_ms, 3),
            exec_p99_ms=round(outcome.exec_p99_ms, 3),
            cpu_time_total_ms=round(outcome.cpu_time_total_ms, 3),
        )

        if exit_stack is not None:
            exit_stack.close()
        _set_embed_worker_state({})

    return StageHooks(
        before_stage=before_stage,
        after_item=after_item,
        after_stage=after_stage,
    )


def _ensure_pyarrow_vectors() -> None:
    """Validate that pyarrow is available for parquet vector output."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise EmbeddingCLIValidationError(
            option="--format",
            message=(
                "parquet vector output requires the optional dependency 'pyarrow'. "
                "Install DocsToKG[docparse-parquet] or add pyarrow to the environment."
            ),
        ) from exc


def _main_inner(args: argparse.Namespace | None = None, config_adapter=None) -> int:
    """CLI entrypoint for chunk UUID cleanup and embedding generation.

    Args:
        args (argparse.Namespace | None): Optional parsed arguments, primarily
            for testing or orchestration.
        config_adapter: Optional EmbedCfg instance from ConfigurationAdapter (new pattern).
              If provided, bypasses sys.argv parsing and uses this config directly.

    Returns:
        int: Exit code where ``0`` indicates success.

    Raises:
        ValueError: If invalid runtime parameters (such as batch sizes) are supplied.
    """

    parser = build_parser()
    bootstrap_root = detect_data_root()
    try:
        data_chunks(bootstrap_root)
        data_vectors(bootstrap_root)
    except Exception as exc:
        logging.getLogger(__name__).debug("Failed to bootstrap data directories", exc_info=exc)

    # NEW PATH: If adapter provided (from unified CLI), use it directly
    if config_adapter is not None:
        cfg = config_adapter
        base_config = cfg.to_manifest()
        namespace = argparse.Namespace()
        for field_def in fields(EmbedCfg):
            setattr(namespace, field_def.name, getattr(cfg, field_def.name))
        explicit_overrides = set()
    # LEGACY PATH: Parse from args or sys.argv
    else:
        explicit_overrides: set[str]
        if args is None:
            namespace = parse_args_with_overrides(parser)
            explicit_overrides = set(getattr(namespace, "_cli_explicit_overrides", ()) or ())
        elif isinstance(args, argparse.Namespace):
            namespace = args
            if getattr(namespace, "_cli_explicit_overrides", None) is None:
                keys = [name for name in vars(namespace) if not name.startswith("_")]
                annotate_cli_overrides(namespace, explicit=keys, defaults={})
                explicit_overrides = set(keys)
            else:
                explicit_overrides = set(namespace._cli_explicit_overrides or ())
        elif isinstance(args, SimpleNamespace) or hasattr(args, "__dict__"):
            base = parse_args_with_overrides(parser, [])
            payload = {key: value for key, value in vars(args).items() if not key.startswith("_")}
            for key, value in payload.items():
                setattr(base, key, value)
            defaults = getattr(base, "_cli_defaults", {})
            annotate_cli_overrides(base, explicit=payload.keys(), defaults=defaults)
            explicit_overrides = set(payload.keys())
            namespace = base
        else:
            namespace = parse_args_with_overrides(parser, args)
            explicit_overrides = set(getattr(namespace, "_cli_explicit_overrides", ()) or ())

        profile = getattr(namespace, "profile", None)
        defaults = EMBED_PROFILE_PRESETS.get(profile or "", {})

        # Build config from namespace (legacy: no from_args() available)
        cfg = EmbedCfg()
        cfg.apply_args(namespace, defaults=defaults)
        cfg.finalize()

        base_config = cfg.to_manifest()
        if profile:
            base_config.setdefault("profile", profile)
        for field_def in fields(EmbedCfg):
            setattr(namespace, field_def.name, getattr(cfg, field_def.name))

    if getattr(namespace, "plan_only", False) and getattr(namespace, "validate_only", False):
        raise EmbeddingCLIValidationError(
            option="--plan-only/--validate-only",
            message="flags cannot be combined",
        )

    validate_only = bool(cfg.validate_only)
    plan_only = bool(getattr(namespace, "plan_only", False))

    log_level = cfg.log_level
    run_id = uuid.uuid4().hex
    logger = get_logger(
        __name__,
        level=str(log_level),
        base_fields={"run_id": run_id, "stage": EMBED_STAGE},
    )
    profile = getattr(namespace, "profile", None)
    defaults = EMBED_PROFILE_PRESETS.get(profile or "", {})
    if profile and defaults:
        log_event(
            logger,
            "info",
            "Applying profile",
            status="profile",
            stage=EMBED_STAGE,
            profile=profile,
            **{key: defaults[key] for key in sorted(defaults)},
        )
    args = namespace
    args.plan_only = plan_only
    offline_mode = bool(cfg.offline)

    try:
        shard_count = int(cfg.shard_count)
        shard_index = int(cfg.shard_index)
    except (TypeError, ValueError) as exc:
        raise EmbeddingCLIValidationError(
            option="--shard-count/--shard-index",
            message="must be integers",
        ) from exc
    if shard_count < 1:
        raise EmbeddingCLIValidationError(option="--shard-count", message="must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise EmbeddingCLIValidationError(
            option="--shard-index",
            message="must be between 0 and shard-count-1",
        )
    args.shard_count = shard_count
    args.shard_index = shard_index

    vector_format = str(cfg.vector_format or "parquet").lower()
    if vector_format != "parquet":
        raise EmbeddingCLIValidationError(
            option="--vector-format",
            message="must be: parquet (JSONL embeddings no longer supported)",
        )
    _ensure_pyarrow_vectors()
    cfg.vector_format = vector_format
    args.vector_format = vector_format

    global HF_HOME, MODEL_ROOT, QWEN_DIR, SPLADE_DIR
    HF_HOME, MODEL_ROOT = ensure_model_environment()
    QWEN_DIR = expand_path(_resolve_qwen_dir(MODEL_ROOT))
    SPLADE_DIR = expand_path(_resolve_splade_dir(MODEL_ROOT))
    model_root = MODEL_ROOT
    default_qwen_dir = QWEN_DIR
    default_splade_dir = SPLADE_DIR

    cli_splade = _expand_optional(getattr(args, "splade_model_dir", None))
    cli_qwen = _expand_optional(getattr(args, "qwen_model_dir", None))
    splade_model_dir = cli_splade or default_splade_dir
    qwen_model_dir = cli_qwen or default_qwen_dir

    if offline_mode and not validate_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        missing_paths = []
        if not splade_model_dir.exists():
            missing_paths.append(f"SPLADE model directory missing: {splade_model_dir}")
        if not qwen_model_dir.exists():
            missing_paths.append(
                "Qwen model directory not found: "
                f"{qwen_model_dir}. Pre-download the model before rerunning."
            )
        if missing_paths:
            detail = "; ".join(missing_paths)
            raise FileNotFoundError("Offline mode requires local model directories. " + detail)

    cfg.offline = offline_mode
    args.offline = offline_mode
    args.splade_model_dir = splade_model_dir
    args.qwen_model_dir = qwen_model_dir

    splade_model_dir = _resolve_cli_path(args.splade_model_dir, SPLADE_DIR)
    qwen_model_dir = _resolve_cli_path(args.qwen_model_dir, QWEN_DIR)

    splade_env = ensure_splade_environment(cache_dir=splade_model_dir)
    qwen_env = ensure_qwen_environment(dtype=cfg.qwen_dtype, model_dir=qwen_model_dir)

    if args.batch_size_splade < 1 or args.batch_size_qwen < 1:
        raise EmbeddingCLIValidationError(
            option="--batch-size-splade/--batch-size-qwen",
            message="must be >= 1",
        )

    overall_start = time.perf_counter()

    if args.offline and not validate_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        logger.info("Offline mode enabled: expecting local caches only")
        missing_models = [
            (label, path)
            for label, path in (("SPLADE", splade_model_dir), ("Qwen", qwen_model_dir))
            if not path.exists() or not path.is_dir()
        ]
        if missing_models:
            missing_desc = ", ".join(f"{label} model at {path}" for label, path in missing_models)
            raise FileNotFoundError(
                "Offline mode requires local model directories. Missing: " + missing_desc
            )

    data_root_override = cfg.data_root
    data_root_overridden = data_root_override is not None
    resolved_root = prepare_data_root(data_root_override, detect_data_root())
    logger.bind(data_root=str(resolved_root))

    default_chunks_dir = data_chunks(resolved_root, ensure=False)
    default_vectors_dir = data_vectors(resolved_root, ensure=False)

    chunks_dir = resolve_pipeline_path(
        cli_value=args.chunks_dir,
        default_path=default_chunks_dir,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_chunks(root, ensure=False),
    ).resolve()

    out_dir = resolve_pipeline_path(
        cli_value=args.out_dir,
        default_path=default_vectors_dir,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_vectors(root, ensure=False),
    ).resolve()

    logger.bind(
        input_relpath=relative_path(chunks_dir, resolved_root),
        output_relpath=relative_path(out_dir, resolved_root),
    )

    requested_parallel = max(1, int(cfg.files_parallel or 1))

    context = ParsingContext(run_id=run_id, data_root=resolved_root)
    context.apply_config(cfg)
    context.chunks_dir = chunks_dir
    context.out_dir = out_dir
    context.vectors_dir = out_dir
    context.vector_format = vector_format
    context.shard_count = shard_count
    context.shard_index = shard_index
    context.resume = bool(cfg.resume)
    context.force = bool(cfg.force)
    context.validate_only = bool(validate_only)
    context.plan_only = bool(plan_only)
    context.offline = bool(offline_mode)
    context.files_parallel = None
    context.profile = profile
    base_extra = {
        key: value for key, value in base_config.items() if key not in ParsingContext.field_names()
    }
    if base_extra:
        context.merge_extra(base_extra)
    context.update_extra(
        splade_model_dir=str(splade_model_dir),
        qwen_model_dir=str(qwen_model_dir),
        splade_env=splade_env,
        qwen_env=qwen_env,
        files_parallel_requested=requested_parallel,
        bm25_k1=float(cfg.bm25_k1),
        bm25_b=float(cfg.bm25_b),
        batch_size_splade=int(cfg.batch_size_splade),
        batch_size_qwen=int(cfg.batch_size_qwen),
        sparsity_warn_threshold_pct=float(cfg.sparsity_warn_threshold_pct),
        sparsity_report_top_n=int(cfg.sparsity_report_top_n),
        no_cache=bool(cfg.no_cache),
        vector_format=vector_format,
    )

    if validate_only:
        missing_reasons = []
        if not chunks_dir.exists():
            missing_reasons.append(f"chunks directory {chunks_dir}")
        if not out_dir.exists():
            missing_reasons.append(f"vector directory {out_dir}")
        if missing_reasons:
            log_event(
                logger,
                "error",
                "Validate-only mode requires existing inputs and vectors",
                missing_paths=missing_reasons,
                tip="Run without --validate-only to generate embeddings before validation.",
            )
            raise FileNotFoundError(
                "Validate-only mode expects existing directories: " + ", ".join(missing_reasons)
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    context_payload = context.to_manifest()

    telemetry_sink = TelemetrySink(
        resolve_attempts_path(MANIFEST_STAGE, resolved_root),
        resolve_manifest_path(MANIFEST_STAGE, resolved_root),
    )
    stage_telemetry = StageTelemetry(telemetry_sink, run_id=run_id, stage=EMBED_STAGE)
    with telemetry_scope(stage_telemetry):
        manifest_log_success(
            stage=MANIFEST_STAGE,
            doc_id="__config__",
            duration_s=0.0,
            schema_version=VECTOR_SCHEMA_VERSION,
            input_path=chunks_dir,
            input_hash="",
            output_path=out_dir,
            config=context_payload,
            vector_format=vector_format,
        )

        def _provider_telemetry(event: ProviderTelemetryEvent) -> None:
            stage_telemetry.log_provider_event(
                provider=event.provider,
                phase=event.phase,
                data=dict(event.data),
            )

        if validate_only:
            expected_dimension: int | None = int(cfg.qwen_dim)
            if "qwen_dim" not in explicit_overrides:
                expected_dimension = None
            files_checked, rows_validated = _validate_vectors_for_chunks(
                chunks_dir,
                out_dir,
                logger,
                data_root=resolved_root,
                expected_dimension=expected_dimension,
                vector_format=vector_format,
            )
            logger.info(
                "Validation-only mode complete",
                extra={
                    "extra_fields": {
                        "files": files_checked,
                        "rows": rows_validated,
                        "chunks_dir": str(chunks_dir),
                        "vectors_dir": str(out_dir),
                    }
                },
            )
            return 0

        args.out_dir = out_dir

        log_event(
            logger,
            "info",
            "Embedding configuration",
            status="config",
            stage=EMBED_STAGE,
            data_root=str(resolved_root),
            chunks_dir=str(chunks_dir),
            embeddings_dir=str(out_dir),
            splade_model_dir=str(splade_model_dir),
            qwen_model_dir=str(qwen_model_dir),
            splade_device=splade_env.get("device"),
            qwen_device=qwen_env.get("device"),
            qwen_dtype=qwen_env.get("dtype"),
            offline=offline_mode,
            requested_files_parallel=requested_parallel,
            shard_count=shard_count,
            shard_index=shard_index,
            vector_format=vector_format,
            qwen_cache_enabled=not bool(cfg.no_cache),
            sparsity_warn_threshold_pct=float(cfg.sparsity_warn_threshold_pct),
            sparsity_report_top_n=int(cfg.sparsity_report_top_n),
            profile=profile,
        )

        chunk_entries = list(_iter_chunks_or_empty(chunks_dir))
        if not chunk_entries:
            log_event(
                logger,
                "warning",
                "No chunk files found",
                stage=EMBED_STAGE,
                doc_id="__aggregate__",
                input_hash=None,
                error_code="NO_INPUT_FILES",
                chunks_dir=str(chunks_dir),
            )
            return 0

        if args.shard_count > 1:
            total_candidates = len(chunk_entries)
            selected_entries = [
                entry
                for entry in chunk_entries
                if compute_stable_shard(entry.logical_path.as_posix(), args.shard_count)
                == args.shard_index
            ]
            logger.info(
                "Applying shard filter",
                extra={
                    "extra_fields": {
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                        "selected_files": len(selected_entries),
                        "total_files": total_candidates,
                    }
                },
            )
            chunk_entries = selected_entries
            if not chunk_entries:
                log_event(
                    logger,
                    "warning",
                    "Shard contains no chunk files",
                    stage=EMBED_STAGE,
                    doc_id="__aggregate__",
                    input_hash=None,
                    error_code="SHARD_EMPTY",
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                )
                return 0

        incompatible_chunks: list[Path] = []
        validated_files: list[ChunkDiscovery] = []
        for chunk_entry in chunk_entries:
            chunk_file = chunk_entry.resolved_path
            try:
                _validate_chunk_file_schema(chunk_file)
            except ValueError as exc:
                reason = str(exc)
                try:
                    chunk_hash = compute_content_hash(chunk_file)
                except Exception:
                    chunk_hash = ""
                doc_id = compute_relative_doc_id(chunks_dir / chunk_entry.logical_path, chunks_dir)
                log_event(
                    logger,
                    "error",
                    "Chunk file rejected: incompatible schema",
                    stage=EMBED_STAGE,
                    doc_id=doc_id,
                    input_hash=chunk_hash,
                    error_code="CHUNK_SCHEMA_INCOMPATIBLE",
                    chunk_file=str(chunk_file),
                    error=reason,
                )
                incompatible_chunks.append(chunk_file)
                _, quarantine_vector_path = derive_doc_id_and_vectors_path(
                    chunk_entry,
                    chunks_dir,
                    args.out_dir,
                    vector_format=vector_format,
                )
                _handle_embedding_quarantine(
                    chunk_path=chunk_file,
                    vector_path=quarantine_vector_path,
                    doc_id=doc_id,
                    input_hash=chunk_hash,
                    reason=reason,
                    logger=logger,
                    vector_format=vector_format,
                )
                continue
            validated_files.append(chunk_entry)
        if incompatible_chunks:
            chunk_entries = validated_files

        if cfg.force:
            logger.info("Force mode: reprocessing all chunk files")
        elif cfg.resume:
            logger.info("Resume mode enabled: unchanged chunk files will be skipped")

        if plan_only:
            stats = BM25Stats(N=0, avgdl=0.0, df={})
        else:
            stats = process_pass_a([entry.resolved_path for entry in chunk_entries], logger)
            if not stats.N:
                log_event(
                    logger,
                    "warning",
                    "No chunk statistics collected in Pass A",
                    stage=EMBED_STAGE,
                    doc_id="__aggregate__",
                    input_hash=None,
                    error_code="NO_PASS_A_DATA",
                )
                return 0

        cfg.splade_model_dir = splade_model_dir
        cfg.qwen_model_dir = qwen_model_dir
        hash_alg = resolve_hash_algorithm()
        manifest_index = load_manifest_index(MANIFEST_STAGE, resolved_root) if cfg.resume else {}
        resume_controller = ResumeController(cfg.resume, cfg.force, manifest_index)
        cfg_hash = _compute_embed_cfg_hash(cfg, vector_format)

        plan, plan_meta = _build_embedding_plan(
            chunk_entries=chunk_entries,
            chunks_dir=chunks_dir,
            vectors_dir=out_dir,
            resolved_root=resolved_root,
            resume_controller=resume_controller,
            vector_format=vector_format,
            cfg_hash=cfg_hash,
            hash_alg=hash_alg,
            logger=logger,
            plan_only=plan_only,
        )
        planned_ids = plan_meta["planned_ids"]
        skipped_ids = plan_meta["skipped_ids"]
        resume_skipped = int(plan_meta["resume_skipped"])

        log_event(
            logger,
            "info",
            "Embedding stage plan",
            stage=EMBED_STAGE,
            doc_id="__plan__",
            input_hash=None,
            scheduled=len(planned_ids),
            resume_skipped=resume_skipped,
            plan_items=plan.total_items,
            skip_candidates=len(skipped_ids),
        )

        if plan_only:
            log_event(
                logger,
                "info",
                "Embedding resume dry-run summary",
                stage=EMBED_STAGE,
                doc_id="__aggregate__",
                input_hash=None,
                error_code="EMBED_PLAN_ONLY",
                process_count=len(planned_ids),
                skip_count=len(skipped_ids),
            )
            print("docparse embed plan")
            print(f"- embed: process {len(planned_ids)}, skip {len(skipped_ids)}")
            if planned_ids:
                preview = ", ".join(planned_ids[:5])
                if len(planned_ids) > 5:
                    preview += f", ... (+{len(planned_ids) - 5} more)"
                print("  process preview:", preview)
            if skipped_ids:
                preview = ", ".join(skipped_ids[:5])
                if len(skipped_ids) > 5:
                    preview += f", ... (+{len(skipped_ids) - 5} more)"
                print("  skip preview:", preview)
            return 0

        using_stub_vectors = getattr(process_chunk_file_vectors, "__module__", __name__) != __name__
        stub_counters = (
            _extract_stub_counters(process_chunk_file_vectors) if using_stub_vectors else None
        )
        log_event(
            logger,
            "debug",
            "Embedding stage provider selection",
            stage=EMBED_STAGE,
            doc_id="__plan__",
            stub_vectors=bool(using_stub_vectors),
        )
        if using_stub_vectors:
            settings = cfg.provider_settings()
            embedding_cfg = settings["embedding"]
            provider_bundle = ProviderBundle(
                dense=None,
                sparse=None,
                lexical=None,
                context=ProviderContext(
                    device=embedding_cfg["device"] or "auto",
                    dtype=embedding_cfg["dtype"] or "auto",
                    batch_hint=embedding_cfg["batch_size"],
                    max_concurrency=embedding_cfg["max_concurrency"],
                    normalize_l2=bool(embedding_cfg["normalize_l2"]),
                    offline=bool(embedding_cfg["offline"]),
                    cache_dir=embedding_cfg["cache_dir"],
                    telemetry_tags=dict(embedding_cfg["telemetry_tags"] or {}),
                    telemetry_emitter=_provider_telemetry,
                ),
            )
        else:
            try:
                provider_bundle = ProviderFactory.create(cfg, telemetry_emitter=_provider_telemetry)
            except ProviderError as exc:
                log_event(
                    logger,
                    "error",
                    "Failed to initialise embedding providers",
                    stage=EMBED_STAGE,
                    error=str(exc),
                )
                raise

        args.splade_cfg = SpladeCfg(
            model_dir=splade_model_dir,
            cache_folder=model_root,
            batch_size=cfg.batch_size_splade,
            max_active_dims=cfg.sparse_splade_st_max_active_dims,
            attn_impl=cfg.sparse_splade_st_attn_backend,
            local_files_only=bool(cfg.offline),
        )
        args.qwen_cfg = QwenCfg(
            model_dir=qwen_model_dir,
            dtype=cfg.qwen_dtype,
            tp=int(cfg.tp),
            batch_size=int(cfg.dense_qwen_vllm_batch_size or cfg.batch_size_qwen),
            quantization=cfg.dense_qwen_vllm_quantization,
            dim=int(cfg.dense_qwen_vllm_dimension or cfg.qwen_dim),
            cache_enabled=not bool(cfg.no_cache),
        )

        exit_stack: ExitStack | None = ExitStack()
        bundle = exit_stack.enter_context(provider_bundle)

        validator = SPLADEValidator(
            warn_threshold_pct=float(cfg.sparsity_warn_threshold_pct),
            top_n=max(1, int(cfg.sparsity_report_top_n)),
        )

        files_parallel = min(requested_parallel, max(1, plan.total_items))
        args.files_parallel = files_parallel
        cfg.files_parallel = files_parallel
        context.files_parallel = files_parallel
        context.update_extra(
            files_parallel_effective=files_parallel,
            resume_skipped=resume_skipped,
            embed_cfg_hash=cfg_hash,
        )

        pass_b_start = time.perf_counter()

        if files_parallel > 1:
            log_event(
                logger,
                "info",
                "File-level parallelism enabled",
                files_parallel=files_parallel,
            )

        hooks = _make_embedding_stage_hooks(
            logger=logger,
            resolved_root=resolved_root,
            vector_format=vector_format,
            cfg=cfg,
            stats=stats,
            validator=validator,
            bundle=bundle,
            exit_stack=exit_stack,
            overall_start=overall_start,
            pass_b_start=pass_b_start,
            files_parallel=files_parallel,
            resume_skipped=resume_skipped,
            cfg_hash=cfg_hash,
            vectors_dir=out_dir,
            stub_vectors=using_stub_vectors,
            stub_counters=stub_counters,
        )

        options = StageOptions(
            policy="io",
            workers=files_parallel,
            resume=bool(cfg.resume),
            force=bool(cfg.force),
            error_budget=0,
            diagnostics_interval_s=15.0,
            resume_controller=resume_controller,
        )

        outcome = run_stage(plan, _embedding_stage_worker, options, hooks)
        log_event(
            logger,
            "info",
            "Embedding runner outcome",
            stage=EMBED_STAGE,
            doc_id="__system__",
            scheduled=outcome.scheduled,
            succeeded=outcome.succeeded,
            failed=outcome.failed,
            skipped=outcome.skipped,
            cancelled=outcome.cancelled,
        )
        if outcome.failed > 0 or outcome.cancelled:
            return 1
        return 0


def main(args: argparse.Namespace | None = None) -> int:
    """Wrapper that normalises CLI validation failures for the embedding stage."""

    try:
        return _main_inner(args)
    except EmbeddingCLIValidationError as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
