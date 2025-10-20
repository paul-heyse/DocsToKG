#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.embedding.runtime",
#   "purpose": "Embedding pipelines for DocParsing",
#   "sections": [
#     {"id": "s1", "name": "_build_bm25_vector", "anchor": "S1", "kind": "function"},
#     {"id": "s2", "name": "_build_splade_vector", "anchor": "S2", "kind": "function"},
#     {"id": "s3", "name": "_build_dense_vector", "anchor": "S3", "kind": "function"},
#     {"id": "s4", "name": "_build_vector_row", "anchor": "S4", "kind": "function"},
#     {"id": "s5", "name": "_get_sparse_encoder_cls", "anchor": "S5", "kind": "function"},
#     {"id": "s6", "name": "_get_vllm_components", "anchor": "S6", "kind": "function"},
#     {"id": "s7", "name": "_shutdown_llm_instance", "anchor": "S7", "kind": "function"},
#     {"id": "s8", "name": "_LRUCache", "anchor": "S8", "kind": "class"},
#     {"id": "s9", "name": "flush_llm_cache", "anchor": "S9", "kind": "function"},
#     {"id": "s10", "name": "close_all_qwen", "anchor": "S10", "kind": "function"},
#     {"id": "s11", "name": "_qwen_cache_key", "anchor": "S11", "kind": "function"},
#     {"id": "s12", "name": "_resolve_qwen_dir", "anchor": "S12", "kind": "function"},
#     {"id": "s13", "name": "_resolve_splade_dir", "anchor": "S13", "kind": "function"},
#     {"id": "s14", "name": "_expand_optional", "anchor": "S14", "kind": "function"},
#     {"id": "s15", "name": "_resolve_cli_path", "anchor": "S15", "kind": "function"},
#     {"id": "s16", "name": "_percentile", "anchor": "S16", "kind": "function"},
#     {"id": "s17", "name": "_ensure_splade_dependencies", "anchor": "S17", "kind": "function"},
#     {"id": "s18", "name": "_ensure_qwen_dependencies", "anchor": "S18", "kind": "function"},
#     {"id": "s19", "name": "ensure_uuid", "anchor": "S19", "kind": "function"},
#     {"id": "s20", "name": "_legacy_chunk_uuid", "anchor": "S20", "kind": "function"},
#     {"id": "s21", "name": "tokens", "anchor": "S21", "kind": "function"},
#     {"id": "s22", "name": "BM25StatsAccumulator", "anchor": "S22", "kind": "class"},
#     {"id": "s23", "name": "print_bm25_summary", "anchor": "S23", "kind": "function"},
#     {"id": "s24", "name": "bm25_vector", "anchor": "S24", "kind": "function"},
#     {"id": "s25", "name": "splade_encode", "anchor": "S25", "kind": "function"},
#     {"id": "s26", "name": "_detect_splade_backend", "anchor": "S26", "kind": "function"},
#     {"id": "s27", "name": "_get_splade_encoder", "anchor": "S27", "kind": "function"},
#     {"id": "s28", "name": "_get_splade_backend_used", "anchor": "S28", "kind": "function"},
#     {"id": "s29", "name": "SPLADEValidator", "anchor": "S29", "kind": "class"},
#     {"id": "s30", "name": "_qwen_embed_direct", "anchor": "S30", "kind": "function"},
#     {"id": "s31", "name": "qwen_embed", "anchor": "S31", "kind": "function"},
#     {"id": "s32", "name": "QwenEmbeddingQueue", "anchor": "S32", "kind": "class"},
#     {"id": "s33", "name": "EmbeddingProcessingError", "anchor": "S33", "kind": "class"},
#     {"id": "s34", "name": "process_pass_a", "anchor": "S34", "kind": "function"},
#     {"id": "s35", "name": "iter_rows_in_batches", "anchor": "S35", "kind": "function"},
#     {"id": "s36", "name": "iter_rows_in_batches_with_hash", "anchor": "S36", "kind": "function"},
#     {"id": "s37", "name": "_validate_chunk_file_schema", "anchor": "S37", "kind": "function"},
#     {"id": "s38", "name": "iter_chunk_files", "anchor": "S38", "kind": "function"},
#     {"id": "s39", "name": "_iter_chunks_or_empty", "anchor": "S39", "kind": "function"},
#     {"id": "s40", "name": "_ensure_pyarrow_vectors", "anchor": "S40", "kind": "function"},
#     {"id": "s41", "name": "_vector_arrow_schema", "anchor": "S41", "kind": "function"},
#     {"id": "s42", "name": "_prepare_vector_row_for_arrow", "anchor": "S42", "kind": "function"},
#     {"id": "s43", "name": "_rows_to_arrow_table", "anchor": "S43", "kind": "function"},
#     {"id": "s44", "name": "VectorWriter", "anchor": "S44", "kind": "class"},
#     {"id": "s45", "name": "JsonlVectorWriter", "anchor": "S45", "kind": "class"},
#     {"id": "s46", "name": "ParquetVectorWriter", "anchor": "S46", "kind": "class"},
#     {"id": "s47", "name": "create_vector_writer", "anchor": "S47", "kind": "function"},
#     {"id": "s48", "name": "_iter_vector_rows", "anchor": "S48", "kind": "function"},
#     {"id": "s49", "name": "process_chunk_file_vectors", "anchor": "S49", "kind": "function"},
#     {"id": "s50", "name": "write_vectors", "anchor": "S50", "kind": "function"},
#     {"id": "s51", "name": "_handle_embedding_quarantine", "anchor": "S51", "kind": "function"},
#     {"id": "s52", "name": "_validate_vectors_for_chunks", "anchor": "S52", "kind": "function"},
#     {"id": "s53", "name": "_main_inner", "anchor": "S53", "kind": "function"},
#     {"id": "s54", "name": "main", "anchor": "S54", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""
Hybrid Embedding Pipeline

Generates BM25, SPLADE, and Qwen embeddings for DocsToKG chunk files while
maintaining manifest entries, UUID hygiene, and data quality metrics. The
pipeline runs in two passes: the first ensures chunk UUID integrity and builds
BM25 corpus statistics; the second executes SPLADE and Qwen models to emit
vector JSONL artefacts ready for downstream search.

Key Features:
- Auto-detect DocsToKG data directories and manage resume/force semantics
- Stream SPLADE sparse encoding and Qwen dense embeddings from local caches
- Validate vector schemas, norms, and dimensions before writing outputs
- Record manifest metadata for observability and auditing
- Explain SPLADE attention backend fallbacks (auto→FlashAttention2→SDPA→eager)

Usage:
    python -m DocsToKG.DocParsing.core embed --resume

Dependencies:
- sentence_transformers (optional): Provides SPLADE sparse encoders.
- vllm (optional): Hosts the Qwen embedding model with pooling support.
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
import threading
import json
import logging
import math
import os
import re
import statistics
import time
import tracemalloc
import unicodedata
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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


from DocsToKG.DocParsing.cli_errors import EmbeddingCLIValidationError, format_cli_error
from DocsToKG.DocParsing.config import annotate_cli_overrides, parse_args_with_overrides
from DocsToKG.DocParsing.context import ParsingContext
from contextlib import ExitStack

from DocsToKG.DocParsing.core import (
    DEFAULT_TOKENIZER,
    UUID_NAMESPACE,
    Batcher,
    BM25Stats,
    ChunkDiscovery,
    QwenCfg,
    ResumeController,
    SpladeCfg,
    acquire_lock,
    compute_relative_doc_id,
    compute_stable_shard,
    derive_doc_id_and_vectors_path,
    iter_chunks,
    should_skip_output,
)
from DocsToKG.DocParsing.embedding.backends import (
    ProviderBundle,
    ProviderContext,
    ProviderError,
    ProviderFactory,
    ProviderIdentity,
    ProviderTelemetryEvent,
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
    SchemaKind,
    ensure_chunk_schema,
    validate_schema_version,
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
    cfg: SpladeCfg, texts: List[str], batch_size: Optional[int] = None
) -> Tuple[List[List[str]], List[List[float]]]:
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
        token_lists: List[List[str]] = []
        weight_lists: List[List[float]] = []
        for row in encoded:
            tokens = [str(token) for token, _weight in row]
            weights = [float(weight) for _token, weight in row]
            token_lists.append(tokens)
            weight_lists.append(weights)
        return token_lists, weight_lists
    finally:
        provider.close()


def qwen_embed(
    cfg: QwenCfg, texts: List[str], batch_size: Optional[int] = None
) -> List[List[float]]:
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
    "iter_chunk_files",
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


def _expand_optional(path: Optional[Path]) -> Optional[Path]:
    """Expand optional :class:`Path` values to absolutes when provided.

    Args:
        path: Optional path reference supplied by the caller.

    Returns:
        ``None`` when ``path`` is ``None``; otherwise the expanded absolute path.
    """

    if path is None:
        return None
    return path.expanduser().resolve()


def _resolve_cli_path(value: Optional[Path], default: Path) -> Path:
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


# --- BM25 Tokenizer ---

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


def ensure_uuid(rows: List[dict]) -> bool:
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


def tokens(text: str) -> List[str]:
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

    Attributes:
        N: Number of documents processed so far.
        total_tokens: Total token count across processed documents.
        df: Document frequency map collected to date.

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
) -> Tuple[List[str], List[float]]:
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
    tf = Counter(toks)
    terms, weights = [], []
    for t, f in tf.items():
        n_qi = stats.df.get(t, 0)
        idf = math.log((stats.N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)  # RSJ-smoothed IDF
        denom = f + k1 * (1.0 - b + b * (dl / stats.avgdl))
        w = idf * (f * (k1 + 1.0)) / denom
        if w > 0:
            terms.append(t)
            weights.append(w)
    return terms, weights


class SPLADEValidator:
    """Track SPLADE sparsity metrics across the corpus.

    Attributes:
        total_chunks: Total number of chunks inspected.
        zero_nnz_chunks: UUIDs whose SPLADE vector has zero active terms.
        nnz_counts: Non-zero counts per processed chunk.

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
        self.zero_nnz_chunks: List[str] = []
        self.nnz_counts: List[int] = []
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
    start: Optional[int] = None,
    end: Optional[int] = None,
    skip_invalid: bool = True,
    max_errors: int = 10,
) -> Iterator[List[dict]]:
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

    buf: List[dict] = []
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


def iter_rows_in_batches_with_hash(
    path: Path,
    batch_size: int,
    *,
    content_hasher: StreamingContentHasher,
) -> Iterator[List[dict]]:
    """Iterate JSONL rows while incrementally updating ``content_hasher``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    buf: List[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            content_hasher.update(raw_line)
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - malformed rows
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            buf.append(record)
            if len(buf) >= batch_size:
                yield buf
                buf = []
    if buf:
        yield buf


def _validate_chunk_file_schema(path: Path) -> None:
    """Stream chunk file rows and assert schema compatibility."""

    for index, row in enumerate(iter_jsonl(path), start=1):
        version = row.get("schema_version")
        if not version:
            continue
        validate_schema_version(str(version), SchemaKind.CHUNK, context=f"{path}:{index}")


def iter_chunk_files(directory: Path) -> Iterator[Path]:
    """Deprecated shim that forwards to :func:`iter_chunks`.

    Args:
        directory: Directory to scan for chunk artifacts.

    Returns:
        Iterator over chunk files.
    """

    yield from iter_chunks(directory)


def _iter_chunks_or_empty(directory: Path) -> Iterable[ChunkDiscovery]:
    """Return chunk entries for ``directory`` or an empty iterable when missing."""

    chunk_iter = iter_chunks(directory)
    if chunk_iter is None:
        return ()
    return chunk_iter


_PYARROW_MODULE: Any | None = None
_PYARROW_PARQUET: Any | None = None
_PYARROW_VECTOR_SCHEMA = None


def _ensure_pyarrow_vectors() -> tuple[Any, Any]:
    """Return ``(pyarrow, pyarrow.parquet)`` or raise a CLI validation error."""

    global _PYARROW_MODULE, _PYARROW_PARQUET
    if _PYARROW_MODULE is not None and _PYARROW_PARQUET is not None:
        return _PYARROW_MODULE, _PYARROW_PARQUET
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via CLI error path
        raise EmbeddingCLIValidationError(
            option="--format",
            message=(
                "parquet vector output requires the optional dependency 'pyarrow'. "
                "Install DocsToKG[docparse-parquet] or add pyarrow to the environment."
            ),
        ) from exc
    _PYARROW_MODULE = pa
    _PYARROW_PARQUET = pq
    return pa, pq


def _vector_arrow_schema(pa_module: Any):
    """Return (and cache) the Arrow schema used for parquet vector rows."""

    global _PYARROW_VECTOR_SCHEMA
    if _PYARROW_VECTOR_SCHEMA is not None:
        return _PYARROW_VECTOR_SCHEMA

    pa = pa_module
    string_list = pa.list_(pa.string())
    float_list = pa.list_(pa.float32())

    _PYARROW_VECTOR_SCHEMA = pa.schema(
        [
            pa.field("UUID", pa.string(), nullable=False),
            pa.field(
                "BM25",
                pa.struct(
                    [
                        pa.field("terms", string_list, nullable=True),
                        pa.field("weights", float_list, nullable=True),
                        pa.field("avgdl", pa.float64(), nullable=True),
                        pa.field("N", pa.int64(), nullable=True),
                    ]
                ),
                nullable=False,
            ),
            pa.field(
                "SPLADEv3",
                pa.struct(
                    [
                        pa.field("tokens", string_list, nullable=True),
                        pa.field("weights", float_list, nullable=True),
                    ]
                ),
                nullable=False,
            ),
            pa.field(
                "Qwen3-4B",
                pa.struct(
                    [
                        pa.field("model_id", pa.string(), nullable=False),
                        pa.field("vector", float_list, nullable=False),
                        pa.field("dimension", pa.int32(), nullable=True),
                    ]
                ),
                nullable=False,
            ),
            pa.field("model_metadata", pa.string(), nullable=True),
            pa.field("schema_version", pa.string(), nullable=False),
        ]
    )
    return _PYARROW_VECTOR_SCHEMA


def _prepare_vector_row_for_arrow(row: dict) -> dict:
    """Normalise a vector row dictionary for Arrow conversion."""

    bm25 = dict(row.get("BM25") or {})
    splade = dict(row.get("SPLADEv3") or {})
    qwen = dict(row.get("Qwen3-4B") or row.get("Qwen3_4B") or {})
    metadata = row.get("model_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    def _to_float_list(values: Any) -> list[float]:
        """Return ``values`` coerced into a list of floats."""
        if not isinstance(values, (list, tuple)):
            return []
        return [float(value) for value in values]

    prepared = {
        "UUID": str(row.get("UUID", "")),
        "BM25": {
            "terms": list(bm25.get("terms") or []),
            "weights": _to_float_list(bm25.get("weights")),
            "avgdl": float(bm25["avgdl"]) if bm25.get("avgdl") is not None else None,
            "N": int(bm25["N"]) if bm25.get("N") is not None else None,
        },
        "SPLADEv3": {
            "tokens": list(splade.get("tokens") or []),
            "weights": _to_float_list(splade.get("weights")),
        },
        "Qwen3-4B": {
            "model_id": str(qwen.get("model_id") or ""),
            "vector": _to_float_list(qwen.get("vector")),
            "dimension": int(qwen["dimension"]) if qwen.get("dimension") is not None else None,
        },
        "model_metadata": (
            json.dumps(metadata, ensure_ascii=False, sort_keys=True) if metadata else None
        ),
        "schema_version": str(row.get("schema_version") or VECTOR_SCHEMA_VERSION),
    }
    return prepared


def _rows_to_arrow_table(rows: Sequence[dict]) -> Any:
    """Convert ``rows`` into an Arrow table suitable for parquet writes."""

    if not rows:
        raise ValueError("rows must contain at least one vector payload")
    pa, _pq = _ensure_pyarrow_vectors()
    schema = _vector_arrow_schema(pa)
    prepared = [_prepare_vector_row_for_arrow(row) for row in rows]
    return pa.Table.from_pylist(prepared, schema=schema)


class VectorWriter:
    """Abstract base class for vector writers."""

    path: Path

    def write_rows(self, rows: Sequence[dict]) -> None:  # pragma: no cover - interface
        """Persist a batch of vector rows to the underlying storage medium."""
        raise NotImplementedError


class JsonlVectorWriter(VectorWriter):
    """Context manager that writes vector rows to JSONL atomically."""

    def __init__(self, path: Path) -> None:
        """Initialise the writer with the destination ``path``."""
        self.path = path
        self._context = None
        self._handle = None
        self._writes = 0

    def __enter__(self) -> "JsonlVectorWriter":
        """Open the underlying atomic writer and return ``self`` for chaining."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._context = atomic_write(self.path)
        self._handle = self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Close the atomic writer context, propagating exceptions."""
        if self._context is None:
            return False
        return self._context.__exit__(exc_type, exc, tb)

    def write_rows(self, rows: Sequence[dict]) -> None:
        """Append ``rows`` to the active JSONL artifact created by ``__enter__``."""
        if self._handle is None:
            raise RuntimeError("JsonlVectorWriter not initialised; call __enter__ first.")
        crash_after = getattr(sys.modules[__name__], "_crash_after_write", None)
        for row in rows:
            self._writes += 1
            if isinstance(crash_after, int) and self._writes > crash_after:
                raise RuntimeError("Simulated crash")
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class ParquetVectorWriter(VectorWriter):
    """Write vector rows to a temporary parquet file before atomic promotion."""

    def __init__(self, path: Path) -> None:
        """Initialise a parquet writer targeting ``path``."""
        self.path = path
        self._tmp_path: Path | None = None
        self._writer = None
        self._writes = 0

    def __enter__(self) -> "ParquetVectorWriter":
        """Create the underlying parquet writer and prepare a temp file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pa, pq = _ensure_pyarrow_vectors()
        schema = _vector_arrow_schema(pa)
        self._tmp_path = self.path.with_name(f"{self.path.name}.tmp.{uuid.uuid4().hex}")
        self._writer = pq.ParquetWriter(self._tmp_path.as_posix(), schema=schema)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Close the parquet writer and promote the temporary file atomically."""

        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            self._writer = None
        if exc_type is not None:
            if self._tmp_path is not None:
                self._tmp_path.unlink(missing_ok=True)
            return False
        if self._tmp_path is None:
            return False
        try:
            with self._tmp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            self._tmp_path.replace(self.path)
        finally:
            self._tmp_path = None
        return False

    def write_rows(self, rows: Sequence[dict]) -> None:
        """Append ``rows`` to the staged parquet file."""
        if not rows:
            return
        if self._writer is None or self._tmp_path is None:
            raise RuntimeError("ParquetVectorWriter not initialised; call __enter__ first.")
        crash_after = getattr(sys.modules[__name__], "_crash_after_write", None)
        for _ in rows:
            self._writes += 1
            if isinstance(crash_after, int) and self._writes > crash_after:
                raise RuntimeError("Simulated crash")
        table = _rows_to_arrow_table(rows)
        self._writer.write_table(table)


def create_vector_writer(path: Path, fmt: str) -> VectorWriter:
    """Factory returning the appropriate vector writer for ``fmt``."""

    fmt_normalized = fmt.lower()
    if fmt_normalized == "jsonl":
        return JsonlVectorWriter(path)
    if fmt_normalized == "parquet":
        return ParquetVectorWriter(path)
    raise ValueError(f"Unsupported vector format: {fmt}")


def _iter_vector_rows(path: Path, fmt: str, *, batch_size: int = 4096) -> Iterator[List[dict]]:
    """Yield batches of vector rows for ``path`` respecting the selected format."""

    fmt_normalized = str(fmt or "jsonl").lower()
    if fmt_normalized == "jsonl":
        yield from iter_rows_in_batches(path, batch_size)
        return
    if fmt_normalized != "parquet":
        raise ValueError(f"Unsupported vector format: {fmt}")

    _, pq = _ensure_pyarrow_vectors()
    parquet_file = pq.ParquetFile(path)
    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        rows = record_batch.to_pylist()
        normalised: List[dict] = []
        for entry in rows:
            metadata = entry.get("model_metadata")
            if isinstance(metadata, str) and metadata:
                try:
                    entry["model_metadata"] = json.loads(metadata)
                except json.JSONDecodeError:
                    entry["model_metadata"] = {}
            elif metadata in (None, ""):
                entry["model_metadata"] = {}
            normalised.append(entry)
        if normalised:
            yield normalised


def process_chunk_file_vectors(
    chunk_file: Path,
    out_path: Path,
    bundle: ProviderBundle,
    cfg: EmbedCfg,
    stats: BM25Stats,
    validator: SPLADEValidator,
    logger,
    *,
    content_hasher: Optional[StreamingContentHasher] = None,
    vector_format: str = "jsonl",
) -> Tuple[int, List[int], List[float]]:
    """Generate vectors for a single chunk file and persist them to disk."""

    if not isinstance(out_path, Path):
        raise TypeError("out_path must be a Path")
    resolved_out_path = out_path
    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)
    vector_format = str(vector_format or "jsonl").lower()

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
    nnz_all: List[int] = []
    norms_all: List[float] = []

    with create_vector_writer(resolved_out_path, vector_format) as writer:
        if content_hasher is None:
            row_batches: Iterator[List[dict]] = iter_rows_in_batches(chunk_file, row_batch_size)
        else:
            row_batches = iter_rows_in_batches_with_hash(
                chunk_file,
                row_batch_size,
                content_hasher=content_hasher,
            )

        for rows in row_batches:
            if not rows:
                continue

            uuids: List[str] = []
            texts: List[str] = []
            for index, row in enumerate(rows, start=1):
                ensure_chunk_schema(row, context=f"{chunk_file}:{index}")
                uuid_value = row.get("uuid")
                if not uuid_value:
                    raise ValueError(f"Chunk row missing UUID in {chunk_file}")
                uuids.append(str(uuid_value))
                texts.append(str(row.get("text", "")))

            lexical_vectors: List[Tuple[Sequence[str], Sequence[float]]] = []
            for text in texts:
                terms, weights = lexical_provider.vector(text, stats)
                lexical_vectors.append((list(terms), list(weights)))

            sparse_encoded = sparse_provider.encode(texts)
            sparse_vectors: List[Tuple[Sequence[str], Sequence[float]]] = []
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
    writer: VectorWriter,
    uuids: Sequence[str],
    texts: Sequence[str],
    lexical_results: Sequence[Tuple[Sequence[str], Sequence[float]]],
    splade_results: Sequence[Tuple[Sequence[str], Sequence[float]]],
    dense_results: Sequence[Sequence[float]],
    stats: BM25Stats,
    cfg: EmbedCfg,
    *,
    rows: Sequence[dict],
    validator: SPLADEValidator,
    logger,
    provider_identities: Dict[str, ProviderIdentity],
    output_path: Optional[Path] = None,
    vector_format: str = "jsonl",
) -> Tuple[int, List[int], List[float]]:
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

    splade_nnz: List[int] = []
    dense_norms: List[float] = []
    output_ref = output_path or getattr(writer, "path", None)
    payloads: List[dict] = []

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
        output_ref = str(output_path) if output_path is not None else ""
        failure_logger = getattr(sys.modules[__name__], "manifest_log_failure")
        for row in rows:
            doc_id = row.get("doc_id", "unknown") if isinstance(row, dict) else "unknown"
            failure_logger(
                stage="embeddings",
                doc_id=doc_id,
                duration_s=0.0,
                schema_version=VECTOR_SCHEMA_VERSION,
                input_path=(
                    row.get("source_path", "unknown") if isinstance(row, dict) else "unknown"
                ),
                input_hash=row.get("input_hash", "") if isinstance(row, dict) else "",
                output_path=output_ref,
                vector_format=vector_format,
                error=str(exc),
            )
        raise

    return len(uuids), splade_nnz, dense_norms


def _handle_embedding_quarantine(
    *,
    chunk_path: Path,
    vector_path: Path,
    doc_id: str,
    input_hash: str,
    reason: str,
    logger,
    data_root: Optional[Path] = None,
    vector_format: str = "jsonl",
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
        vector_format=str(vector_format or "jsonl").lower(),
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
        vector_format=str(vector_format or "jsonl").lower(),
    )


# --- Main Driver ---


def _validate_vectors_for_chunks(
    chunks_dir: Path,
    vectors_dir: Path,
    logger,
    *,
    data_root: Optional[Path] = None,
    expected_dimension: Optional[int] = None,
    vector_format: str = "jsonl",
) -> tuple[int, int]:
    """Validate vectors associated with chunk files without recomputing models.

    Returns:
        (files_checked, rows_validated)
    """
    fmt_normalised = str(vector_format or "jsonl").lower()
    files_checked = 0
    rows_validated = 0
    missing: List[tuple[str, Path]] = []
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
            for batch in _iter_vector_rows(vector_path, fmt_normalised, batch_size=4096):
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


def _main_inner(args: argparse.Namespace | None = None) -> int:
    """CLI entrypoint for chunk UUID cleanup and embedding generation.

    Args:
        args (argparse.Namespace | None): Optional parsed arguments, primarily
            for testing or orchestration.

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

    if getattr(namespace, "plan_only", False) and getattr(namespace, "validate_only", False):
        raise EmbeddingCLIValidationError(
            option="--plan-only/--validate-only",
            message="flags cannot be combined",
        )

    profile = getattr(namespace, "profile", None)
    defaults = EMBED_PROFILE_PRESETS.get(profile or "", {})
    cfg = EmbedCfg.from_args(namespace, defaults=defaults)
    base_config = cfg.to_manifest()
    if profile:
        base_config.setdefault("profile", profile)
    for field_def in fields(EmbedCfg):
        setattr(namespace, field_def.name, getattr(cfg, field_def.name))

    validate_only = bool(cfg.validate_only)
    plan_only = bool(getattr(namespace, "plan_only", False))

    log_level = cfg.log_level
    run_id = uuid.uuid4().hex
    logger = get_logger(
        __name__,
        level=str(log_level),
        base_fields={"run_id": run_id, "stage": EMBED_STAGE},
    )
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

    vector_format = str(cfg.vector_format or "jsonl").lower()
    if vector_format not in {"jsonl", "parquet"}:
        raise EmbeddingCLIValidationError(
            option="--format",
            message="must be one of: jsonl, parquet",
        )
    if vector_format == "parquet":
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

        incompatible_chunks: List[Path] = []
        validated_files: List[ChunkDiscovery] = []
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
        tracemalloc.start()
        pass_b_start = time.perf_counter()
        total_vectors = 0
        splade_nnz_all: List[int] = []
        dense_norms_all: List[float] = []

        manifest_index = load_manifest_index(MANIFEST_STAGE, resolved_root) if cfg.resume else {}
        resume_controller = ResumeController(cfg.resume, cfg.force, manifest_index)
        file_entries: List[Tuple[Path, Path, str, str]] = []
        skipped_files = 0
        quarantined_files = 0
        skipped_ids: List[str] = []
        planned_ids: List[str] = []
        resume_needs_hash = resume_controller.resume and not resume_controller.force
        for chunk_entry in chunk_entries:
            chunk_file = chunk_entry.resolved_path
            doc_id, out_path = derive_doc_id_and_vectors_path(
                chunk_entry,
                chunks_dir,
                args.out_dir,
                vector_format=vector_format,
            )
            manifest_entry = resume_controller.entry(doc_id) if resume_controller.resume else None
            vectors_exist = out_path.exists()
            has_manifest_entry = manifest_entry is not None
            should_hash = resume_needs_hash and has_manifest_entry and vectors_exist
            input_hash = compute_content_hash(chunk_file) if should_hash else ""
            entry_format = (
                str(manifest_entry.get("vector_format") or "jsonl").lower()
                if manifest_entry
                else None
            )
            format_mismatch = bool(manifest_entry) and entry_format != vector_format
            if resume_controller.resume and not format_mismatch:
                skip_file = should_skip_output(
                    out_path,
                    manifest_entry,
                    input_hash,
                    resume_controller.resume,
                    resume_controller.force,
                )
            else:
                skip_file = False
            if skip_file:
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
                        output_relpath=relative_path(out_path, resolved_root),
                        vector_format=vector_format,
                    )
                    manifest_log_skip(
                        stage=MANIFEST_STAGE,
                        doc_id=doc_id,
                        input_path=chunk_file,
                        input_hash=input_hash,
                        output_path=out_path,
                        schema_version=VECTOR_SCHEMA_VERSION,
                        vector_format=vector_format,
                    )
                skipped_files += 1
                continue
            if format_mismatch and not plan_only:
                log_event(
                    logger,
                    "info",
                    "Regenerating vectors due to format mismatch",
                    status="process",
                    stage=EMBED_STAGE,
                    doc_id=doc_id,
                    previous_format=entry_format,
                    requested_format=vector_format,
                    input_relpath=relative_path(chunk_file, resolved_root),
                    output_relpath=relative_path(out_path, resolved_root),
                    vector_format=vector_format,
                )
            planned_ids.append(doc_id)
            if plan_only:
                continue
            file_entries.append((chunk_file, out_path, input_hash, doc_id))

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
            should_stop_tracing = True
            if hasattr(tracemalloc, "is_tracing"):
                should_stop_tracing = tracemalloc.is_tracing()
            if should_stop_tracing:
                try:
                    tracemalloc.stop()
                except RuntimeError:
                    pass
            return 0

        files_parallel = min(requested_parallel, max(1, len(file_entries)))
        args.files_parallel = files_parallel
        cfg.files_parallel = files_parallel
        context.files_parallel = files_parallel
        context.update_extra(files_parallel_effective=files_parallel)

        if files_parallel > 1:
            log_event(
                logger, "info", "File-level parallelism enabled", files_parallel=files_parallel
            )

        provider_identities_summary: Dict[str, ProviderIdentity] = {}
        try:
            def _process_entry(
                entry: Tuple[Path, Path, str, str],
            ) -> Tuple[int, List[int], List[float], float, bool, str]:
                """Encode vectors for a chunk file and report per-file metrics."""

                chunk_path, vectors_path, input_hash, doc_id = entry
                start = time.perf_counter()
                hasher: StreamingContentHasher | None = None
                if not input_hash:
                    hasher = StreamingContentHasher()
                try:
                    with acquire_lock(vectors_path):
                        count, nnz, norms = process_chunk_file_vectors(
                            chunk_path,
                            vectors_path,
                            bundle,
                            cfg,
                            stats,
                            validator,
                            logger,
                            content_hasher=hasher,
                            vector_format=vector_format,
                        )
                except ValueError as exc:
                    duration = time.perf_counter() - start
                    computed_hash = input_hash or (hasher.hexdigest() if hasher else "")
                    _handle_embedding_quarantine(
                        chunk_path=chunk_path,
                        vector_path=vectors_path,
                        doc_id=doc_id,
                        input_hash=computed_hash,
                        reason=str(exc),
                        logger=logger,
                        data_root=resolved_root,
                        vector_format=vector_format,
                    )
                    return 0, [], [], duration, True, computed_hash
                except Exception as exc:  # pragma: no cover - propagated to caller
                    duration = time.perf_counter() - start
                    computed_hash = input_hash or (hasher.hexdigest() if hasher else "")
                    raise EmbeddingProcessingError(exc, duration, computed_hash) from exc
                duration = time.perf_counter() - start
                computed_hash = input_hash or (hasher.hexdigest() if hasher else "")
                return count, nnz, norms, duration, False, computed_hash

            if not file_entries:
                pass
            elif files_parallel > 1:
                with (
                    ThreadPoolExecutor(max_workers=files_parallel) as executor,
                    tqdm(
                        total=len(file_entries), desc="Pass B: Encoding vectors", unit="file"
                    ) as bar,
                ):
                    future_map = {
                        executor.submit(_process_entry, entry): entry for entry in file_entries
                    }
                    for future in as_completed(future_map):
                        chunk_file, out_path, input_hash, doc_id = future_map[future]
                        try:
                            count, nnz, norms, duration, quarantined, resolved_hash = (
                                future.result()
                            )
                        except EmbeddingProcessingError as exc:
                            manifest_log_failure(
                                stage=MANIFEST_STAGE,
                                doc_id=doc_id,
                                duration_s=round(exc.duration, 3),
                                schema_version=VECTOR_SCHEMA_VERSION,
                                input_path=chunk_file,
                                input_hash=exc.input_hash,
                                output_path=out_path,
                                vector_format=vector_format,
                                error=str(exc.original),
                            )
                            bar.update(1)
                            raise exc.original from exc
                        except Exception as exc:
                            manifest_log_failure(
                                stage=MANIFEST_STAGE,
                                doc_id=doc_id,
                                duration_s=0.0,
                                schema_version=VECTOR_SCHEMA_VERSION,
                                input_path=chunk_file,
                                input_hash=input_hash,
                                output_path=out_path,
                                vector_format=vector_format,
                                error=str(exc),
                            )
                            bar.update(1)
                            raise
                        if quarantined:
                            quarantined_files += 1
                            bar.update(1)
                            continue
                        total_vectors += count
                        splade_nnz_all.extend(nnz)
                        dense_norms_all.extend(norms)
                        manifest_log_success(
                            stage=MANIFEST_STAGE,
                            doc_id=doc_id,
                            duration_s=round(duration, 3),
                            schema_version=VECTOR_SCHEMA_VERSION,
                            input_path=chunk_file,
                            input_hash=resolved_hash,
                            output_path=out_path,
                            vector_format=vector_format,
                            vector_count=count,
                        )
                        avg_nnz_file = statistics.mean(nnz) if nnz else 0.0
                        avg_norm_file = statistics.mean(norms) if norms else 0.0
                        log_event(
                            logger,
                            "info",
                            "Embedding file written",
                            status="success",
                            stage=EMBED_STAGE,
                            doc_id=doc_id,
                            input_relpath=relative_path(chunk_file, resolved_root),
                            output_relpath=relative_path(out_path, resolved_root),
                            elapsed_ms=int(duration * 1000),
                            vectors=count,
                            splade_avg_nnz=round(avg_nnz_file, 3),
                            qwen_avg_norm=round(avg_norm_file, 4),
                            vector_format=vector_format,
                        )
                        bar.update(1)
            else:
                for entry in tqdm(file_entries, desc="Pass B: Encoding vectors", unit="file"):
                    chunk_file, out_path, input_hash, doc_id = entry
                    try:
                        count, nnz, norms, duration, quarantined, resolved_hash = _process_entry(
                            entry
                        )
                    except EmbeddingProcessingError as exc:
                        manifest_log_failure(
                            stage=MANIFEST_STAGE,
                            doc_id=doc_id,
                            duration_s=round(exc.duration, 3),
                            schema_version=VECTOR_SCHEMA_VERSION,
                            input_path=chunk_file,
                            input_hash=exc.input_hash,
                            output_path=out_path,
                            vector_format=vector_format,
                            error=str(exc.original),
                        )
                        raise exc.original from exc
                    except Exception as exc:
                        manifest_log_failure(
                            stage=MANIFEST_STAGE,
                            doc_id=doc_id,
                            duration_s=0.0,
                            schema_version=VECTOR_SCHEMA_VERSION,
                            input_path=chunk_file,
                            input_hash=input_hash,
                            output_path=out_path,
                            vector_format=vector_format,
                            error=str(exc),
                        )
                        raise
                    if quarantined:
                        quarantined_files += 1
                        continue
                    total_vectors += count
                    splade_nnz_all.extend(nnz)
                    dense_norms_all.extend(norms)
                    manifest_log_success(
                        stage=MANIFEST_STAGE,
                        doc_id=doc_id,
                        duration_s=round(duration, 3),
                        schema_version=VECTOR_SCHEMA_VERSION,
                        input_path=chunk_file,
                        input_hash=resolved_hash,
                        output_path=out_path,
                        vector_format=vector_format,
                        vector_count=count,
                    )
                    avg_nnz_file = statistics.mean(nnz) if nnz else 0.0
                    avg_norm_file = statistics.mean(norms) if norms else 0.0
                    log_event(
                        logger,
                        "info",
                        "Embedding file written",
                        status="success",
                        stage=EMBED_STAGE,
                        doc_id=doc_id,
                        input_relpath=relative_path(chunk_file, resolved_root),
                        output_relpath=relative_path(out_path, resolved_root),
                        elapsed_ms=int(duration * 1000),
                        vectors=count,
                        splade_avg_nnz=round(avg_nnz_file, 3),
                        qwen_avg_norm=round(avg_norm_file, 4),
                        vector_format=vector_format,
                    )
            provider_identities_summary = bundle.identities()
        finally:
            if exit_stack is not None:
                exit_stack.close()

        if quarantined_files:
            log_event(
                logger,
                "warning",
                "Quarantined chunk inputs during embedding",
                status="quarantine-summary",
                stage=EMBED_STAGE,
                quarantined=quarantined_files,
            )

        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_b = time.perf_counter() - pass_b_start

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

        sparse_identity = provider_identities_summary.get("sparse")
        dense_identity = provider_identities_summary.get("dense")
        lexical_identity = provider_identities_summary.get("lexical")

        logger.info(
            "Embedding summary",
            extra={
                "extra_fields": {
                    "total_vectors": total_vectors,
                    "splade_avg_nnz": round(avg_nnz, 3),
                    "splade_median_nnz": round(median_nnz, 3),
                    "splade_p95_nnz": round(splade_p95, 3),
                    "splade_p99_nnz": round(splade_p99, 3),
                    "splade_zero_pct": round(zero_pct, 2),
                    "qwen_avg_norm": round(avg_norm, 4),
                    "qwen_std_norm": round(std_norm, 4),
                    "qwen_norm_p95": round(norm_p95, 4),
                    "qwen_norm_p99": round(norm_p99, 4),
                    "qwen_norm_low_outliers": norm_low_outliers,
                    "qwen_norm_high_outliers": norm_high_outliers,
                    "pass_b_seconds": round(elapsed_b, 3),
                    "skipped_files": skipped_files,
                    "quarantined_files": quarantined_files,
                    "files_parallel": files_parallel,
                    "splade_attn_backend_used": cfg.sparse_splade_st_attn_backend or "auto",
                    "sparsity_warn_threshold_pct": float(cfg.sparsity_warn_threshold_pct),
                    "dense_provider": dense_identity.name if dense_identity else None,
                    "dense_provider_version": dense_identity.version if dense_identity else None,
                    "sparse_provider": sparse_identity.name if sparse_identity else None,
                    "lexical_provider": lexical_identity.name if lexical_identity else None,
                }
            },
        )
        logger.info("Peak memory: %.2f GB", peak / 1024**3)

        manifest_log_success(
            stage=MANIFEST_STAGE,
            doc_id="__corpus__",
            duration_s=round(time.perf_counter() - overall_start, 3),
            schema_version=VECTOR_SCHEMA_VERSION,
            input_path="__corpus__",
            input_hash="",
            output_path=out_dir,
            warnings=(
                validator.zero_nnz_chunks[: validator.top_n] if validator.zero_nnz_chunks else []
            ),
            total_vectors=total_vectors,
            splade_avg_nnz=avg_nnz,
            splade_median_nnz=median_nnz,
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
            skipped_files=skipped_files,
            quarantined_files=quarantined_files,
            files_parallel=files_parallel,
            splade_attn_backend_used=backend_used,
            sparsity_warn_threshold_pct=float(cfg.sparsity_warn_threshold_pct),
        )

        log_event(
            logger,
            "info",
            "[DONE] Saved vectors",
            status="complete",
            stage=EMBED_STAGE,
            embeddings_dir=str(out_dir),
            processed_files=len(file_entries),
            skipped_files=skipped_files,
            quarantined_files=quarantined_files,
            total_vectors=total_vectors,
        )

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
