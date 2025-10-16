#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.EmbeddingV2",
#   "purpose": "Embedding pipelines for DocParsing",
#   "sections": [
#     {
#       "id": "qwen-cache-key",
#       "name": "_qwen_cache_key",
#       "anchor": "function-qwen-cache-key",
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
#       "id": "missing-splade-dependency-message",
#       "name": "_missing_splade_dependency_message",
#       "anchor": "function-missing-splade-dependency-message",
#       "kind": "function"
#     },
#     {
#       "id": "missing-qwen-dependency-message",
#       "name": "_missing_qwen_dependency_message",
#       "anchor": "function-missing-qwen-dependency-message",
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
#       "id": "ensure-uuid",
#       "name": "ensure_uuid",
#       "anchor": "function-ensure-uuid",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-chunk-schema",
#       "name": "ensure_chunk_schema",
#       "anchor": "function-ensure-chunk-schema",
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
#       "id": "splade-encode",
#       "name": "splade_encode",
#       "anchor": "function-splade-encode",
#       "kind": "function"
#     },
#     {
#       "id": "detect-splade-backend",
#       "name": "_detect_splade_backend",
#       "anchor": "function-detect-splade-backend",
#       "kind": "function"
#     },
#     {
#       "id": "get-splade-encoder",
#       "name": "_get_splade_encoder",
#       "anchor": "function-get-splade-encoder",
#       "kind": "function"
#     },
#     {
#       "id": "get-splade-backend-used",
#       "name": "_get_splade_backend_used",
#       "anchor": "function-get-splade-backend-used",
#       "kind": "function"
#     },
#     {
#       "id": "spladevalidator",
#       "name": "SPLADEValidator",
#       "anchor": "class-spladevalidator",
#       "kind": "class"
#     },
#     {
#       "id": "qwen-embed-direct",
#       "name": "_qwen_embed_direct",
#       "anchor": "function-qwen-embed-direct",
#       "kind": "function"
#     },
#     {
#       "id": "qwen-embed",
#       "name": "qwen_embed",
#       "anchor": "function-qwen-embed",
#       "kind": "function"
#     },
#     {
#       "id": "qwenembeddingqueue",
#       "name": "QwenEmbeddingQueue",
#       "anchor": "class-qwenembeddingqueue",
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
#       "id": "iter-chunk-files",
#       "name": "iter_chunk_files",
#       "anchor": "function-iter-chunk-files",
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
#       "id": "validate-vectors-for-chunks",
#       "name": "_validate_vectors_for_chunks",
#       "anchor": "function-validate-vectors-for-chunks",
#       "kind": "function"
#     },
#     {
#       "id": "build-parser",
#       "name": "build_parser",
#       "anchor": "function-build-parser",
#       "kind": "function"
#     },
#     {
#       "id": "parse-args",
#       "name": "parse_args",
#       "anchor": "function-parse-args",
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
vector JSONL artefacts ready for downstream search.

Key Features:
- Auto-detect DocsToKG data directories and manage resume/force semantics
- Stream SPLADE sparse encoding and Qwen dense embeddings from local caches
- Validate vector schemas, norms, and dimensions before writing outputs
- Record manifest metadata for observability and auditing
- Explain SPLADE attention backend fallbacks (auto→FlashAttention2→SDPA→eager)

Usage:
    python -m DocsToKG.DocParsing.EmbeddingV2 --resume

Dependencies:
- sentence_transformers (optional): Provides SPLADE sparse encoders.
- vllm (optional): Hosts the Qwen embedding model with pooling support.
- tqdm: Surface user-friendly progress bars across pipeline phases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import statistics
import time
import tracemalloc
import threading
import unicodedata
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# Third-party imports
from tqdm import tqdm

from DocsToKG.DocParsing._common import (
    UUID_NAMESPACE,
    Batcher,
    BM25Stats,
    QwenCfg,
    SpladeCfg,
    acquire_lock,
    atomic_write,
    compute_content_hash,
    data_chunks,
    data_vectors,
    derive_doc_id_and_vectors_path,
    detect_data_root,
    expand_path,
    get_logger,
    init_hf_env,
    iter_chunks,
    jsonl_append_iter,
    jsonl_load,
    load_manifest_index,
    log_event,
    manifest_append,
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
    should_skip_output,
)
from DocsToKG.DocParsing.pipelines import (
    add_data_root_option,
    add_resume_force_options,
    prepare_data_root,
    resolve_pipeline_path,
)
from DocsToKG.DocParsing.schemas import (
    COMPATIBLE_CHUNK_VERSIONS,
    VECTOR_SCHEMA_VERSION,
    BM25Vector,
    DenseVector,
    SPLADEVector,
    VectorRow,
    validate_schema_version,
)

# --- Globals ---

__all__ = (
    "BM25Stats",
    "BM25StatsAccumulator",
    "QwenCfg",
    "SPLADEValidator",
    "SpladeCfg",
    "bm25_vector",
    "build_parser",
    "ensure_chunk_schema",
    "ensure_uuid",
    "iter_chunk_files",
    "iter_rows_in_batches",
    "main",
    "parse_args",
    "print_bm25_summary",
    "process_chunk_file_vectors",
    "process_pass_a",
    "qwen_embed",
    "QwenEmbeddingQueue",
    "splade_encode",
    "tokens",
    "write_vectors",
)


# --- Public Functions ---

try:  # Optional dependency used for SPLADE sparse embeddings
    from sentence_transformers import (
        SparseEncoder,
    )  # loads from local dir if given (cache_folder supported)

    _SENTENCE_TRANSFORMERS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via tests with stubs
    SparseEncoder = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc

try:  # Optional dependency used for Qwen dense embeddings
    from vllm import (
        LLM,
        PoolingParams,
    )  # PoolingParams(dimensions=...) selects output dim if model supports MRL

    _VLLM_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via tests with stubs
    LLM = None  # type: ignore[assignment]
    PoolingParams = None  # type: ignore[assignment]
    _VLLM_IMPORT_ERROR = exc


_QWEN_LLM_CACHE: Dict[Tuple[str, str, int, float, str | None], LLM] = {}


def _qwen_cache_key(cfg: QwenCfg) -> Tuple[str, str, int, float, str | None]:
    """Return cache key tuple for Qwen LLM instances."""

    quant = cfg.quantization if cfg.quantization else None
    return (
        str(cfg.model_dir),
        cfg.dtype,
        int(cfg.tp),
        float(cfg.gpu_mem_util),
        quant,
    )


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


HF_HOME, MODEL_ROOT = init_hf_env()
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


DEFAULT_DATA_ROOT = detect_data_root()
DEFAULT_CHUNKS_DIR = data_chunks(DEFAULT_DATA_ROOT)
DEFAULT_VECTORS_DIR = data_vectors(DEFAULT_DATA_ROOT)
MANIFEST_STAGE = "embeddings"
SPLADE_SPARSITY_WARN_THRESHOLD_PCT = 1.0


def _missing_splade_dependency_message() -> str:
    """Return a human-readable installation hint for SPLADE extras."""

    return (
        "Optional dependency 'sentence-transformers' is required for SPLADE "
        "embeddings. Install it with `pip install sentence-transformers` or "
        "disable SPLADE generation."
    )


def _missing_qwen_dependency_message() -> str:
    """Return a human-readable installation hint for Qwen/vLLM extras."""

    return (
        "Optional dependency 'vllm' is required for Qwen dense embeddings. "
        "Install it with `pip install vllm` and ensure GPU drivers are "
        "available before running the embedding pipeline."
    )


def _ensure_splade_dependencies() -> None:
    """Validate that SPLADE optional dependencies are importable."""

    if SparseEncoder is None:
        raise ImportError(
            _missing_splade_dependency_message()
        ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR


def _ensure_qwen_dependencies() -> None:
    """Validate that Qwen/vLLM optional dependencies are importable."""

    if LLM is None or PoolingParams is None:
        raise ImportError(_missing_qwen_dependency_message()) from _VLLM_IMPORT_ERROR


# --- BM25 Tokenizer ---

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


def ensure_uuid(rows: List[dict]) -> bool:
    """Populate missing chunk UUIDs in-place using deterministic UUIDv5 derivation.

    Args:
        rows: Chunk dictionaries that should include a `uuid` key.

    Returns:
        True when at least one UUID was newly assigned; otherwise False. UUIDs are
        derived from a namespace, doc ID, source chunk indices, and the first 16 hex
        characters of the chunk text SHA-1 digest. If deterministic generation fails,
        a UUID4 is assigned as a fallback.
    """

    updated = False
    for row in rows:
        if not row.get("uuid"):
            try:
                doc_id = row.get("doc_id") or ""
                src_raw = row.get("source_chunk_idxs") or []
                if isinstance(src_raw, (str, bytes)):
                    src_iterable = [src_raw]
                elif isinstance(src_raw, Sequence):
                    src_iterable = list(src_raw)
                else:
                    src_iterable = [src_raw] if src_raw is not None else []
                src = ",".join(str(idx) for idx in src_iterable)

                text_raw = row.get("text", "")
                if isinstance(text_raw, bytes):
                    text_value = text_raw.decode("utf-8", errors="ignore")
                else:
                    text_value = str(text_raw) if text_raw is not None else ""
                text_value = unicodedata.normalize("NFKC", text_value)
                digest = hashlib.sha1(text_value.encode("utf-8")).hexdigest()[:16]

                name = f"{doc_id}|{src}|{digest}"
                row["uuid"] = str(uuid.uuid5(UUID_NAMESPACE, name))
            except Exception:
                row["uuid"] = str(uuid.uuid4())
            updated = True
    return updated


def ensure_chunk_schema(rows: Sequence[dict], source: Path) -> None:
    """Assert that chunk rows declare a compatible schema version.

    Args:
        rows: Iterable of chunk dictionaries to validate.
        source: Path to the originating chunk file, used for error context.

    Returns:
        None

    Raises:
        ValueError: Propagated when an incompatible schema version is detected.
    """

    for index, row in enumerate(rows, start=1):
        version = row.get("schema_version")
        if not version:
            # Older chunk artifacts omitted explicit schema versions. Default
            # them to the newest compatible revision so downstream validators
            # continue to operate without forcing regeneration.
            row["schema_version"] = COMPATIBLE_CHUNK_VERSIONS[-1]
            continue
        validate_schema_version(
            version,
            COMPATIBLE_CHUNK_VERSIONS,
            kind="chunk",
            source=f"{source}:{index}",
        )


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


# --- SPLADE-v3 (GPU) ---


def splade_encode(
    cfg: SpladeCfg, texts: List[str], batch_size: Optional[int] = None
) -> Tuple[List[List[str]], List[List[float]]]:
    """Encode text with SPLADE to obtain sparse lexical vectors.

    Args:
        cfg: SPLADE configuration describing device, batch size, and cache.
        texts: Batch of input strings to encode.
        batch_size: Optional override for the encoding batch size.

    Returns:
        Tuple of token lists and weight lists aligned per input text.
    """
    effective_batch = batch_size or cfg.batch_size
    enc = _get_splade_encoder(cfg)
    token_lists, weight_lists = [], []
    for i in range(0, len(texts), effective_batch):
        batch = texts[i : i + effective_batch]
        # returns a torch.sparse tensor (rows = batch)
        s = enc.encode(batch)  # shape: (B, |vocab|)
        for r in range(s.shape[0]):
            # decode all non-zeros for this row
            # decodes to list[(token, weight)] sorted by weight
            nnz = s[r].coalesce().values().numel()
            decoded = enc.decode(s[r], top_k=int(nnz))
            toks, wts = zip(*decoded) if decoded else ([], [])
            token_lists.append(list(toks))
            weight_lists.append([float(w) for w in wts])
    return token_lists, weight_lists


_SPLADE_ENCODER_CACHE: Dict[Tuple[str, str, Optional[str], Optional[int]], SparseEncoder] = {}
_SPLADE_ENCODER_BACKENDS: Dict[Tuple[str, str, Optional[str], Optional[int]], str] = {}


def _detect_splade_backend(encoder: SparseEncoder, requested: str | None) -> str:
    """Best-effort detection of the attention backend used by SPLADE."""

    candidates = (
        ("model", "model", "config", "attn_implementation"),
        ("model", "config", "attn_implementation"),
        ("config", "attn_implementation"),
        ("model", "model", "attn_implementation"),
    )
    for path in candidates:
        value = encoder
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        else:
            if isinstance(value, str) and value:
                return value

    if requested in {"sdpa", "eager", "flash_attention_2"}:
        return requested
    return "auto" if requested is None else requested


def _get_splade_encoder(cfg: SpladeCfg) -> SparseEncoder:
    """Retrieve (or create) a cached SPLADE encoder instance.

    Args:
        cfg: SPLADE configuration describing model location and runtime options.

    Returns:
        Cached :class:`SparseEncoder` ready for SPLADE inference.

    Raises:
        ValueError: If the encoder cannot be initialised with the supplied configuration.
        ImportError: If required SPLADE dependencies are unavailable.
    """

    _ensure_splade_dependencies()

    key = (str(cfg.model_dir), cfg.device, cfg.attn_impl, cfg.max_active_dims)
    if key in _SPLADE_ENCODER_CACHE:
        if key not in _SPLADE_ENCODER_BACKENDS:
            _SPLADE_ENCODER_BACKENDS[key] = cfg.attn_impl or "auto"
        return _SPLADE_ENCODER_CACHE[key]

    model_kwargs: Dict[str, object] = {}
    if cfg.attn_impl:
        model_kwargs["attn_implementation"] = cfg.attn_impl
    if cfg.max_active_dims is not None:
        model_kwargs["max_active_dims"] = cfg.max_active_dims

    backend_used: str | None = cfg.attn_impl
    try:
        encoder = SparseEncoder(
            str(cfg.model_dir),
            device=cfg.device,
            cache_folder=str(cfg.cache_folder),
            model_kwargs=model_kwargs,
            local_files_only=cfg.local_files_only,
        )
        backend_used = _detect_splade_backend(encoder, backend_used)
    except (ValueError, ImportError) as exc:
        if cfg.attn_impl == "flash_attention_2" and "Flash Attention 2" in str(exc):
            print("[SPLADE] FlashAttention 2 unavailable; retrying with standard attention.")
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["attn_implementation"] = "sdpa"
            encoder = SparseEncoder(
                str(cfg.model_dir),
                device=cfg.device,
                cache_folder=str(cfg.cache_folder),
                model_kwargs=fallback_kwargs,
                local_files_only=cfg.local_files_only,
            )
            backend_used = _detect_splade_backend(encoder, "sdpa")
        else:
            raise

    _SPLADE_ENCODER_CACHE[key] = encoder
    _SPLADE_ENCODER_BACKENDS[key] = backend_used or "auto"
    return encoder


def _get_splade_backend_used(cfg: SpladeCfg) -> str:
    """Return the backend string recorded for a given SPLADE configuration."""

    key = (str(cfg.model_dir), cfg.device, cfg.attn_impl, cfg.max_active_dims)
    backend = _SPLADE_ENCODER_BACKENDS.get(key)
    if backend:
        return backend
    if cfg.attn_impl:
        return cfg.attn_impl
    return "auto"


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
            logger.warning(
                "SPLADE sparsity warning (threshold %.1f%%): %s / %s (%.1f%%) chunks have zero non-zero elements.",
                threshold,
                len(self.zero_nnz_chunks),
                total,
                pct,
            )
            logger.warning(
                "Affected UUIDs (first %s of %s): %s",
                top_n,
                len(zero_chunks),
                zero_chunks[:top_n],
            )


# --- Qwen3 Embeddings ---


def _qwen_embed_direct(
    cfg: QwenCfg, texts: List[str], batch_size: Optional[int] = None
) -> List[List[float]]:
    """Produce dense embeddings using a local Qwen3 model served by vLLM.

    Args:
        cfg: Configuration describing model path, dtype, and batching.
        texts: Batch of documents to embed.
        batch_size: Optional override for inference batch size.

    Returns:
        List of embedding vectors, one per input text.
    """
    _ensure_qwen_dependencies()

    effective_batch = batch_size or cfg.batch_size
    cache_key = _qwen_cache_key(cfg)
    llm = _QWEN_LLM_CACHE.get(cache_key)
    if llm is None:
        llm = LLM(
            model=str(cfg.model_dir),  # local path
            task="embed",
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tp,
            gpu_memory_utilization=cfg.gpu_mem_util,
            quantization=cfg.quantization,  # None or 'awq' (if a matching AWQ checkpoint exists)
            download_dir=str(HF_HOME),  # belt & suspenders: keep any aux files in your cache
        )
        _QWEN_LLM_CACHE[cache_key] = llm
    pool = PoolingParams(normalize=True, dimensions=int(cfg.dim))
    out: List[List[float]] = []
    for i in range(0, len(texts), effective_batch):
        batch = texts[i : i + effective_batch]
        res = llm.embed(batch, pooling_params=pool)
        for r in res:
            out.append([float(x) for x in r.outputs.embedding])
    return out


def qwen_embed(
    cfg: QwenCfg, texts: List[str], batch_size: Optional[int] = None
) -> List[List[float]]:
    """Public wrapper around the direct Qwen embedding implementation."""

    return _qwen_embed_direct(cfg, texts, batch_size=batch_size)


class QwenEmbeddingQueue:
    """Serialize Qwen embedding requests across worker threads."""

    def __init__(self, cfg: QwenCfg, *, maxsize: int = 8):
        self._cfg = cfg
        self._queue: "queue.Queue[tuple[List[str], int, Future[List[List[float]]]] | None]" = queue.Queue(
            maxsize=max(1, maxsize)
        )
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="QwenEmbeddingQueue", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            texts, batch_size, future = item
            try:
                result = _qwen_embed_direct(self._cfg, texts, batch_size=batch_size)
            except Exception as exc:  # pragma: no cover - propagates to caller
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._queue.task_done()

    def embed(self, texts: Sequence[str], batch_size: int) -> List[List[float]]:
        """Queue an embedding request and block until the result is ready."""

        if self._closed:
            raise RuntimeError("QwenEmbeddingQueue has been shut down")
        future: Future[List[List[float]]] = Future()
        self._queue.put((list(texts), int(batch_size), future))
        return future.result()

    def shutdown(self, wait: bool = True) -> None:
        """Flush pending requests and terminate the worker thread."""

        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        if wait:
            self._queue.join()
            self._thread.join()


class EmbeddingProcessingError(RuntimeError):
    """Wrap exceptions raised during per-file embedding with timing metadata."""

    def __init__(self, original: Exception, duration: float) -> None:
        super().__init__(str(original))
        self.original = original
        self.duration = duration


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

                # Ensure schema version
                version = row.get("schema_version")
                if not version:
                    row["schema_version"] = COMPATIBLE_CHUNK_VERSIONS[-1]
                    updated = True
                else:
                    validate_schema_version(
                        version,
                        COMPATIBLE_CHUNK_VERSIONS,
                        kind="chunk",
                        source=f"{chunk_file}:{line_no}",
                    )

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


def iter_rows_in_batches(path: Path, batch_size: int) -> Iterator[List[dict]]:
    """Iterate over JSONL rows in batches to reduce memory usage.

    Args:
        path: Path to JSONL file to read.
        batch_size: Number of rows to yield per batch.

    Returns:
        Iterator[List[dict]]: Lazy iterator producing batched chunk rows.

    Yields:
        Lists of row dictionaries, each containing up to batch_size items.
    """
    buf: List[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            buf.append(row)
            if len(buf) >= batch_size:
                yield buf
                buf = []
    if buf:
        yield buf


def iter_chunk_files(directory: Path) -> Iterator[Path]:
    """Deprecated shim that forwards to :func:`iter_chunks`.

    Args:
        directory: Directory to scan for chunk artifacts.

    Returns:
        Iterator over chunk files.
    """

    yield from iter_chunks(directory)


def process_chunk_file_vectors(
    chunk_file: Path,
    out_path: Path,
    stats: Optional[BM25Stats] = None,
    args: Optional[argparse.Namespace] = None,
    validator: Optional[SPLADEValidator] = None,
    logger=None,
) -> Tuple[int, List[int], List[float]]:
    """Generate vectors for a single chunk file and persist them to disk.

    Args:
        chunk_file: Chunk JSONL file to process.
        out_path: Destination path for vectors.
        stats: Precomputed BM25 statistics.
        args: Parsed CLI arguments with runtime configuration.
        validator: SPLADE validator for sparsity metrics.
        logger: Logger for structured output.

    Returns:
        Tuple of ``(vector_count, splade_nnz_list, qwen_norms)``.

    Raises:
        ValueError: Propagated if vector dimensions or norms fail validation.
    """

    if not isinstance(stats, BM25Stats) or args is None or validator is None:
        raise TypeError("process_chunk_file_vectors received invalid arguments")

    if not isinstance(out_path, Path):
        raise TypeError("out_path must be a Path")
    resolved_out_path = out_path
    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine batch size for streaming
    batch_size = max(args.batch_size_qwen, args.batch_size_splade)

    total_count = 0
    nnz_all: List[int] = []
    norms_all: List[float] = []

    with atomic_write(resolved_out_path) as handle:
        for rows in iter_rows_in_batches(chunk_file, batch_size):
            if not rows:
                continue
            ensure_chunk_schema(rows, chunk_file)

            uuids: List[str] = []
            texts: List[str] = []
            for row in rows:
                uuid_value = row.get("uuid")
                if not uuid_value:
                    raise ValueError(f"Chunk row missing UUID in {chunk_file}")
                uuids.append(uuid_value)
                texts.append(str(row.get("text", "")))

            # SPLADE encoding for this batch
            splade_results: List[Tuple[Sequence[str], Sequence[float]]] = []
            for batch in Batcher(texts, args.batch_size_splade):
                tokens_batch, weights_batch = splade_encode(
                    args.splade_cfg, list(batch), batch_size=args.batch_size_splade
                )
                splade_results.extend(zip(tokens_batch, weights_batch))

            # Qwen embedding for this batch
            qwen_queue = getattr(args, "qwen_queue", None)
            if qwen_queue is not None:
                qwen_results = qwen_queue.embed(texts, int(args.batch_size_qwen))
            else:
                qwen_results = qwen_embed(args.qwen_cfg, texts, batch_size=args.batch_size_qwen)

            # Write vectors for this batch immediately
            count, nnz, norms = write_vectors(
                handle,
                uuids,
                texts,
                splade_results,
                qwen_results,
                stats,
                args,
                rows=rows,
                validator=validator,
                logger=logger,
                output_path=resolved_out_path,
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
    destination,
    uuids: Sequence[str],
    texts: Sequence[str],
    splade_results: Sequence[Tuple[Sequence[str], Sequence[float]]],
    qwen_results: Sequence[Sequence[float]],
    stats: BM25Stats,
    args: argparse.Namespace,
    *,
    rows: Sequence[dict],
    validator: SPLADEValidator,
    logger,
    output_path: Optional[Path] = None,
) -> Tuple[int, List[int], List[float]]:
    """Write validated vector rows to disk with schema enforcement.

    Args:
        path_or_handle: Destination JSONL path or file handle for vector rows.
        uuids: Sequence of chunk UUIDs aligned with the other inputs.
        texts: Chunk text bodies.
        splade_results: SPLADE token and weight pairs per chunk.
        qwen_results: Dense embedding vectors per chunk.
        stats: BM25 statistics used to generate sparse vectors.
        args: Parsed CLI arguments for runtime configuration.
        rows: Original chunk row dictionaries.
        validator: SPLADE validator capturing sparsity data.
        logger: Logger used to emit structured diagnostics.

    Returns:
        Tuple containing the number of vectors written, SPLADE nnz counts,
        and Qwen vector norms.

    Raises:
        ValueError: If vector lengths are inconsistent or fail validation.
    """

    if not (len(uuids) == len(texts) == len(splade_results) == len(qwen_results) == len(rows)):
        raise ValueError("Mismatch between chunk, SPLADE, or Qwen result lengths")

    bm25_k1 = float(args.bm25_k1)
    bm25_b = float(args.bm25_b)
    splade_nnz: List[int] = []
    qwen_norms: List[float] = []

    dest_is_handle = hasattr(destination, "write")
    if not dest_is_handle and isinstance(destination, Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
    output_ref = output_path or (destination if isinstance(destination, Path) else None)
    payloads: List[dict] = []
    for uuid_value, text, splade_pair, qwen_vector, row in zip(
        uuids, texts, splade_results, qwen_results, rows
    ):
        tokens_list = list(splade_pair[0])
        weight_list = [float(w) for w in splade_pair[1]]
        validator.validate(uuid_value, tokens_list, weight_list)
        nnz = sum(1 for weight in weight_list if weight > 0)
        splade_nnz.append(nnz)

        if len(qwen_vector) != int(args.qwen_dim):
            message = (
                f"Qwen dimension mismatch for UUID={uuid_value}: expected {int(args.qwen_dim)}, "
                f"got {len(qwen_vector)}"
            )
            doc_id = row.get("doc_id", "unknown")
            manifest_log_failure(
                stage="embeddings",
                doc_id=doc_id,
                duration_s=0.0,
                schema_version=VECTOR_SCHEMA_VERSION,
                input_path=row.get("source_path", "unknown"),
                input_hash=row.get("input_hash", ""),
                output_path=output_ref,
                error=message,
            )
            raise ValueError(message)

        norm = math.sqrt(sum(float(x) * float(x) for x in qwen_vector))
        if norm <= 0:
            doc_id = row.get("doc_id", "unknown")
            message = f"Invalid Qwen vector (zero norm) for UUID={uuid_value}"
            logger.error(message)
            manifest_log_failure(
                stage="embeddings",
                doc_id=doc_id,
                duration_s=0.0,
                schema_version=VECTOR_SCHEMA_VERSION,
                input_path=row.get("source_path", "unknown"),
                input_hash=row.get("input_hash", ""),
                output_path=output_ref,
                error=message,
            )
            raise ValueError(message)

        if abs(norm - 1.0) > 0.01:
            logger.warning("Qwen norm for UUID=%s: %.4f (expected ~1.0)", uuid_value, norm)
        qwen_norms.append(norm)

        terms, weights = bm25_vector(text, stats, k1=bm25_k1, b=bm25_b)

        try:
            vector_row = VectorRow(
                UUID=uuid_value,
                BM25=BM25Vector(
                    terms=terms,
                    weights=weights,
                    k1=bm25_k1,
                    b=bm25_b,
                    avgdl=stats.avgdl,
                    N=stats.N,
                ),
                SPLADEv3=SPLADEVector(tokens=tokens_list, weights=weight_list),
                Qwen3_4B=DenseVector(
                    model_id="Qwen/Qwen3-Embedding-4B",
                    vector=[float(x) for x in qwen_vector],
                    dimension=int(args.qwen_dim),
                ),
                model_metadata={
                    "splade": {"batch_size": args.batch_size_splade},
                    "qwen": {
                        "dtype": args.qwen_dtype,
                        "batch_size": args.batch_size_qwen,
                    },
                },
            )
        except Exception as exc:
            doc_id = row.get("doc_id", "unknown")
            logger.error(
                "Vector row validation failed",
                extra={
                    "extra_fields": {
                        "uuid": uuid_value,
                        "doc_id": doc_id,
                        "error": str(exc),
                    }
                },
            )
            manifest_append(
                stage="embeddings",
                doc_id=doc_id,
                status="failure",
                error=str(exc),
                schema_version=VECTOR_SCHEMA_VERSION,
            )
            raise

        payloads.append(vector_row.model_dump(by_alias=True))

    jsonl_append_iter(destination, payloads, atomic=not dest_is_handle)

    return len(uuids), splade_nnz, qwen_norms


# --- Main Driver ---


def _validate_vectors_for_chunks(chunks_dir: Path, vectors_dir: Path, logger) -> tuple[int, int]:
    """Validate vectors associated with chunk files without recomputing models.

    Returns:
        (files_checked, rows_validated)
    """
    from DocsToKG.DocParsing.schemas import validate_vector_row

    files_checked = 0
    rows_validated = 0
    missing: List[tuple[str, Path]] = []

    for chunk_path in iter_chunks(chunks_dir):
        doc_id, vector_path = derive_doc_id_and_vectors_path(chunk_path, chunks_dir, vectors_dir)
        if not vector_path.exists():
            missing.append((doc_id, vector_path))
            continue
        files_checked += 1
        for batch in iter_rows_in_batches(vector_path, 4096):
            for row in batch:
                # Raises on error
                validate_vector_row(row)
                rows_validated += 1

    if missing:
        preview = ", ".join(doc for doc, _ in missing[:5])
        if len(missing) > 5:
            preview += ", ..."
        logger.error(
            "Missing vector files for chunk documents",
            extra={
                "extra_fields": {
                    "missing": [str(path) for _, path in missing],
                    "missing_count": len(missing),
                    "chunks_dir": str(chunks_dir),
                    "vectors_dir": str(vectors_dir),
                }
            },
        )
        raise FileNotFoundError("Vector files not found for documents: " + preview)

    logger.info(
        "Validated vector files",
        extra={
            "extra_fields": {
                "files_checked": files_checked,
                "rows_validated": rows_validated,
                "chunks_dir": str(chunks_dir),
                "vectors_dir": str(vectors_dir),
            }
        },
    )
    print(
        f"Validated {rows_validated} rows across {files_checked} vector files under {vectors_dir}"
    )
    return files_checked, rows_validated


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the embedding pipeline.

    Args:
        None

    Returns:
        argparse.ArgumentParser: Parser configured for embedding options.

    Raises:
        None
    """

    parser = argparse.ArgumentParser()
    add_data_root_option(parser)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--batch-size-splade", type=int, default=32)
    parser.add_argument("--batch-size-qwen", type=int, default=64)
    parser.add_argument("--splade-max-active-dims", type=int, default=None)
    parser.add_argument(
        "--splade-model-dir",
        type=Path,
        default=None,
        help=(
            "Override SPLADE model directory (CLI > $DOCSTOKG_SPLADE_DIR > "
            f"{SPLADE_DIR})."
            "model root/naver/splade-v3)."
        ),
    )
    parser.add_argument(
        "--splade-attn",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        help=(
            "Attention backend for SPLADE. 'auto' tries FlashAttention 2, then "
            "SDPA, then eager. 'flash_attention_2' requires the Flash Attention "
            "2 package. 'sdpa' forces PyTorch scaled dot-product attention. "
            "'eager' uses the standard attention implementation."
        ),
    )
    parser.add_argument("--qwen-dtype", type=str, default="bfloat16")
    parser.add_argument("--qwen-quant", type=str, default=None)
    parser.add_argument(
        "--qwen-model-dir",
        type=Path,
        default=None,
        help=(
            "Override Qwen model directory (CLI > $DOCSTOKG_QWEN_DIR > "
            f"{QWEN_DIR})."
            "model root/Qwen/Qwen3-Embedding-4B)."
        ),
    )
    parser.add_argument(
        "--splade-sparsity-warn-pct",
        dest="sparsity_warn_threshold_pct",
        type=float,
        default=SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
        help=(
            "Warn if percentage of zero-NNZ SPLADE vectors exceeds this threshold "
            f"(default: {SPLADE_SPARSITY_WARN_THRESHOLD_PCT})."
        ),
    )
    parser.add_argument(
        "--splade-zero-pct-warn-threshold",
        dest="sparsity_warn_threshold_pct",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qwen-dim",
        type=int,
        default=2560,
        help="Dimension of the dense embedding head (model dependent).",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--sparsity-report-top-n",
        type=int,
        default=10,
        help=(
            "Number of zero-NNZ SPLADE chunk UUIDs to list when sparsity exceeds the warning threshold "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--files-parallel",
        type=int,
        default=1,
        help="Process up to N chunk files concurrently during embedding (default: 1 for serial runs).",
    )
    add_resume_force_options(
        parser,
        resume_help="Skip chunk files whose vector outputs already exist with matching hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing vectors in --out-dir and exit",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Disable network access by setting TRANSFORMERS_OFFLINE=1. "
            "All models must already exist in local caches."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone embedding execution.

    Args:
        argv (list[str] | None): Optional CLI argument vector. When ``None`` the
            current process arguments are used.

    Returns:
        argparse.Namespace: Parsed embedding configuration.

    Raises:
        SystemExit: Propagated if ``argparse`` reports invalid options.
    """

    return build_parser().parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    """CLI entrypoint for chunk UUID cleanup and embedding generation.

    Args:
        args (argparse.Namespace | None): Optional parsed arguments, primarily
            for testing or orchestration.

    Returns:
        int: Exit code where ``0`` indicates success.

    Raises:
        ValueError: If invalid runtime parameters (such as batch sizes) are supplied.
    """

    logger = get_logger(__name__)

    parser = build_parser()
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, argparse.Namespace):
        pass
    elif isinstance(args, SimpleNamespace) or hasattr(args, "__dict__"):
        defaults = parser.parse_args([])
        for key, value in vars(args).items():
            setattr(defaults, key, value)
        args = defaults
    else:
        args = parser.parse_args(args)
    offline_mode = bool(getattr(args, "offline", False))

    global HF_HOME, MODEL_ROOT, QWEN_DIR, SPLADE_DIR
    HF_HOME, MODEL_ROOT = init_hf_env()
    QWEN_DIR = expand_path(_resolve_qwen_dir(MODEL_ROOT))
    SPLADE_DIR = expand_path(_resolve_splade_dir(MODEL_ROOT))
    model_root = MODEL_ROOT
    default_qwen_dir = QWEN_DIR
    default_splade_dir = SPLADE_DIR

    cli_splade = _expand_optional(getattr(args, "splade_model_dir", None))
    cli_qwen = _expand_optional(getattr(args, "qwen_model_dir", None))
    splade_model_dir = cli_splade or default_splade_dir
    qwen_model_dir = cli_qwen or default_qwen_dir

    validate_only = bool(getattr(args, "validate_only", False))

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

    args.offline = offline_mode
    args.splade_model_dir = splade_model_dir
    args.qwen_model_dir = qwen_model_dir

    splade_model_dir = _resolve_cli_path(args.splade_model_dir, SPLADE_DIR)
    qwen_model_dir = _resolve_cli_path(args.qwen_model_dir, QWEN_DIR)

    if args.batch_size_splade < 1 or args.batch_size_qwen < 1:
        raise ValueError("Batch sizes must be >= 1")

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

    if not validate_only:
        try:
            _ensure_splade_dependencies()
            _ensure_qwen_dependencies()
        except ImportError as exc:
            logger.error("Embedding dependencies unavailable: %s", exc)
            raise

    data_root_override = args.data_root
    data_root_overridden = data_root_override is not None
    resolved_root = prepare_data_root(data_root_override, DEFAULT_DATA_ROOT)

    chunks_dir = resolve_pipeline_path(
        cli_value=args.chunks_dir,
        default_path=DEFAULT_CHUNKS_DIR,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=data_chunks,
    ).resolve()

    out_dir = resolve_pipeline_path(
        cli_value=args.out_dir,
        default_path=DEFAULT_VECTORS_DIR,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=data_vectors,
    ).resolve()

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

    if validate_only:
        files_checked, rows_validated = _validate_vectors_for_chunks(chunks_dir, out_dir, logger)
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

    requested_parallel = max(1, int(getattr(args, "files_parallel", 1)))

    logger.info(
        "Embedding configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_root),
                "chunks_dir": str(chunks_dir),
                "embeddings_dir": str(out_dir),
                "splade_model_dir": str(splade_model_dir),
                "qwen_model_dir": str(qwen_model_dir),
                "offline": offline_mode,
                "requested_files_parallel": requested_parallel,
                "sparsity_warn_threshold_pct": getattr(
                    args,
                    "sparsity_warn_threshold_pct",
                    SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
                ),
                "sparsity_report_top_n": int(getattr(args, "sparsity_report_top_n", 10)),
            }
        },
    )

    files = list(iter_chunks(chunks_dir))
    if not files:
        logger.warning(
            "No chunk files found",
            extra={"extra_fields": {"chunks_dir": str(chunks_dir)}},
        )
        return 0

    incompatible_chunks: List[Path] = []
    for chunk_file in files:
        rows = jsonl_load(chunk_file)
        try:
            ensure_chunk_schema(rows, chunk_file)
        except ValueError as exc:
            incompatible_chunks.append(chunk_file)
            logger.error(
                "Chunk file rejected: incompatible schema",
                extra={
                    "extra_fields": {
                        "chunk_file": str(chunk_file),
                        "error": str(exc),
                    }
                },
            )
    if incompatible_chunks:
        summary = ", ".join(str(path) for path in incompatible_chunks[:5])
        if len(incompatible_chunks) > 5:
            summary += ", ..."
        raise ValueError(
            "Incompatible chunk schema detected; review chunk files before proceeding: " + summary
        )

    if args.force:
        logger.info("Force mode: reprocessing all chunk files")
    elif args.resume:
        logger.info("Resume mode enabled: unchanged chunk files will be skipped")

    attn_impl = None if args.splade_attn == "auto" else args.splade_attn
    args.splade_cfg = SpladeCfg(
        model_dir=splade_model_dir,
        cache_folder=model_root,
        batch_size=args.batch_size_splade,
        max_active_dims=args.splade_max_active_dims,
        attn_impl=attn_impl,
        local_files_only=bool(args.offline),
    )
    args.qwen_cfg = QwenCfg(
        model_dir=qwen_model_dir,
        dtype=args.qwen_dtype,
        tp=int(args.tp),
        batch_size=int(args.batch_size_qwen),
        quantization=args.qwen_quant,
        dim=int(args.qwen_dim),
    )

    stats = process_pass_a(files, logger)
    if not stats.N:
        logger.warning("No chunks found after Pass A")
        return 0

    validator = SPLADEValidator(
        warn_threshold_pct=getattr(
            args,
            "sparsity_warn_threshold_pct",
            SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
        ),
        top_n=max(1, int(getattr(args, "sparsity_report_top_n", 10))),
    )
    tracemalloc.start()
    pass_b_start = time.perf_counter()
    total_vectors = 0
    splade_nnz_all: List[int] = []
    qwen_norms_all: List[float] = []

    manifest_index = load_manifest_index(MANIFEST_STAGE, resolved_root) if args.resume else {}
    file_entries: List[Tuple[Path, Path, str, str]] = []
    skipped_files = 0
    for chunk_file in files:
        doc_id, out_path = derive_doc_id_and_vectors_path(chunk_file, chunks_dir, args.out_dir)
        input_hash = compute_content_hash(chunk_file)
        entry = manifest_index.get(doc_id)
        if should_skip_output(out_path, entry, input_hash, args.resume, args.force):
            log_event(
                logger,
                "info",
                "Skipping chunk file: output exists and input unchanged",
                doc_id=doc_id,
                output_path=str(out_path),
            )
            manifest_log_skip(
                stage=MANIFEST_STAGE,
                doc_id=doc_id,
                input_path=chunk_file,
                input_hash=input_hash,
                output_path=out_path,
                schema_version=VECTOR_SCHEMA_VERSION,
            )
            skipped_files += 1
            continue
        file_entries.append((chunk_file, out_path, input_hash, doc_id))

    files_parallel = min(requested_parallel, max(1, len(file_entries)))
    args.files_parallel = files_parallel

    if files_parallel > 1:
        log_event(logger, "info", "File-level parallelism enabled", files_parallel=files_parallel)

    qwen_queue: QwenEmbeddingQueue | None = None
    try:
        if files_parallel > 1 and file_entries:
            qwen_queue = QwenEmbeddingQueue(args.qwen_cfg, maxsize=files_parallel * 2)
            args.qwen_queue = qwen_queue
        else:
            args.qwen_queue = None

        def _process_entry(entry: Tuple[Path, Path, str, str]) -> Tuple[int, List[int], List[float], float]:
            chunk_path, vectors_path, input_hash, doc_id = entry
            start = time.perf_counter()
            try:
                with acquire_lock(vectors_path):
                    count, nnz, norms = process_chunk_file_vectors(
                        chunk_path, vectors_path, stats, args, validator, logger
                    )
            except Exception as exc:  # pragma: no cover - propagated to caller
                duration = time.perf_counter() - start
                raise EmbeddingProcessingError(exc, duration) from exc
            duration = time.perf_counter() - start
            return count, nnz, norms, duration

        if not file_entries:
            pass
        elif files_parallel > 1:
            with ThreadPoolExecutor(max_workers=files_parallel) as executor, tqdm(
                total=len(file_entries), desc="Pass B: Encoding vectors", unit="file"
            ) as bar:
                future_map = {
                    executor.submit(_process_entry, entry): entry for entry in file_entries
                }
                for future in as_completed(future_map):
                    chunk_file, out_path, input_hash, doc_id = future_map[future]
                    try:
                        count, nnz, norms, duration = future.result()
                    except EmbeddingProcessingError as exc:
                        manifest_log_failure(
                            stage=MANIFEST_STAGE,
                            doc_id=doc_id,
                            duration_s=round(exc.duration, 3),
                            schema_version=VECTOR_SCHEMA_VERSION,
                            input_path=chunk_file,
                            input_hash=input_hash,
                            output_path=out_path,
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
                            error=str(exc),
                        )
                        bar.update(1)
                        raise
                    total_vectors += count
                    splade_nnz_all.extend(nnz)
                    qwen_norms_all.extend(norms)
                    manifest_log_success(
                        stage=MANIFEST_STAGE,
                        doc_id=doc_id,
                        duration_s=round(duration, 3),
                        schema_version=VECTOR_SCHEMA_VERSION,
                        input_path=chunk_file,
                        input_hash=input_hash,
                        output_path=out_path,
                        vector_count=count,
                    )
                    bar.update(1)
        else:
            for entry in tqdm(file_entries, desc="Pass B: Encoding vectors", unit="file"):
                chunk_file, out_path, input_hash, doc_id = entry
                try:
                    count, nnz, norms, duration = _process_entry(entry)
                except EmbeddingProcessingError as exc:
                    manifest_log_failure(
                        stage=MANIFEST_STAGE,
                        doc_id=doc_id,
                        duration_s=round(exc.duration, 3),
                        schema_version=VECTOR_SCHEMA_VERSION,
                        input_path=chunk_file,
                        input_hash=input_hash,
                        output_path=out_path,
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
                        error=str(exc),
                    )
                    raise
                total_vectors += count
                splade_nnz_all.extend(nnz)
                qwen_norms_all.extend(norms)
                manifest_log_success(
                    stage=MANIFEST_STAGE,
                    doc_id=doc_id,
                    duration_s=round(duration, 3),
                    schema_version=VECTOR_SCHEMA_VERSION,
                    input_path=chunk_file,
                    input_hash=input_hash,
                    output_path=out_path,
                    vector_count=count,
                )
    finally:
        args.qwen_queue = None
        if qwen_queue is not None:
            qwen_queue.shutdown()

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_b = time.perf_counter() - pass_b_start

    validator.report(logger)

    zero_pct = (
        100.0 * len([n for n in splade_nnz_all if n == 0]) / total_vectors if total_vectors else 0.0
    )
    avg_nnz = statistics.mean(splade_nnz_all) if splade_nnz_all else 0.0
    median_nnz = statistics.median(splade_nnz_all) if splade_nnz_all else 0.0
    avg_norm = statistics.mean(qwen_norms_all) if qwen_norms_all else 0.0
    std_norm = statistics.pstdev(qwen_norms_all) if len(qwen_norms_all) > 1 else 0.0

    backend_used = _get_splade_backend_used(args.splade_cfg)

    logger.info(
        "Embedding summary",
        extra={
            "extra_fields": {
                "total_vectors": total_vectors,
                "splade_avg_nnz": round(avg_nnz, 3),
                "splade_median_nnz": round(median_nnz, 3),
                "splade_zero_pct": round(zero_pct, 2),
                "qwen_avg_norm": round(avg_norm, 4),
                "qwen_std_norm": round(std_norm, 4),
                "pass_b_seconds": round(elapsed_b, 3),
                "skipped_files": skipped_files,
                "files_parallel": files_parallel,
                "splade_attn_backend_used": backend_used,
                "sparsity_warn_threshold_pct": getattr(
                    args,
                    "sparsity_warn_threshold_pct",
                    SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
                ),
            }
        },
    )
    logger.info("Peak memory: %.2f GB", peak / 1024**3)

    manifest_append(
        stage=MANIFEST_STAGE,
        doc_id="__corpus__",
        status="success",
        duration_s=round(time.perf_counter() - overall_start, 3),
        warnings=validator.zero_nnz_chunks[: validator.top_n] if validator.zero_nnz_chunks else [],
        schema_version=VECTOR_SCHEMA_VERSION,
        total_vectors=total_vectors,
        splade_avg_nnz=avg_nnz,
        splade_median_nnz=median_nnz,
        splade_zero_pct=zero_pct,
        qwen_avg_norm=avg_norm,
        qwen_std_norm=std_norm,
        peak_memory_gb=peak / 1024**3,
        skipped_files=skipped_files,
        files_parallel=files_parallel,
        splade_attn_backend_used=backend_used,
        sparsity_warn_threshold_pct=getattr(
            args,
            "sparsity_warn_threshold_pct",
            SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
        ),
    )

    logger.info(
        "[DONE] Saved vectors",
        extra={
            "extra_fields": {
                "embeddings_dir": str(out_dir),
                "processed_files": len(file_entries),
                "skipped_files": skipped_files,
            }
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
