#!/usr/bin/env python3
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
import json
import math
import os
import re
import statistics
import time
import tracemalloc
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

# Third-party imports
from tqdm import tqdm

from DocsToKG.DocParsing._common import (
    Batcher,
    atomic_write,
    compute_content_hash,
    data_chunks,
    data_vectors,
    detect_data_root,
    expand_path,
    get_logger,
    iter_chunks,
    jsonl_load,
    jsonl_save,
    load_manifest_index,
    manifest_append,
    resolve_hash_algorithm,
    resolve_hf_home,
    resolve_model_root,
)
from DocsToKG.DocParsing.pipelines import (
    add_data_root_option,
    add_resume_force_options,
    prepare_data_root,
    resolve_pipeline_path,
)
from DocsToKG.DocParsing.schemas import (
    COMPATIBLE_CHUNK_VERSIONS,
    BM25Vector,
    DenseVector,
    SPLADEVector,
    VectorRow,
    validate_schema_version,
)

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

# ---- Cache / model path resolution ----

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


HF_HOME = resolve_hf_home()
MODEL_ROOT = resolve_model_root(HF_HOME)
QWEN_DIR = expand_path(_resolve_qwen_dir(MODEL_ROOT))
SPLADE_DIR = expand_path(_resolve_splade_dir(MODEL_ROOT))


def _derive_doc_id_and_output_path(
    chunk_file: Path, chunks_root: Path, vectors_root: Path
) -> tuple[str, Path]:
    """Return manifest doc_id and vector output path for a chunk artifact."""

    relative = chunk_file.relative_to(chunks_root)
    base = relative
    if base.suffix == ".jsonl":
        base = base.with_suffix("")
    if base.suffix == ".chunks":
        base = base.with_suffix("")
    doc_id = base.with_suffix(".doctags").as_posix()
    vector_relative = base.with_suffix(".vectors.jsonl")
    return doc_id, vectors_root / vector_relative


def _derive_doc_id_and_output_path(
    chunk_file: Path, chunks_root: Path, vectors_root: Path
) -> tuple[str, Path]:
    """Return manifest doc_id and vector output path for a chunk artifact."""

    relative = chunk_file.relative_to(chunks_root)
    base = relative
    if base.suffix == ".jsonl":
        base = base.with_suffix("")
    if base.suffix == ".chunks":
        base = base.with_suffix("")
    doc_id = base.with_suffix(".doctags").as_posix()
    vector_relative = base.with_suffix(".vectors.jsonl")
    return doc_id, vectors_root / vector_relative


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

# Make sure every lib (Transformers / HF Hub / Sentence-Transformers) honors this cache.
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_ROOT))
os.environ.setdefault("DOCSTOKG_MODEL_ROOT", str(MODEL_ROOT))


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


# ---- simple tokenizer for BM25 ----

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


def ensure_uuid(rows: List[dict]) -> bool:
    """Populate missing chunk UUIDs in-place.

    Args:
        rows: Chunk dictionaries that should include a `uuid` key.

    Returns:
        True when at least one UUID was newly assigned; otherwise False.
    """

    updated = False
    for row in rows:
        if not row.get("uuid"):
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
    return [t.lower() for t in TOKEN_RE.findall(text)]


# ---- BM25 (global) ----
@dataclass
class BM25Stats:
    """Corpus-wide statistics required for BM25 weighting.

    Attributes:
        N: Total number of documents (chunks) in the corpus.
        avgdl: Average document length in tokens.
        df: Document frequency per token.

    Examples:
        >>> stats = BM25Stats(N=100, avgdl=120.5, df={"hybrid": 5})
        >>> stats.df["hybrid"]
        5
    """

    N: int
    avgdl: float
    df: Dict[str, int]


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


# ---- SPLADE-v3 (GPU) ----
@dataclass
class SpladeCfg:
    """Runtime configuration for SPLADE sparse encoding.

    Attributes:
        model_dir: Path to the SPLADE model directory.
        device: Torch device identifier to run inference on.
        batch_size: Number of texts encoded per batch.
        cache_folder: Directory where transformer weights are cached.
        max_active_dims: Optional cap on active sparse dimensions.
        attn_impl: Preferred attention implementation override.

    Examples:
        >>> cfg = SpladeCfg(batch_size=8, device="cuda:1")
        >>> cfg.batch_size
        8
    """

    model_dir: Path = SPLADE_DIR
    device: str = "cuda"
    batch_size: int = 32
    cache_folder: Path = MODEL_ROOT
    max_active_dims: int | None = None
    # Leave None to let HF/torch pick (usually SDPA); set to "flash_attention_2" if available.
    attn_impl: str | None = None


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
            local_files_only=True,
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
                local_files_only=True,
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

    def __init__(self) -> None:
        """Initialise internal counters for SPLADE sparsity tracking.

        Args:
            self: Validator instance being initialised.

        Returns:
            None
        """
        self.total_chunks = 0
        self.zero_nnz_chunks: List[str] = []
        self.nnz_counts: List[int] = []

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

        self.total_chunks += 1
        nnz = sum(1 for weight in weights if weight > 0)
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

        if not self.total_chunks:
            return
        pct = 100 * len(self.zero_nnz_chunks) / self.total_chunks
        if pct > 1.0:
            logger.warning(
                "SPLADE sparsity warning: %s / %s (%.1f%%) chunks have zero non-zero elements.",
                len(self.zero_nnz_chunks),
                self.total_chunks,
                pct,
            )
            logger.warning(
                "Affected UUIDs (first 10): %s",
                self.zero_nnz_chunks[:10],
            )


# ---- Qwen3-Embedding-4B via vLLM (2560-d) ----
@dataclass
class QwenCfg:
    """Configuration for generating dense embeddings with Qwen via vLLM.

    Attributes:
        model_dir: Path to the local Qwen model.
        dtype: Torch dtype used during inference.
        tp: Tensor parallelism degree.
        gpu_mem_util: Target GPU memory utilization for vLLM.
        batch_size: Number of texts processed per embedding batch.
        quantization: Optional quantization mode (e.g., `awq`).

    Examples:
        >>> cfg = QwenCfg(batch_size=64, tp=2)
        >>> cfg.tp
        2
    """

    model_dir: Path = QWEN_DIR
    dtype: str = "bfloat16"  # good default on Ada/Hopper
    tp: int = 1
    gpu_mem_util: float = 0.60
    batch_size: int = 32
    quantization: str | None = None  # 'awq' if you have an AWQ checkpoint


def qwen_embed(
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
    pool = PoolingParams(normalize=True)
    out: List[List[float]] = []
    for i in range(0, len(texts), effective_batch):
        batch = texts[i : i + effective_batch]
        res = llm.embed(batch, pooling_params=pool)
        for r in res:
            out.append([float(x) for x in r.outputs.embedding])
    return out


def process_pass_a(files: Sequence[Path], logger) -> BM25Stats:
    """Assign UUIDs and build BM25 statistics for a corpus of chunk files.

    Args:
        files: Sequence of chunk file paths to process.
        logger: Logger used for structured progress output.

    Returns:
        Aggregated BM25 statistics for the supplied chunk corpus.

    Raises:
        ValueError: Propagated when chunk rows are missing required fields.
    """

    accumulator = BM25StatsAccumulator()

    for chunk_file in tqdm(files, desc="Pass A: UUID + BM25 stats", unit="file"):
        rows = jsonl_load(chunk_file)
        if not rows:
            continue
        ensure_chunk_schema(rows, chunk_file)
        if ensure_uuid(rows):
            jsonl_save(chunk_file, rows)
        for row in rows:
            text = row.get("text", "")
            accumulator.add_document(text)

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


def process_chunk_file_vectors(
    chunk_file: Path,
    out_path: Path,
    stats: BM25Stats,
    args: argparse.Namespace,
    validator: SPLADEValidator,
    logger,
) -> Tuple[int, List[int], List[float]]:
    """Generate vectors for a single chunk file and persist them to disk.

    Args:
        chunk_file: Chunk JSONL file to process.
        stats: Precomputed BM25 statistics.
        args: Parsed CLI arguments with runtime configuration.
        validator: SPLADE validator for sparsity metrics.
        logger: Logger for structured output.

    Returns:
        Tuple of ``(vector_count, splade_nnz_list, qwen_norms)``.

    Raises:
        ValueError: Propagated if vector dimensions or norms fail validation.
    """

    rows = jsonl_load(chunk_file)
    if not rows:
        logger.warning("Chunk file empty", extra={"extra_fields": {"chunk_file": str(chunk_file)}})
        return 0, [], []
    ensure_chunk_schema(rows, chunk_file)

    uuids: List[str] = []
    texts: List[str] = []
    for row in rows:
        uuid_value = row.get("uuid")
        if not uuid_value:
            raise ValueError(f"Chunk row missing UUID in {chunk_file}")
        uuids.append(uuid_value)
        texts.append(str(row.get("text", "")))
    splade_results: List[Tuple[Sequence[str], Sequence[float]]] = []
    for batch in Batcher(texts, args.batch_size_splade):
        tokens_batch, weights_batch = splade_encode(
            args.splade_cfg, list(batch), batch_size=args.batch_size_splade
        )
        splade_results.extend(zip(tokens_batch, weights_batch))
    qwen_results = qwen_embed(args.qwen_cfg, texts, batch_size=args.batch_size_qwen)

    count, nnz, norms = write_vectors(
        out_path,
        uuids,
        texts,
        splade_results,
        qwen_results,
        stats,
        args,
        rows=rows,
        validator=validator,
        logger=logger,
    )

    logger.info(
        "Embeddings written",
        extra={
            "extra_fields": {
                "chunk_file": str(chunk_file),
                "vectors_file": str(out_path),
                "rows": count,
            }
        },
    )
    return count, nnz, norms


def write_vectors(
    path: Path,
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
) -> Tuple[int, List[int], List[float]]:
    """Write validated vector rows to disk with schema enforcement.

    Args:
        path: Destination JSONL path for vector rows.
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

    path.parent.mkdir(parents=True, exist_ok=True)

    with atomic_write(path) as handle:
        for uuid_value, text, splade_pair, qwen_vector, row in zip(
            uuids, texts, splade_results, qwen_results, rows
        ):
            tokens_list = list(splade_pair[0])
            weight_list = [float(w) for w in splade_pair[1]]
            validator.validate(uuid_value, tokens_list, weight_list)
            nnz = sum(1 for weight in weight_list if weight > 0)
            splade_nnz.append(nnz)

            if len(qwen_vector) != 2560:
                message = (
                    f"Qwen dimension mismatch for UUID={uuid_value}: expected 2560, "
                    f"got {len(qwen_vector)}"
                )
                doc_id = row.get("doc_id", "unknown")
                manifest_append(
                    stage="embeddings",
                    doc_id=doc_id,
                    status="failure",
                    error=message,
                    schema_version="embeddings/1.0.0",
                )
                raise ValueError(message)

            norm = math.sqrt(sum(float(x) * float(x) for x in qwen_vector))
            if norm <= 0:
                doc_id = row.get("doc_id", "unknown")
                message = f"Invalid Qwen vector (zero norm) for UUID={uuid_value}"
                logger.error(message)
                manifest_append(
                    stage="embeddings",
                    doc_id=doc_id,
                    status="failure",
                    error=message,
                    schema_version="embeddings/1.0.0",
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
                        dimension=2560,
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
                    schema_version="embeddings/1.0.0",
                )
                raise

            payload = vector_row.model_dump(by_alias=True)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return len(uuids), splade_nnz, qwen_norms


# ---- Main driver ----
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
    parser.add_argument("--tp", type=int, default=1)
    add_resume_force_options(
        parser,
        resume_help="Skip chunk files whose vector outputs already exist with matching hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
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

    hf_home = resolve_hf_home()
    model_root = resolve_model_root(hf_home)
    default_splade_dir = expand_path(_resolve_splade_dir(model_root))
    default_qwen_dir = expand_path(_resolve_qwen_dir(model_root))
    cli_splade = _expand_optional(getattr(args, "splade_model_dir", None))
    cli_qwen = _expand_optional(getattr(args, "qwen_model_dir", None))
    splade_model_dir = cli_splade or default_splade_dir
    qwen_model_dir = cli_qwen or default_qwen_dir

    global HF_HOME, MODEL_ROOT, QWEN_DIR, SPLADE_DIR
    HF_HOME = hf_home
    MODEL_ROOT = model_root
    QWEN_DIR = default_qwen_dir
    SPLADE_DIR = default_splade_dir

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(model_root)

    if offline_mode:
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

    if args.offline:
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

    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = out_dir

    logger.info(
        "Embedding configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_root),
                "chunks_dir": str(chunks_dir),
                "vectors_dir": str(out_dir),
                "splade_model_dir": str(splade_model_dir),
                "qwen_model_dir": str(qwen_model_dir),
                "offline": offline_mode,
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
    )
    args.qwen_cfg = QwenCfg(
        model_dir=qwen_model_dir,
        dtype=args.qwen_dtype,
        tp=int(args.tp),
        batch_size=int(args.batch_size_qwen),
        quantization=args.qwen_quant,
    )

    stats = process_pass_a(files, logger)
    if not stats.N:
        logger.warning("No chunks found after Pass A")
        return 0

    validator = SPLADEValidator()
    tracemalloc.start()
    pass_b_start = time.perf_counter()
    total_vectors = 0
    splade_nnz_all: List[int] = []
    qwen_norms_all: List[float] = []

    manifest_index = load_manifest_index(MANIFEST_STAGE, resolved_root) if args.resume else {}
    file_entries = []
    skipped_files = 0
    for chunk_file in files:
        doc_id, out_path = _derive_doc_id_and_output_path(
            chunk_file, chunks_dir, args.out_dir
        )
        input_hash = compute_content_hash(chunk_file)
        entry = manifest_index.get(doc_id)
        if entry is None:
            legacy_key = chunk_file.relative_to(chunks_dir).as_posix()
            entry = manifest_index.get(legacy_key)
        if (
            args.resume
            and not args.force
            and out_path.exists()
            and entry
            and entry.get("input_hash") == input_hash
        ):
            logger.info(
                "Skipping chunk file: output exists and input unchanged",
                extra={
                    "extra_fields": {
                        "doc_id": doc_id,
                        "output_path": str(out_path),
                    }
                },
            )
            manifest_append(
                stage=MANIFEST_STAGE,
                doc_id=doc_id,
                status="skip",
                duration_s=0.0,
                schema_version="embeddings/1.0.0",
                input_path=str(chunk_file),
                input_hash=input_hash,
                hash_alg=resolve_hash_algorithm(),
                output_path=str(out_path),
            )
            skipped_files += 1
            continue
        file_entries.append((chunk_file, out_path, input_hash, doc_id))

    for chunk_file, out_path, input_hash, doc_id in tqdm(
        file_entries, desc="Pass B: Encoding vectors", unit="file"
    ):
        start = time.perf_counter()
        try:
            count, nnz, norms = process_chunk_file_vectors(
                chunk_file, out_path, stats, args, validator, logger
            )
        except Exception as exc:
            duration = time.perf_counter() - start
            manifest_append(
                stage=MANIFEST_STAGE,
                doc_id=doc_id,
                status="failure",
                duration_s=round(duration, 3),
                schema_version="embeddings/1.0.0",
                input_path=str(chunk_file),
                input_hash=input_hash,
                hash_alg=resolve_hash_algorithm(),
                output_path=str(out_path),
                error=str(exc),
            )
            raise

        duration = time.perf_counter() - start
        total_vectors += count
        splade_nnz_all.extend(nnz)
        qwen_norms_all.extend(norms)
        manifest_append(
            stage=MANIFEST_STAGE,
            doc_id=doc_id,
            status="success",
            duration_s=round(duration, 3),
            schema_version="embeddings/1.0.0",
            input_path=str(chunk_file),
            input_hash=input_hash,
            hash_alg=resolve_hash_algorithm(),
            output_path=str(out_path),
            vector_count=count,
        )

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
                "splade_attn_backend_used": backend_used,
                "sparsity_warn_threshold_pct": SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
            }
        },
    )
    logger.info("Peak memory: %.2f GB", peak / 1024**3)

    manifest_append(
        stage=MANIFEST_STAGE,
        doc_id="__corpus__",
        status="success",
        duration_s=round(time.perf_counter() - overall_start, 3),
        warnings=validator.zero_nnz_chunks[:10] if validator.zero_nnz_chunks else [],
        schema_version="embeddings/1.0.0",
        total_vectors=total_vectors,
        splade_avg_nnz=avg_nnz,
        splade_median_nnz=median_nnz,
        splade_zero_pct=zero_pct,
        qwen_avg_norm=avg_norm,
        qwen_std_norm=std_norm,
        peak_memory_gb=peak / 1024**3,
        skipped_files=skipped_files,
        splade_attn_backend_used=backend_used,
        sparsity_warn_threshold_pct=SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
    )

    logger.info(
        "[DONE] Saved vectors",
        extra={
            "extra_fields": {
                "vectors_dir": str(out_dir),
                "processed_files": len(file_entries),
                "skipped_files": skipped_files,
            }
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
