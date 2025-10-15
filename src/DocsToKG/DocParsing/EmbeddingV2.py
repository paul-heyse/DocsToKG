#!/usr/bin/env python3
"""
EmbedVectors.py

- Reads chunked JSONL from /home/paul/DocsToKG/Data/ChunkedDocTagFiles
- Ensures each chunk has a UUID (writes back to the same files)
- Emits vectors JSONL to /home/paul/DocsToKG/Data/Vectors

Uses local HF cache at /home/paul/hf-cache/:
  - Qwen3-Embedding-4B at   /home/paul/hf-cache/models/Qwen/Qwen3-Embedding-4B
  - SPLADE-v3 at            /home/paul/hf-cache/models/naver/splade-v3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import tracemalloc
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Third-party imports
from sentence_transformers import (
    SparseEncoder,
)  # loads from local dir if given (cache_folder supported)
from tqdm import tqdm

from vllm import (
    LLM,
    PoolingParams,
)  # PoolingParams(dimensions=...) selects output dim if model supports MRL

from DocsToKG.DocParsing._common import (
    Batcher,
    atomic_write,
    compute_content_hash,
    data_chunks,
    data_manifests,
    data_vectors,
    detect_data_root,
    get_logger,
    jsonl_load,
    jsonl_save,
    load_manifest_index,
    manifest_append,
)
from DocsToKG.DocParsing.schemas import BM25Vector, DenseVector, SPLADEVector, VectorRow

# ---- Fixed locations ----
HF_HOME = Path("/home/paul/hf-cache")
MODEL_ROOT = HF_HOME
QWEN_DIR = MODEL_ROOT / "Qwen" / "Qwen3-Embedding-4B"
SPLADE_DIR = MODEL_ROOT / "naver" / "splade-v3"

DEFAULT_DATA_ROOT = detect_data_root()
DEFAULT_CHUNKS_DIR = data_chunks(DEFAULT_DATA_ROOT)
DEFAULT_VECTORS_DIR = data_vectors(DEFAULT_DATA_ROOT)
MANIFEST_STAGE = "embeddings"

# Make sure every lib (Transformers / HF Hub / Sentence-Transformers) honors this cache.
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_ROOT))

# ---- simple tokenizer for BM25 ----

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


@dataclass
class Chunk:
    """Minimal representation of a DocTags chunk stored on disk.

    Attributes:
        uuid: Stable identifier for the chunk.
        text: Textual content extracted from the DocTags document.
        doc_id: Identifier of the source document for manifest reporting.

    Examples:
        >>> chunk = Chunk(uuid="chunk-1", text="Hybrid search is powerful.", doc_id="doc")
        >>> chunk.uuid
        'chunk-1'
    """

    uuid: str
    text: str
    doc_id: str


def iter_chunk_files(d: Path) -> List[Path]:
    """Enumerate chunked DocTags JSONL files in a directory.

    Args:
        d: Directory containing `*.chunks.jsonl` files.

    Returns:
        Sorted list of chunk file paths.
    """
    return sorted(d.glob("*.chunks.jsonl"))


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


def build_bm25_stats(chunks: Iterable[Chunk]) -> BM25Stats:
    """Compute corpus statistics required for BM25 weighting.

    Args:
        chunks: Iterable of text chunks with identifiers.

    Returns:
        BM25Stats containing document frequency counts and average length.
    """
    accumulator = BM25StatsAccumulator()
    for chunk in chunks:
        accumulator.add_document(chunk.text)
    return accumulator.finalize()


class BM25StatsAccumulator:
    """Streaming accumulator for BM25 corpus statistics."""

    def __init__(self) -> None:
        self.N = 0
        self.total_tokens = 0
        self.df = Counter()

    def add_document(self, text: str) -> None:
        """Add document to statistics without retaining text."""

        toks = tokens(text)
        self.N += 1
        self.total_tokens += len(toks)
        self.df.update(set(toks))

    def finalize(self) -> BM25Stats:
        """Compute final statistics."""

        avgdl = self.total_tokens / max(self.N, 1)
        return BM25Stats(N=self.N, avgdl=avgdl, df=dict(self.df))


def print_bm25_summary(stats: BM25Stats) -> None:
    """Print corpus-level BM25 statistics."""

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


def _get_splade_encoder(cfg: SpladeCfg) -> SparseEncoder:
    """Retrieve (or create) a cached SPLADE encoder instance."""

    key = (str(cfg.model_dir), cfg.device, cfg.attn_impl, cfg.max_active_dims)
    if key in _SPLADE_ENCODER_CACHE:
        return _SPLADE_ENCODER_CACHE[key]

    model_kwargs: Dict[str, object] = {}
    if cfg.attn_impl:
        model_kwargs["attn_implementation"] = cfg.attn_impl
    if cfg.max_active_dims is not None:
        model_kwargs["max_active_dims"] = cfg.max_active_dims

    try:
        encoder = SparseEncoder(
            str(cfg.model_dir),
            device=cfg.device,
            cache_folder=str(cfg.cache_folder),
            model_kwargs=model_kwargs,
            local_files_only=True,
        )
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
        else:
            raise

    _SPLADE_ENCODER_CACHE[key] = encoder
    return encoder


class SPLADEValidator:
    """Track SPLADE sparsity metrics across the corpus."""

    def __init__(self) -> None:
        self.total_chunks = 0
        self.zero_nnz_chunks: List[str] = []
        self.nnz_counts: List[int] = []

    def validate(self, uuid: str, tokens: Sequence[str], weights: Sequence[float]) -> None:
        """Record sparsity information for a single chunk."""

        self.total_chunks += 1
        nnz = sum(1 for weight in weights if weight > 0)
        self.nnz_counts.append(nnz)
        if nnz == 0:
            self.zero_nnz_chunks.append(uuid)

    def report(self, logger) -> None:
        """Emit warnings if sparsity metrics exceed thresholds."""

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
    effective_batch = batch_size or cfg.batch_size
    llm = LLM(
        model=str(cfg.model_dir),  # local path
        task="embed",
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tp,
        gpu_memory_utilization=cfg.gpu_mem_util,
        quantization=cfg.quantization,  # None or 'awq' (if a matching AWQ checkpoint exists)
        download_dir=str(HF_HOME),  # belt & suspenders: keep any aux files in your cache
    )
    pool = PoolingParams(normalize=True)
    out: List[List[float]] = []
    for i in range(0, len(texts), effective_batch):
        batch = texts[i : i + effective_batch]
        res = llm.embed(batch, pooling_params=pool)
        for r in res:
            out.append([float(x) for x in r.outputs.embedding])
    return out


def process_pass_a(files: Sequence[Path], logger) -> Tuple[Dict[str, Chunk], BM25Stats]:
    """Assign UUIDs and build BM25 statistics for a corpus of chunk files."""

    uuid_to_chunk: Dict[str, Chunk] = {}
    accumulator = BM25StatsAccumulator()

    for chunk_file in tqdm(files, desc="Pass A: UUID + BM25 stats", unit="file"):
        rows = jsonl_load(chunk_file)
        if not rows:
            continue
        if ensure_uuid(rows):
            jsonl_save(chunk_file, rows)
        for row in rows:
            text = row.get("text", "")
            uuid_value = row["uuid"]
            doc_id = row.get("doc_id", "unknown")
            uuid_to_chunk[uuid_value] = Chunk(uuid=uuid_value, text=text, doc_id=doc_id)
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
    return uuid_to_chunk, stats


def process_chunk_file_vectors(
    chunk_file: Path,
    uuid_to_chunk: Dict[str, Chunk],
    stats: BM25Stats,
    args: argparse.Namespace,
    validator: SPLADEValidator,
    logger,
) -> Tuple[int, List[int], List[float]]:
    """Generate vectors for a single chunk file and persist them to disk."""

    rows = jsonl_load(chunk_file)
    if not rows:
        logger.warning(
            "Chunk file empty", extra={"extra_fields": {"chunk_file": str(chunk_file)}}
        )
        return 0, [], []

    uuids = [row["uuid"] for row in rows]
    texts = [uuid_to_chunk[uuid].text for uuid in uuids]
    splade_results: List[Tuple[Sequence[str], Sequence[float]]] = []
    for batch in Batcher(texts, args.batch_size_splade):
        tokens_batch, weights_batch = splade_encode(
            args.splade_cfg, list(batch), batch_size=args.batch_size_splade
        )
        splade_results.extend(zip(tokens_batch, weights_batch))
    qwen_results = qwen_embed(
        args.qwen_cfg, texts, batch_size=args.batch_size_qwen
    )

    out_path = args.out_dir / f"{chunk_file.stem.replace('.chunks', '')}.vectors.jsonl"
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
        "Vectors written",
        extra={
            "extra_fields": {
                "chunk_file": str(chunk_file.name),
                "vectors_file": out_path.name,
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
    """Write validated vector rows to disk with schema enforcement."""

    if not (
        len(uuids)
        == len(texts)
        == len(splade_results)
        == len(qwen_results)
        == len(rows)
    ):
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
                logger.warning(
                    "Qwen norm for UUID=%s: %.4f (expected ~1.0)", uuid_value, norm
                )
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
def main():
    """CLI entrypoint for chunk UUID cleanup and embedding generation."""

    logger = get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Override DocsToKG Data directory. Defaults to auto-detection or "
            "$DOCSTOKG_DATA_ROOT."
        ),
    )
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--batch-size-splade", type=int, default=32)
    parser.add_argument("--batch-size-qwen", type=int, default=64)
    parser.add_argument("--splade-max-active-dims", type=int, default=None)
    parser.add_argument(
        "--splade-attn",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        help="Attention backend for SPLADE transformer (default: auto/SDPA).",
    )
    parser.add_argument("--qwen-dtype", type=str, default="bfloat16")
    parser.add_argument("--qwen-quant", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunk files whose vector outputs already exist with matching hash",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even when resume criteria are satisfied",
    )
    args = parser.parse_args()

    if args.batch_size_splade < 1 or args.batch_size_qwen < 1:
        raise ValueError("Batch sizes must be >= 1")

    overall_start = time.perf_counter()

    data_root_override = args.data_root
    resolved_root = (
        detect_data_root(data_root_override)
        if data_root_override is not None
        else DEFAULT_DATA_ROOT
    )

    if data_root_override is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved_root)

    data_manifests(resolved_root)

    if args.chunks_dir == DEFAULT_CHUNKS_DIR and data_root_override is not None:
        chunks_dir = data_chunks(resolved_root)
    else:
        chunks_dir = args.chunks_dir.resolve()

    if args.out_dir == DEFAULT_VECTORS_DIR and data_root_override is not None:
        out_dir = data_vectors(resolved_root)
    else:
        out_dir = args.out_dir.resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = out_dir

    logger.info(
        "Embedding configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_root),
                "chunks_dir": str(chunks_dir),
                "vectors_dir": str(out_dir),
            }
        },
    )

    files = iter_chunk_files(chunks_dir)
    if not files:
        logger.warning(
            "No chunk files found",
            extra={"extra_fields": {"chunks_dir": str(chunks_dir)}},
        )
        return

    if args.force:
        logger.info("Force mode: reprocessing all chunk files")
    elif args.resume:
        logger.info("Resume mode enabled: unchanged chunk files will be skipped")

    attn_impl = None if args.splade_attn == "auto" else args.splade_attn
    args.splade_cfg = SpladeCfg(
        batch_size=args.batch_size_splade,
        max_active_dims=args.splade_max_active_dims,
        attn_impl=attn_impl,
    )
    args.qwen_cfg = QwenCfg(
        dtype=args.qwen_dtype,
        tp=int(args.tp),
        batch_size=int(args.batch_size_qwen),
        quantization=args.qwen_quant,
    )

    uuid_to_chunk, stats = process_pass_a(files, logger)
    if not uuid_to_chunk:
        logger.warning("No chunks found after Pass A")
        return

    validator = SPLADEValidator()
    tracemalloc.start()
    pass_b_start = time.perf_counter()
    total_vectors = 0
    splade_nnz_all: List[int] = []
    qwen_norms_all: List[float] = []

    manifest_index = (
        load_manifest_index(MANIFEST_STAGE, resolved_root) if args.resume else {}
    )
    file_entries = []
    skipped_files = 0
    for chunk_file in files:
        doc_id = chunk_file.relative_to(chunks_dir).as_posix()
        out_path = args.out_dir / f"{chunk_file.stem.replace('.chunks', '')}.vectors.jsonl"
        input_hash = compute_content_hash(chunk_file)
        entry = manifest_index.get(doc_id)
        if (
            args.resume
            and not args.force
            and out_path.exists()
            and entry
            and entry.get("input_hash") == input_hash
        ):
            logger.info("Skipping %s: output exists and input unchanged", doc_id)
            manifest_append(
                stage=MANIFEST_STAGE,
                doc_id=doc_id,
                status="skip",
                duration_s=0.0,
                schema_version="embeddings/1.0.0",
                input_path=str(chunk_file),
                input_hash=input_hash,
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
                chunk_file, uuid_to_chunk, stats, args, validator, logger
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
            output_path=str(out_path),
            vector_count=count,
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
    avg_norm = statistics.mean(qwen_norms_all) if qwen_norms_all else 0.0
    std_norm = (
        statistics.pstdev(qwen_norms_all) if len(qwen_norms_all) > 1 else 0.0
    )

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


if __name__ == "__main__":
    main()
