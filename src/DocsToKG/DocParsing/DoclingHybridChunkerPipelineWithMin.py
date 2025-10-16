#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin",
#   "purpose": "CLI entry points for DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin workflows",
#   "sections": [
#     {
#       "id": "dedupe-preserve-order",
#       "name": "_dedupe_preserve_order",
#       "anchor": "function-dedupe-preserve-order",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-str-list",
#       "name": "_ensure_str_list",
#       "anchor": "function-ensure-str-list",
#       "kind": "function"
#     },
#     {
#       "id": "load-structural-marker-config",
#       "name": "_load_structural_marker_config",
#       "anchor": "function-load-structural-marker-config",
#       "kind": "function"
#     },
#     {
#       "id": "validate-chunk-files",
#       "name": "_validate_chunk_files",
#       "anchor": "function-validate-chunk-files",
#       "kind": "function"
#     },
#     {
#       "id": "read-utf8",
#       "name": "read_utf8",
#       "anchor": "function-read-utf8",
#       "kind": "function"
#     },
#     {
#       "id": "build-doc",
#       "name": "build_doc",
#       "anchor": "function-build-doc",
#       "kind": "function"
#     },
#     {
#       "id": "extract-refs-and-pages",
#       "name": "extract_refs_and_pages",
#       "anchor": "function-extract-refs-and-pages",
#       "kind": "function"
#     },
#     {
#       "id": "summarize-image-metadata",
#       "name": "summarize_image_metadata",
#       "anchor": "function-summarize-image-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "rec",
#       "name": "Rec",
#       "anchor": "class-rec",
#       "kind": "class"
#     },
#     {
#       "id": "chunkworkerconfig",
#       "name": "ChunkWorkerConfig",
#       "anchor": "class-chunkworkerconfig",
#       "kind": "class"
#     },
#     {
#       "id": "chunktask",
#       "name": "ChunkTask",
#       "anchor": "class-chunktask",
#       "kind": "class"
#     },
#     {
#       "id": "chunkresult",
#       "name": "ChunkResult",
#       "anchor": "class-chunkresult",
#       "kind": "class"
#     },
#     {
#       "id": "chunk-worker-initializer",
#       "name": "_chunk_worker_initializer",
#       "anchor": "function-chunk-worker-initializer",
#       "kind": "function"
#     },
#     {
#       "id": "process-chunk-task",
#       "name": "_process_chunk_task",
#       "anchor": "function-process-chunk-task",
#       "kind": "function"
#     },
#     {
#       "id": "merge-rec",
#       "name": "merge_rec",
#       "anchor": "function-merge-rec",
#       "kind": "function"
#     },
#     {
#       "id": "is-structural-boundary",
#       "name": "is_structural_boundary",
#       "anchor": "function-is-structural-boundary",
#       "kind": "function"
#     },
#     {
#       "id": "coalesce-small-runs",
#       "name": "coalesce_small_runs",
#       "anchor": "function-coalesce-small-runs",
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
Docling Hybrid Chunker with Minimum Token Coalescence

Transforms DocTags documents into chunked records with topic-aware coalescence.
The module exposes a CLI (`python -m DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin`)
and reusable helpers for other pipelines.

Key Features:
- Token-aware chunk merging that respects structural boundaries and image metadata.
- Shared CLI configuration via :func:`DocsToKG.DocParsing.pipelines.add_data_root_option`.
- Manifest logging that records chunk counts, parsing engines, and durations.

Dependencies:
- docling_core: Provides chunkers, serializers, and DocTags parsing.
- transformers: Supplies HuggingFace tokenizers.
- tqdm: Optional progress reporting when imported by callers.

Usage:
    python -m DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin \\
        --data-root /datasets/Data --min-tokens 256 --max-tokens 512

Tokenizer Alignment:
    The default tokenizer (``Qwen/Qwen3-Embedding-4B``) aligns with the dense
    embedder used by the embeddings pipeline. When experimenting with other
    tokenizers (for example, legacy BERT models), run the calibration utility
    beforehand to understand token count deltas::

        python scripts/calibrate_tokenizers.py --doctags-dir Data/DocTagsFiles

    The calibration script reports relative token ratios and recommends
    adjustments to ``--min-tokens`` so chunk sizes remain compatible with the
    embedding stage.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Third-party imports
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument, DocTagsDocument
from transformers import AutoTokenizer


# --- Globals ---

__all__ = (
    "Rec",
    "build_doc",
    "build_parser",
    "coalesce_small_runs",
    "compute_relative_doc_id",
    "extract_refs_and_pages",
    "is_structural_boundary",
    "main",
    "merge_rec",
    "parse_args",
    "read_utf8",
    "summarize_image_metadata",
)

from DocsToKG.DocParsing._common import (
    acquire_lock,
    atomic_write,
    compute_content_hash,
    compute_relative_doc_id,
    data_chunks,
    data_doctags,
    detect_data_root,
    get_logger,
    iter_doctags,
    load_manifest_index,
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
    resolve_hash_algorithm,
    should_skip_output,
)
from DocsToKG.DocParsing.pipelines import (
    add_data_root_option,
    add_resume_force_options,
    prepare_data_root,
    resolve_pipeline_path,
)
from DocsToKG.DocParsing.schemas import (
    CHUNK_SCHEMA_VERSION,
    ChunkRow,
    ProvenanceMetadata,
    get_docling_version,
    validate_chunk_row,
)
from DocsToKG.DocParsing.serializers import RichSerializerProvider

SOFT_BARRIER_MARGIN = 64
DEFAULT_HEADING_MARKERS: Tuple[str, ...] = ("#",)
DEFAULT_CAPTION_MARKERS: Tuple[str, ...] = (
    "Figure caption:",
    "Table:",
    "Picture description:",
    "<!-- image -->",
)


def _dedupe_preserve_order(markers: Sequence[str]) -> Tuple[str, ...]:
    """Return a tuple with duplicates removed while preserving original order."""

    seen: set[str] = set()
    ordered: List[str] = []
    for marker in markers:
        if marker not in seen:
            seen.add(marker)
            ordered.append(marker)
    return tuple(ordered)


def _ensure_str_list(value: object, label: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected a list of strings for '{label}'")
    return [item for item in value if item]


def _load_structural_marker_config(path: Path) -> Tuple[List[str], List[str]]:
    """Load custom heading and caption markers from a YAML or JSON file."""

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    data: object
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Loading YAML heading markers requires the 'PyYAML' package"
            ) from exc
        data = yaml.safe_load(raw) or {}
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ValueError(
                    f"Unable to parse heading markers file {path}: not valid JSON."
                ) from exc
            data = yaml.safe_load(raw) or {}

    if isinstance(data, list):
        headings = _ensure_str_list(data, "headings")
        captions: List[str] = []
    elif isinstance(data, dict):
        headings = _ensure_str_list(data.get("headings"), "headings")
        captions = _ensure_str_list(data.get("captions"), "captions")
    else:
        raise ValueError(
            f"Unsupported heading markers format in {path}: expected list or mapping."
        )

    return headings, captions


def _validate_chunk_files(files: Sequence[Path], logger) -> tuple[int, int]:
    """Validate chunk JSONL rows across supplied files."""

    total_files = 0
    total_rows = 0
    errors: List[str] = []

    for path in files:
        total_files += 1
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    validate_chunk_row(json.loads(line))
                except ValueError as exc:
                    message = f"{path}:{line_no}: {exc}"
                    logger.error("Chunk validation error: %s", message)
                    errors.append(message)

    if errors:
        preview = ", ".join(errors[:5])
        if len(errors) > 5:
            preview += ", ..."
        raise ValueError(
            "Chunk validation failed; fix the highlighted rows before continuing: " + preview
        )

    logger.info(
        "Chunk validation complete",
        extra={"extra_fields": {"files": total_files, "rows": total_rows}},
    )
    print(f"Validated {total_rows} rows across {total_files} chunk files.")
    return total_files, total_rows

# --- Defaults ---

DEFAULT_DATA_ROOT = detect_data_root()
DEFAULT_IN_DIR = data_doctags(DEFAULT_DATA_ROOT)
DEFAULT_OUT_DIR = data_chunks(DEFAULT_DATA_ROOT)
MANIFEST_STAGE = "chunks"


_LOGGER = get_logger(__name__)


# --- Public Functions ---

def read_utf8(p: Path) -> str:
    """Load text from disk using UTF-8 with replacement for invalid bytes.

    Args:
        p: Path to the text file.

    Returns:
        String contents of the file.
    """
    return p.read_text(encoding="utf-8", errors="replace")


def build_doc(doc_name: str, doctags_text: str) -> DoclingDocument:
    """Construct a Docling document from serialized DocTags text.

    Args:
        doc_name: Human-readable document identifier for logging.
        doctags_text: Serialized DocTags payload.

    Returns:
        Loaded DoclingDocument ready for chunking.
    """
    dt = DocTagsDocument.from_doctags_and_image_pairs([doctags_text], [None])
    return DoclingDocument.load_from_doctags(dt, document_name=doc_name)


def extract_refs_and_pages(chunk: BaseChunk) -> Tuple[List[str], List[int]]:
    """Collect self-references and page numbers associated with a chunk.

    Args:
        chunk: Chunk object produced by the hybrid chunker.

    Returns:
        Tuple containing a list of reference identifiers and sorted page numbers.

    Raises:
        None
    """
    refs, pages = [], set()
    try:
        for it in chunk.meta.doc_items:
            if getattr(it, "self_ref", None):
                refs.append(it.self_ref)
            for pv in getattr(it, "prov", []) or []:
                pn = getattr(pv, "page_no", None)
                if pn is not None:
                    pages.add(int(pn))
    except Exception:
        pass
    return refs, sorted(pages)


def summarize_image_metadata(chunk: BaseChunk, text: str) -> Tuple[bool, bool, int]:
    """Infer image annotation flags and counts from chunk metadata and text.

    Args:
        chunk: Chunk metadata object containing image annotations.
        text: Chunk text used to detect fallback caption cues.

    Returns:
        Tuple of ``(has_caption, has_classification, num_images)`` describing
        inferred image metadata.
    """

    has_caption = False
    has_classification = False
    num_images = 0

    try:
        doc_items = getattr(chunk.meta, "doc_items", []) or []
    except Exception:  # pragma: no cover - defensive catch
        doc_items = []

    for doc_item in doc_items:
        picture = getattr(doc_item, "doc_item", doc_item)
        flags = getattr(picture, "_docstokg_flags", None)
        if isinstance(flags, dict):
            has_caption = has_caption or bool(flags.get("has_image_captions"))
            has_classification = has_classification or bool(flags.get("has_image_classification"))
        if getattr(picture, "__class__", type(None)).__name__.lower().startswith("picture"):
            num_images += 1

    text_has_caption = any(
        marker in text for marker in ("Figure caption:", "Picture description:", "SMILES:")
    )
    text_has_classification = "Picture type:" in text
    has_caption = has_caption or text_has_caption
    has_classification = has_classification or text_has_classification

    if num_images == 0:
        num_images = (
            text.count("<!-- image -->")
            + text.count("Figure caption:")
            + text.count("Picture description:")
        )
    if num_images == 0 and (has_caption or has_classification):
        num_images = 1

    return has_caption, has_classification, num_images


# --- Public Classes ---


@dataclass(slots=True)
class Rec:
    """Intermediate record tracking chunk text and provenance.

    Attributes:
        text: Chunk text content.
        n_tok: Token count computed by the tokenizer.
        src_idxs: Source chunk indices contributing to this record.
        refs: List of inline reference identifiers.
        pages: Page numbers associated with the chunk.

    Examples:
        >>> rec = Rec(text="Example", n_tok=5, src_idxs=[0], refs=[], pages=[1])
        >>> rec.n_tok
        5
    """

    text: str
    n_tok: int
    src_idxs: List[int]
    refs: List[str]
    pages: List[int]
    has_image_captions: bool = False
    has_image_classification: bool = False
    num_images: int = 0


@dataclass(slots=True)
class ChunkWorkerConfig:
    """Configuration shared across worker processes performing chunking.

    Attributes:
        tokenizer_model: HuggingFace identifier used for tokenisation.
        min_tokens: Minimum tokens per chunk before spillover occurs.
        max_tokens: Hard ceiling on chunk size accepted by embedding models.
        soft_barrier_margin: Buffer applied when soft-limiting chunk size.
        heading_markers: Strings that designate heading boundaries.
        caption_markers: Strings that mark caption boundaries.
        docling_version: Docling version used to produce metadata.
    """

    tokenizer_model: str
    min_tokens: int
    max_tokens: int
    soft_barrier_margin: int
    heading_markers: Tuple[str, ...]
    caption_markers: Tuple[str, ...]
    docling_version: str


@dataclass(slots=True)
class ChunkTask:
    """Work item describing a single document chunking request.

    Attributes:
        doc_path: Source document that must be chunked.
        output_path: Destination path for generated chunk JSONL.
        doc_id: Unique identifier assigned to the document.
        doc_stem: Filesystem-friendly stem used for derived artefacts.
        input_hash: Content hash captured during ingestion.
        parse_engine: Docling engine used to produce structured content.
    """

    doc_path: Path
    output_path: Path
    doc_id: str
    doc_stem: str
    input_hash: str
    parse_engine: str


@dataclass(slots=True)
class ChunkResult:
    """Result returned by chunking workers to the orchestrator.

    Attributes:
        doc_id: Unique identifier assigned to the document.
        doc_stem: Filesystem-friendly stem used for derived artefacts.
        status: Worker outcome (``success``, ``skip``, or ``failure``).
        duration_s: Processing duration measured in seconds.
        input_path: Source document path.
        output_path: Chunk JSONL destination.
        input_hash: Content hash associated with the document.
        chunk_count: Number of chunks produced.
        parse_engine: Docling engine used to produce structured content.
        error: Optional failure description supplied on error.
    """

    doc_id: str
    doc_stem: str
    status: str
    duration_s: float
    input_path: Path
    output_path: Path
    input_hash: str
    chunk_count: int
    parse_engine: str
    error: Optional[str] = None


_CHUNK_WORKER_CONFIG: Optional[ChunkWorkerConfig] = None
_CHUNK_WORKER_TOKENIZER: Optional[HuggingFaceTokenizer] = None
_CHUNK_WORKER_CHUNKER: Optional[HybridChunker] = None


def _chunk_worker_initializer(cfg: ChunkWorkerConfig) -> None:
    global _CHUNK_WORKER_CONFIG, _CHUNK_WORKER_TOKENIZER, _CHUNK_WORKER_CHUNKER
    _CHUNK_WORKER_CONFIG = cfg
    hf = AutoTokenizer.from_pretrained(cfg.tokenizer_model, use_fast=True)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf, max_tokens=cfg.max_tokens)
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        serializer_provider=RichSerializerProvider(),
    )
    _CHUNK_WORKER_TOKENIZER = tokenizer
    _CHUNK_WORKER_CHUNKER = chunker


def _process_chunk_task(task: ChunkTask) -> ChunkResult:
    if _CHUNK_WORKER_CONFIG is None or _CHUNK_WORKER_TOKENIZER is None or _CHUNK_WORKER_CHUNKER is None:
        raise RuntimeError("Chunk worker not initialised")

    cfg = _CHUNK_WORKER_CONFIG
    tokenizer = _CHUNK_WORKER_TOKENIZER
    chunker = _CHUNK_WORKER_CHUNKER

    start = time.perf_counter()
    try:
        doctags_text = read_utf8(task.doc_path)
        doc = build_doc(doc_name=task.doc_stem, doctags_text=doctags_text)

        chunks = list(chunker.chunk(dl_doc=doc))
        recs: List[Rec] = []
        for idx, ch in enumerate(chunks):
            text = chunker.contextualize(ch)
            n_tok = tokenizer.count_tokens(text=text)
            refs, pages = extract_refs_and_pages(ch)
            has_caption, has_classification, num_images = summarize_image_metadata(ch, text)
            recs.append(
                Rec(
                    text=text,
                    n_tok=n_tok,
                    src_idxs=[idx],
                    refs=refs,
                    pages=pages,
                    has_image_captions=has_caption,
                    has_image_classification=has_classification,
                    num_images=num_images,
                )
            )

        final_recs = coalesce_small_runs(
            records=recs,
            tokenizer=tokenizer,
            min_tokens=cfg.min_tokens,
            max_tokens=cfg.max_tokens,
            soft_barrier_margin=cfg.soft_barrier_margin,
            heading_markers=cfg.heading_markers,
            caption_markers=cfg.caption_markers,
        )

        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        with acquire_lock(task.output_path):
            with atomic_write(task.output_path) as handle:
                for cid, r in enumerate(final_recs):
                    provenance = ProvenanceMetadata(
                        parse_engine=task.parse_engine,
                        docling_version=cfg.docling_version,
                        has_image_captions=r.has_image_captions,
                        has_image_classification=r.has_image_classification,
                        num_images=r.num_images,
                    )
                    row = ChunkRow(
                        doc_id=task.doc_id,
                        source_path=str(task.doc_path),
                        chunk_id=cid,
                        source_chunk_idxs=r.src_idxs,
                        num_tokens=r.n_tok,
                        text=r.text,
                        doc_items_refs=r.refs,
                        page_nos=r.pages,
                        schema_version=CHUNK_SCHEMA_VERSION,
                        has_image_captions=r.has_image_captions,
                        has_image_classification=r.has_image_classification,
                        num_images=r.num_images,
                        provenance=provenance,
                    )
                    payload = row.model_dump(mode="json", exclude_none=True)
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        duration = time.perf_counter() - start
        return ChunkResult(
            doc_id=task.doc_id,
            doc_stem=task.doc_stem,
            status="success",
            duration_s=duration,
            input_path=task.doc_path,
            output_path=task.output_path,
            input_hash=task.input_hash,
            chunk_count=len(final_recs),
            parse_engine=task.parse_engine,
        )
    except Exception as exc:  # pragma: no cover - propagated to main for handling
        duration = time.perf_counter() - start
        return ChunkResult(
            doc_id=task.doc_id,
            doc_stem=task.doc_stem,
            status="failure",
            duration_s=duration,
            input_path=task.doc_path,
            output_path=task.output_path,
            input_hash=task.input_hash,
            chunk_count=0,
            parse_engine=task.parse_engine,
            error=str(exc),
        )
def merge_rec(
    a: Rec,
    b: Rec,
    tokenizer: HuggingFaceTokenizer,
    *,
    recount: bool = True,
) -> Rec:
    """Merge two chunk records, updating token counts and provenance metadata.

    Args:
        a: First record to merge.
        b: Second record to merge.
        tokenizer: Tokenizer used to recompute token counts for combined text.
        recount: When ``True`` the merged text is re-tokenized; otherwise token
            counts are summed from inputs.

    Returns:
        New `Rec` instance containing fused text, token counts, and metadata.
    """
    text = a.text + "\n\n" + b.text
    n_tok = tokenizer.count_tokens(text=text) if recount else a.n_tok + b.n_tok
    refs = a.refs + [r for r in b.refs if r not in a.refs]
    pages = sorted(set(a.pages).union(b.pages))
    return Rec(
        text=text,
        n_tok=n_tok,
        src_idxs=a.src_idxs + b.src_idxs,
        refs=refs,
        pages=pages,
        has_image_captions=a.has_image_captions or b.has_image_captions,
        has_image_classification=a.has_image_classification or b.has_image_classification,
        num_images=a.num_images + b.num_images,
    )


# --- Topic-aware boundary detection ---

def is_structural_boundary(
    rec: Rec,
    heading_markers: Optional[Sequence[str]] = None,
    caption_markers: Optional[Sequence[str]] = None,
) -> bool:
    """Detect whether a chunk begins with a structural heading or caption marker.

    Args:
        rec: Chunk record to inspect.
        heading_markers: Optional prefixes treated as section headings.
        caption_markers: Optional prefixes treated as caption markers.

    Returns:
        ``True`` when ``rec.text`` starts with a heading indicator (``#``) or a
        recognised caption prefix, otherwise ``False``.

    Examples:
        >>> is_structural_boundary(Rec(text="# Introduction", n_tok=2, src_idxs=[], refs=[], pages=[]))
        True
        >>> is_structural_boundary(Rec(text="Regular paragraph", n_tok=2, src_idxs=[], refs=[], pages=[]))
        False
    """

    text = rec.text.lstrip()
    heading_prefixes = tuple(heading_markers) if heading_markers else DEFAULT_HEADING_MARKERS
    if any(text.startswith(marker) for marker in heading_prefixes):
        return True

    caption_prefixes = tuple(caption_markers) if caption_markers else DEFAULT_CAPTION_MARKERS
    return any(text.startswith(marker) for marker in caption_prefixes)


# --- Smart coalescence of SMALL-RUNS (< min_tokens) ---

def coalesce_small_runs(
    records: List[Rec],
    tokenizer: HuggingFaceTokenizer,
    min_tokens: int = 256,
    max_tokens: int = 512,
    soft_barrier_margin: int = SOFT_BARRIER_MARGIN,
    heading_markers: Optional[Sequence[str]] = None,
    caption_markers: Optional[Sequence[str]] = None,
) -> List[Rec]:
    """Merge contiguous short chunks until they satisfy minimum token thresholds.

    Args:
        records: Ordered list of chunk records to normalize.
        tokenizer: Tokenizer used to recompute token counts for merged chunks.
        min_tokens: Target minimum tokens per chunk after coalescing.
        max_tokens: Hard ceiling to avoid producing overly large chunks.
        soft_barrier_margin: Margin applied when respecting structural boundaries.
        heading_markers: Optional heading prefixes treated as structural boundaries.
        caption_markers: Optional caption prefixes treated as structural boundaries.

    Returns:
        New list of records where small runs are merged while preserving order.

    Note:
        Strategy:
            • Identify contiguous runs where every chunk has fewer than `min_tokens`.
            • Greedily pack neighbors within a run to exceed `min_tokens` without
              surpassing `max_tokens`.
            • Merge trailing fragments into adjacent groups when possible,
              preferring same-run neighbors to maintain topical cohesion.
            • Leave chunks outside small runs unchanged.
    """
    out: List[Rec] = []
    i, N = 0, len(records)
    margin = max(0, soft_barrier_margin)
    heading_prefixes = tuple(heading_markers) if heading_markers is not None else DEFAULT_HEADING_MARKERS
    caption_prefixes = (
        tuple(caption_markers) if caption_markers is not None else DEFAULT_CAPTION_MARKERS
    )

    def is_small(idx: int) -> bool:
        """Return True when the chunk at `idx` is below the minimum token threshold.

        Args:
            idx: Index of the chunk under evaluation.

        Returns:
            True if the chunk length is less than `min_tokens`, else False.
        """
        return records[idx].n_tok < min_tokens

    while i < N:
        if not is_small(i):
            out.append(records[i])
            i += 1
            continue

        # find small-run [s, e)
        s = i
        while i < N and is_small(i):
            i += 1
        e = i

        # pack within [s, e)
        groups: List[Rec] = []
        j = s
        while j < e:
            g = records[j]
            k = j + 1
            while k < e and g.n_tok < min_tokens:
                next_rec = records[k]
                combined_size = g.n_tok + next_rec.n_tok
                threshold = max_tokens - margin

                if (
                    is_structural_boundary(
                        next_rec, heading_markers=heading_prefixes, caption_markers=caption_prefixes
                    )
                    and combined_size > threshold
                ):
                    _LOGGER.debug(
                        "Soft barrier at chunk %s: boundary detected, combined size %s > %s",
                        k,
                        combined_size,
                        threshold,
                        extra={
                            "extra_fields": {
                                "combined_tokens": combined_size,
                                "threshold": threshold,
                                "max_tokens": max_tokens,
                                "context": "intra_run",
                            }
                        },
                    )
                    break

                if combined_size <= max_tokens:
                    g = merge_rec(g, next_rec, tokenizer, recount=False)
                    k += 1
                else:
                    break
            groups.append(g)
            j = k

        # handle tiny trailing group if present
        trailing_small = len(groups) >= 1 and groups[-1].n_tok < min_tokens
        # add all but possibly the last (we may re-route it)
        for g in groups[:-1] if trailing_small else groups:
            out.append(g)

        if trailing_small:
            tail = groups[-1]
            threshold = max_tokens - margin

            if len(groups) >= 2:
                combined_size = groups[-2].n_tok + tail.n_tok
                if (
                    is_structural_boundary(
                        tail, heading_markers=heading_prefixes, caption_markers=caption_prefixes
                    )
                    and combined_size > threshold
                ):
                    _LOGGER.debug(
                        "Soft barrier prevented intra-run merge: combined size %s > %s",
                        combined_size,
                        threshold,
                        extra={
                            "extra_fields": {
                                "combined_tokens": combined_size,
                                "threshold": threshold,
                                "max_tokens": max_tokens,
                                "context": "intra_run_tail",
                            }
                        },
                    )
                elif combined_size <= max_tokens:
                    merged = merge_rec(groups[-2], tail, tokenizer, recount=False)
                    if out and out[-1].src_idxs == groups[-2].src_idxs:
                        out[-1] = merged
                    else:
                        out.append(merged)
                    continue

            left_can = len(out) >= 1
            right_can = e < N
            left_ok = False
            right_ok = False

            if left_can:
                combined_size = out[-1].n_tok + tail.n_tok
                if (
                    is_structural_boundary(
                        tail, heading_markers=heading_prefixes, caption_markers=caption_prefixes
                    )
                    and combined_size > threshold
                ):
                    _LOGGER.debug(
                        "Soft barrier prevented left merge: combined size %s > %s",
                        combined_size,
                        threshold,
                        extra={
                            "extra_fields": {
                                "combined_tokens": combined_size,
                                "threshold": threshold,
                                "max_tokens": max_tokens,
                                "context": "left_neighbor",
                            }
                        },
                    )
                else:
                    left_ok = combined_size <= max_tokens

            if right_can:
                combined_size = records[e].n_tok + tail.n_tok
                if (
                    is_structural_boundary(
                        tail, heading_markers=heading_prefixes, caption_markers=caption_prefixes
                    )
                    and combined_size > threshold
                ):
                    _LOGGER.debug(
                        "Soft barrier prevented right merge: combined size %s > %s",
                        combined_size,
                        threshold,
                        extra={
                            "extra_fields": {
                                "combined_tokens": combined_size,
                                "threshold": threshold,
                                "max_tokens": max_tokens,
                                "context": "right_neighbor",
                            }
                        },
                    )
                else:
                    right_ok = combined_size <= max_tokens

            if left_ok and right_ok:
                if out[-1].n_tok <= records[e].n_tok:
                    out[-1] = merge_rec(out[-1], tail, tokenizer, recount=False)
                else:
                    records[e] = merge_rec(tail, records[e], tokenizer, recount=False)
            elif left_ok:
                out[-1] = merge_rec(out[-1], tail, tokenizer, recount=False)
            elif right_ok:
                records[e] = merge_rec(tail, records[e], tokenizer, recount=False)
            else:
                out.append(tail)

    for rec in out:
        rec.n_tok = tokenizer.count_tokens(text=rec.text)

    return out


# --- Main ---

def build_parser() -> argparse.ArgumentParser:
    """Construct an argument parser for the chunking pipeline.

    Args:
        None

    Returns:
        argparse.ArgumentParser: Parser configured with chunking options.

    Raises:
        None
    """

    parser = argparse.ArgumentParser()
    add_data_root_option(parser)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="HuggingFace tokenizer model (default aligns with dense embedder)",
    )
    parser.add_argument(
        "--soft-barrier-margin",
        type=int,
        default=SOFT_BARRIER_MARGIN,
        help="Token margin applied when respecting structural boundaries (default: 64).",
    )
    parser.add_argument(
        "--heading-markers",
        type=Path,
        default=None,
        help=(
            "Optional YAML/JSON file listing additional heading prefixes "
            "(and optionally caption prefixes) to treat as structural boundaries."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for chunking (default: 1).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate chunk files and exit without producing new outputs.",
    )
    add_resume_force_options(
        parser,
        resume_help="Skip DocTags whose chunk outputs already exist with matching hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone chunking execution.

    Args:
        argv (list[str] | None): Optional CLI argument vector. When ``None`` the
            process arguments are parsed.

    Returns:
        argparse.Namespace: Parsed CLI options.

    Raises:
        SystemExit: Propagated if ``argparse`` reports invalid arguments.
    """

    return build_parser().parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    """CLI driver that chunks DocTags files and enforces minimum token thresholds.

    Args:
        args (argparse.Namespace | None): Optional CLI namespace supplied during
            testing or orchestration.

    Returns:
        int: Exit code where ``0`` indicates success.
    """

    logger = get_logger(__name__)
    args = args if args is not None else build_parser().parse_args()

    data_root_override = args.data_root
    data_root_overridden = data_root_override is not None
    resolved_data_root = prepare_data_root(data_root_override, DEFAULT_DATA_ROOT)

    html_manifest_index = load_manifest_index("doctags-html", resolved_data_root)
    pdf_manifest_index = load_manifest_index("doctags-pdf", resolved_data_root)
    parse_engine_lookup = {
        doc_id: entry.get("parse_engine", "docling-html")
        for doc_id, entry in html_manifest_index.items()
    }
    parse_engine_lookup.update(
        {
            doc_id: entry.get("parse_engine", "docling-vlm")
            for doc_id, entry in pdf_manifest_index.items()
        }
    )
    docling_version = get_docling_version()

    in_dir = resolve_pipeline_path(
        cli_value=args.in_dir,
        default_path=DEFAULT_IN_DIR,
        resolved_data_root=resolved_data_root,
        data_root_overridden=data_root_overridden,
        resolver=data_doctags,
    )
    out_dir = resolve_pipeline_path(
        cli_value=args.out_dir,
        default_path=DEFAULT_OUT_DIR,
        resolved_data_root=resolved_data_root,
        data_root_overridden=data_root_overridden,
        resolver=data_chunks,
    )

    heading_markers: Tuple[str, ...] = DEFAULT_HEADING_MARKERS
    caption_markers: Tuple[str, ...] = DEFAULT_CAPTION_MARKERS
    custom_heading_markers: List[str] = []
    custom_caption_markers: List[str] = []
    if getattr(args, "heading_markers", None) is not None:
        markers_path = args.heading_markers.expanduser().resolve()
        if not markers_path.exists():
            raise FileNotFoundError(f"Heading markers file not found: {markers_path}")
        extra_headings, extra_captions = _load_structural_marker_config(markers_path)
        custom_heading_markers = extra_headings
        custom_caption_markers = extra_captions
        if extra_headings:
            heading_markers = _dedupe_preserve_order(
                (*heading_markers, *tuple(extra_headings))
            )
        if extra_captions:
            caption_markers = _dedupe_preserve_order(
                (*caption_markers, *tuple(extra_captions))
            )
        args.heading_markers = markers_path

    logger.info(
        "Chunking configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_data_root),
                "input_dir": str(in_dir),
                "output_dir": str(out_dir),
                "min_tokens": args.min_tokens,
                "max_tokens": args.max_tokens,
                "soft_barrier_margin": args.soft_barrier_margin,
                "custom_heading_markers": custom_heading_markers,
                "custom_caption_markers": custom_caption_markers,
                "workers": int(getattr(args, "workers", 1)),
            }
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_doctags(in_dir))
    if not files:
        logger.warning(
            "No .doctags files found",
            extra={"extra_fields": {"input_dir": str(in_dir)}},
        )
        return 0

    if args.force:
        logger.info("Force mode: reprocessing all DocTags files")
    elif args.resume:
        logger.info("Resume mode enabled: unchanged inputs will be skipped")

    chunk_manifest_index = (
        load_manifest_index(MANIFEST_STAGE, resolved_data_root) if args.resume else {}
    )

    tokenizer_model = args.tokenizer_model
    logger.info(
        "Loading tokenizer",
        extra={"extra_fields": {"tokenizer_model": tokenizer_model}},
    )

    chunk_manifest_index = (
        load_manifest_index(MANIFEST_STAGE, resolved_data_root) if args.resume else {}
    )

    if getattr(args, "validate_only", False):
        _validate_chunk_files(files, logger)
        return 0

    chunk_config = ChunkWorkerConfig(
        tokenizer_model=tokenizer_model,
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        soft_barrier_margin=int(args.soft_barrier_margin),
        heading_markers=heading_markers,
        caption_markers=caption_markers,
        docling_version=docling_version,
    )

    if "bert" in tokenizer_model.lower():
        logger.warning(
            "BERT tokenizer may not align with Qwen embedder. Consider running "
            "scripts/calibrate_tokenizers.py or using --tokenizer-model "
            "Qwen/Qwen3-Embedding-4B.",
            extra={"extra_fields": {"tokenizer_model": tokenizer_model}},
        )

    worker_count = max(1, int(getattr(args, "workers", 1)))

    tasks: List[ChunkTask] = []
    for path in files:
        rel_id = compute_relative_doc_id(path, in_dir)
        name = path.stem
        relative_target = Path(rel_id)
        out_path = (out_dir / relative_target).with_suffix(".chunks.jsonl")
        input_hash = compute_content_hash(path)
        manifest_entry = chunk_manifest_index.get(rel_id)
        parse_engine = parse_engine_lookup.get(rel_id, "docling-html")
        if rel_id not in parse_engine_lookup:
            logger.debug(
                "Parse engine defaulted to docling-html",
                extra={"extra_fields": {"doc_id": rel_id}},
            )

        if should_skip_output(out_path, manifest_entry, input_hash, args.resume, args.force):
            manifest_log_skip(
                stage=MANIFEST_STAGE,
                doc_id=rel_id,
                input_path=path,
                input_hash=input_hash,
                output_path=out_path,
                schema_version=CHUNK_SCHEMA_VERSION,
                parse_engine=parse_engine,
            )
            continue

        tasks.append(
            ChunkTask(
                doc_path=path,
                output_path=out_path,
                doc_id=rel_id,
                doc_stem=name,
                input_hash=input_hash,
                parse_engine=parse_engine,
            )
        )

    if not tasks:
        return 0

    def handle_result(result: ChunkResult) -> None:
        """Persist manifest information and raise on worker failure.

        Args:
            result: Structured outcome emitted by the chunking worker.
        """
        duration = round(result.duration_s, 3)
        if result.status != "success":
            manifest_log_failure(
                stage=MANIFEST_STAGE,
                doc_id=result.doc_id,
                duration_s=duration,
                schema_version=CHUNK_SCHEMA_VERSION,
                input_path=result.input_path,
                input_hash=result.input_hash,
                output_path=result.output_path,
                error=result.error or "unknown error",
                parse_engine=result.parse_engine,
            )
            raise RuntimeError(
                f"Chunking failed for {result.doc_id}: {result.error or 'unknown error'}"
            )

        logger.info(
            "Chunk file written",
            extra={
                "extra_fields": {
                    "doc_id": result.doc_id,
                    "doc_stem": result.doc_stem,
                    "chunks": result.chunk_count,
                    "output_file": result.output_path.name,
                }
            },
        )
        manifest_log_success(
            stage=MANIFEST_STAGE,
            doc_id=result.doc_id,
            duration_s=duration,
            schema_version=CHUNK_SCHEMA_VERSION,
            input_path=result.input_path,
            input_hash=result.input_hash,
            output_path=result.output_path,
            chunk_count=result.chunk_count,
            parse_engine=result.parse_engine,
        )

    if worker_count == 1:
        _chunk_worker_initializer(chunk_config)
        for task in tasks:
            handle_result(_process_chunk_task(task))
    else:
        logger.info(
            "Parallel chunking enabled",
            extra={"extra_fields": {"workers": worker_count}},
        )
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_chunk_worker_initializer,
            initargs=(chunk_config,),
        ) as pool:
            for result in pool.map(_process_chunk_task, tasks):
                handle_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
