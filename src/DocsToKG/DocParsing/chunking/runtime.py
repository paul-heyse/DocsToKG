#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.chunking.runtime",
#   "purpose": "CLI entry points for DocsToKG.DocParsing.chunking workflows",
#   "sections": [
#     {
#       "id": "rec",
#       "name": "Rec",
#       "anchor": "class-rec",
#       "kind": "class"
#     },
#     {
#       "id": "read-utf8",
#       "name": "read_utf8",
#       "anchor": "function-read-utf8",
#       "kind": "function"
#     },
#     {
#       "id": "hash-doctags-text",
#       "name": "_hash_doctags_text",
#       "anchor": "function-hash-doctags-text",
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
#       "id": "is-structural-boundary",
#       "name": "is_structural_boundary",
#       "anchor": "function-is-structural-boundary",
#       "kind": "function"
#     },
#     {
#       "id": "summarize-image-metadata",
#       "name": "summarize_image_metadata",
#       "anchor": "function-summarize-image-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "extract-chunk-start",
#       "name": "_extract_chunk_start",
#       "anchor": "function-extract-chunk-start",
#       "kind": "function"
#     },
#     {
#       "id": "merge-rec",
#       "name": "merge_rec",
#       "anchor": "function-merge-rec",
#       "kind": "function"
#     },
#     {
#       "id": "coalesce-small-runs",
#       "name": "coalesce_small_runs",
#       "anchor": "function-coalesce-small-runs",
#       "kind": "function"
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
#       "id": "process-indexed-chunk-task",
#       "name": "_process_indexed_chunk_task",
#       "anchor": "function-process-indexed-chunk-task",
#       "kind": "function"
#     },
#     {
#       "id": "ordered-results",
#       "name": "_ordered_results",
#       "anchor": "function-ordered-results",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-serializer-provider",
#       "name": "_resolve_serializer_provider",
#       "anchor": "function-resolve-serializer-provider",
#       "kind": "function"
#     },
#     {
#       "id": "inspect-parquet-chunk-file",
#       "name": "_inspect_parquet_chunk_file",
#       "anchor": "function-inspect-parquet-chunk-file",
#       "kind": "function"
#     },
#     {
#       "id": "normalise-validation-targets",
#       "name": "_normalise_validation_targets",
#       "anchor": "function-normalise-validation-targets",
#       "kind": "function"
#     },
#     {
#       "id": "validate-chunk-files",
#       "name": "_validate_chunk_files",
#       "anchor": "function-validate-chunk-files",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-parquet-chunk-artifact",
#       "name": "_resolve_parquet_chunk_artifact",
#       "anchor": "function-resolve-parquet-chunk-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "collect-chunk-artifacts",
#       "name": "_collect_chunk_artifacts",
#       "anchor": "function-collect-chunk-artifacts",
#       "kind": "function"
#     },
#     {
#       "id": "compute-worker-cfg-hash",
#       "name": "_compute_worker_cfg_hash",
#       "anchor": "function-compute-worker-cfg-hash",
#       "kind": "function"
#     },
#     {
#       "id": "write-fingerprint-for-output",
#       "name": "_write_fingerprint_for_output",
#       "anchor": "function-write-fingerprint-for-output",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-worker-initialised",
#       "name": "_ensure_worker_initialised",
#       "anchor": "function-ensure-worker-initialised",
#       "kind": "function"
#     },
#     {
#       "id": "build-chunk-plan",
#       "name": "_build_chunk_plan",
#       "anchor": "function-build-chunk-plan",
#       "kind": "function"
#     },
#     {
#       "id": "chunk-stage-worker",
#       "name": "_chunk_stage_worker",
#       "anchor": "function-chunk-stage-worker",
#       "kind": "function"
#     },
#     {
#       "id": "make-chunk-stage-hooks",
#       "name": "_make_chunk_stage_hooks",
#       "anchor": "function-make-chunk-stage-hooks",
#       "kind": "function"
#     },
#     {
#       "id": "main-inner",
#       "name": "_main_inner",
#       "anchor": "function-main-inner",
#       "kind": "function"
#     },
#     {
#       "id": "run-validate-only",
#       "name": "_run_validate_only",
#       "anchor": "function-run-validate-only",
#       "kind": "function"
#     },
#     {
#       "id": "write-chunks-atomic",
#       "name": "_write_chunks_atomic",
#       "anchor": "function-write-chunks-atomic",
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
The module exposes a CLI (``python -m DocsToKG.DocParsing.core chunk``)
and reusable helpers for other pipelines.

Key Features:
- Token-aware chunk merging that respects structural boundaries and image metadata.
- Shared CLI configuration via :func:`DocsToKG.DocParsing.doctags.add_data_root_option`.
- Manifest logging that records chunk counts, parsing engines, and durations.
- Atomic writes using JsonlWriter for concurrent-safe JSONL appending.
- Deterministic chunk IDs based on content hash for idempotent processing.

Concurrency & Durability:
- Chunk JSONL written atomically via JsonlWriter for concurrent-safe appending.
- Manifest entries written atomically to prevent corruption under parallel loads.
- Process-safe locking ensures reliable multi-worker pipelines.

Dependencies:
- docling_core: Provides chunkers, serializers, and DocTags parsing.
- transformers: Supplies HuggingFace tokenizers.
- tqdm: Optional progress reporting when imported by callers.

Usage:
    python -m DocsToKG.DocParsing.core chunk \\
        --data-root /datasets/Data --min-tokens 256 --max-tokens 512

Tokenizer Alignment:
    The default tokenizer (``DEFAULT_TOKENIZER`` from :mod:`DocsToKG.DocParsing.core`) aligns with the dense
    embedder used by the embeddings pipeline. When experimenting with other
    tokenizers (for example, legacy BERT models), run the calibration utility
    beforehand to understand token count deltas::

        python -m DocsToKG.DocParsing.token_profiles --doctags-dir Data/DocTagsFiles

    The calibration script reports relative token ratios and recommends
    adjustments to ``--min-tokens`` so chunk sizes remain compatible with the
    embedding stage.
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
import importlib
import json
import logging
import statistics
import time
import unicodedata
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq

# Third-party imports
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument, DocTagsDocument
from transformers import AutoTokenizer

# --- Globals ---

__all__ = (
    "Rec",
    "ChunkerCfg",
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

from DocsToKG.DocParsing.cli_errors import ChunkingCLIValidationError, format_cli_error
from DocsToKG.DocParsing.config import annotate_cli_overrides, parse_args_with_overrides
from DocsToKG.DocParsing.context import ParsingContext
from DocsToKG.DocParsing.core import (
    DEFAULT_CAPTION_MARKERS,
    DEFAULT_HEADING_MARKERS,
    DEFAULT_SERIALIZER_PROVIDER,
    DEFAULT_TOKENIZER,
    ChunkResult,
    ChunkTask,
    ChunkWorkerConfig,
    ItemFingerprint,
    ItemOutcome,
    StageContext,
    StageError,
    StageHooks,
    StageOptions,
    StageOutcome,
    StagePlan,
    WorkItem,
    compute_relative_doc_id,
    compute_stable_shard,
    dedupe_preserve_order,
    derive_doc_id_and_chunks_path,
    load_structural_marker_config,
    run_stage,
    set_spawn_or_warn,
)
from DocsToKG.DocParsing.env import (
    data_chunks,
    data_doctags,
    detect_data_root,
    ensure_model_environment,
    prepare_data_root,
    resolve_pipeline_path,
)
from DocsToKG.DocParsing.formats import (
    CHUNK_SCHEMA_VERSION,
    ChunkRow,
    ProvenanceMetadata,
    get_docling_version,
    validate_chunk_row,
)
from DocsToKG.DocParsing.interfaces import ChunkingSerializerProvider
from DocsToKG.DocParsing.io import (
    atomic_write,
    compute_chunk_uuid,
    compute_content_hash,
    iter_doctags,
    load_manifest_index,
    make_hasher,
    quarantine_artifact,
    relative_path,
    resolve_attempts_path,
    resolve_hash_algorithm,
    resolve_manifest_path,
)
from DocsToKG.DocParsing.logging import (
    get_logger,
    log_event,
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
    telemetry_scope,
)
from DocsToKG.DocParsing.storage import parquet_schemas
from DocsToKG.DocParsing.storage import paths as storage_paths
from DocsToKG.DocParsing.storage.chunks_writer import ParquetChunksWriter
from DocsToKG.DocParsing.telemetry import StageTelemetry, TelemetrySink

from .cli import build_parser, parse_args
from .config import CHUNK_PROFILE_PRESETS, ChunkerCfg

CHUNK_STAGE = "chunking"

_LOGGER = get_logger(__name__)


@dataclass(slots=True)
class Rec:
    """Chunk record used by coalescence helpers during validation."""

    text: str
    n_tok: int
    src_idxs: list[int]
    refs: list[str]
    pages: list[int]
    start_offset: int | None = None
    has_image_captions: bool = False
    has_image_classification: bool = False
    num_images: int = 0
    image_confidence: float | None = None
    picture_meta: list[dict[str, Any]] | None = None


def read_utf8(path: Path) -> str:
    """Load UTF-8 text from ``path`` replacing undecodable bytes."""

    return Path(path).read_text(encoding="utf-8", errors="replace")


def _hash_doctags_text(text: str) -> str:
    """Return a normalised content hash for DocTags ``text``."""

    algorithm = resolve_hash_algorithm()
    hasher = make_hasher(name=algorithm)
    if text:
        normalised = unicodedata.normalize("NFKC", text)
        if normalised:
            hasher.update(normalised.encode("utf-8"))
    return hasher.hexdigest()


def build_doc(doc_name: str, doctags_text: str) -> DoclingDocument:
    """Construct a Docling document from serialized DocTags markup."""

    tags = DocTagsDocument.from_doctags_and_image_pairs([doctags_text], images=None)
    return DoclingDocument.load_from_doctags(tags, document_name=doc_name)


def extract_refs_and_pages(chunk: BaseChunk) -> tuple[list[str], list[int]]:
    """Extract inline references and page numbers from a Docling chunk."""

    refs: list[str] = []
    pages: list[int] = []
    doc_id = getattr(getattr(chunk, "meta", None), "document_id", "__unknown__")

    for item in getattr(getattr(chunk, "meta", None), "doc_items", []) or []:
        if getattr(item, "self_ref", None):
            if item.self_ref not in refs:
                refs.append(item.self_ref)
        for prov in getattr(item, "prov", []) or []:
            page_value = getattr(prov, "page_no", None)
            if page_value is None:
                continue
            try:
                page_int = int(page_value)
            except (TypeError, ValueError):
                log_event(
                    _LOGGER.logger,
                    "warning",
                    "Invalid page metadata encountered",
                    stage=CHUNK_STAGE,
                    doc_id=doc_id,
                    error_code="CHUNK_PAGE_INVALID",
                    page_value=page_value,
                )
                continue
            if page_int > 0 and page_int not in pages:
                pages.append(page_int)
    pages.sort()
    return refs, pages


def is_structural_boundary(
    record: Rec,
    heading_markers: tuple[str, ...] = DEFAULT_HEADING_MARKERS,
    caption_markers: tuple[str, ...] = DEFAULT_CAPTION_MARKERS,
) -> bool:
    """Return True when ``record`` begins with a structural marker."""

    text = record.text.lstrip()
    for marker in heading_markers:
        if marker and text.startswith(marker):
            return True
    for marker in caption_markers:
        if marker and text.startswith(marker):
            return True
    return False


def summarize_image_metadata(
    chunk: BaseChunk, text: str
) -> tuple[bool, bool, int, float | None, list[dict[str, Any]]]:
    """Summarise image annotations associated with ``chunk``."""

    has_caption = any(
        marker and marker in text
        for marker in ("Figure caption:", "Table:", "Picture description:")
    ) or text.strip().startswith("<!-- image -->")
    has_classification = False
    num_images = 0
    confidence: float | None = None
    extras: list[dict[str, Any]] = []
    doc_id = getattr(getattr(chunk, "meta", None), "document_id", "__unknown__")

    for entry in getattr(getattr(chunk, "meta", None), "doc_items", []) or []:
        doc_item = getattr(entry, "doc_item", entry)
        flags = getattr(doc_item, "_docstokg_flags", None) or {}
        if flags:
            num_images += 1
            if flags.get("has_image_captions"):
                has_caption = True
            if flags.get("has_image_classification"):
                has_classification = True
            if flags.get("image_confidence") is not None:
                try:
                    score = float(flags["image_confidence"])
                    confidence = score if confidence is None else max(confidence, score)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    pass

        annotations = getattr(doc_item, "annotations", getattr(entry, "annotations", None))
        if annotations is None:
            continue
        if not isinstance(annotations, (list, tuple)):
            log_event(
                _LOGGER.logger,
                "warning",
                "Chunk annotations not iterable",
                stage=CHUNK_STAGE,
                doc_id=doc_id,
                error_code="CHUNK_ANNOTATIONS_NON_ITERABLE",
            )
            continue
        for ann in annotations:
            predicted = getattr(ann, "predicted_classes", None)
            if predicted is None:
                continue
            if not isinstance(predicted, (list, tuple)):
                log_event(
                    _LOGGER.logger,
                    "warning",
                    "Predicted classes not iterable",
                    stage=CHUNK_STAGE,
                    doc_id=doc_id,
                    error_code="CHUNK_PREDICTED_CLASSES_NON_ITERABLE",
                )
                continue
            has_classification = True
            for cls in predicted:
                label = getattr(cls, "class_name", None)
                conf = getattr(cls, "confidence", None)
                if conf is not None:
                    try:
                        score = float(conf)
                        confidence = score if confidence is None else max(confidence, score)
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        pass
                if label:
                    extras.append({"class": label, "confidence": conf})

    return has_caption, has_classification, num_images, confidence, extras


def _extract_chunk_start(chunk: BaseChunk) -> int | None:
    """Attempt to extract the starting character offset for ``chunk``."""

    try:
        doc_items = getattr(getattr(chunk, "meta", None), "doc_items", None) or []
        if not doc_items:
            return None
        first = doc_items[0]
        for attr in ("start_offset", "offset", "char_start"):
            value = getattr(first, attr, None)
            if isinstance(value, int) and value >= 0:
                return value
        char_range = getattr(first, "char_range", None)
        if isinstance(char_range, (tuple, list)) and char_range:
            maybe = char_range[0]
            if isinstance(maybe, int) and maybe >= 0:
                return maybe
        start = getattr(getattr(first, "meta", None), "char_start", None)
        if isinstance(start, int) and start >= 0:
            return start
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def merge_rec(a: Rec, b: Rec, tokenizer: Any) -> Rec:
    """Merge two chunk records into a single aggregate record."""

    text = (a.text.rstrip() + "\n\n" + b.text.lstrip()).strip("\n")
    token_count = tokenizer.count_tokens(text=text)
    refs = list(dict.fromkeys(a.refs + [ref for ref in b.refs if ref not in a.refs]))
    pages = sorted(set(a.pages + b.pages))
    start_offset_candidates = [
        offset for offset in (a.start_offset, b.start_offset) if isinstance(offset, int)
    ]
    start_offset = min(start_offset_candidates) if start_offset_candidates else None

    picture_meta: list[dict[str, Any]] = []
    if a.picture_meta:
        picture_meta.extend(a.picture_meta)
    if b.picture_meta:
        picture_meta.extend(b.picture_meta)

    confidences = [
        score
        for score in (a.image_confidence, b.image_confidence)
        if isinstance(score, (int, float))
    ]
    image_confidence = max(confidences) if confidences else None

    return Rec(
        text=text,
        n_tok=token_count,
        src_idxs=a.src_idxs + b.src_idxs,
        refs=refs,
        pages=pages,
        start_offset=start_offset,
        has_image_captions=a.has_image_captions or b.has_image_captions,
        has_image_classification=a.has_image_classification or b.has_image_classification,
        num_images=a.num_images + b.num_images,
        image_confidence=image_confidence,
        picture_meta=picture_meta or None,
    )


def coalesce_small_runs(
    records: list[Rec],
    tokenizer: Any,
    *,
    min_tokens: int,
    max_tokens: int | None = None,
    soft_barrier_margin: int = 64,
    heading_markers: tuple[str, ...] = DEFAULT_HEADING_MARKERS,
    caption_markers: tuple[str, ...] = DEFAULT_CAPTION_MARKERS,
) -> list[Rec]:
    """Merge contiguous undersized chunks while respecting structural boundaries."""

    if not records:
        return []

    max_threshold = max_tokens if max_tokens is not None else 512

    output: list[Rec] = []
    i = 0
    total = len(records)
    while i < total:
        current = records[i]
        if current.n_tok >= min_tokens:
            output.append(current)
            i += 1
            continue

        merged = current
        j = i + 1
        while j < total:
            candidate = records[j]
            boundary = is_structural_boundary(candidate, heading_markers, caption_markers)
            if boundary and soft_barrier_margin > 0:
                break
            tentative = merge_rec(merged, candidate, tokenizer)
            if tentative.n_tok > max_threshold:
                break
            merged = tentative
            j += 1
            if merged.n_tok >= min_tokens:
                break

        output.append(merged)
        if j == i + 1 and merged is current:
            i += 1
        else:
            i = j

    return output


_WORKER_STATE: dict[str, Any] = {}


def _chunk_worker_initializer(cfg: ChunkWorkerConfig) -> None:
    """Initialise shared tokenizer/chunker state for worker processes."""

    tokenizer_cls = HuggingFaceTokenizer
    auto_tokenizer = AutoTokenizer
    tokenizer = tokenizer_cls(
        tokenizer=auto_tokenizer.from_pretrained(cfg.tokenizer_model, use_fast=True),
        max_tokens=cfg.max_tokens,
    )
    provider_cls = _resolve_serializer_provider(cfg.serializer_provider_spec)
    provider = provider_cls()
    chunker_cls = HybridChunker
    chunker = chunker_cls(
        tokenizer=tokenizer,
        merge_peers=True,
        serializer_provider=provider,
    )
    global _WORKER_STATE
    _WORKER_STATE = {
        "config": cfg,
        "tokenizer": tokenizer,
        "chunker": chunker,
        "heading_markers": tuple(cfg.heading_markers),
        "caption_markers": tuple(cfg.caption_markers),
        "config_hash": _compute_worker_cfg_hash(cfg),
    }


def _process_chunk_task(task: ChunkTask) -> ChunkResult:
    """Chunk a single DocTags file using worker-local state."""

    if not _WORKER_STATE:
        raise RuntimeError("Chunk worker initialiser was not executed")

    cfg: ChunkWorkerConfig = _WORKER_STATE["config"]
    tokenizer: HuggingFaceTokenizer = _WORKER_STATE["tokenizer"]
    chunker: HybridChunker = _WORKER_STATE["chunker"]
    heading_markers: tuple[str, ...] = _WORKER_STATE["heading_markers"]
    caption_markers: tuple[str, ...] = _WORKER_STATE["caption_markers"]

    start_time = time.perf_counter()
    chunk_count = 0
    total_tokens = 0
    try:
        text = read_utf8(task.doc_path)
        input_hash = task.input_hash or _hash_doctags_text(text)
        doc = build_doc(task.doc_stem, text)
        raw_chunks = list(chunker.chunk(dl_doc=doc))

        records: list[Rec] = []
        for idx, chunk in enumerate(raw_chunks):
            chunk_text = chunker.contextualize(chunk)
            token_count = tokenizer.count_tokens(text=chunk_text)
            refs, pages = extract_refs_and_pages(chunk)
            has_caption, has_classification, num_images, confidence, picture_meta = (
                summarize_image_metadata(chunk, chunk_text)
            )
            records.append(
                Rec(
                    text=chunk_text,
                    n_tok=token_count,
                    src_idxs=[idx],
                    refs=refs,
                    pages=pages or [],
                    start_offset=_extract_chunk_start(chunk),
                    has_image_captions=has_caption,
                    has_image_classification=has_classification,
                    num_images=num_images,
                    image_confidence=confidence,
                    picture_meta=picture_meta or None,
                )
            )

        merged = coalesce_small_runs(
            records=records,
            tokenizer=tokenizer,
            min_tokens=cfg.min_tokens,
            max_tokens=cfg.max_tokens,
            soft_barrier_margin=cfg.soft_barrier_margin,
            heading_markers=heading_markers,
            caption_markers=caption_markers,
        )

        planned_output_path = task.output_path
        planned_output_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_count = len(merged)
        total_tokens = sum(rec.n_tok for rec in merged)

        artifact_paths: list[Path] = []
        parquet_bytes: int | None = None
        row_group_count: int | None = None
        rows_written: int | None = None

        # Collect chunk rows (shared for both formats)
        chunk_rows = []
        for chunk_id, rec in enumerate(merged):
            text_body = rec.text
            if cfg.inject_anchors:
                anchor = f"<<chunk:{task.doc_id}:{chunk_id}>>"
                if not text_body.startswith(anchor):
                    text_body = f"{anchor}\n{text_body}"

            provenance = ProvenanceMetadata(
                parse_engine=task.parse_engine,
                docling_version=cfg.docling_version,
                has_image_captions=rec.has_image_captions,
                has_image_classification=rec.has_image_classification,
                num_images=rec.num_images,
                image_confidence=rec.image_confidence,
            )

            row = ChunkRow(
                doc_id=task.doc_id,
                source_path=relative_path(task.doc_path, cfg.data_root),
                chunk_id=chunk_id,
                source_chunk_idxs=rec.src_idxs,
                num_tokens=rec.n_tok,
                text=text_body,
                doc_items_refs=rec.refs,
                page_nos=rec.pages,
                schema_version=CHUNK_SCHEMA_VERSION,
                start_offset=rec.start_offset,
                has_image_captions=rec.has_image_captions,
                has_image_classification=rec.has_image_classification,
                num_images=rec.num_images,
                image_confidence=rec.image_confidence,
                provenance=provenance,
                uuid=compute_chunk_uuid(task.doc_id, rec.start_offset or 0, text_body),
            ).model_dump(mode="python")

            validate_chunk_row(row)
            chunk_rows.append(row)

        # Route by format (Parquet default, JSONL legacy)
        if cfg.format == "parquet":
            try:
                # Use ParquetChunksWriter for Parquet output
                from DocsToKG.DocParsing.storage.chunks_writer import ParquetChunksWriter

                writer = ParquetChunksWriter()
                rel_id = storage_paths.normalize_rel_id(planned_output_path.stem)
                # Compute cfg_hash for this worker config
                cfg_hash = _compute_worker_cfg_hash(cfg)
                write_result = writer.write(
                    chunk_rows,
                    data_root=cfg.data_root,
                    rel_id=rel_id,
                    cfg_hash=cfg_hash,
                    created_by="DocsToKG-DocParsing",
                )
                artifact_paths = list(write_result.paths)
                if write_result.paths:
                    primary_path = write_result.paths[0]
                else:
                    primary_path = planned_output_path
                parquet_bytes = write_result.parquet_bytes
                row_group_count = write_result.row_group_count
                rows_written = write_result.rows_written
            except Exception as exc:
                # Fallback to JSONL on Parquet write failure
                _LOGGER.warning(
                    f"Parquet write failed for {task.doc_id}, falling back to JSONL: {exc}"
                )
                primary_path = planned_output_path
                with atomic_write(primary_path) as handle:
                    for row in chunk_rows:
                        handle.write(json.dumps(row, ensure_ascii=False))
                        handle.write("\n")
                artifact_paths = [primary_path]
                rows_written = len(chunk_rows)
        else:
            # Legacy JSONL format
            primary_path = planned_output_path
            with atomic_write(primary_path) as handle:
                for row in chunk_rows:
                    handle.write(json.dumps(row, ensure_ascii=False))
                    handle.write("\n")
            artifact_paths = [primary_path]
            rows_written = len(chunk_rows)

        duration = time.perf_counter() - start_time
        return ChunkResult(
            doc_id=task.doc_id,
            doc_stem=task.doc_stem,
            status="success",
            duration_s=duration,
            input_path=task.doc_path,
            output_path=primary_path,
            input_hash=input_hash,
            chunk_count=chunk_count,
            total_tokens=total_tokens,
            parse_engine=task.parse_engine,
            sanitizer_profile=task.sanitizer_profile,
            anchors_injected=cfg.inject_anchors,
            artifact_paths=tuple(artifact_paths),
            parquet_bytes=parquet_bytes,
            row_group_count=row_group_count,
            rows_written=rows_written,
        )
    except Exception as exc:  # pragma: no cover - exercised in failure paths
        duration = time.perf_counter() - start_time
        input_hash = locals().get("input_hash", task.input_hash)
        return ChunkResult(
            doc_id=task.doc_id,
            doc_stem=task.doc_stem,
            status="error",
            duration_s=duration,
            input_path=task.doc_path,
            output_path=planned_output_path,
            input_hash=input_hash,
            chunk_count=0,
            total_tokens=0,
            parse_engine=task.parse_engine,
            sanitizer_profile=task.sanitizer_profile,
            anchors_injected=cfg.inject_anchors,
            error=str(exc),
        )


def _process_indexed_chunk_task(payload: tuple[int, ChunkTask]) -> tuple[int, ChunkResult]:
    """Execute ``_process_chunk_task`` and preserve submission ordering."""

    index, task = payload
    return index, _process_chunk_task(task)


def _ordered_results(results: Iterable[tuple[int, ChunkResult]]) -> Iterator[ChunkResult]:
    """Yield chunk results in their original submission order."""

    pending: dict[int, ChunkResult] = {}
    next_index = 0
    for index, result in results:
        pending[index] = result
        while next_index in pending:
            yield pending.pop(next_index)
            next_index += 1


def _resolve_serializer_provider(spec: str) -> type[ChunkingSerializerProvider]:
    """Return the serializer provider class referenced by ``spec``."""

    if ":" not in spec:
        raise ChunkingCLIValidationError(
            option="--serializer-provider",
            message=f"expected 'module:Class' but received {spec!r}",
        )
    module_name, class_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Unable to import serializer provider module '{module_name}'") from exc
    try:
        provider_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Serializer provider '{class_name}' not found in module '{module_name}'"
        ) from exc
    if not isinstance(provider_cls, type):
        raise TypeError(f"{spec!r} did not resolve to a class")
    if not issubclass(provider_cls, ChunkingSerializerProvider):  # type: ignore[arg-type]
        raise TypeError(f"{spec!r} is not a ChunkingSerializerProvider")
    return provider_cls  # type: ignore[return-value]


def _inspect_parquet_chunk_file(path: Path) -> tuple[int, int, tuple[str, ...]]:
    """Return metadata for a Parquet chunk artifact or raise ``ValueError``."""

    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:  # pragma: no cover - exercised during error handling
        raise ValueError(f"Unable to open Parquet file: {exc}") from exc

    schema = parquet_file.schema_arrow
    expected = parquet_schemas.chunks_schema(include_optional=False)
    for field in expected:
        try:
            actual = schema.field(field.name)
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Missing required column: {field.name}") from exc
        if actual.type != field.type:
            raise ValueError(f"Column {field.name} has type {actual.type}, expected {field.type}")

    footer = parquet_schemas.validate_parquet_file(str(path))
    if not footer.ok:
        detail = "; ".join(footer.errors) or "invalid footer metadata"
        raise ValueError(detail)

    row_count = int(parquet_file.metadata.num_rows) if parquet_file.metadata else 0
    row_groups = int(parquet_file.metadata.num_row_groups) if parquet_file.metadata else 0
    return row_count, row_groups, footer.warnings


def _normalise_validation_targets(
    targets: Sequence[Path | tuple[str, Path]],
) -> list[tuple[str, Path]]:
    """Coerce validation targets into ``(doc_id, Path)`` tuples."""

    normalised: list[tuple[str, Path]] = []
    for entry in targets:
        if isinstance(entry, tuple):
            doc_id, path = entry
            normalised.append((str(doc_id), Path(path)))
        else:
            path = Path(entry)
            normalised.append((path.stem, path))
    return normalised


def _validate_chunk_files(
    targets: Sequence[Path | tuple[str, Path]],
    logger,
    *,
    data_root: Path | None = None,
    telemetry: StageTelemetry | None = None,
    format: str = "jsonl",
) -> dict[str, int]:
    """Validate chunk artifacts and return aggregate statistics."""

    chunk_format = str(format or "jsonl").lower()
    validated_files = 0
    validated_rows = 0
    validated_row_groups = 0
    quarantined_files = 0
    missing_files = 0

    for doc_id, path in _normalise_validation_targets(targets):
        if not path.exists():
            missing_files += 1
            log_event(
                logger,
                "warning",
                "Chunk artifact missing",
                status="missing",
                stage=CHUNK_STAGE,
                doc_id=doc_id,
                input_relpath=relative_path(path, data_root),
            )
            continue

        if chunk_format == "parquet":
            try:
                file_rows, row_groups, warnings = _inspect_parquet_chunk_file(path)
            except ValueError as exc:
                reason = str(exc)
                try:
                    input_hash = compute_content_hash(path)
                except Exception:
                    input_hash = ""
                hash_alg = resolve_hash_algorithm()
                quarantine_path = quarantine_artifact(path, reason=reason, logger=logger)
                if telemetry is not None:
                    telemetry.log_failure(
                        doc_id=doc_id,
                        input_path=path,
                        duration_s=0.0,
                        reason=reason,
                        metadata={
                            "input_path": str(path),
                            "input_hash": input_hash,
                            "quarantine": True,
                            "status": "failure",
                            "hash_alg": hash_alg,
                            "format": chunk_format,
                        },
                        manifest_metadata={
                            "output_path": str(quarantine_path),
                            "schema_version": CHUNK_SCHEMA_VERSION,
                            "input_path": str(path),
                            "input_hash": input_hash,
                            "error": reason,
                            "quarantine": True,
                            "status": "failure",
                            "hash_alg": hash_alg,
                            "format": chunk_format,
                        },
                    )
                log_event(
                    logger,
                    "warning",
                    "Chunk artifact quarantined",
                    status="quarantine",
                    stage=CHUNK_STAGE,
                    doc_id=doc_id,
                    input_relpath=relative_path(path, data_root),
                    output_relpath=relative_path(quarantine_path, data_root),
                    error_class="ValidationError",
                    reason=reason,
                )
                quarantined_files += 1
                continue

            validated_files += 1
            validated_rows += file_rows
            validated_row_groups += row_groups
            for warning in warnings:
                log_event(
                    logger,
                    "warning",
                    "Chunk artifact warning",
                    status="validate-only-warning",
                    stage=CHUNK_STAGE,
                    doc_id=doc_id,
                    input_relpath=relative_path(path, data_root),
                    detail=warning,
                )
            continue

        file_rows = 0
        file_errors: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                file_rows += 1
                try:
                    validate_chunk_row(json.loads(line))
                except ValueError as exc:
                    message = f"{path}:{line_no}: {exc}"
                    log_event(
                        logger,
                        "error",
                        "Chunk validation error",
                        status="invalid",
                        stage=CHUNK_STAGE,
                        doc_id=doc_id,
                        input_relpath=relative_path(path, data_root),
                        error_class="ValidationError",
                        detail=message,
                    )
                    file_errors.append(message)

        if file_errors:
            reason = ", ".join(file_errors[:3])
            if len(file_errors) > 3:
                reason += ", ..."
            try:
                input_hash = compute_content_hash(path)
            except Exception:
                input_hash = ""
            hash_alg = resolve_hash_algorithm()
            quarantine_path = quarantine_artifact(path, reason=reason, logger=logger)
            if telemetry is not None:
                telemetry.log_failure(
                    doc_id=doc_id,
                    input_path=path,
                    duration_s=0.0,
                    reason=reason,
                    metadata={
                        "input_path": str(path),
                        "input_hash": input_hash,
                        "quarantine": True,
                        "status": "failure",
                        "hash_alg": hash_alg,
                        "format": chunk_format,
                    },
                    manifest_metadata={
                        "output_path": str(quarantine_path),
                        "schema_version": CHUNK_SCHEMA_VERSION,
                        "input_path": str(path),
                        "input_hash": input_hash,
                        "error": reason,
                        "quarantine": True,
                        "status": "failure",
                        "hash_alg": hash_alg,
                        "format": chunk_format,
                    },
                )
            log_event(
                logger,
                "warning",
                "Chunk artifact quarantined",
                status="quarantine",
                stage=CHUNK_STAGE,
                doc_id=doc_id,
                input_relpath=relative_path(path, data_root),
                output_relpath=relative_path(quarantine_path, data_root),
                error_class="ValidationError",
                reason=reason,
            )
            quarantined_files += 1
            continue

        validated_files += 1
        validated_rows += file_rows

    return {
        "files": validated_files,
        "rows": validated_rows,
        "row_groups": validated_row_groups,
        "quarantined": quarantined_files,
        "missing": missing_files,
    }


# --- Defaults ---


def _resolve_parquet_chunk_artifact(
    *, expected_jsonl_path: Path, out_dir: Path, data_root: Path | None
) -> Path:
    """Best-effort resolution of a Parquet chunk artifact for validation."""

    if data_root is None:
        return expected_jsonl_path

    dataset_root = Path(data_root) / "Chunks" / "fmt=parquet"
    candidates: list[str] = []
    try:
        relative = expected_jsonl_path.relative_to(out_dir)
        candidates.append(storage_paths.normalize_rel_id(relative))
    except ValueError:
        pass
    candidates.append(storage_paths.normalize_rel_id(expected_jsonl_path.stem))

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        rel_pattern = Path(candidate).as_posix()
        pattern = f"**/{rel_pattern}.parquet"
        matches = sorted(dataset_root.glob(pattern))
        if matches:
            return matches[0]

    if candidates:
        return dataset_root / Path(candidates[0]).with_suffix(".parquet")

    fallback = storage_paths.normalize_rel_id(expected_jsonl_path.stem)
    return dataset_root / Path(fallback).with_suffix(".parquet")


def _collect_chunk_artifacts(
    doctag_files: Sequence[Path],
    *,
    in_dir: Path,
    out_dir: Path,
    data_root: Path | None,
    format: str,
) -> list[tuple[str, Path]]:
    """Return ``(doc_id, chunk_path)`` tuples for downstream validation."""

    chunk_format = str(format or "jsonl").lower()
    artifacts: list[tuple[str, Path]] = []
    for doc_path in doctag_files:
        doc_id, chunk_jsonl = derive_doc_id_and_chunks_path(doc_path, in_dir, out_dir)
        if chunk_format == "parquet":
            chunk_path = _resolve_parquet_chunk_artifact(
                expected_jsonl_path=chunk_jsonl,
                out_dir=out_dir,
                data_root=data_root,
            )
        else:
            chunk_path = chunk_jsonl
        artifacts.append((doc_id, chunk_path))
    return artifacts


# --- Defaults ---

MANIFEST_STAGE = "chunks"
_ACTIVE_CONFIG_HASH: str | None = None


def _compute_worker_cfg_hash(config: ChunkWorkerConfig) -> str:
    """Return a stable hash representing the worker configuration."""

    payload = {
        "tokenizer_model": config.tokenizer_model,
        "min_tokens": config.min_tokens,
        "max_tokens": config.max_tokens,
        "soft_barrier_margin": config.soft_barrier_margin,
        "heading_markers": list(config.heading_markers),
        "caption_markers": list(config.caption_markers),
        "docling_version": config.docling_version,
        "serializer_provider_spec": config.serializer_provider_spec,
        "inject_anchors": bool(config.inject_anchors),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_fingerprint_for_output(output_path: Path, *, input_sha256: str, cfg_hash: str) -> Path:
    """Persist the resume fingerprint alongside ``output_path`` and return its location."""

    fingerprint_path = output_path.with_name(f"{output_path.name}.fp.json")
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_sha256": input_sha256,
        "cfg_hash": cfg_hash,
    }
    with atomic_write(fingerprint_path) as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    return fingerprint_path


def _ensure_worker_initialised(config: ChunkWorkerConfig, cfg_hash: str) -> None:
    """Initialise worker state lazily per process."""

    global _ACTIVE_CONFIG_HASH, _WORKER_STATE
    if _WORKER_STATE and cfg_hash == _ACTIVE_CONFIG_HASH:
        return
    _chunk_worker_initializer(config)
    _WORKER_STATE["config_hash"] = cfg_hash
    _ACTIVE_CONFIG_HASH = cfg_hash


def _build_chunk_plan(
    *,
    files: Sequence[Path],
    in_dir: Path,
    out_dir: Path,
    resolved_root: Path,
    worker_config: ChunkWorkerConfig,
    cfg_hash: str,
    hash_alg: str,
    parse_engine_lookup: Mapping[str, str],
    manifest_lookup: Mapping[str, Mapping[str, Any]],
) -> StagePlan:
    """Construct a StagePlan covering every DocTags file slated for chunking."""

    plan_items: list[WorkItem] = []
    for doc_path in files:
        doc_id, planned_output_path = derive_doc_id_and_chunks_path(doc_path, in_dir, out_dir)
        planned_output_path.parent.mkdir(parents=True, exist_ok=True)
        input_hash = compute_content_hash(doc_path, algorithm=hash_alg)
        entry = manifest_lookup.get(doc_id, {})
        parse_engine = parse_engine_lookup.get(doc_id, entry.get("parse_engine", "docling-html"))
        sanitizer_profile = entry.get("sanitizer_profile")
        manifest_output_path_str = entry.get("output_path") if isinstance(entry, Mapping) else None
        manifest_output_path = (
            Path(str(manifest_output_path_str)) if manifest_output_path_str else planned_output_path
        )
        fingerprint_path = manifest_output_path.with_name(f"{manifest_output_path.name}.fp.json")
        metadata: dict[str, Any] = {
            "doc_id": doc_id,
            "input_path": str(doc_path),
            "output_path": str(planned_output_path),
            "resume_output_path": str(manifest_output_path),
            "input_hash": input_hash,
            "hash_alg": hash_alg,
            "parse_engine": parse_engine,
            "sanitizer_profile": sanitizer_profile,
            "worker_config": worker_config,
            "input_relpath": relative_path(doc_path, resolved_root),
            "output_relpath": relative_path(manifest_output_path, resolved_root),
            "planned_output_relpath": relative_path(planned_output_path, resolved_root),
        }
        try:
            size = doc_path.stat().st_size
        except OSError:
            size = 1
        plan_items.append(
            WorkItem(
                item_id=doc_id,
                inputs={"doctags": doc_path},
                outputs={"chunks": manifest_output_path},
                cfg_hash=cfg_hash,
                cost_hint=max(float(size), 1.0),
                metadata=metadata,
                fingerprint=ItemFingerprint(
                    path=fingerprint_path,
                    input_sha256=input_hash,
                    cfg_hash=cfg_hash,
                ),
            )
        )
    return StagePlan(stage_name=MANIFEST_STAGE, items=plan_items, total_items=len(plan_items))


def _chunk_stage_worker(item: WorkItem) -> ItemOutcome:
    """Execute chunking for a single document."""

    metadata = item.metadata
    config: ChunkWorkerConfig = metadata["worker_config"]
    cfg_hash = item.cfg_hash
    _ensure_worker_initialised(config, cfg_hash)

    input_path = Path(metadata["input_path"])
    planned_output_path = Path(metadata["output_path"])
    resume_output_path = Path(metadata.get("resume_output_path", metadata["output_path"]))
    parse_engine = metadata.get("parse_engine", "docling-html")
    sanitizer_profile = metadata.get("sanitizer_profile")
    input_hash = metadata["input_hash"]
    hash_alg = metadata["hash_alg"]

    task = ChunkTask(
        doc_path=input_path,
        output_path=planned_output_path,
        doc_id=metadata["doc_id"],
        doc_stem=input_path.stem,
        input_hash=input_hash,
        parse_engine=parse_engine,
        sanitizer_profile=sanitizer_profile,
    )

    try:
        result = _process_chunk_task(task)
    except Exception as exc:  # pragma: no cover - defensive
        duration = 0.0
        err = StageError(
            stage=MANIFEST_STAGE,
            item_id=metadata["doc_id"],
            category="runtime",
            message=str(exc),
            retryable=False,
        )
        manifest = {
            "input_path": str(input_path),
            "input_hash": input_hash,
            "hash_alg": hash_alg,
            "output_path": str(resume_output_path),
            "schema_version": CHUNK_SCHEMA_VERSION,
            "error": err.message,
            "parse_engine": parse_engine,
            "sanitizer_profile": sanitizer_profile,
            "anchors_injected": bool(config.inject_anchors),
        }
        return ItemOutcome(
            status="failure", duration_s=duration, manifest=manifest, result={}, error=err
        )

    if result.status != "success":
        error_message = result.error or "unknown error"
        err = StageError(
            stage=MANIFEST_STAGE,
            item_id=result.doc_id,
            category="runtime",
            message=error_message,
            retryable=False,
        )
        manifest = {
            "input_path": str(result.input_path),
            "input_hash": result.input_hash,
            "hash_alg": hash_alg,
            "output_path": str(result.output_path),
            "schema_version": CHUNK_SCHEMA_VERSION,
            "error": error_message,
            "parse_engine": result.parse_engine,
            "sanitizer_profile": result.sanitizer_profile,
            "chunk_count": result.chunk_count,
            "anchors_injected": result.anchors_injected,
        }
        return ItemOutcome(
            status="failure", duration_s=result.duration_s, manifest=manifest, result={}, error=err
        )

    fingerprint_path = _write_fingerprint_for_output(
        result.output_path, input_sha256=input_hash, cfg_hash=cfg_hash
    )
    metadata["resume_output_path"] = str(result.output_path)
    metadata["output_path"] = str(result.output_path)
    metadata["fingerprint_path"] = str(fingerprint_path)
    if config.data_root is not None:
        metadata["output_relpath"] = relative_path(result.output_path, config.data_root)

    manifest = {
        "input_path": str(result.input_path),
        "input_hash": result.input_hash,
        "hash_alg": hash_alg,
        "output_path": str(result.output_path),
        "schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_count": result.chunk_count,
        "total_tokens": result.total_tokens,
        "parse_engine": result.parse_engine,
        "anchors_injected": result.anchors_injected,
        "sanitizer_profile": result.sanitizer_profile,
        "chunks_format": config.format,
    }
    if result.parquet_bytes is not None:
        manifest["parquet_bytes"] = result.parquet_bytes
    if result.row_group_count is not None:
        manifest["row_group_count"] = result.row_group_count
    if result.rows_written is not None:
        manifest["rows_written"] = result.rows_written
    if result.artifact_paths:
        manifest["output_paths"] = [str(path) for path in result.artifact_paths]
    result_payload = {
        "chunk_count": result.chunk_count,
        "total_tokens": result.total_tokens,
        "anchors_injected": result.anchors_injected,
    }
    if result.parquet_bytes is not None:
        result_payload["parquet_bytes"] = result.parquet_bytes
    if result.row_group_count is not None:
        result_payload["row_group_count"] = result.row_group_count
    if result.artifact_paths:
        result_payload["output_paths"] = [str(path) for path in result.artifact_paths]
    return ItemOutcome(
        status="success",
        duration_s=result.duration_s,
        manifest=manifest,
        result=result_payload,
        error=None,
    )


def _make_chunk_stage_hooks(
    *,
    logger,
    resolved_root: Path,
) -> StageHooks:
    """Return StageHooks that log manifests and telemetry for chunking."""

    def before_stage(context: StageContext) -> None:
        context.metadata["logger"] = logger
        context.metadata["resolved_root"] = resolved_root
        context.metadata["schema_version"] = CHUNK_SCHEMA_VERSION

        if context.options.workers > 1:
            log_event(
                logger,
                "info",
                "Parallel chunking enabled",
                workers=context.options.workers,
            )

    def after_item(
        item: WorkItem,
        outcome_or_error: ItemOutcome | StageError,
        context: StageContext,
    ) -> None:
        stage_logger = context.metadata.get("logger", logger)
        root = context.metadata.get("resolved_root", resolved_root)
        schema_version = context.metadata.get("schema_version", CHUNK_SCHEMA_VERSION)
        metadata = item.metadata
        doc_id = metadata["doc_id"]
        input_path = Path(metadata["input_path"])
        output_path = Path(metadata.get("resume_output_path", metadata["output_path"]))
        input_hash = metadata["input_hash"]
        hash_alg = metadata["hash_alg"]
        parse_engine = metadata.get("parse_engine", "docling-html")
        rel_fields = {
            "stage": MANIFEST_STAGE,
            "doc_id": doc_id,
            "input_relpath": metadata.get("input_relpath", relative_path(input_path, root)),
            "output_relpath": metadata.get(
                "output_relpath",
                metadata.get("planned_output_relpath", relative_path(output_path, root)),
            ),
        }

        if isinstance(outcome_or_error, ItemOutcome):
            if outcome_or_error.status == "success":
                chunk_count = outcome_or_error.result.get("chunk_count")
                log_event(
                    stage_logger,
                    "info",
                    "Chunk file written",
                    status="success",
                    elapsed_ms=int(outcome_or_error.duration_s * 1000),
                    chunk_count=chunk_count,
                    parse_engine=parse_engine,
                    **rel_fields,
                )
                payload = dict(outcome_or_error.manifest)
                for key in (
                    "input_path",
                    "input_hash",
                    "output_path",
                    "schema_version",
                    "hash_alg",
                ):
                    payload.pop(key, None)
                manifest_log_success(
                    stage=MANIFEST_STAGE,
                    doc_id=doc_id,
                    duration_s=outcome_or_error.duration_s,
                    schema_version=schema_version,
                    input_path=input_path,
                    input_hash=input_hash,
                    output_path=output_path,
                    hash_alg=hash_alg,
                    **payload,
                )
                return

            if outcome_or_error.status == "skip":
                reason = outcome_or_error.result.get("reason", "resume-satisfied")
                log_event(
                    stage_logger,
                    "info",
                    "Skipping chunk file: output exists and input unchanged",
                    status="skip",
                    reason=reason,
                    **rel_fields,
                )
                manifest_log_skip(
                    stage=MANIFEST_STAGE,
                    doc_id=doc_id,
                    input_path=input_path,
                    input_hash=input_hash,
                    output_path=output_path,
                    hash_alg=hash_alg,
                    schema_version=schema_version,
                    reason=reason,
                    parse_engine=parse_engine,
                )
                return

            error_message = (
                outcome_or_error.error.message if outcome_or_error.error else "unknown error"
            )
            log_event(
                stage_logger,
                "error",
                "Chunking failed",
                status="failure",
                error=error_message,
                elapsed_ms=int(outcome_or_error.duration_s * 1000),
                parse_engine=parse_engine,
                **rel_fields,
            )
            payload = dict(outcome_or_error.manifest)
            for key in (
                "input_path",
                "input_hash",
                "output_path",
                "schema_version",
                "hash_alg",
                "error",
            ):
                payload.pop(key, None)
            manifest_log_failure(
                stage=MANIFEST_STAGE,
                doc_id=doc_id,
                duration_s=outcome_or_error.duration_s,
                schema_version=schema_version,
                input_path=input_path,
                input_hash=input_hash,
                output_path=output_path,
                hash_alg=hash_alg,
                error=error_message,
                **payload,
            )
            return

        # StageError surfaced from the runner.
        error = outcome_or_error
        log_event(
            stage_logger,
            "error",
            "Chunking failed",
            status="failure",
            error=error.message,
            **rel_fields,
        )
        manifest_log_failure(
            stage=MANIFEST_STAGE,
            doc_id=doc_id,
            duration_s=0.0,
            schema_version=schema_version,
            input_path=input_path,
            input_hash=input_hash,
            output_path=output_path,
            hash_alg=hash_alg,
            error=error.message,
        )

    def after_stage(outcome: StageOutcome, context: StageContext) -> None:
        stage_logger = context.metadata.get("logger", logger)
        log_event(
            stage_logger,
            "info",
            "Chunk stage summary",
            scheduled=outcome.scheduled,
            succeeded=outcome.succeeded,
            failed=outcome.failed,
            skipped=outcome.skipped,
            cancelled=outcome.cancelled,
            wall_ms=round(outcome.wall_ms, 3),
            stage=MANIFEST_STAGE,
            doc_id="__summary__",
        )

    return StageHooks(
        before_stage=before_stage,
        after_item=after_item,
        after_stage=after_stage,
    )


def _main_inner(
    args: argparse.Namespace | SimpleNamespace | Sequence[str] | None = None,
    config_adapter=None,
) -> int:
    """CLI driver that chunks DocTags files and enforces minimum token thresholds.

    Args:
        args (argparse.Namespace | None): Optional CLI namespace supplied during
            testing or orchestration.
        config_adapter: Optional ChunkerCfg instance from ConfigurationAdapter (new pattern).
              If provided, bypasses sys.argv parsing and uses this config directly.

    Returns:
        int: Exit code where ``0`` indicates success.
    """

    parser = build_parser()
    bootstrap_root = detect_data_root()
    try:
        data_doctags(bootstrap_root)
        data_chunks(bootstrap_root)
    except Exception as exc:
        logging.getLogger(__name__).debug("Failed to bootstrap chunking directories", exc_info=exc)

    # NEW PATH: If adapter provided (from unified CLI), use it directly
    if config_adapter is not None:
        cfg = config_adapter
        base_config = cfg.to_manifest()
        namespace = argparse.Namespace()
        for field_def in fields(ChunkerCfg):
            setattr(namespace, field_def.name, getattr(cfg, field_def.name))
    # LEGACY PATH: Parse from args or sys.argv
    else:
        if args is None:
            namespace = parse_args_with_overrides(parser)
        elif isinstance(args, argparse.Namespace):
            namespace = args
            if getattr(namespace, "_cli_explicit_overrides", None) is None:
                keys = [name for name in vars(namespace) if not name.startswith("_")]
                annotate_cli_overrides(namespace, explicit=keys, defaults={})
        elif isinstance(args, SimpleNamespace) or hasattr(args, "__dict__"):
            base = parse_args_with_overrides(parser, [])
            payload = {key: value for key, value in vars(args).items() if not key.startswith("_")}
            for key, value in payload.items():
                setattr(base, key, value)
            annotate_cli_overrides(base, explicit=payload.keys(), defaults={})
            namespace = base
        else:
            namespace = parse_args_with_overrides(parser, args)

        profile = getattr(namespace, "profile", None)
        defaults = CHUNK_PROFILE_PRESETS.get(profile or "", {})

        # Build config from namespace (legacy: no from_args() available)
        cfg = ChunkerCfg()
        cfg.apply_args(namespace, defaults=defaults)
        cfg.finalize()

        base_config = cfg.to_manifest()
        if profile:
            base_config.setdefault("profile", profile)
        for field_def in fields(ChunkerCfg):
            setattr(namespace, field_def.name, getattr(cfg, field_def.name))

    log_level = getattr(namespace, "log_level", "INFO")
    run_id = uuid.uuid4().hex
    logger = get_logger(
        __name__,
        level=str(log_level),
        base_fields={"run_id": run_id, "stage": CHUNK_STAGE},
    )
    if profile and defaults:
        log_event(
            logger,
            "info",
            "Applying profile",
            status="profile",
            profile=profile,
            **{key: defaults[key] for key in sorted(defaults)},
        )
    set_spawn_or_warn(logger)
    args = namespace

    if args.min_tokens < 0 or args.max_tokens < 0:
        log_event(
            logger,
            "error",
            "Token thresholds must be non-negative",
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        raise ChunkingCLIValidationError(
            option="--min-tokens/--max-tokens",
            message="values must be non-negative",
        )
    if args.min_tokens > args.max_tokens:
        log_event(
            logger,
            "error",
            "Invalid token range",
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        raise ChunkingCLIValidationError(
            option="--min-tokens/--max-tokens",
            message="minimum threshold cannot exceed maximum",
        )
    if args.soft_barrier_margin < 0:
        log_event(
            logger,
            "error",
            "Soft barrier margin must be non-negative",
            soft_barrier_margin=args.soft_barrier_margin,
        )
        raise ChunkingCLIValidationError(
            option="--soft-barrier-margin",
            message="must be >= 0",
        )

    serializer_spec = getattr(args, "serializer_provider", DEFAULT_SERIALIZER_PROVIDER)
    try:
        _resolve_serializer_provider(serializer_spec)
    except Exception as exc:
        log_event(
            logger,
            "error",
            "Serializer provider import failed",
            serializer_provider=serializer_spec,
            error=str(exc),
        )
        raise ChunkingCLIValidationError(
            option="--serializer-provider",
            message=f"invalid provider: {exc}",
        ) from exc

    try:
        min_tokens = int(args.min_tokens)
        max_tokens = int(args.max_tokens)
        soft_margin = int(args.soft_barrier_margin)
    except (TypeError, ValueError) as exc:
        raise ChunkingCLIValidationError(
            option="--min-tokens/--max-tokens/--soft-barrier-margin",
            message="must be integers",
        ) from exc

    args.min_tokens = min_tokens
    args.max_tokens = max_tokens
    args.soft_barrier_margin = soft_margin
    args.serializer_provider = serializer_spec
    log_event(
        logger,
        "info",
        "Serializer provider selected",
        serializer_provider=args.serializer_provider,
    )

    try:
        shard_count = int(getattr(args, "shard_count", 1))
        shard_index = int(getattr(args, "shard_index", 0))
    except (TypeError, ValueError) as exc:
        raise ChunkingCLIValidationError(
            option="--shard-count/--shard-index",
            message="must be integers",
        ) from exc
    if shard_count < 1:
        raise ChunkingCLIValidationError(option="--shard-count", message="must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ChunkingCLIValidationError(
            option="--shard-index",
            message="must be between 0 and shard-count-1",
        )
    args.shard_count = shard_count
    args.shard_index = shard_index

    data_root_override = args.data_root
    data_root_overridden = data_root_override is not None
    resolved_data_root = prepare_data_root(data_root_override, detect_data_root())
    ensure_model_environment()
    logger.bind(data_root=str(resolved_data_root))

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

    default_in_dir = data_doctags(resolved_data_root, ensure=False)
    default_out_dir = data_chunks(resolved_data_root, ensure=False)

    in_dir = resolve_pipeline_path(
        cli_value=args.in_dir,
        default_path=default_in_dir,
        resolved_data_root=resolved_data_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_doctags(root, ensure=False),
    )
    out_dir = resolve_pipeline_path(
        cli_value=args.out_dir,
        default_path=default_out_dir,
        resolved_data_root=resolved_data_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_chunks(root, ensure=False),
    )
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    args.in_dir = in_dir
    args.out_dir = out_dir
    logger.bind(
        input_relpath=relative_path(in_dir, resolved_data_root),
        output_relpath=relative_path(out_dir, resolved_data_root),
    )

    context = ParsingContext(run_id=run_id, data_root=resolved_data_root)
    context.apply_config(cfg)
    context.in_dir = in_dir
    context.out_dir = out_dir
    context.doctags_dir = in_dir
    context.min_tokens = min_tokens
    context.max_tokens = max_tokens
    context.soft_barrier_margin = soft_margin
    context.serializer_provider = serializer_spec
    context.shard_count = shard_count
    context.shard_index = shard_index
    context.resume = bool(args.resume)
    context.force = bool(args.force)
    context.validate_only = bool(args.validate_only)
    context.inject_anchors = bool(args.inject_anchors)
    context.profile = profile
    base_extra = {
        key: value for key, value in base_config.items() if key not in ParsingContext.field_names()
    }
    if base_extra:
        context.merge_extra(base_extra)

    heading_markers: tuple[str, ...] = DEFAULT_HEADING_MARKERS
    caption_markers: tuple[str, ...] = DEFAULT_CAPTION_MARKERS
    custom_heading_markers: list[str] = []
    custom_caption_markers: list[str] = []
    markers_override = getattr(args, "structural_markers", None)
    if markers_override is not None:
        markers_path = markers_override.expanduser().resolve()
        if not markers_path.exists():
            log_event(
                logger,
                "error",
                "Structural marker configuration not found",
                structural_markers=str(markers_path),
            )
            raise FileNotFoundError(f"Heading markers file not found: {markers_path}")
        extra_headings, extra_captions = load_structural_marker_config(markers_path)
        custom_heading_markers = extra_headings
        custom_caption_markers = extra_captions
        if extra_headings:
            heading_markers = dedupe_preserve_order((*heading_markers, *tuple(extra_headings)))
        if extra_captions:
            caption_markers = dedupe_preserve_order((*caption_markers, *tuple(extra_captions)))
        args.structural_markers = markers_path
        context.update_extra(structural_markers=str(markers_path))

    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_doctags(in_dir))
    if args.shard_count > 1:
        total_candidates = len(files)
        selected_files = [
            path
            for path in files
            if compute_stable_shard(compute_relative_doc_id(path, in_dir), args.shard_count)
            == args.shard_index
        ]
        logger.info(
            "Applying shard filter",
            extra={
                "extra_fields": {
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "selected_files": len(selected_files),
                    "total_files": total_candidates,
                }
            },
        )
        files = selected_files
        if not files:
            log_event(
                logger,
                "warning",
                "Shard contains no DocTags files",
                stage=CHUNK_STAGE,
                doc_id="__aggregate__",
                input_hash=None,
                error_code="SHARD_EMPTY",
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
            return 0
    if not files:
        log_event(
            logger,
            "warning",
            "No .doctags files found",
            stage=CHUNK_STAGE,
            doc_id="__aggregate__",
            input_hash=None,
            error_code="NO_INPUT_FILES",
            input_dir=str(in_dir),
        )
        return 0

    if args.force:
        logger.info("Force mode: reprocessing all DocTags files")
    elif args.resume:
        logger.info("Resume mode enabled: unchanged inputs will be skipped")

    tokenizer_model = args.tokenizer_model
    logger.info(
        "Loading tokenizer",
        extra={"extra_fields": {"tokenizer_model": tokenizer_model}},
    )

    telemetry_sink = TelemetrySink(
        resolve_attempts_path(MANIFEST_STAGE, resolved_data_root),
        resolve_manifest_path(MANIFEST_STAGE, resolved_data_root),
    )
    stage_telemetry = StageTelemetry(telemetry_sink, run_id=run_id, stage=MANIFEST_STAGE)
    with telemetry_scope(stage_telemetry):
        if getattr(args, "validate_only", False):
            chunk_artifacts = _collect_chunk_artifacts(
                files,
                in_dir=in_dir,
                out_dir=out_dir,
                data_root=resolved_data_root,
                format=cfg.format,
            )
            _run_validate_only(
                doctag_files=files,
                chunk_artifacts=chunk_artifacts,
                expected_artifacts=len(files),
                logger=logger,
                cfg=cfg,
                tokenizer_model=tokenizer_model,
                heading_markers=heading_markers,
                caption_markers=caption_markers,
                data_root=resolved_data_root,
                in_dir=in_dir,
                out_dir=out_dir,
                telemetry=stage_telemetry,
                format=cfg.format,
            )
            return 0

        chunk_config = ChunkWorkerConfig(
            tokenizer_model=tokenizer_model,
            min_tokens=int(args.min_tokens),
            max_tokens=int(args.max_tokens),
            soft_barrier_margin=int(args.soft_barrier_margin),
            heading_markers=heading_markers,
            caption_markers=caption_markers,
            docling_version=docling_version,
            serializer_provider_spec=str(args.serializer_provider),
            inject_anchors=bool(cfg.inject_anchors),
            data_root=resolved_data_root,
        )

        if "bert" in tokenizer_model.lower():
            log_event(
                logger,
                "warning",
                "Tokenizer may not align with downstream embedder; consider calibration",
                stage=CHUNK_STAGE,
                doc_id="__config__",
                input_hash=None,
                error_code="TOKENIZER_MISMATCH",
                tokenizer_model=tokenizer_model,
                recommended_tokenizer=DEFAULT_TOKENIZER,
            )

        worker_count = max(1, int(getattr(args, "workers", 1)))
        if worker_count > 1 and str(args.serializer_provider) != DEFAULT_SERIALIZER_PROVIDER:
            log_event(
                logger,
                "warning",
                "Falling back to single worker because serializer provider may be stateful",
                stage=CHUNK_STAGE,
                doc_id="__config__",
                input_hash=None,
                error_code="STATEFUL_SERIALIZER",
                requested_workers=int(getattr(args, "workers", 1)),
                serializer_provider=str(args.serializer_provider),
            )
            worker_count = 1

        cfg_hash = _compute_worker_cfg_hash(chunk_config)
        context.workers = worker_count
        context.resume = bool(args.resume)
        context.force = bool(args.force)
        context.validate_only = bool(args.validate_only)
        context.inject_anchors = bool(args.inject_anchors)
        context.update_extra(
            docling_version=docling_version,
            custom_heading_markers=custom_heading_markers,
            custom_caption_markers=custom_caption_markers,
        )
        context.update_extra(worker_config_hash=cfg_hash)
        context_payload = context.to_manifest()
        log_event(
            logger,
            "info",
            "Chunking configuration",
            status="config",
            stage=CHUNK_STAGE,
            **context_payload,
        )
        stage_telemetry.log_config(
            output_path=out_dir,
            schema_version=CHUNK_SCHEMA_VERSION,
            metadata={
                "input_path": str(in_dir),
                "input_hash": "",
                "config": context_payload,
            },
        )

        set_spawn_or_warn(logger)

        hash_alg = resolve_hash_algorithm()
        manifest_lookup: dict[str, Mapping[str, Any]] = dict(html_manifest_index)
        manifest_lookup.update(pdf_manifest_index)
        plan = _build_chunk_plan(
            files=files,
            in_dir=in_dir,
            out_dir=out_dir,
            resolved_root=resolved_data_root,
            worker_config=chunk_config,
            cfg_hash=cfg_hash,
            hash_alg=hash_alg,
            parse_engine_lookup=parse_engine_lookup,
            manifest_lookup=manifest_lookup,
        )

        hooks = _make_chunk_stage_hooks(logger=logger, resolved_root=resolved_data_root)
        options = StageOptions(
            policy="cpu",
            workers=worker_count,
            resume=bool(args.resume),
            force=bool(args.force),
            diagnostics_interval_s=15.0,
        )

        outcome = run_stage(plan, _chunk_stage_worker, options, hooks)
        if outcome.failed > 0 or outcome.cancelled:
            return 1
        return 0


def _run_validate_only(
    *,
    doctag_files: Sequence[Path],
    chunk_artifacts: Sequence[tuple[str, Path]],
    expected_artifacts: int,
    logger,
    cfg: ChunkerCfg,
    tokenizer_model: str,
    heading_markers: tuple[str, ...],
    caption_markers: tuple[str, ...],
    data_root: Path | None,
    in_dir: Path,
    out_dir: Path,
    telemetry: StageTelemetry,
    format: str,
) -> None:
    """Validate chunk outputs and report statistics without writing artifacts."""

    chunk_format = str(format or "jsonl").lower()
    stats = _validate_chunk_files(
        chunk_artifacts,
        logger,
        data_root=data_root,
        telemetry=telemetry,
        format=chunk_format,
    )
    logger.bind(mode="validate-only", chunk_format=chunk_format)

    validated_artifacts = stats["files"]
    examined_artifacts = validated_artifacts + stats["quarantined"]

    if validated_artifacts == 0:
        log_event(
            logger,
            "info",
            "No chunk artifacts validated",
            status="validate-only",
            stage=CHUNK_STAGE,
            doc_id="__aggregate__",
            input_hash=None,
            expected_artifacts=expected_artifacts,
            examined_artifacts=examined_artifacts,
            chunk_format=chunk_format,
            **stats,
        )
        return

    log_event(
        logger,
        "info",
        "Chunk validation complete",
        status="validate-only",
        stage=CHUNK_STAGE,
        doc_id="__aggregate__",
        input_hash=None,
        expected_artifacts=expected_artifacts,
        examined_artifacts=examined_artifacts,
        chunk_format=chunk_format,
        **stats,
    )

    ensure_model_environment()
    hf = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf, max_tokens=cfg.max_tokens)
    provider_cls = _resolve_serializer_provider(str(cfg.serializer_provider))
    provider = provider_cls()
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        serializer_provider=provider,
    )

    total_chunks = 0
    total_records = 0
    token_counts: list[int] = []
    boundary_violations = 0
    heading_hits = 0
    caption_hits = 0

    for path in doctag_files:
        if not path.exists():
            continue
        doctags_text = read_utf8(path)
        doc = build_doc(doc_name=path.stem, doctags_text=doctags_text)
        chunks = list(chunker.chunk(dl_doc=doc))
        total_chunks += len(chunks)
        recs: list[Rec] = []
        for idx, ch in enumerate(chunks):
            text = chunker.contextualize(ch)
            n_tok = tokenizer.count_tokens(text=text)
            has_caption, has_classification, num_images, image_confidence, picture_meta = (
                summarize_image_metadata(ch, text)
            )
            recs.append(
                Rec(
                    text=text,
                    n_tok=n_tok,
                    src_idxs=[idx],
                    refs=[],
                    pages=[],
                    has_image_captions=has_caption,
                    has_image_classification=has_classification,
                    num_images=num_images,
                    image_confidence=image_confidence,
                    start_offset=_extract_chunk_start(ch),
                    picture_meta=picture_meta,
                )
            )
        coalesced = coalesce_small_runs(
            records=recs,
            tokenizer=tokenizer,
            min_tokens=cfg.min_tokens,
            max_tokens=cfg.max_tokens,
            soft_barrier_margin=cfg.soft_barrier_margin,
            heading_markers=heading_markers,
            caption_markers=caption_markers,
        )
        total_records += len(coalesced)
        token_counts.extend(rec.n_tok for rec in coalesced)
        boundary_violations += sum(
            1 for rec in coalesced if is_structural_boundary(rec, heading_markers, caption_markers)
        )
        heading_hits += sum(
            1 for rec in coalesced if is_structural_boundary(rec, heading_markers, ())
        )
        caption_hits += sum(
            1 for rec in coalesced if is_structural_boundary(rec, (), caption_markers)
        )

    avg_tokens = statistics.mean(token_counts) if token_counts else 0.0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    validated_files = stats["files"]
    validated_rows = stats["rows"]

    log_event(
        logger,
        "info",
        "Validate-only summary",
        status="validate-only-summary",
        stage=CHUNK_STAGE,
        validated_files=validated_files,
        validated_rows=validated_rows,
        examined_artifacts=examined_artifacts,
        expected_artifacts=expected_artifacts,
        missing_artifacts=stats["missing"],
        quarantined_artifacts=stats["quarantined"],
        row_groups=stats["row_groups"],
        chunk_format=chunk_format,
        generated_chunks=total_records,
        avg_tokens=round(avg_tokens, 2),
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        heading_hits=heading_hits,
        caption_hits=caption_hits,
        boundary_violations=boundary_violations,
        input_relpath=relative_path(in_dir, data_root),
        output_relpath=relative_path(out_dir, data_root),
    )


def _write_chunks_atomic(
    output_path: Path,
    chunk_rows: list[dict[str, Any]],
    format: str,
    doc_id: str,
    data_root: Path | None,
    cfg_hash: str,
) -> None:
    """
    Write chunk rows to output path in specified format.

    Supports both JSONL (legacy) and Parquet (default) formats.

    Args:
        output_path: Path for output file.
        chunk_rows: List of chunk row dictionaries.
        format: Output format ("parquet" or "jsonl").
        doc_id: Document ID for error messages.
        data_root: Data root directory (for Parquet output).
        cfg_hash: Configuration hash for Parquet footer.

    Raises:
        ValueError: If format is unsupported.
    """
    if format == "parquet":
        # Use Parquet writer with partitioned layout
        if data_root is None:
            raise ValueError("data_root is required for Parquet output")

        # Normalize rel_id from output_path
        rel_id = storage_paths.normalize_rel_id(output_path.stem)

        # Convert ChunkRow format to Parquet format
        parquet_rows = []
        for row in chunk_rows:
            parquet_row = {
                "doc_id": row.get("doc_id", doc_id),
                "chunk_id": row.get("chunk_id", 0),
                "text": row.get("text", ""),
                "tokens": row.get("num_tokens", 0),
                "span": {
                    "start": row.get("start_offset", 0),
                    "end": row.get("start_offset", 0) + len(row.get("text", "")),
                },
                "created_at": {"ts": "now"},  # Will be replaced by writer
                "schema_version": "docparse/chunks/1.0.0",
            }
            # Optional fields
            if "meta" in row or row.get("provenance"):
                parquet_row["meta"] = row.get("meta", {})
            parquet_rows.append(parquet_row)

        writer = ParquetChunksWriter()
        writer.write(
            parquet_rows,
            data_root=data_root,
            rel_id=rel_id,
            cfg_hash=cfg_hash,
            created_by="DocsToKG-DocParsing",
        )
    elif format == "jsonl":
        # Legacy JSONL format
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_write(output_path) as handle:
            for row in chunk_rows:
                validate_chunk_row(row)
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
    else:
        raise ValueError(f"Unsupported chunk format: {format}")


def main(args: argparse.Namespace | SimpleNamespace | Sequence[str] | None = None) -> int:
    """Wrapper that normalises CLI validation failures for the chunk stage."""

    try:
        return _main_inner(args)
    except ChunkingCLIValidationError as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
