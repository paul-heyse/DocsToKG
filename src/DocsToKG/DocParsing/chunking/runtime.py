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
#       "id": "resolve-serializer-provider",
#       "name": "_resolve_serializer_provider",
#       "anchor": "function-resolve-serializer-provider",
#       "kind": "function"
#     },
#     {
#       "id": "validate-chunk-files",
#       "name": "_validate_chunk_files",
#       "anchor": "function-validate-chunk-files",
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
import importlib
import itertools
import json
import logging
import statistics
import unicodedata
import time
import uuid
from dataclasses import dataclass, fields
from multiprocessing import get_context
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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
    ResumeController,
    compute_relative_doc_id,
    compute_stable_shard,
    dedupe_preserve_order,
    derive_doc_id_and_chunks_path,
    load_structural_marker_config,
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
    make_hasher,
    load_manifest_index,
    quarantine_artifact,
    relative_path,
    resolve_attempts_path,
    resolve_hash_algorithm,
    resolve_manifest_path,
)
from DocsToKG.DocParsing.logging import get_logger, log_event, manifest_log_skip, telemetry_scope
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
    src_idxs: List[int]
    refs: List[str]
    pages: List[int]
    start_offset: Optional[int] = None
    has_image_captions: bool = False
    has_image_classification: bool = False
    num_images: int = 0
    image_confidence: Optional[float] = None
    picture_meta: Optional[List[Dict[str, Any]]] = None


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


def extract_refs_and_pages(chunk: BaseChunk) -> Tuple[List[str], List[int]]:
    """Extract inline references and page numbers from a Docling chunk."""

    refs: List[str] = []
    pages: List[int] = []
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
    heading_markers: Tuple[str, ...] = DEFAULT_HEADING_MARKERS,
    caption_markers: Tuple[str, ...] = DEFAULT_CAPTION_MARKERS,
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
) -> Tuple[bool, bool, int, Optional[float], List[Dict[str, Any]]]:
    """Summarise image annotations associated with ``chunk``."""

    has_caption = any(
        marker and marker in text
        for marker in ("Figure caption:", "Table:", "Picture description:")
    ) or text.strip().startswith("<!-- image -->")
    has_classification = False
    num_images = 0
    confidence: Optional[float] = None
    extras: List[Dict[str, Any]] = []
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


def _extract_chunk_start(chunk: BaseChunk) -> Optional[int]:
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

    picture_meta: List[Dict[str, Any]] = []
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
    records: List[Rec],
    tokenizer: Any,
    *,
    min_tokens: int,
    max_tokens: int | None = None,
    soft_barrier_margin: int = 64,
    heading_markers: Tuple[str, ...] = DEFAULT_HEADING_MARKERS,
    caption_markers: Tuple[str, ...] = DEFAULT_CAPTION_MARKERS,
) -> List[Rec]:
    """Merge contiguous undersized chunks while respecting structural boundaries."""

    if not records:
        return []

    max_threshold = max_tokens if max_tokens is not None else 512

    output: List[Rec] = []
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


_WORKER_STATE: Dict[str, Any] = {}


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
    }


def _process_chunk_task(task: ChunkTask) -> ChunkResult:
    """Chunk a single DocTags file using worker-local state."""

    if not _WORKER_STATE:
        raise RuntimeError("Chunk worker initialiser was not executed")

    cfg: ChunkWorkerConfig = _WORKER_STATE["config"]
    tokenizer: HuggingFaceTokenizer = _WORKER_STATE["tokenizer"]
    chunker: HybridChunker = _WORKER_STATE["chunker"]
    heading_markers: Tuple[str, ...] = _WORKER_STATE["heading_markers"]
    caption_markers: Tuple[str, ...] = _WORKER_STATE["caption_markers"]

    start_time = time.perf_counter()
    chunk_count = 0
    total_tokens = 0
    try:
        text = read_utf8(task.doc_path)
        input_hash = task.input_hash or _hash_doctags_text(text)
        doc = build_doc(task.doc_stem, text)
        raw_chunks = list(chunker.chunk(dl_doc=doc))

        records: List[Rec] = []
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

        output_path = task.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_count = len(merged)
        total_tokens = sum(rec.n_tok for rec in merged)

        with atomic_write(output_path) as handle:
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
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        duration = time.perf_counter() - start_time
        return ChunkResult(
            doc_id=task.doc_id,
            doc_stem=task.doc_stem,
            status="success",
            duration_s=duration,
            input_path=task.doc_path,
            output_path=task.output_path,
            input_hash=input_hash,
            chunk_count=chunk_count,
            total_tokens=total_tokens,
            parse_engine=task.parse_engine,
            sanitizer_profile=task.sanitizer_profile,
            anchors_injected=cfg.inject_anchors,
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
            output_path=task.output_path,
            input_hash=input_hash,
            chunk_count=0,
            total_tokens=0,
            parse_engine=task.parse_engine,
            sanitizer_profile=task.sanitizer_profile,
            anchors_injected=cfg.inject_anchors,
            error=str(exc),
        )


def _process_indexed_chunk_task(payload: Tuple[int, ChunkTask]) -> Tuple[int, ChunkResult]:
    """Execute ``_process_chunk_task`` and preserve submission ordering."""

    index, task = payload
    return index, _process_chunk_task(task)


def _ordered_results(results: Iterable[Tuple[int, ChunkResult]]) -> Iterator[ChunkResult]:
    """Yield chunk results in their original submission order."""

    pending: Dict[int, ChunkResult] = {}
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


def _validate_chunk_files(
    files: Sequence[Path],
    logger,
    *,
    data_root: Optional[Path] = None,
    telemetry: Optional[StageTelemetry] = None,
) -> Dict[str, int]:
    """Validate chunk JSONL rows across supplied files.

    Returns a dictionary summarising file, row, and quarantine counts. Detailed
    log events for individual errors are emitted within the function; callers
    are responsible for logging the aggregate summary so they can attach
    run-specific context.
    """

    validated_files = 0
    validated_rows = 0
    quarantined_files = 0

    for path in files:
        if not path.exists():
            continue
        file_rows = 0
        file_errors: List[str] = []
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
                        doc_id=path.stem,
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
            doc_id = path.stem
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
                    },
                )
            log_event(
                logger,
                "warning",
                "Chunk file quarantined",
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

    if quarantined_files:
        log_event(
            logger,
            "warning",
            "Quarantined chunk files",
            stage=CHUNK_STAGE,
            doc_id="__aggregate__",
            input_hash=None,
            error_code="QUARANTINE_DETECTED",
            quarantined=quarantined_files,
        )

    return {
        "files": validated_files,
        "rows": validated_rows,
        "quarantined": quarantined_files,
    }


# --- Defaults ---

MANIFEST_STAGE = "chunks"


def _main_inner(
    args: argparse.Namespace | SimpleNamespace | Sequence[str] | None = None,
) -> int:
    """CLI driver that chunks DocTags files and enforces minimum token thresholds.

    Args:
        args (argparse.Namespace | None): Optional CLI namespace supplied during
            testing or orchestration.

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
    cfg = ChunkerCfg.from_args(namespace, defaults=defaults)
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

    heading_markers: Tuple[str, ...] = DEFAULT_HEADING_MARKERS
    caption_markers: Tuple[str, ...] = DEFAULT_CAPTION_MARKERS
    custom_heading_markers: List[str] = []
    custom_caption_markers: List[str] = []
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

    chunk_manifest_index: Dict[str, Any] = {}
    if args.resume:
        chunk_manifest_index = load_manifest_index(MANIFEST_STAGE, resolved_data_root)

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
            _run_validate_only(
                files=files,
                logger=logger,
                cfg=cfg,
                tokenizer_model=tokenizer_model,
                heading_markers=heading_markers,
                caption_markers=caption_markers,
                data_root=resolved_data_root,
                in_dir=in_dir,
                out_dir=out_dir,
                telemetry=stage_telemetry,
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

        resume_controller = ResumeController(args.resume, args.force, chunk_manifest_index)

        def iter_chunk_tasks() -> Iterator[ChunkTask]:
            """Generate chunk tasks for processing, respecting resume/force settings."""
            resume_enabled = bool(args.resume)
            force_enabled = bool(args.force)
            resume_needs_hash = resume_enabled and not force_enabled
            for path in files:
                doc_id, out_path = derive_doc_id_and_chunks_path(path, in_dir, out_dir)
                name = path.stem
                input_hash = ""
                manifest_entry = resume_controller.entry(doc_id) if resume_enabled else None
                output_exists = out_path.exists() if resume_enabled else False
                if resume_needs_hash and manifest_entry and output_exists:
                    input_hash = compute_content_hash(path)
                parse_engine = parse_engine_lookup.get(doc_id, "docling-html")
                if doc_id not in parse_engine_lookup:
                    logger.debug(
                        "Parse engine defaulted to docling-html",
                        extra={"extra_fields": {"doc_id": doc_id}},
                    )

                if resume_needs_hash and manifest_entry and output_exists:
                    skip_doc, _ = resume_controller.should_skip(doc_id, out_path, input_hash)
                else:
                    skip_doc = False
                if skip_doc:
                    log_event(
                        logger,
                        "info",
                        "Skipping chunk file: output exists and input unchanged",
                        status="skip",
                        stage=CHUNK_STAGE,
                        doc_id=doc_id,
                        input_relpath=relative_path(path, resolved_data_root),
                        output_relpath=relative_path(out_path, resolved_data_root),
                    )
                    manifest_log_skip(
                        stage=MANIFEST_STAGE,
                        doc_id=doc_id,
                        input_path=path,
                        input_hash=input_hash,
                        output_path=out_path,
                        schema_version=CHUNK_SCHEMA_VERSION,
                        reason="unchanged-input",
                        parse_engine=parse_engine,
                    )
                    continue

                yield ChunkTask(
                    doc_path=path,
                    output_path=out_path,
                    doc_id=doc_id,
                    doc_stem=name,
                    input_hash=input_hash,
                    parse_engine=parse_engine,
                )

        task_iterator = iter_chunk_tasks()
        try:
            first_task = next(task_iterator)
        except StopIteration:
            return 0

        def handle_result(result: ChunkResult) -> None:
            """Persist manifest information and raise on worker failure.

            Args:
                result: Structured outcome emitted by the chunking worker.
            """
            duration = round(result.duration_s, 3)
            if result.status != "success":
                error_message = result.error or "unknown error"
                failure_metadata = {
                    "status": "failure",
                    "duration_s": duration,
                    "schema_version": CHUNK_SCHEMA_VERSION,
                    "input_path": str(result.input_path),
                    "input_hash": result.input_hash,
                    "hash_alg": resolve_hash_algorithm(),
                    "output_path": str(result.output_path),
                    "parse_engine": result.parse_engine,
                    "chunk_count": result.chunk_count,
                    "anchors_injected": result.anchors_injected,
                    "sanitizer_profile": result.sanitizer_profile,
                    "error": error_message,
                }
                log_event(
                    logger,
                    "error",
                    "Chunking failed",
                    status="failure",
                    stage=CHUNK_STAGE,
                    doc_id=result.doc_id,
                    input_relpath=relative_path(result.input_path, resolved_data_root),
                    output_relpath=relative_path(result.output_path, resolved_data_root),
                    elapsed_ms=int(result.duration_s * 1000),
                    error_class="RuntimeError",
                    error=error_message,
                )
                stage_telemetry.log_failure(
                    doc_id=result.doc_id,
                    input_path=result.input_path,
                    duration_s=duration,
                    reason=error_message,
                    metadata=failure_metadata.copy(),
                    manifest_metadata=failure_metadata,
                )
                raise RuntimeError(
                    f"Chunking failed for {result.doc_id}: {error_message}"
                )

            log_event(
                logger,
                "info",
                "Chunk file written",
                status="success",
                stage=CHUNK_STAGE,
                doc_id=result.doc_id,
                input_relpath=relative_path(result.input_path, resolved_data_root),
                output_relpath=relative_path(result.output_path, resolved_data_root),
                elapsed_ms=int(result.duration_s * 1000),
                chunk_count=result.chunk_count,
                parse_engine=result.parse_engine,
            )
            stage_telemetry.log_success(
                doc_id=result.doc_id,
                input_path=result.input_path,
                output_path=result.output_path,
                tokens=result.total_tokens,
                schema_version=CHUNK_SCHEMA_VERSION,
                duration_s=duration,
                metadata={
                    "status": "success",
                    "input_path": str(result.input_path),
                    "input_hash": result.input_hash,
                    "chunk_count": result.chunk_count,
                    "total_tokens": result.total_tokens,
                    "parse_engine": result.parse_engine,
                    "hash_alg": resolve_hash_algorithm(),
                    "anchors_injected": result.anchors_injected,
                    "sanitizer_profile": result.sanitizer_profile,
                },
            )

        if worker_count == 1:
            _chunk_worker_initializer(chunk_config)
            for task in itertools.chain((first_task,), task_iterator):
                handle_result(_process_chunk_task(task))
        else:
            logger.info(
                "Parallel chunking enabled",
                extra={"extra_fields": {"workers": worker_count}},
            )
            ctx = get_context("spawn")
            with ctx.Pool(
                processes=worker_count,
                initializer=_chunk_worker_initializer,
                initargs=(chunk_config,),
            ) as pool:
                indexed_results = pool.imap_unordered(
                    _process_indexed_chunk_task,
                    enumerate(itertools.chain((first_task,), task_iterator)),
                    chunksize=1,
                )
                for result in _ordered_results(indexed_results):
                    handle_result(result)

        return 0


def _run_validate_only(
    *,
    files: Sequence[Path],
    logger,
    cfg: ChunkerCfg,
    tokenizer_model: str,
    heading_markers: Tuple[str, ...],
    caption_markers: Tuple[str, ...],
    data_root: Optional[Path],
    in_dir: Path,
    out_dir: Path,
    telemetry: StageTelemetry,
) -> None:
    """Validate chunk inputs and report statistics without writing outputs."""

    stats = _validate_chunk_files(
        files,
        logger,
        data_root=data_root,
        telemetry=telemetry,
    )
    logger.bind(mode="validate-only")

    if not stats["files"]:
        log_event(
            logger,
            "info",
            "No chunk files validated",
            status="validate-only",
            stage=CHUNK_STAGE,
            doc_id="__aggregate__",
            input_hash=None,
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
    token_counts: List[int] = []
    boundary_violations = 0
    heading_hits = 0
    caption_hits = 0

    for path in files:
        if not path.exists():
            continue
        doctags_text = read_utf8(path)
        doc = build_doc(doc_name=path.stem, doctags_text=doctags_text)
        chunks = list(chunker.chunk(dl_doc=doc))
        total_chunks += len(chunks)
        recs: List[Rec] = []
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


def main(args: argparse.Namespace | SimpleNamespace | Sequence[str] | None = None) -> int:
    """Wrapper that normalises CLI validation failures for the chunk stage."""

    try:
        return _main_inner(args)
    except ChunkingCLIValidationError as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
