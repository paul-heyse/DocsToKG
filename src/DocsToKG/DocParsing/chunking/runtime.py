#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.chunking",
#   "purpose": "CLI entry points for DocsToKG.DocParsing.chunking workflows",
#   "sections": [
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
#       "id": "chunkercfg",
#       "name": "ChunkerCfg",
#       "anchor": "class-chunkercfg",
#       "kind": "class"
#     },
#     {
#       "id": "chunk-log-context",
#       "name": "_chunk_log_context",
#       "anchor": "function-chunk-log-context",
#       "kind": "function"
#     },
#     {
#       "id": "log-chunk-metadata-issue",
#       "name": "_log_chunk_metadata_issue",
#       "anchor": "function-log-chunk-metadata-issue",
#       "kind": "function"
#     },
#     {
#       "id": "collect-doc-items",
#       "name": "_collect_doc_items",
#       "anchor": "function-collect-doc-items",
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
#       "id": "extract-chunk-start",
#       "name": "_extract_chunk_start",
#       "anchor": "function-extract-chunk-start",
#       "kind": "function"
#     },
#     {
#       "id": "rec",
#       "name": "Rec",
#       "anchor": "class-rec",
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
#     },
#     {
#       "id": "run-validate-only",
#       "name": "_run_validate_only",
#       "anchor": "function-run-validate-only",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Docling Hybrid Chunker with Minimum Token Coalescence

Transforms DocTags documents into chunked records with topic-aware coalescence.
The module exposes a CLI (`python -m DocsToKG.DocParsing.chunking`)
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
    python -m DocsToKG.DocParsing.chunking \\
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
import json
import logging
import os
import statistics
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields
from types import SimpleNamespace
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

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
    acquire_lock,
    compute_relative_doc_id,
    compute_stable_shard,
    dedupe_preserve_order,
    derive_doc_id_and_chunks_path,
    load_structural_marker_config,
    set_spawn_or_warn,
)
from DocsToKG.DocParsing.doctags import (
    add_data_root_option,
    add_resume_force_options,
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
    quarantine_artifact,
    relative_path,
    resolve_attempts_path,
    resolve_hash_algorithm,
    resolve_manifest_path,
)
from DocsToKG.DocParsing.logging import (
    get_logger,
    log_event,
    telemetry_scope,
)
from DocsToKG.DocParsing.telemetry import StageTelemetry, TelemetrySink

from .cli import build_parser, parse_args
from .config import CHUNK_PROFILE_PRESETS, ChunkerCfg, SOFT_BARRIER_MARGIN

CHUNK_STAGE = "chunking"


def _resolve_serializer_provider(spec: str) -> type[ChunkingSerializerProvider]:
    """Return the serializer provider class referenced by ``spec``."""

    if ":" not in spec:
        raise ValueError(
            "Serializer provider must be specified as 'module:Class', received %r" % (spec,)
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
                    },
                    manifest_metadata={
                        "output_path": str(quarantine_path),
                        "schema_version": CHUNK_SCHEMA_VERSION,
                        "input_path": str(path),
                        "input_hash": input_hash,
                        "error": reason,
                        "quarantine": True,
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


def main(args: argparse.Namespace | SimpleNamespace | Sequence[str] | None = None) -> int:
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
        logging.getLogger(__name__).debug(
            "Failed to bootstrap chunking directories", exc_info=exc
        )
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
        raise ValueError("--min-tokens and --max-tokens must be non-negative")
    if args.min_tokens > args.max_tokens:
        log_event(
            logger,
            "error",
            "Invalid token range",
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        raise ValueError("--min-tokens must be less than or equal to --max-tokens")
    if args.soft_barrier_margin < 0:
        log_event(
            logger,
            "error",
            "Soft barrier margin must be non-negative",
            soft_barrier_margin=args.soft_barrier_margin,
        )
        raise ValueError("--soft-barrier-margin must be >= 0")

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
        raise ValueError(f"Invalid serializer provider '{serializer_spec}': {exc}") from exc

    try:
        min_tokens = int(args.min_tokens)
        max_tokens = int(args.max_tokens)
        soft_margin = int(args.soft_barrier_margin)
    except (TypeError, ValueError) as exc:
        raise ValueError("Token thresholds must be integers") from exc

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
        raise ValueError("--shard-count and --shard-index must be integers") from exc
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index must be between 0 and shard-count-1")
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
    stage_telemetry = StageTelemetry(telemetry_sink, run_id=run_id, stage=CHUNK_STAGE)
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

        tasks: List[ChunkTask] = []
        for path in files:
            doc_id, out_path = derive_doc_id_and_chunks_path(path, in_dir, out_dir)
            name = path.stem
            input_hash = compute_content_hash(path)
            parse_engine = parse_engine_lookup.get(doc_id, "docling-html")
            if doc_id not in parse_engine_lookup:
                logger.debug(
                    "Parse engine defaulted to docling-html",
                    extra={"extra_fields": {"doc_id": doc_id}},
                )

            skip_doc, manifest_entry = resume_controller.should_skip(doc_id, out_path, input_hash)
            if skip_doc:
                stage_telemetry.log_skip(
                    doc_id=doc_id,
                    input_path=path,
                    reason="unchanged-input",
                    metadata={
                        "input_path": str(path),
                        "input_hash": input_hash,
                        "output_path": str(out_path),
                        "schema_version": CHUNK_SCHEMA_VERSION,
                        "parse_engine": parse_engine,
                    },
                )
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
                continue

            tasks.append(
                ChunkTask(
                    doc_path=path,
                    output_path=out_path,
                    doc_id=doc_id,
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
                    error=result.error or "unknown error",
                )
                stage_telemetry.log_failure(
                    doc_id=result.doc_id,
                    input_path=result.input_path,
                    duration_s=duration,
                    reason=result.error or "unknown error",
                    metadata={
                        "input_path": str(result.input_path),
                        "input_hash": result.input_hash,
                        "output_path": str(result.output_path),
                        "parse_engine": result.parse_engine,
                    },
                )
                raise RuntimeError(
                    f"Chunking failed for {result.doc_id}: {result.error or 'unknown error'}"
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
                tokens=result.chunk_count,
                schema_version=CHUNK_SCHEMA_VERSION,
                duration_s=duration,
                metadata={
                    "input_path": str(result.input_path),
                    "input_hash": result.input_hash,
                    "chunk_count": result.chunk_count,
                    "parse_engine": result.parse_engine,
                    "hash_alg": resolve_hash_algorithm(),
                    "anchors_injected": result.anchors_injected,
                    "sanitizer_profile": result.sanitizer_profile,
                },
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
                1 for rec in coalesced if rec.text.strip().startswith(heading_markers)
            )
            heading_hits += sum(1 for rec in coalesced if rec.text.strip().startswith(heading_markers))
            caption_hits += sum(
                1
                for rec in coalesced
                if any(marker in rec.text for marker in caption_markers if marker)
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
