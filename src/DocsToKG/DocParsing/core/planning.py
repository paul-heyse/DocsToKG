"""Planner orchestration utilities for DocParsing stages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

from DocsToKG.DocParsing.cli_errors import DoctagsCLIValidationError
from DocsToKG.DocParsing.env import (
    data_chunks,
    data_doctags,
    data_vectors,
    detect_data_root,
)
from DocsToKG.DocParsing.io import (
    compute_content_hash,
    iter_doctags,
    load_manifest_index,
)

from .discovery import (
    derive_doc_id_and_chunks_path,
    derive_doc_id_and_doctags_path,
    derive_doc_id_and_vectors_path,
    iter_chunks,
)
from .manifest import ResumeController, should_skip_output

PLAN_PREVIEW_LIMIT = 5


def _new_bucket() -> Dict[str, Any]:
    """Return a new mutable bucket for tracking plan membership."""

    return {"count": 0, "preview": []}


def _record_bucket(bucket: Dict[str, Any], doc_id: str) -> None:
    """Update ``bucket`` with ``doc_id`` while respecting preview bounds."""

    preview: List[str] = bucket.setdefault("preview", [])
    if len(preview) < PLAN_PREVIEW_LIMIT:
        preview.append(doc_id)
    bucket["count"] = bucket.get("count", 0) + 1


def _bucket_counts(entry: Dict[str, Any], key: str) -> Tuple[int, List[str]]:
    """Return ``(count, preview)`` for ``key`` within ``entry``."""

    value = entry.get(key)
    if isinstance(value, dict):
        count = int(value.get("count", 0))
        preview_list_value = value.get("preview", [])
        if isinstance(preview_list_value, (list, tuple)):
            preview = list(preview_list_value[:PLAN_PREVIEW_LIMIT])
        else:
            preview = []
        return count, preview
    if not value:
        return 0, []
    items = list(value)
    count = len(items)
    preview = items[:PLAN_PREVIEW_LIMIT]
    return count, preview


def _render_preview(preview: List[str], count: int) -> str:
    """Render a preview string that includes remainder hints when applicable."""

    items = list(preview)
    remainder = max(0, count - len(preview))
    if remainder:
        items.append(f"... (+{remainder} more)")
    return ", ".join(items)

__all__ = [
    "display_plan",
    "plan_chunk",
    "plan_doctags",
    "plan_embed",
]


def plan_doctags(argv: Sequence[str]) -> Dict[str, Any]:
    """Compute which DocTags inputs would be processed."""

    from DocsToKG.DocParsing import doctags as doctags_module

    from .cli import _resolve_doctags_paths, build_doctags_parser

    parser = build_doctags_parser()

    args, _unknown = parser.parse_known_args(argv)
    raw_log_level = getattr(args, "log_level", None)
    log_level = str(raw_log_level).upper() if raw_log_level is not None else None

    try:
        mode, input_dir, output_dir, resolved_root_str = _resolve_doctags_paths(args)
    except DoctagsCLIValidationError as exc:
        hint_suffix = f" (Hint: {exc.hint})" if exc.hint else ""
        return {
            "stage": "doctags",
            "mode": args.mode,
            "input_dir": str(args.in_dir) if args.in_dir is not None else None,
            "output_dir": str(args.out_dir) if args.out_dir is not None else None,
            "process": [],
            "skip": [],
            "notes": [f"{exc.message}{hint_suffix}"],
            "log_level": log_level,
            "error": {
                "option": exc.option,
                "message": exc.message,
                "hint": exc.hint,
            },
        }

    resolved_root = Path(resolved_root_str)

    if not input_dir.exists():
        return {
            "stage": "doctags",
            "mode": mode,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "process": _new_bucket(),
            "skip": _new_bucket(),
            "notes": ["Input directory missing"],
            "log_level": log_level,
        }

    if mode == "html":
        iter_htmls = getattr(doctags_module, "iter_htmls", None)
        files = (
            iter_htmls(input_dir)
            if callable(iter_htmls)
            else iter(doctags_module.list_htmls(input_dir))
        )
        manifest_stage = getattr(doctags_module, "HTML_MANIFEST_STAGE", "doctags-html")
        overwrite = bool(getattr(args, "overwrite", False))
    else:
        iter_pdfs = getattr(doctags_module, "iter_pdfs", None)
        files = (
            iter_pdfs(input_dir)
            if callable(iter_pdfs)
            else iter(doctags_module.list_pdfs(input_dir))
        )
        manifest_stage = doctags_module.MANIFEST_STAGE
        overwrite = False

    manifest_index = (
        load_manifest_index(manifest_stage, resolved_root) if args.resume else {}
    )
    resume_controller = ResumeController(args.resume, args.force, manifest_index)
    planned = _new_bucket()
    skipped = _new_bucket()

    for path in files:
        doc_id, out_path = derive_doc_id_and_doctags_path(path, input_dir, output_dir)
        manifest_entry = resume_controller.entry(doc_id)
        should_hash = bool(args.resume and not args.force and manifest_entry)
        skip = False
        if should_hash:
            input_hash = compute_content_hash(path)
            skip = should_skip_output(
                out_path,
                manifest_entry,
                input_hash,
                resume_controller.resume,
                resume_controller.force,
            )
        if mode == "html" and overwrite:
            skip = False
        if skip:
            _record_bucket(skipped, doc_id)
        else:
            _record_bucket(planned, doc_id)

    return {
        "stage": "doctags",
        "mode": mode,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "process": planned,
        "skip": skipped,
        "notes": [],
        "log_level": log_level,
    }


def plan_chunk(argv: Sequence[str]) -> Dict[str, Any]:
    """Compute which DocTags files the chunk stage would touch."""

    from DocsToKG.DocParsing import chunking as chunk_module
    from DocsToKG.DocParsing import doctags as doctags_module

    parser = chunk_module.build_parser()
    args, _unknown = parser.parse_known_args(argv)
    resolved_root = doctags_module.prepare_data_root(args.data_root, detect_data_root())
    data_root_overridden = args.data_root is not None

    default_in_dir = data_doctags(resolved_root, ensure=False)
    default_out_dir = data_chunks(resolved_root, ensure=False)

    in_dir = doctags_module.resolve_pipeline_path(
        cli_value=args.in_dir,
        default_path=default_in_dir,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_doctags(root, ensure=False),
    ).resolve()

    out_dir = doctags_module.resolve_pipeline_path(
        cli_value=args.out_dir,
        default_path=default_out_dir,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_chunks(root, ensure=False),
    ).resolve()

    if not in_dir.exists():
        return {
            "stage": "chunk",
            "input_dir": str(in_dir),
            "output_dir": str(out_dir),
            "process": _new_bucket(),
            "skip": _new_bucket(),
            "notes": ["DocTags directory missing"],
        }

    manifest_index = (
        load_manifest_index(chunk_module.MANIFEST_STAGE, resolved_root) if args.resume else {}
    )
    resume_controller = ResumeController(args.resume, args.force, manifest_index)
    planned = _new_bucket()
    skipped = _new_bucket()

    for path in iter_doctags(in_dir):
        rel_id, out_path = derive_doc_id_and_chunks_path(path, in_dir, out_dir)
        manifest_entry = resume_controller.entry(rel_id)
        should_hash = bool(args.resume and not args.force and manifest_entry)
        skip = False
        if should_hash:
            if not out_path.exists():
                _record_bucket(planned, rel_id)
                continue
            input_hash = compute_content_hash(path)
            skip = should_skip_output(
                out_path,
                manifest_entry,
                input_hash,
                resume_controller.resume,
                resume_controller.force,
            )
        if skip:
            _record_bucket(skipped, rel_id)
        else:
            _record_bucket(planned, rel_id)

    return {
        "stage": "chunk",
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "process": planned,
        "skip": skipped,
        "notes": [],
    }


def plan_embed(argv: Sequence[str]) -> Dict[str, Any]:
    """Compute which chunk files the embed stage would process or validate."""

    from DocsToKG.DocParsing import doctags as doctags_module
    from DocsToKG.DocParsing import embedding as embedding_module

    parser = embedding_module.build_parser()
    args, _unknown = parser.parse_known_args(argv)
    resolved_root = doctags_module.prepare_data_root(args.data_root, detect_data_root())
    data_root_overridden = args.data_root is not None

    default_chunks_dir = data_chunks(resolved_root, ensure=False)
    default_vectors_dir = data_vectors(resolved_root, ensure=False)

    chunks_dir = doctags_module.resolve_pipeline_path(
        cli_value=args.chunks_dir,
        default_path=default_chunks_dir,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_chunks(root, ensure=False),
    ).resolve()

    vectors_dir = doctags_module.resolve_pipeline_path(
        cli_value=args.out_dir,
        default_path=default_vectors_dir,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_vectors(root, ensure=False),
    ).resolve()

    chunks_missing = not chunks_dir.exists()
    vectors_missing = not vectors_dir.exists()

    if args.validate_only:
        validate_bucket = _new_bucket()
        missing_bucket = _new_bucket()
        notes: List[str] = []
        if chunks_missing:
            notes.append("Chunks directory missing")
        if vectors_missing:
            notes.append("Vectors directory missing")
        if notes:
            return {
                "stage": "embed",
                "action": "validate",
                "chunks_dir": str(chunks_dir),
                "vectors_dir": str(vectors_dir),
                "validate": validate_bucket,
                "missing": missing_bucket,
                "notes": notes,
            }
        for chunk in iter_chunks(chunks_dir):
            doc_id, vector_path = derive_doc_id_and_vectors_path(
                chunk, chunks_dir, vectors_dir
            )
            if vector_path.exists():
                _record_bucket(validate_bucket, doc_id)
            else:
                _record_bucket(missing_bucket, doc_id)
        return {
            "stage": "embed",
            "action": "validate",
            "chunks_dir": str(chunks_dir),
            "vectors_dir": str(vectors_dir),
            "validate": validate_bucket,
            "missing": missing_bucket,
            "notes": [],
        }

    manifest_index = (
        load_manifest_index(embedding_module.MANIFEST_STAGE, resolved_root) if args.resume else {}
    )
    resume_controller = ResumeController(args.resume, args.force, manifest_index)
    planned = _new_bucket()
    skipped = _new_bucket()

    notes: List[str] = []
    if chunks_missing:
        return {
            "stage": "embed",
            "action": "generate",
            "chunks_dir": str(chunks_dir),
            "vectors_dir": str(vectors_dir),
            "process": _new_bucket(),
            "skip": _new_bucket(),
            "notes": ["Chunks directory missing"],
        }
    if vectors_missing:
        notes.append("Vectors directory not found; outputs will be created during generation")

    files = iter_chunks(chunks_dir)
    for chunk in files:
        doc_id, vector_path = derive_doc_id_and_vectors_path(
            chunk, chunks_dir, vectors_dir
        )
        manifest_entry = resume_controller.entry(doc_id)
        should_hash = bool(args.resume and not args.force and manifest_entry)
        skip = False
        if should_hash:
            if not vector_path.exists():
                _record_bucket(planned, doc_id)
                continue
            input_hash = compute_content_hash(chunk.resolved_path)
            skip = should_skip_output(
                vector_path,
                manifest_entry,
                input_hash,
                resume_controller.resume,
                resume_controller.force,
            )
        if skip:
            _record_bucket(skipped, doc_id)
        else:
            _record_bucket(planned, doc_id)

    return {
        "stage": "embed",
        "action": "generate",
        "chunks_dir": str(chunks_dir),
        "vectors_dir": str(vectors_dir),
        "process": planned,
        "skip": skipped,
        "notes": notes,
    }


def display_plan(plans: Sequence[Dict[str, Any]], stream: Optional[TextIO] = None) -> List[str]:
    """Pretty-print plan summaries and return the rendered lines."""

    lines: List[str] = ["docparse all plan"]
    for entry in plans:
        stage = entry.get("stage", "unknown")
        notes = entry.get("notes", [])
        if stage == "doctags":
            desc = f"doctags (mode={entry.get('mode')})"
            process_count, process_preview = _bucket_counts(entry, "process")
            skip_count, skip_preview = _bucket_counts(entry, "skip")
            lines.append(f"- {desc}: process {process_count}, skip {skip_count}")
            lines.append(f"  input:  {entry.get('input_dir')}")
            lines.append(f"  output: {entry.get('output_dir')}")
            log_level_value = entry.get("log_level")
            if log_level_value:
                lines.append(f"  log_level: {log_level_value}")
            if process_count:
                lines.append(
                    "  process preview: "
                    + _render_preview(process_preview, process_count)
                )
            if skip_count:
                lines.append(
                    "  skip preview: " + _render_preview(skip_preview, skip_count)
                )
        elif stage == "chunk":
            process_count, process_preview = _bucket_counts(entry, "process")
            skip_count, skip_preview = _bucket_counts(entry, "skip")
            lines.append(f"- chunk: process {process_count}, skip {skip_count}")
            lines.append(f"  input:  {entry.get('input_dir')}")
            lines.append(f"  output: {entry.get('output_dir')}")
            if process_count:
                lines.append(
                    "  process preview: "
                    + _render_preview(process_preview, process_count)
                )
            if skip_count:
                lines.append(
                    "  skip preview: " + _render_preview(skip_preview, skip_count)
                )
        elif stage == "embed" and entry.get("action") == "validate":
            validate_count, validate_preview = _bucket_counts(entry, "validate")
            missing_count, missing_preview = _bucket_counts(entry, "missing")
            lines.append(
                "- embed (validate-only): validate"
                f" {validate_count}, missing vectors {missing_count}"
            )
            lines.append(f"  chunks:  {entry.get('chunks_dir')}")
            lines.append(f"  vectors: {entry.get('vectors_dir')}")
            if validate_count:
                lines.append(
                    "  validate preview: "
                    + _render_preview(validate_preview, validate_count)
                )
            if missing_count:
                lines.append(
                    "  missing preview: "
                    + _render_preview(missing_preview, missing_count)
                )
        elif stage == "embed":
            process_count, process_preview = _bucket_counts(entry, "process")
            skip_count, skip_preview = _bucket_counts(entry, "skip")
            lines.append(f"- embed: process {process_count}, skip {skip_count}")
            lines.append(f"  chunks:  {entry.get('chunks_dir')}")
            lines.append(f"  vectors: {entry.get('vectors_dir')}")
            if process_count:
                lines.append(
                    "  process preview: "
                    + _render_preview(process_preview, process_count)
                )
            if skip_count:
                lines.append(
                    "  skip preview: " + _render_preview(skip_preview, skip_count)
                )
        else:
            lines.append(f"- {stage}: no actionable items")
        if notes:
            lines.append("  notes: " + "; ".join(notes))
    lines.append("")

    output = stream or sys.stdout
    for line in lines:
        print(line, file=output)
    return lines
