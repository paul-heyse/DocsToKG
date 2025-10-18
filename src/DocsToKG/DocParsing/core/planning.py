"""Planner orchestration utilities for DocParsing stages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

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

from .cli_utils import preview_list
from .discovery import (
    derive_doc_id_and_chunks_path,
    derive_doc_id_and_doctags_path,
    derive_doc_id_and_vectors_path,
    iter_chunks,
)
from .manifest import ResumeController

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
            "process": [],
            "skip": [],
            "notes": ["Input directory missing"],
        }

    if mode == "html":
        files = doctags_module.list_htmls(input_dir)
        manifest_stage = getattr(doctags_module, "HTML_MANIFEST_STAGE", "doctags-html")
        overwrite = bool(getattr(args, "overwrite", False))
    else:
        files = doctags_module.list_pdfs(input_dir)
        manifest_stage = doctags_module.MANIFEST_STAGE
        overwrite = False

    manifest_index = (
        load_manifest_index(manifest_stage, resolved_root) if args.resume else {}
    )
    resume_controller = ResumeController(args.resume, args.force, manifest_index)
    planned: List[str] = []
    skipped: List[str] = []

    for path in files:
        doc_id, out_path = derive_doc_id_and_doctags_path(path, input_dir, output_dir)
        input_hash = compute_content_hash(path)
        skip, _ = resume_controller.should_skip(doc_id, out_path, input_hash)
        if mode == "html" and overwrite:
            skip = False
        if skip:
            skipped.append(doc_id)
        else:
            planned.append(doc_id)

    return {
        "stage": "doctags",
        "mode": mode,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "process": planned,
        "skip": skipped,
        "notes": [],
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
            "process": [],
            "skip": [],
            "notes": ["DocTags directory missing"],
        }

    files = list(iter_doctags(in_dir))
    manifest_index = (
        load_manifest_index(chunk_module.MANIFEST_STAGE, resolved_root) if args.resume else {}
    )
    resume_controller = ResumeController(args.resume, args.force, manifest_index)
    planned: List[str] = []
    skipped: List[str] = []

    for path in files:
        rel_id, out_path = derive_doc_id_and_chunks_path(path, in_dir, out_dir)
        input_hash = compute_content_hash(path)
        skip, _ = resume_controller.should_skip(rel_id, out_path, input_hash)
        if skip:
            skipped.append(rel_id)
        else:
            planned.append(rel_id)

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

    if args.validate_only:
        validate: List[str] = []
        missing: List[str] = []
        for chunk_path in iter_chunks(chunks_dir):
            doc_id, vector_path = derive_doc_id_and_vectors_path(
                chunk_path, chunks_dir, vectors_dir
            )
            if vector_path.exists():
                validate.append(doc_id)
            else:
                missing.append(doc_id)
        return {
            "stage": "embed",
            "action": "validate",
            "chunks_dir": str(chunks_dir),
            "vectors_dir": str(vectors_dir),
            "validate": validate,
            "missing": missing,
            "notes": [],
        }

    files = list(iter_chunks(chunks_dir))
    manifest_index = (
        load_manifest_index(embedding_module.MANIFEST_STAGE, resolved_root) if args.resume else {}
    )
    resume_controller = ResumeController(args.resume, args.force, manifest_index)
    planned: List[str] = []
    skipped: List[str] = []

    for chunk_path in files:
        doc_id, vector_path = derive_doc_id_and_vectors_path(chunk_path, chunks_dir, vectors_dir)
        input_hash = compute_content_hash(chunk_path)
        skip, _ = resume_controller.should_skip(doc_id, vector_path, input_hash)
        if skip:
            skipped.append(doc_id)
        else:
            planned.append(doc_id)

    return {
        "stage": "embed",
        "action": "generate",
        "chunks_dir": str(chunks_dir),
        "vectors_dir": str(vectors_dir),
        "process": planned,
        "skip": skipped,
        "notes": [],
    }


def display_plan(plans: Sequence[Dict[str, Any]], stream: Optional[TextIO] = None) -> List[str]:
    """Pretty-print plan summaries and return the rendered lines."""

    lines: List[str] = ["docparse all plan"]
    for entry in plans:
        stage = entry.get("stage", "unknown")
        notes = entry.get("notes", [])
        if stage == "doctags":
            desc = f"doctags (mode={entry.get('mode')})"
            process = entry.get("process", [])
            skip = entry.get("skip", [])
            lines.append(f"- {desc}: process {len(process)}, skip {len(skip)}")
            lines.append(f"  input:  {entry.get('input_dir')}")
            lines.append(f"  output: {entry.get('output_dir')}")
            if process:
                lines.append("  process preview: " + ", ".join(preview_list(process)))
            if skip:
                lines.append("  skip preview: " + ", ".join(preview_list(skip)))
        elif stage == "chunk":
            process = entry.get("process", [])
            skip = entry.get("skip", [])
            lines.append(f"- chunk: process {len(process)}, skip {len(skip)}")
            lines.append(f"  input:  {entry.get('input_dir')}")
            lines.append(f"  output: {entry.get('output_dir')}")
            if process:
                lines.append("  process preview: " + ", ".join(preview_list(process)))
            if skip:
                lines.append("  skip preview: " + ", ".join(preview_list(skip)))
        elif stage == "embed" and entry.get("action") == "validate":
            validate = entry.get("validate", [])
            missing = entry.get("missing", [])
            lines.append(
                f"- embed (validate-only): validate {len(validate)}, missing vectors {len(missing)}"
            )
            lines.append(f"  chunks:  {entry.get('chunks_dir')}")
            lines.append(f"  vectors: {entry.get('vectors_dir')}")
            if validate:
                lines.append("  validate preview: " + ", ".join(preview_list(validate)))
            if missing:
                lines.append("  missing preview: " + ", ".join(preview_list(missing)))
        elif stage == "embed":
            process = entry.get("process", [])
            skip = entry.get("skip", [])
            lines.append(f"- embed: process {len(process)}, skip {len(skip)}")
            lines.append(f"  chunks:  {entry.get('chunks_dir')}")
            lines.append(f"  vectors: {entry.get('vectors_dir')}")
            if process:
                lines.append("  process preview: " + ", ".join(preview_list(process)))
            if skip:
                lines.append("  skip preview: " + ", ".join(preview_list(skip)))
        else:
            lines.append(f"- {stage}: no actionable items")
        if notes:
            lines.append("  notes: " + "; ".join(notes))
    lines.append("")

    output = stream or sys.stdout
    for line in lines:
        print(line, file=output)
    return lines
