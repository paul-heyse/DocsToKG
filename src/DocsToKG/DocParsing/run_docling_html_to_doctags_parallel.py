#!/usr/bin/env python3
"""Parallel HTML → DocTags conversion with manifest-aware resume support."""

from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

from tqdm import tqdm

from DocsToKG.DocParsing._common import (
    compute_content_hash,
    data_doctags,
    data_html,
    data_manifests,
    detect_data_root,
    get_logger,
    load_manifest_index,
    manifest_append,
)

DEFAULT_INPUT_DIR = data_html()
DEFAULT_OUTPUT_DIR = data_doctags()
MANIFEST_STAGE = "doctags-html"

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

warnings.warn(
    "Direct invocation of run_docling_html_to_doctags_parallel.py is deprecated. "
    "Use unified CLI: python -m DocsToKG.DocParsing.cli.doctags_convert --mode html",
    DeprecationWarning,
    stacklevel=2,
)

_LOGGER = get_logger(__name__)

# keep numeric libs polite; also ensure nothing touches CUDA by mistake
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only

# per-process converter cache
_CONVERTER = None


def build_parser() -> argparse.ArgumentParser:
    """Construct an argument parser for the HTML → DocTags converter.

    Args:
        None: Parser initialization does not require inputs.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.

    Raises:
        None
    """

    parser = argparse.ArgumentParser(
        description="Convert HTML corpora to DocTags using Docling",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Override DocsToKG Data directory. Defaults to auto-detection or "
            "$DOCSTOKG_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder with HTML files (recurses)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for .doctags",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) - 1),
        help="Parallel workers",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing .doctags files"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip documents whose outputs already exist with matching content hash",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even when resume criteria are satisfied",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional CLI argument vector. When ``None`` the values from
            :data:`sys.argv` are used.

    Returns:
        Namespace containing parsed CLI options.

    Raises:
        SystemExit: Propagated if ``argparse`` detects invalid options.
    """

    return build_parser().parse_args(argv)


@dataclass
class HtmlTask:
    """Work item describing a single HTML conversion job.

    Attributes:
        html_path: Absolute path to the HTML file to be converted.
        relative_id: Relative identifier for manifest entries.
        output_path: Destination DocTags path.
        input_hash: Content hash used for resume detection.
        overwrite: Flag indicating whether existing outputs should be replaced.

    Examples:
        >>> HtmlTask(Path("/tmp/a.html"), "doc", Path("/tmp/doc.doctags"), "hash", False)
        HtmlTask(html_path=PosixPath('/tmp/a.html'), relative_id='doc', output_path=PosixPath('/tmp/doc.doctags'), input_hash='hash', overwrite=False)
    """

    html_path: Path
    relative_id: str
    output_path: Path
    input_hash: str
    overwrite: bool


@dataclass
class ConversionResult:
    """Structured result emitted by worker processes.

    Attributes:
        doc_id: Document identifier matching manifest entries.
        status: Conversion outcome (``"success"``, ``"skip"``, or ``"failure"``).
        duration_s: Time in seconds spent converting.
        input_path: Source HTML path recorded for auditing.
        input_hash: Content hash captured prior to conversion.
        output_path: Destination DocTags path.
        error: Optional error detail for failures.

    Examples:
        >>> ConversionResult("doc", "success", 1.0, "in.html", "hash", "out.doctags")
        ConversionResult(doc_id='doc', status='success', duration_s=1.0, input_path='in.html', input_hash='hash', output_path='out.doctags', error=None)
    """

    doc_id: str
    status: str
    duration_s: float
    input_path: str
    input_hash: str
    output_path: str
    error: str | None = None


def _get_converter() -> "DocumentConverter":
    """Instantiate and cache a Docling HTML converter per worker process.

    Returns:
        DocumentConverter configured for HTML input, cached for reuse within
        the worker process.
    """
    from docling.backend.html_backend import HTMLDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, HTMLFormatOption

    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = DocumentConverter(
            format_options={InputFormat.HTML: HTMLFormatOption(backend=HTMLDocumentBackend)}
        )
    return _CONVERTER


def list_htmls(root: Path) -> List[Path]:
    """Enumerate HTML-like files beneath a directory tree.

    Args:
        root: Directory whose subtree should be searched for HTML files.

    Returns:
        Sorted list of discovered HTML file paths excluding normalized outputs.
    """
    exts = {".html", ".htm", ".xhtml"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and not p.name.endswith(".normalized.html"):
            out.append(p)
    return sorted(out)


def convert_one(task: HtmlTask) -> ConversionResult:
    """Convert a single HTML file to DocTags, honoring overwrite semantics.

    Args:
        task: Conversion details including paths, hash, and overwrite policy.

    Returns:
        :class:`ConversionResult` capturing the conversion status.

    Raises:
        ValueError: Propagated when Docling validation fails prior to internal handling.
    """

    start = time.perf_counter()
    try:
        out_path = task.output_path
        if out_path.exists() and not task.overwrite:
            return ConversionResult(
                doc_id=task.relative_id,
                status="skip",
                duration_s=0.0,
                input_path=str(task.html_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
            )

        converter = _get_converter()
        result = converter.convert(task.html_path, raises_on_error=False)

        if result.document is None:
            return ConversionResult(
                doc_id=task.relative_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(task.html_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error="empty-document",
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.document.save_as_doctags(out_path)
        return ConversionResult(
            doc_id=task.relative_id,
            status="success",
            duration_s=time.perf_counter() - start,
            input_path=str(task.html_path),
            input_hash=task.input_hash,
            output_path=str(out_path),
        )

    except Exception as exc:  # pragma: no cover - integration failure path
        return ConversionResult(
            doc_id=task.relative_id,
            status="failure",
            duration_s=time.perf_counter() - start,
            input_path=str(task.html_path),
            input_hash=task.input_hash,
            output_path=str(task.output_path),
            error=str(exc),
        )


def main(args: argparse.Namespace | None = None) -> int:
    """Entrypoint for parallel HTML-to-DocTags conversion across a dataset.

    Args:
        args: Optional pre-parsed CLI namespace to override command-line inputs.

    Returns:
        Process exit code, where ``0`` denotes success.
    """

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    parser = build_parser()
    defaults = parser.parse_args([])
    provided = parse_args() if args is None else args
    for key, value in vars(provided).items():
        if value is not None:
            setattr(defaults, key, value)
    args = defaults

    data_root_override = args.data_root
    resolved_root = (
        detect_data_root(data_root_override)
        if data_root_override is not None
        else detect_data_root()
    )

    if data_root_override is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved_root)

    data_manifests(resolved_root)

    if args.input == DEFAULT_INPUT_DIR and data_root_override is not None:
        input_dir: Path = data_html(resolved_root)
    else:
        input_dir = (args.input or DEFAULT_INPUT_DIR).resolve()

    if args.output == DEFAULT_OUTPUT_DIR and data_root_override is not None:
        output_dir: Path = data_doctags(resolved_root)
    else:
        output_dir = (args.output or DEFAULT_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info(
        "HTML conversion configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_root),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "workers": args.workers,
            }
        },
    )

    if args.force:
        _LOGGER.info("Force mode: reprocessing all documents")
    elif args.resume:
        _LOGGER.info("Resume mode enabled: unchanged outputs will be skipped")

    if args.force:
        print("Force mode: reprocessing all documents")
    elif args.resume:
        print("Resume mode enabled: unchanged outputs will be skipped")

    files = list_htmls(input_dir)
    if not files:
        _LOGGER.warning(
            "No HTML files found", extra={"extra_fields": {"input_dir": str(input_dir)}}
        )
        return 0

    manifest_index = load_manifest_index(MANIFEST_STAGE, resolved_root) if args.resume else {}

    tasks: List[HtmlTask] = []
    ok = fail = skip = 0
    for path in files:
        rel_path = path.relative_to(input_dir)
        doc_id = rel_path.as_posix()
        out_path = (output_dir / rel_path).with_suffix(".doctags")
        input_hash = compute_content_hash(path)
        manifest_entry = manifest_index.get(doc_id)
        if (
            args.resume
            and not args.force
            and not args.overwrite
            and out_path.exists()
            and manifest_entry
            and manifest_entry.get("input_hash") == input_hash
        ):
            _LOGGER.info(
                "Skipping HTML document",
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
                schema_version="docparse/1.1.0",
                input_path=str(path),
                input_hash=input_hash,
                output_path=str(out_path),
                parse_engine="docling-html",
            )
            skip += 1
            continue
        tasks.append(
            HtmlTask(
                html_path=path,
                relative_id=doc_id,
                output_path=out_path,
                input_hash=input_hash,
                overwrite=args.overwrite,
            )
        )

    if not tasks:
        _LOGGER.info(
            "HTML conversion summary",
            extra={
                "extra_fields": {
                    "ok": 0,
                    "skip": skip,
                    "fail": 0,
                }
            },
        )
        return 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(convert_one, task) for task in tasks]
        for fut in tqdm(
            as_completed(futures), total=len(futures), unit="file", desc="HTML → DocTags"
        ):
            result = fut.result()
            if result.status == "success":
                ok += 1
            elif result.status == "skip":
                skip += 1
            else:
                fail += 1
                _LOGGER.error(
                    "HTML conversion failure",
                    extra={
                        "extra_fields": {
                            "doc_id": result.doc_id,
                            "error": result.error or "conversion failed",
                        }
                    },
                )

            manifest_append(
                stage=MANIFEST_STAGE,
                doc_id=result.doc_id,
                status=result.status,
                duration_s=round(result.duration_s, 3),
                schema_version="docparse/1.1.0",
                input_path=result.input_path,
                input_hash=result.input_hash,
                output_path=result.output_path,
                error=result.error,
                parse_engine="docling-html",
            )

    _LOGGER.info(
        "HTML conversion summary",
        extra={
            "extra_fields": {
                "ok": ok,
                "skip": skip,
                "fail": fail,
            }
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
