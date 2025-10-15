#!/usr/bin/env python3
"""
HTML → DocTags (parallel, CPU-only; no captioning/classification, no HF auth)

- Input : Data/HTML/ (recurses; excludes *.normalized.html)
- Output: Data/DocTagsFiles/<mirrored_subdirs>/*.doctags

Example:
  python run_docling_html_to_doctags_parallel.py \
      --input  Data/HTML \
      --output Data/DocTagsFiles \
      --workers 12
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from DocsToKG.DocParsing._common import data_doctags, data_html, detect_data_root

DEFAULT_INPUT_DIR = data_html()
DEFAULT_OUTPUT_DIR = data_doctags()

# keep numeric libs polite; also ensure nothing touches CUDA by mistake
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only

# docling imports (safe on CPU for HTML)
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption

# per-process converter cache
_CONVERTER = None


def _get_converter() -> DocumentConverter:
    """Instantiate and cache a Docling HTML converter per worker process.

    Returns:
        DocumentConverter configured for HTML input, cached for reuse within
        the worker process.
    """
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


def convert_one(
    html_path: Path, input_root: Path, output_root: Path, overwrite: bool
) -> Tuple[str, str]:
    """Convert a single HTML file to DocTags, honoring overwrite semantics.

    Args:
        html_path: Path to the source HTML document.
        input_root: Root directory used to compute relative paths for logging.
        output_root: Base directory where generated `.doctags` files are stored.
        overwrite: Whether to replace existing outputs.

    Returns:
        Tuple of `(relative_path, status)` where status is `ok`, `skip`, or a
        `fail:<reason>` string describing an error condition.
    """
    rel = html_path.relative_to(input_root)
    try:
        out_path = (output_root / rel).with_suffix(".doctags")
        if out_path.exists() and not overwrite:
            return (rel.as_posix(), "skip")

        converter = _get_converter()
        result = converter.convert(html_path, raises_on_error=False)

        if result.document is None:
            return (rel.as_posix(), "fail: empty-document")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.document.save_as_doctags(out_path)
        return (rel.as_posix(), "ok")

    except Exception as e:
        return (rel.as_posix(), f"fail: {e}")


def main():
    """Entrypoint for parallel HTML-to-DocTags conversion across a dataset.

    Args:
        None

    Returns:
        None
    """
    import argparse
    import multiprocessing as mp

    # safer start method for multi-proc even though we're CPU-only
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

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
        "--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1), help="Parallel workers"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing .doctags files"
    )
    args = parser.parse_args()

    data_root_override = args.data_root
    resolved_root = (
        detect_data_root(data_root_override)
        if data_root_override is not None
        else detect_data_root()
    )

    if args.input == DEFAULT_INPUT_DIR and data_root_override is not None:
        input_dir: Path = data_html(resolved_root)
    else:
        input_dir = args.input.resolve()

    if args.output == DEFAULT_OUTPUT_DIR and data_root_override is not None:
        output_dir: Path = data_doctags(resolved_root)
    else:
        output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")

    files = list_htmls(input_dir)
    if not files:
        print("No HTML files found. Exiting.")
        return

    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(convert_one, p, input_dir, output_dir, args.overwrite) for p in files]
        for fut in tqdm(
            as_completed(futures), total=len(futures), unit="file", desc="HTML → DocTags"
        ):
            rel, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"[FAIL] {rel}: {status}")

    print(f"\nDone. ok={ok}, skip={skip}, fail={fail}")


if __name__ == "__main__":
    main()
