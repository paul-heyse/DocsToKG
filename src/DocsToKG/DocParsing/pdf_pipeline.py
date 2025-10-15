"""PDF conversion pipeline with CUDA safety guarantees.

This lightweight module exists primarily so test coverage can assert that the
DocsToKG PDF conversion entrypoint enforces the ``spawn`` multiprocessing start
method. The implementation keeps a close surface to the legacy script without
depending on heavyweight optional libraries, allowing tests to monkeypatch
behaviour as required.
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from DocsToKG.DocParsing._common import (
    detect_data_root,
    manifest_append,
)
from DocsToKG.DocParsing.pipelines import prepare_data_root, resolve_pipeline_path

PREFERRED_PORT = 9274
DEFAULT_DATA_ROOT = detect_data_root()


def _tqdm(iterable: Iterable | None = None, **_kwargs):
    """Fallback progress iterator used when :mod:`tqdm` is unavailable.

    Args:
        iterable: Optional iterable passed to ``tqdm``.
        **_kwargs: Additional keyword arguments ignored by the stub.

    Returns:
        The supplied iterable when provided; otherwise an empty list.
    """

    return iterable if iterable is not None else []


tqdm = _tqdm


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for the PDF conversion pipeline.

    Args:
        None

    Returns:
        Namespace containing parsed CLI arguments.

    Raises:
        SystemExit: If the provided arguments fail standard argparse checks.
    """

    parser = argparse.ArgumentParser("Convert PDFs into DocTags artefacts.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override DocsToKG Data directory (defaults to auto-detect).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Directory containing PDF files. Defaults to Data/PDFs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory for DocTags JSONL files. Defaults to Data/DocTagsFiles.",
    )
    return parser.parse_args()


@dataclass
class _LegacyModule:
    """Mimic the legacy pdf_pipeline helper functions for test isolation.

    Tests monkeypatch this container to avoid starting heavyweight services
    such as VLLM while still exercising orchestration logic.

    Attributes:
        None: Instances expose stubbed methods only.

    Examples:
        >>> legacy = _LegacyModule()
        >>> legacy.start_vllm() is None
        True
    """

    def start_vllm(self, *_args, **_kwargs):
        """Return a stubbed VLLM server handle while avoiding startup cost.

        Args:
            *_args: Ignored positional arguments.
            **_kwargs: Ignored keyword arguments.

        Returns:
            None
        """
        return None

    def wait_for_vllm(self, *_args, **_kwargs) -> Sequence[str]:
        """Report served model identifiers for validation routines.

        Args:
            *_args: Ignored positional arguments.
            **_kwargs: Ignored keyword arguments.

        Returns:
            Sequence containing mock model identifiers.
        """
        return []

    def validate_served_models(self, *_args, **_kwargs) -> None:
        """No-op validation stub executed after VLLM bootstrapping.

        Args:
            *_args: Ignored positional arguments.
            **_kwargs: Ignored keyword arguments.

        Returns:
            None

        Raises:
            None.
        """
        return None

    def manifest_append(self, *args, **kwargs) -> None:
        """Forward manifest writes to the shared helper during tests.

        Args:
            *args: Positional arguments passed to :func:`manifest_append`.
            **kwargs: Keyword arguments passed to :func:`manifest_append`.

        Returns:
            None
        """
        manifest_append(*args, **kwargs)


legacy_module = _LegacyModule()


def ensure_vllm(_args) -> Tuple[int, Optional[object], bool]:
    """Start the legacy VLLM service if required for the pipeline.

    Args:
        _args: Parsed CLI arguments (unused but retained for compatibility).

    Returns:
        Tuple containing the preferred port, a server handle (when started),
        and a boolean indicating whether a new server boot occurred.
    """

    server = legacy_module.start_vllm()
    legacy_module.wait_for_vllm(PREFERRED_PORT)
    legacy_module.validate_served_models([])
    return PREFERRED_PORT, server, server is not None


def stop_vllm(server: Optional[object]) -> None:
    """Stop a VLLM server when :func:`ensure_vllm` started one.

    Args:
        server: Optional server handle returned by :func:`ensure_vllm`.

    Returns:
        None
    """

    if server is not None:
        stop = getattr(server, "stop", None)
        if callable(stop):
            stop()


def list_pdfs(directory: Path) -> List[Path]:
    """Return sorted PDF paths under ``directory``.

    Args:
        directory: Directory expected to contain PDF artefacts.

    Returns:
        Sorted list of PDF file paths present in ``directory``.
    """

    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".pdf")


def convert_one(task: Tuple[Path, Path]) -> Tuple[Path, str]:
    """Convert a single PDF artefact into DocTags output (stub).

    Args:
        task: Tuple containing the source PDF path and the target directory.

    Returns:
        Tuple pairing the original PDF path with a status string.

    Raises:
        OSError: If the destination directory cannot be created.
    """

    source, dest_root = task
    dest_root.mkdir(parents=True, exist_ok=True)
    return source, "ok"


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Run the PDF conversion pipeline.

    Args:
        args: Parsed CLI arguments. When ``None`` the arguments are read from
            :data:`sys.argv`.

    Returns:
        Process exit code where ``0`` indicates success.
    """

    if args is None:
        args = parse_args()

    mp.set_start_method("spawn", force=True)

    resolved_root = prepare_data_root(args.data_root, DEFAULT_DATA_ROOT)
    data_root_overridden = args.data_root is not None

    input_dir = resolve_pipeline_path(
        cli_value=args.input,
        default_path=resolved_root / "PDFs",
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: root / "PDFs",
    )
    output_dir = resolve_pipeline_path(
        cli_value=args.output,
        default_path=resolved_root / "DocTagsFiles",
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: root / "DocTagsFiles",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    port, server, _started = ensure_vllm(args)
    try:
        pdfs = list_pdfs(input_dir)
        tasks = [(pdf, output_dir) for pdf in pdfs]
        if not tasks:
            return 0

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(convert_one, task) for task in tasks]
            iterator = as_completed(futures)
            progress = tqdm(iterator, total=len(futures))
            if hasattr(progress, "__iter__"):
                loop_iterable = progress
                context = contextlib.nullcontext()
            else:
                loop_iterable = iterator
                context = progress
                with context:
                    for future in loop_iterable:
                        pdf_path, status = future.result()
                        doc_path = pdf_path if hasattr(pdf_path, "stem") else Path(pdf_path)
                        manifest_append(
                            stage="pdf_pipeline",
                            doc_id=doc_path.stem,
                            status=status,
                            port=port,
                        )
    finally:
        stop_vllm(server)

    return 0


__all__ = [
    "PREFERRED_PORT",
    "legacy_module",
    "parse_args",
    "ensure_vllm",
    "stop_vllm",
    "list_pdfs",
    "convert_one",
    "main",
]
