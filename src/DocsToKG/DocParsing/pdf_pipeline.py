# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.pdf_pipeline",
#   "purpose": "Legacy compatibility facade wrapping the refactored DocParsing pipelines",
#   "sections": [
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
Compatibility shims for the legacy ``pdf_pipeline`` module.

The PDF conversion logic now lives in :mod:`DocsToKG.DocParsing.pipelines`.
This module re-exports the historical surface so downstream tooling and tests
can continue importing ``DocsToKG.DocParsing.pdf_pipeline`` without changes.
"""

from __future__ import annotations

import argparse
import warnings as _warnings

from DocsToKG.DocParsing import pipelines as _pipelines

_warnings.warn(
    "DocsToKG.DocParsing.pdf_pipeline is deprecated; use DocsToKG.DocParsing.pipelines",
    DeprecationWarning,
    stacklevel=2,
)

# --- Re-exported API ---

__all__ = (
    "legacy_module",
    "PREFERRED_PORT",
    "ProcessPoolExecutor",
    "as_completed",
    "tqdm",
    "ensure_vllm",
    "start_vllm",
    "wait_for_vllm",
    "stop_vllm",
    "validate_served_models",
    "manifest_append",
    "list_pdfs",
    "parse_args",
    "main",
    "convert_one",
    "PdfTask",
    "PdfConversionResult",
)

legacy_module = _pipelines

PREFERRED_PORT = _pipelines.PREFERRED_PORT
ProcessPoolExecutor = _pipelines.ProcessPoolExecutor
as_completed = _pipelines.as_completed
tqdm = _pipelines.tqdm

ensure_vllm = _pipelines.ensure_vllm
start_vllm = _pipelines.start_vllm
wait_for_vllm = _pipelines.wait_for_vllm
stop_vllm = _pipelines.stop_vllm
validate_served_models = _pipelines.validate_served_models
manifest_append = _pipelines.manifest_append
list_pdfs = _pipelines.list_pdfs


# --- Legacy CLI Entry Points ---


def parse_args(argv: object | None = None):
    """Return parsed CLI arguments for the legacy PDF pipeline.

    Args:
        argv: Optional argument vector forwarded to :func:`argparse.ArgumentParser.parse_args`.

    Returns:
        argparse.Namespace: Parsed argument namespace compatible with the refactored pipeline.
    """

    return _pipelines.pdf_parse_args(argv)


def main(args: object | None = None) -> int:
    """Invoke the refactored PDF pipeline using the legacy facade.

    Args:
        args: Optional argument namespace or raw argument list compatible with :func:`parse_args`.

    Returns:
        int: Process exit code returned by :func:`DocsToKG.DocParsing.pipelines.pdf_main`.
    """

    if isinstance(args, argparse.Namespace):
        namespace = args
    else:
        namespace = parse_args() if args is None else parse_args(args)
    if not isinstance(namespace, argparse.Namespace):
        namespace = argparse.Namespace(**vars(namespace))
    if not hasattr(namespace, "vlm_prompt"):
        namespace.vlm_prompt = ""
    if not hasattr(namespace, "vlm_stop"):
        namespace.vlm_stop = []
    overrides = {
        "pdf_convert_one": globals().get("convert_one", _pipelines.pdf_convert_one),
        "ProcessPoolExecutor": globals().get("ProcessPoolExecutor", _pipelines.ProcessPoolExecutor),
        "as_completed": globals().get("as_completed", _pipelines.as_completed),
        "tqdm": globals().get("tqdm", _pipelines.tqdm),
    }
    originals = {name: getattr(_pipelines, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(_pipelines, name, value)
        return _pipelines.pdf_main(namespace)
    finally:
        for name, original in originals.items():
            setattr(_pipelines, name, original)


convert_one = _pipelines.pdf_convert_one
PdfTask = _pipelines.PdfTask
PdfConversionResult = _pipelines.PdfConversionResult
