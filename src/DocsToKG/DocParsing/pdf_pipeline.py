# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.pdf_pipeline",
#   "purpose": "Legacy DocParsing PDF pipeline compatibility surface",
#   "sections": [
#     {
#       "id": "parse_args",
#       "name": "parse_args",
#       "anchor": "PA",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "MAIN",
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

from DocsToKG.DocParsing import pipelines as _pipelines

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


def parse_args(argv: object | None = None):
    """Return parsed CLI arguments for the PDF pipeline."""

    return _pipelines.pdf_parse_args(argv)


def main(args: object | None = None) -> int:
    """Invoke the refactored PDF pipeline using legacy entrypoints."""

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
        "ProcessPoolExecutor": globals().get(
            "ProcessPoolExecutor", _pipelines.ProcessPoolExecutor
        ),
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
