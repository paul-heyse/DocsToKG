# 1. Module: pdf_pipeline

This reference documents the DocsToKG module ``DocsToKG.DocParsing.pdf_pipeline``.

Parallel PDF → DocTags conversion pipeline with vLLM orchestration.

Used by the DocsToKG CLI to launch (or reuse) a local vLLM server and execute
Docling PDF conversions in parallel worker processes. Provides resilience
features such as automatic port selection, manifest-aware resume semantics, and
detailed logging for observability.

## 1. Functions

### `build_parser()`

Construct the argument parser for the PDF → DocTags converter.

### `parse_args(argv=None)`

Parse CLI arguments for standalone execution.

### `convert_one(task)`

Convert a single PDF into DocTags using a remote vLLM-backed pipeline.

### `ensure_vllm(preferred, model_path, served_model_names, gpu_memory_utilization)`

Ensure a vLLM server is available, launching one when necessary.

### `main(args=None)`

Coordinate vLLM startup and parallel DocTags conversion.
