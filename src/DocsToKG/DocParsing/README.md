# DocParsing Pipeline

The DocParsing subsystem converts heterogeneous documents into structured data
that downstream knowledge-graph stages can ingest.  The default workflow relies
on the [Docling](https://github.com/IBM/docling) hybrid conversion stack and
preserves DocTags semantics end-to-end so we remain compatible with existing
metadata and image annotations.

## Architecture Overview

The pipeline is composed of three coarse stages that operate on shared storage:

1. **DocTags conversion** – HTML and PDF inputs are rendered to DocTags using
   Docling engines (`docling-html` and `docling-vlm`).
2. **Chunking & coalescence** – DocTags payloads are segmented into
   topic-aligned text chunks using the hybrid chunker with structural
   boundary detection.
3. **Embedding generation** – The two-pass embedding runner produces dense
   (Qwen3-Embedding-4B) and sparse (SPLADE v3, BM25) vectors while validating
   invariants and emitting manifest telemetry.

Each stage records append-only manifest entries under
`Data/Manifests/docparse.manifest.jsonl`, enabling resumable execution and
auditing across the toolchain.

## Stage Descriptions

### 1. DocTags Conversion

* **Entry point**: `python -m DocsToKG.DocParsing.cli.doctags_convert`
* **Output**: `Data/DocTagsFiles/<doc_id>.doctags`
* **Highlights**:
  - HTML and PDF pipelines use Docling as the default renderer.  PDF conversion
    bootstraps a vLLM-backed visual language model for captioning and picture
    classification.
  - Conversion metadata (parse engine, content hashes, warnings) flows into the
    manifest for provenance tracking.

### 2. Chunking & Coalescence

* **Entry point**: `python -m DocsToKG.DocParsing.cli.chunk_and_coalesce`
* **Output**: `Data/ChunkedDocTagFiles/<doc_id>.chunks.jsonl`
* **Highlights**:
  - The hybrid chunker honors Docling structural annotations and only merges
    across sections when soft-boundary thresholds permit.
  - Chunk rows embed `ProvenanceMetadata` (parse engine, Docling version, image
    flags) to preserve the source context of every span.

### 3. Embedding Generation

* **Entry point**: `python -m DocsToKG.DocParsing.cli.embed_vectors`
* **Output**: `Data/Embeddings/<doc_id>.vectors.jsonl`
* **Highlights**:
  - Streaming two-pass architecture bounds memory usage while collecting BM25
    statistics.
  - Extensive validation guards Qwen vector dimensions, SPLADE sparsity, and
    schema compatibility.

## Configuration

All CLI entry points share a consistent configuration surface.  Key options
include:

| Flag | Stage | Description |
| --- | --- | --- |
| `--data-root` | All | Override the detected `Data/` directory. Defaults to `DOCSTOKG_DATA_ROOT` or ancestor discovery. |
| `--resume` | All | Skip inputs whose outputs exist with matching content hash. |
| `--force` | All | Reprocess all inputs regardless of manifest state. |
| `--workers` | DocTags | Parallel PDF worker count (spawn start method enforced). |
| `--min-tokens` / `--max-tokens` | Chunker | Token window boundaries for chunk coalescence. |
| `--tokenizer-model` | Chunker | HuggingFace tokenizer aligning chunk lengths with the dense embedder. |
| `--batch-size-*` | Embedder | Independent batch sizing for SPLADE and Qwen passes. |

### Environment Variables

| Variable | Purpose |
| --- | --- |
| `DOCSTOKG_DATA_ROOT` | Overrides automatic data-root detection for all stages. |
| `DOCLING_CUDA_USE_FLASH_ATTENTION2` | Enables Flash Attention optimizations within Docling VLM workers. |
| `DOCLING_ARTIFACTS_PATH` | Custom cache directory for Docling-rendered artifacts (bitmaps, intermediate assets). |

## Schema Versioning Strategy

JSONL outputs embed deterministic schema identifiers (`docparse/1.1.0` for
chunks, `embeddings/1.0.0` for vectors).  Reader utilities validate compatibility
against declared allow-lists so we can introduce future schema revisions without
breaking downstream consumers.  When incrementing a schema version:

1. Extend the compatibility matrix in `DocsToKG.DocParsing.schemas`.
2. Preserve validation helpers for older versions until consumers are migrated.
3. Document migration guidance (see below) to keep operators informed.

## CLI Reference & Usage Examples

Typical end-to-end workflow on a small corpus:

```bash
# 1) Convert PDFs to DocTags using the visual language pipeline
python -m DocsToKG.DocParsing.cli.doctags_convert \
  --mode pdf \
  --input Data/PDFs \
  --output Data/DocTagsFiles \
  --resume

# 2) Chunk the DocTags with tokenizer-aligned boundaries
python -m DocsToKG.DocParsing.cli.chunk_and_coalesce \
  --in-dir Data/DocTagsFiles \
  --out-dir Data/ChunkedDocTagFiles \
  --min-tokens 256 \
  --max-tokens 512

# 3) Generate embeddings with streaming batches
python -m DocsToKG.DocParsing.cli.embed_vectors \
  --chunks-dir Data/ChunkedDocTagFiles \
  --vectors-dir Data/Embeddings \
  --batch-size-qwen 24 \
  --batch-size-splade 256
```

Additional tips:

* Use `--resume` across stages for large reruns—skips are recorded with
  explicit manifest entries and summary logging.
* Combine `--force` with focused `--input` / `--in-dir` selectors when
  reprocessing a known-bad document.

## Troubleshooting Guide

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| CUDA error: re-initialization of context | Process forked before CUDA init | Ensure the PDF converter runs with spawn mode (default) or export `CUDA_VISIBLE_DEVICES` to limit visible GPUs. |
| Out-of-memory during embedding | Batch size too large | Reduce `--batch-size-qwen` / `--batch-size-splade` or enable gradient checkpointing on the serving side. |
| vLLM server fails to start | Port conflict or missing model weights | Use `--served-model-name` when launching vLLM and verify the port is free via `lsof -i :8000`. |
| Schema validation errors | Mixed schema versions in output directories | Run the validation CLI (`python -m DocsToKG.DocParsing.cli.embed_vectors --validate-only`) after cleaning stale shards. |
| Missing DocTags | Input hashes mismatch resume manifest | Re-run the converter with `--force` or delete the stale DocTags file to trigger regeneration. |

## Manifest Query Examples

Run these commands from the repository root to inspect manifest telemetry:

```bash
# List failed documents with error messages
jq 'select(.status == "failure") | {doc_id, stage, error}' \
  Data/Manifests/docparse.manifest.jsonl

# Average duration by stage
jq -s 'group_by(.stage) | map({stage: .[0].stage, avg_duration: (map(.duration_s) | add / length)})' \
  Data/Manifests/docparse.manifest.jsonl

# Count of skipped documents per stage
jq 'select(.status == "skip") | .stage' Data/Manifests/docparse.manifest.jsonl \
  | sort | uniq -c
```

## Migration Guidance

When upgrading schema versions or changing tokenizer defaults, update the
OpenSpec change proposal with explicit migration steps.  Provide before/after
examples and recommend recalculating affected manifolds so downstream consumers
can reason about historical data sets.
