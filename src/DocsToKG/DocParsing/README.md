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
`Data/Manifests/docparse.<stage>.manifest.jsonl` (for example,
`docparse.embeddings.manifest.jsonl`), enabling resumable execution and auditing
across the toolchain.

## Stage Descriptions

### 1. DocTags Conversion

* **Entry point**: `python -m DocsToKG.DocParsing.cli doctags`
* **Output**: `Data/DocTagsFiles/<doc_id>.doctags`
* **Highlights**:
  * HTML and PDF pipelines use Docling as the default renderer.  PDF conversion
    bootstraps a vLLM-backed visual language model for captioning and picture
    classification.
  * Conversion metadata (parse engine, content hashes, warnings) flows into the
    manifest for provenance tracking.
  * `--vllm-wait-timeout` lets operators accommodate slow model cold starts
    before marking the DocTags stage unhealthy.

### 2. Chunking & Coalescence

* **Entry point**: `python -m DocsToKG.DocParsing.cli chunk`
* **Output**: `Data/ChunkedDocTagFiles/<relative_path>.chunks.jsonl` (mirrors the DocTags hierarchy)
* **Highlights**:
  * The hybrid chunker honors Docling structural annotations and only merges
    across sections when soft-boundary thresholds permit.
  * Chunk rows embed `ProvenanceMetadata` (parse engine, Docling version, image
    flags) to preserve the source context of every span.
  * Declarative extensions let operators load JSON/YAML/TOML structural markers and
    swap serializer providers (`--structural-markers`, `--serializer-provider`)
    without touching the chunking logic.
  * Document identifiers mirror the relative path within the input directory so
    nested hierarchies never collide on basename alone.
  * Supports deterministic sharding via `--shard-count` / `--shard-index` so
    large corpora can be processed across machines without external tooling.

### 3. Embedding Generation

* **Entry point**: `python -m DocsToKG.DocParsing.cli embed`
* **Output**: `Data/Embeddings/<relative_path>.vectors.jsonl` (mirrors the chunk directory layout)
* **Highlights**:
  * Streaming two-pass architecture bounds memory usage while collecting BM25
    statistics.
  * Extensive validation guards Qwen vector dimensions, SPLADE sparsity, and
    schema compatibility.
  * Vector writers are pluggable (`--format`), and caching controls (`--no-cache`)
    make it easy to debug embedding behaviour without reusing long-lived vLLM
    instances.

## Configuration

All CLI entry points share a consistent configuration surface.  Key options
include:

| Flag | Stage | Description |
| --- | --- | --- |
| `--data-root` | All | Override the detected `Data/` directory. Defaults to `DOCSTOKG_DATA_ROOT` or ancestor discovery. |
| `--resume` | All | Skip inputs whose outputs exist with matching content hash. |
| `--force` | All | Reprocess all inputs regardless of manifest state. |
| `--log-level` | All | Adjust structured log verbosity (`DEBUG`, `INFO`, etc.). |
| `--workers` | DocTags | Parallel PDF worker count (spawn start method enforced). |
| `--vllm-wait-timeout` | DocTags | Seconds to wait for the auxiliary vLLM server before aborting. |
| `--min-tokens` / `--max-tokens` | Chunker | Token window boundaries for chunk coalescence. |
| `--tokenizer-model` | Chunker | HuggingFace tokenizer aligning chunk lengths with the dense embedder. |
| `--structural-markers` | Chunker | Load additional heading/caption prefixes from JSON/YAML/TOML configuration files. |
| `--serializer-provider` | Chunker | Swap in a custom Docling serializer via an import path (`module:Class`). |
| `--shard-count` / `--shard-index` | Chunker, Embedder | Deterministically partition inputs for distributed processing. |
| `--validate-only` | Chunker, Embedder | Validate existing chunk or vector JSONL outputs and exit without writing. |
| `--qwen-dim` | Embedder | Expected Qwen output dimension (use with multi-resolution models). |
| `--batch-size-*` | Embedder | Independent batch sizing for SPLADE and Qwen passes. |
| `--format` | Embedder | Vector output format (`jsonl` today; `parquet` in progress). |
| `--no-cache` | Embedder | Disable Qwen LLM reuse between batches (helpful for debugging). |

### Programmatic configuration loading

The `DocsToKG.DocParsing.config` module exposes public helpers for teams that
manage structural-marker inventories outside the CLI. Use
`load_yaml_markers` / `load_toml_markers` (raising `ConfigLoadError` on malformed
input) to deserialize marker documents before passing the resulting dictionaries
into `load_structural_marker_profile`.

### CLI validation behaviour

All DocParsing CLIs normalise invalid argument handling through
`CLIValidationError`. When a stage rejects input, the process exits with code `2`
and prints a single `[stage] --option: message` line on stderr instead of a
Python traceback. Automation scripts can rely on this format when surfacing
operator guidance.

### Environment Variables

| Variable | Purpose |
| --- | --- |
| `DOCSTOKG_DATA_ROOT` | Overrides automatic data-root detection for all stages. |
| `HF_HOME` | Preferred HuggingFace cache directory used for model discovery when stage-specific overrides are not supplied. |
| `DOCSTOKG_MODEL_ROOT` | Base directory that houses DocParsing models when `DOCLING_PDF_MODEL` is unset. |
| `DOCLING_PDF_MODEL` | Explicit path to the Granite Docling model used by the PDF pipeline. Overrides all other model-location environment variables. |
| `DOCLING_CUDA_USE_FLASH_ATTENTION2` | Enables Flash Attention optimizations within Docling VLM workers. |
| `DOCLING_ARTIFACTS_PATH` | Custom cache directory for Docling-rendered artifacts (bitmaps, intermediate assets). |
| `DOCSTOKG_HASH_ALG` | Forces content hashing algorithm (`sha1` default, set to `sha256`, `sha512`, etc. when compliance requires). Changing this invalidates resume caches created with a different algorithm. |

### PDF model resolution

The PDF DocTags converter resolves its model path using the following
precedence order:

1. `--model` CLI argument
2. `DOCLING_PDF_MODEL` environment variable
3. `DOCSTOKG_MODEL_ROOT/granite-docling-258M`
4. `HF_HOME/granite-docling-258M`
5. `~/.cache/huggingface/granite-docling-258M`

The resolved path is logged at startup so operators can confirm the
expected model directory before conversion begins.

### Serializer Providers

The chunker loads its Markdown-aware serializer via the `--serializer-provider`
flag. Providers are referenced using the import path syntax
`module_path:ClassName` (for example,
`DocsToKG.DocParsing.formats:RichSerializerProvider`, which is the default).

To provide a custom implementation:

1. Subclass :class:`docling_core.transforms.chunker.hierarchical_chunker.ChunkingSerializerProvider`.
2. Implement :meth:`get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer`
   and return a serializer that understands your domain-specific markup.
3. Package the provider on `PYTHONPATH` and point `--serializer-provider` at
   the fully-qualified import path.

If a provider cannot be imported, the chunker now logs a warning with a sample
path so operators can quickly diagnose missing packages. Custom providers should
remain stateless—when a non-default provider is configured, the chunker falls
back to a single worker to avoid sharing mutable state across processes.

## Schema Versioning Strategy

JSONL outputs embed deterministic schema identifiers (`docparse/1.1.0` for
chunks, `embeddings/1.0.0` for vectors). Reader utilities now enforce these
identifiers via ``validate_schema_version`` which raises a ``ValueError`` when a
row is missing the field or advertises an unsupported version. This fail-fast
behaviour allows us to introduce future schema revisions without corrupting
downstream consumers. When incrementing a schema version:

1. Extend the compatibility matrix in `DocsToKG.DocParsing.formats`.
2. Preserve validation helpers for older versions until consumers are migrated.
3. Document migration guidance (see below) to keep operators informed.

### Supported Schema Versions

| Stage | Schema Identifier | Supported Versions | Validation Snippet |
| --- | --- | --- | --- |
| Chunking | `docparse` | `docparse/1.0.0`, `docparse/1.1.0` | `validate_schema_version("docparse/1.1.0", COMPATIBLE_CHUNK_VERSIONS)` |
| Embeddings | `embeddings` | `embeddings/1.0.0` | `validate_schema_version("embeddings/1.0.0", COMPATIBLE_VECTOR_VERSIONS)` |

```python
from DocsToKG.DocParsing.formats import (
    COMPATIBLE_CHUNK_VERSIONS,
    COMPATIBLE_VECTOR_VERSIONS,
    validate_schema_version,
)

# Chunk row check
validate_schema_version("docparse/1.1.0", COMPATIBLE_CHUNK_VERSIONS)

# Vector row check
validate_schema_version("embeddings/1.0.0", COMPATIBLE_VECTOR_VERSIONS)
```

## CLI Reference & Usage Examples

Typical end-to-end workflow on a small corpus:

```bash
# 1) Convert PDFs to DocTags using the visual language pipeline
python -m DocsToKG.DocParsing.cli doctags \
  --mode pdf \
  --input Data/PDFs \
  --output Data/DocTagsFiles \
  --resume

# 2) Chunk the DocTags with tokenizer-aligned boundaries
python -m DocsToKG.DocParsing.cli chunk \
  --in-dir Data/DocTagsFiles \
  --out-dir Data/ChunkedDocTagFiles \
  --min-tokens 256 \
  --max-tokens 512

# 3) Generate embeddings with streaming batches
python -m DocsToKG.DocParsing.cli embed \
  --chunks-dir Data/ChunkedDocTagFiles \
  --out-dir Data/Embeddings \
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
| Schema validation errors | Mixed schema versions in output directories | Run the validation CLI (`python -m DocsToKG.DocParsing.cli embed --validate-only`) after cleaning stale shards. |
| Missing DocTags | Input hashes mismatch resume manifest | Re-run the converter with `--force` or delete the stale DocTags file to trigger regeneration. |
| `sentence-transformers` / `vllm` ImportError | Optional embedding dependencies not installed | Install the extras (`pip install sentence-transformers vllm`) or run `pytest tests/docparsing/test_synthetic_benchmark.py` to verify stubbed dependencies before provisioning GPUs. |

## Manifest Query Examples

Run these commands from the repository root to inspect manifest telemetry:

```bash
# List failed documents with error messages
jq 'select(.status == "failure") | {doc_id, stage, error}' Data/Manifests/docparse.*.manifest.jsonl

# Average duration by stage
jq -s 'group_by(.stage) | map({stage: .[0].stage, avg_duration: (map(.duration_s) | add / length)})' \
  Data/Manifests/docparse.*.manifest.jsonl

# Count of skipped documents per stage
jq 'select(.status == "skip") | .stage' Data/Manifests/docparse.*.manifest.jsonl \
  | sort | uniq -c
```

## Migration Guidance

When upgrading schema versions or changing tokenizer defaults, update the
OpenSpec change proposal with explicit migration steps.  Provide before/after
examples and recommend recalculating affected manifolds so downstream consumers
can reason about historical data sets.

* Embedding outputs now write to `Data/Embeddings/` so deployments with legacy
  `Data/Vectors/` directories should rename them to keep manifests and resumable
  processing aligned with the canonical layout.


### Deprecations

> **Deprecated**  
> `DocsToKG.DocParsing.pdf_pipeline` remains temporarily available as a compatibility shim.
> Importing it emits a :class:`DeprecationWarning` and a structured log pointing to the
> :mod:`DocsToKG.DocParsing.doctags` replacement. The shim will be removed in
> **DocsToKG 2.0.0**. Migrate now to the CLI (`python -m DocsToKG.DocParsing.cli doctags
> --mode pdf`) or call the ``Doctags`` APIs directly to avoid last-minute surprises.

## Synthetic Benchmarking & Test Utilities

The test suite ships lightweight fixtures that allow end-to-end validation
without installing heavyweight dependencies:

* `tests.docparsing.stubs.dependency_stubs()` installs stub implementations of
  Docling, vLLM, and sentence-transformers so the CLI wrappers can be exercised
  on laptops without GPUs. The integration tests reuse these stubs to verify
  manifest updates and schema compliance.
* `tests/docparsing/synthetic.py` contains deterministic factories and a
  synthetic benchmark model. `tests/docparsing/test_synthetic_benchmark.py`
  asserts the expected throughput and memory characteristics as part of CI.
