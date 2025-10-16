# Refactor DocParsing for Robustness and Consistency

## Why

The DocsToKG document parsing pipeline currently exhibits several architectural inconsistencies that compromise auditability, resumability, and operational reliability. Document identifiers are ambiguous due to basename collision risks, output paths do not mirror input directory structures leading to potential overwrites, schema definitions contain duplicate field declarations that create validation ambiguities, and model initialization incurs unnecessary performance penalties through repeated instantiation. Legacy module shims and hard-coded file system paths reduce portability across development and production environments. These issues collectively undermine the system's ability to maintain strict stage boundaries and traceable artifact lineage across the processing chain from raw documents through embeddings generation.

## What Changes

- **Document identifier canonicalization**: Establish relative path as the canonical document identifier across all processing stages, eliminating basename collision risks and ensuring perfect alignment between chunk rows, manifest entries, and output artifacts

- **Output path collision prevention**: Mirror input directory structures in output paths for PDF conversion pipeline to match existing HTML pipeline behavior, preventing filename collisions when multiple subdirectories contain identically-named source documents

- **Schema integrity enforcement**: Remove duplicate field declarations in ChunkRow schema that currently override earlier definitions and create ambiguous validation semantics in Pydantic models

- **Performance optimization through caching**: Introduce LLM instance caching in embedding pipeline to eliminate redundant model initialization overhead when processing multiple chunk files within a single execution context

- **Code deduplication and standardization**: Eliminate redundant variable assignments and consolidate duplicate iterator implementations to use shared utilities from common module

- **Legacy module deprecation**: Phase out legacy pipeline module shims and synthetic submodule injection that create multiple import paths for identical functionality

- **Configuration externalization**: Replace hard-coded file system paths with environment-driven resolution following HuggingFace cache conventions for portable model discovery

- **Logging standardization**: Replace unstructured print statements with structured JSON logging to enable machine-parseable observability across all pipeline stages

- **Directory naming alignment**: Reconcile discrepancies between code-defined output directories and documentation references to establish consistent naming conventions

## Impact

### Affected Capabilities

- Document Parsing Pipeline (new spec)
- Chunking Stage
- Embedding Generation Stage
- Manifest System
- Schema Validation

### Affected Code

- `src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py` - document identifier assignment
- `src/DocsToKG/DocParsing/pipelines.py` - PDF output path construction, model path resolution, logging
- `src/DocsToKG/DocParsing/EmbeddingV2.py` - LLM caching, iterator consolidation, variable assignments
- `src/DocsToKG/DocParsing/schemas.py` - duplicate field removal
- `src/DocsToKG/DocParsing/__init__.py` - legacy module shim deprecation
- `src/DocsToKG/DocParsing/_common.py` - directory name function, spawn setup helper
- `src/DocsToKG/DocParsing/pdf_pipeline.py` - deprecation planning

### Breaking Changes

None. All changes maintain backward compatibility for external consumers while improving internal consistency. Output directory names may change if alignment option is selected, which requires documentation update or code adjustment but does not affect data schemas or APIs.

### Migration Notes

- Existing chunk files with stem-based doc_id will continue to function but new processing runs will generate relative-path identifiers for improved collision resistance
- PDF output paths will adopt mirrored directory structure on first run after deployment; existing flat outputs remain valid for HTML pipeline
- Legacy import paths (`DocsToKG.DocParsing.pdf_pipeline`) will emit deprecation warnings before removal in subsequent release
- Hard-coded model paths replaced with environment resolution; operators should set `HF_HOME`, `DOCSTOKG_MODEL_ROOT`, or `DOCLING_PDF_MODEL` environment variables for non-standard cache locations
