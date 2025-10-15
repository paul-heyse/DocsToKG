# Document Parsing Pipeline Specification

## ADDED Requirements

### Requirement: Unified Path Resolution

The system SHALL provide a centralized path resolution mechanism that discovers the DocsToKG data directory through environment variable lookup or ancestor directory scanning.

#### Scenario: Environment variable takes precedence

- **WHEN** `DOCSTOKG_DATA_ROOT` environment variable is set to `/custom/path/Data`
- **THEN** all scripts SHALL use `/custom/path/Data` as the data root
- **AND** no ancestor directory scanning SHALL occur

#### Scenario: Automatic discovery via ancestor scan

- **WHEN** `DOCSTOKG_DATA_ROOT` is not set
- **AND** the script is invoked from `/home/paul/DocsToKG/src/DocsToKG/DocParsing/`
- **THEN** the system SHALL scan ancestors for a `Data/` directory containing expected subdirectories
- **AND** SHALL return `/home/paul/DocsToKG/Data` if it contains `PDFs/`, `HTML/`, or other expected structure

#### Scenario: Typed path getters prevent hardcoded paths

- **WHEN** a script requires the chunks directory
- **THEN** it SHALL call `data_chunks()` from the common utilities module
- **AND** SHALL NOT hardcode absolute paths like `/home/paul/DocsToKG/Data/ChunkedDocTagFiles`

### Requirement: CUDA-Safe Multiprocessing

The system SHALL enforce the 'spawn' start method for multiprocessing in all scripts that perform GPU operations in worker processes to prevent CUDA re-initialization errors.

#### Scenario: PDF conversion with vLLM workers

- **WHEN** the PDF-to-DocTags converter initializes multiprocessing
- **THEN** it SHALL call `multiprocessing.set_start_method('spawn', force=True)` before creating the ProcessPoolExecutor
- **AND** each worker process SHALL initialize CUDA independently

#### Scenario: Graceful handling of already-set start method

- **WHEN** the start method has already been set by a parent process or framework
- **THEN** the system SHALL catch the RuntimeError from `force=True`
- **AND** SHALL verify the current method is 'spawn'
- **AND** SHALL proceed without error if already spawn

### Requirement: Streaming Embeddings Architecture

The system SHALL implement a two-pass streaming architecture for embedding generation to bound memory usage and enable processing of arbitrarily large document corpora.

#### Scenario: Pass A collects global statistics without retaining text

- **WHEN** the embedding pipeline begins processing
- **THEN** Pass A SHALL iterate through all chunk files exactly once
- **AND** SHALL assign UUIDs to chunks missing identifiers
- **AND** SHALL accumulate BM25 document frequency and token statistics
- **AND** SHALL NOT retain chunk text in memory after processing each file

#### Scenario: Pass B processes chunks in configurable batches

- **WHEN** Pass A completes and BM25 statistics are available
- **THEN** Pass B SHALL iterate through chunk files in document order
- **AND** SHALL load chunks in batches of size `--batch-size-splade` for SPLADE encoding
- **AND** SHALL load chunks in batches of size `--batch-size-qwen` for Qwen encoding
- **AND** SHALL write vector shards atomically with `.tmp` suffix before renaming

#### Scenario: Shard merging produces final vector files

- **WHEN** all batches for a document have been encoded
- **THEN** the system SHALL concatenate all shard files (`*.vectors.partNN.jsonl`)
- **AND** SHALL write the final `*.vectors.jsonl` atomically
- **AND** SHALL delete intermediate shard files after successful merge

### Requirement: Schema Validation with Versioning

The system SHALL define and enforce Pydantic schemas for all JSONL output formats and include deterministic schema version identifiers to track evolution.

#### Scenario: ChunkRow validation on write

- **WHEN** writing a chunk record to JSONL
- **THEN** the system SHALL validate the record against the `ChunkRow` Pydantic model
- **AND** SHALL reject records missing required fields (`doc_id`, `chunk_id`, `text`, `num_tokens`)
- **AND** SHALL include `schema_version` field with value `"docparse/1.1.0"`

#### Scenario: VectorRow validation enforces structure

- **WHEN** writing a vector record to JSONL
- **THEN** the system SHALL validate against the `VectorRow` Pydantic model
- **AND** SHALL enforce nested structure for `BM25`, `SPLADEv3`, and `Qwen3-4B` fields
- **AND** SHALL include `schema_version` field with value `"embeddings/1.0.0"`

#### Scenario: Schema version enables backward compatibility

- **WHEN** a downstream consumer reads a JSONL file
- **THEN** it SHALL parse the `schema_version` field from each record
- **AND** SHALL implement version-specific deserialization logic
- **AND** SHALL fail fast with a clear error message for unsupported versions

### Requirement: Embedding Invariant Validation

The system SHALL validate embedding outputs against expected dimensions, norms, and sparsity constraints to detect silent computation errors.

#### Scenario: Qwen dimension assertion

- **WHEN** Qwen3-4B generates an embedding vector
- **THEN** the system SHALL assert the vector dimension equals 2560
- **AND** SHALL raise an error with the document ID if dimension mismatch occurs

#### Scenario: Qwen L2 norm sanity check

- **WHEN** Qwen3-4B generates an embedding vector
- **THEN** the system SHALL compute the L2 norm
- **AND** SHALL assert the norm is greater than zero
- **AND** SHALL log an error with the document ID if the norm is zero or negative

#### Scenario: SPLADE sparsity validation

- **WHEN** SPLADE-v3 encodes all chunks in a corpus
- **THEN** the system SHALL count chunks with zero non-zero elements (nnz == 0)
- **AND** SHALL emit a warning if more than 1% of chunks have nnz == 0
- **AND** SHALL list the document IDs of affected chunks

#### Scenario: BM25 corpus summary

- **WHEN** Pass A completes BM25 statistics collection
- **THEN** the system SHALL print a summary including total document count (N), average document length (avgdl), and the 10 most frequent tokens
- **AND** SHALL log this summary to the manifest for audit

### Requirement: Tokenizer Alignment

The system SHALL align chunk tokenization with the dense embedding model's tokenizer to ensure consistent token counts between chunking and embedding stages.

#### Scenario: Configurable tokenizer model

- **WHEN** invoking the chunking script
- **THEN** the system SHALL accept a `--tokenizer-model` flag
- **AND** SHALL default to `Qwen/Qwen3-Embedding-4B` to match the dense embedder
- **AND** SHALL initialize the HuggingFace tokenizer from the specified model

#### Scenario: Calibration guidance for legacy tokenizer

- **WHEN** a user specifies `--tokenizer-model bert-base-uncased`
- **THEN** the system SHALL emit a deprecation warning
- **AND** SHALL reference the calibration script for computing token count discrepancies
- **AND** SHALL recommend a tuned `--min-tokens` value based on documented calibration

### Requirement: Topic-Aware Chunk Coalescence

The system SHALL detect structural boundaries (headings, captions) during chunk merging and apply relaxed size constraints to preserve topical coherence.

#### Scenario: Soft boundary at markdown heading

- **WHEN** coalescing small chunks
- **AND** the next candidate chunk starts with a markdown heading (`#`, `##`, etc.)
- **THEN** the system SHALL only merge if the combined size is ≤ (max_tokens - 64)
- **AND** SHALL NOT merge if the combined size exceeds this threshold

#### Scenario: Soft boundary at figure caption

- **WHEN** coalescing small chunks
- **AND** the next candidate chunk starts with "Figure caption:" or "Table:"
- **THEN** the system SHALL apply the soft boundary rule (max_tokens - 64)
- **AND** SHALL preserve the caption with its associated content

#### Scenario: No boundary detected allows normal merge

- **WHEN** coalescing small chunks
- **AND** the next candidate chunk does not start with a structural marker
- **THEN** the system SHALL apply standard merge logic up to `max_tokens`
- **AND** SHALL NOT apply the 64-token buffer

### Requirement: Structured Logging and Progress Manifest

The system SHALL adopt structured JSON logging and maintain a centralized progress manifest for all document processing operations.

#### Scenario: Structured logger configuration

- **WHEN** a DocParsing script initializes
- **THEN** it SHALL call `get_logger(name)` from the common utilities
- **AND** SHALL configure the logger with a JSON formatter
- **AND** SHALL support `extra` fields for context metadata (doc_id, stage, etc.)

#### Scenario: Manifest entry on successful processing

- **WHEN** a document completes processing at any stage
- **THEN** the system SHALL append a manifest entry to `Data/Manifests/docparse.manifest.jsonl`
- **AND** the entry SHALL include fields: `timestamp`, `stage`, `doc_id`, `status`, `duration_s`, `warnings`, `schema_version`
- **AND** the write SHALL be atomic (append-only, no locks required)

#### Scenario: Manifest entry on processing failure

- **WHEN** a document fails processing at any stage
- **THEN** the system SHALL append a manifest entry with `status: "failure"`
- **AND** SHALL include `error_message` and `error_type` in the metadata
- **AND** SHALL ensure the failure is logged for retry/debugging

### Requirement: vLLM Server Lifecycle Management

The system SHALL manage vLLM server startup, health checking, reuse, and validation to prevent silent failures from misconfigured or unavailable servers.

#### Scenario: Model validation before processing

- **WHEN** the PDF converter ensures vLLM is ready
- **THEN** it SHALL query the `/v1/models` endpoint
- **AND** SHALL verify the served model name matches `--served-model-name` flag
- **AND** SHALL refuse to proceed if the model is not available or incorrect

#### Scenario: Health check requires non-empty models list

- **WHEN** reusing an existing vLLM server on the preferred port
- **THEN** the system SHALL probe `/v1/models`
- **AND** SHALL verify the response status is 200
- **AND** SHALL verify the models list is non-empty
- **AND** SHALL start a new server if any check fails

#### Scenario: Configurable GPU memory utilization

- **WHEN** starting a new vLLM server
- **THEN** the system SHALL pass `--gpu-memory-utilization` from the CLI flag to the vLLM command
- **AND** SHALL default to 0.30 if not specified
- **AND** SHALL log the utilization setting for diagnostics

#### Scenario: Early failure detection during warmup

- **WHEN** waiting for vLLM to become ready
- **AND** the vLLM process exits prematurely
- **THEN** the system SHALL detect the exit via `proc.poll()`
- **AND** SHALL print the last 800 characters of stdout/stderr
- **AND** SHALL raise a RuntimeError with the exit code

### Requirement: Idempotent Processing with Content Hashing

The system SHALL support idempotent processing and resume capability by tracking content hashes and output states in the manifest.

#### Scenario: Skip processing when output exists and is valid

- **WHEN** invoking a processing script with `--resume` flag
- **AND** the output file exists for a document
- **AND** the content hash in the manifest matches the current input
- **THEN** the system SHALL skip processing for that document
- **AND** SHALL log a skip event to the manifest

#### Scenario: Reprocess when input content changes

- **WHEN** invoking a processing script with `--resume` flag
- **AND** the output file exists
- **AND** the content hash in the manifest does NOT match the current input
- **THEN** the system SHALL reprocess the document
- **AND** SHALL update the manifest with the new hash

#### Scenario: Lock file prevents concurrent writes

- **WHEN** beginning to write an output file
- **THEN** the system SHALL create a `.lock` sentinel file
- **AND** SHALL check if the lock already exists before proceeding
- **AND** SHALL wait or fail if another process holds the lock
- **AND** SHALL remove the lock on completion or error

### Requirement: Provenance Metadata Enrichment

The system SHALL include provenance metadata in chunk records to enable downstream analysis and quality assessment.

#### Scenario: Parse engine identification

- **WHEN** writing a chunk record
- **THEN** the system SHALL include a `provenance.parse_engine` field
- **AND** SHALL set the value to "docling-html" for HTML conversions
- **AND** SHALL set the value to "docling-vlm" for PDF conversions with VLM

#### Scenario: Docling version tracking

- **WHEN** writing a chunk record
- **THEN** the system SHALL detect the installed docling package version
- **AND** SHALL include `provenance.docling_version` field with the detected version

#### Scenario: Image annotation flags

- **WHEN** writing a chunk record
- **AND** the chunk includes picture serializer output
- **THEN** the system SHALL set `provenance.has_image_captions` to true if captions are present
- **AND** SHALL set `provenance.has_image_classification` to true if classification annotations exist

### Requirement: Unified CLI Entry Point

The system SHALL provide a single command-line interface for DocTags conversion that supports both HTML and PDF inputs via a mode selection parameter.

#### Scenario: PDF mode invocation

- **WHEN** invoking `cli/doctags_convert.py --mode pdf --input Data/PDFs --output Data/DocTagsFiles`
- **THEN** the system SHALL dispatch to the PDF conversion backend
- **AND** SHALL start or reuse a vLLM server as needed
- **AND** SHALL process all PDF files in the input directory

#### Scenario: HTML mode invocation

- **WHEN** invoking `cli/doctags_convert.py --mode html --input Data/HTML --output Data/DocTagsFiles`
- **THEN** the system SHALL dispatch to the HTML conversion backend
- **AND** SHALL NOT attempt to start vLLM
- **AND** SHALL process all HTML files in the input directory

#### Scenario: Auto mode detection

- **WHEN** invoking `cli/doctags_convert.py --mode auto --input Data/Mixed`
- **THEN** the system SHALL scan the input directory for file extensions
- **AND** SHALL group files by type (*.pdf,*.html, *.htm)
- **AND** SHALL process each group with the appropriate backend

#### Scenario: Shared flags apply to all modes

- **WHEN** invoking the unified CLI with `--workers 8 --overwrite --logging-level DEBUG`
- **THEN** these flags SHALL be passed to the backend implementation
- **AND** SHALL apply consistently regardless of mode

### Requirement: Atomic File Operations

The system SHALL perform all file writes atomically using temporary files and rename operations to prevent partial writes and corruption.

#### Scenario: Atomic JSONL write with temporary suffix

- **WHEN** writing a chunk or vector JSONL file
- **THEN** the system SHALL write to a temporary file with `.tmp` suffix
- **AND** SHALL only rename to the final filename after successful write and flush
- **AND** SHALL ensure the rename operation is atomic on POSIX systems

#### Scenario: Cleanup on write failure

- **WHEN** an error occurs during JSONL writing
- **THEN** the system SHALL delete the temporary file
- **AND** SHALL NOT leave partial outputs
- **AND** SHALL log the failure to the manifest with error details

### Requirement: Corpus Statistics Reporting

The system SHALL compute and report corpus-level statistics at the end of each processing stage for validation and monitoring.

#### Scenario: Chunking statistics report

- **WHEN** the chunking stage completes for all documents
- **THEN** the system SHALL print a summary including total chunks, median token count, min/max tokens
- **AND** SHALL log this summary to the manifest

#### Scenario: Embedding statistics report

- **WHEN** the embedding stage completes
- **THEN** the system SHALL print a summary including total vectors, SPLADE average nnz, percentage of chunks with SPLADE zero-vectors
- **AND** SHALL include BM25 corpus statistics (N, avgdl)
- **AND** SHALL log this summary to the manifest

### Requirement: Configurable Batch Sizes

The system SHALL expose batch size parameters for all encoding operations to enable tuning for different hardware configurations and memory constraints.

#### Scenario: SPLADE batch size configuration

- **WHEN** invoking the embedding script with `--batch-size-splade 64`
- **THEN** SPLADE encoding SHALL process texts in batches of 64
- **AND** SHALL not load more than 64 texts into GPU memory simultaneously

#### Scenario: Qwen batch size configuration

- **WHEN** invoking the embedding script with `--batch-size-qwen 32`
- **THEN** Qwen encoding SHALL process texts in batches of 32
- **AND** SHALL call `llm.embed()` with batches of exactly 32 texts

#### Scenario: Automatic batch size adjustment on OOM

- **WHEN** GPU out-of-memory error occurs during encoding
- **THEN** the system SHALL log the error with current batch size
- **AND** SHALL recommend reducing batch size via CLI flag
- **AND** SHALL exit with a clear error message (not continue with corrupted state)

### Requirement: Shared Serializer Providers

The system SHALL extract picture and table serializers into a shared module to eliminate duplication and ensure consistent formatting across processing stages.

#### Scenario: Caption and annotation serializer reuse

- **WHEN** the chunking script needs to serialize picture items
- **THEN** it SHALL import `CaptionPlusAnnotationPictureSerializer` from the serializers module
- **AND** SHALL NOT define a new serializer class locally

#### Scenario: Markdown table serializer reuse

- **WHEN** the chunking script needs to serialize table items
- **THEN** it SHALL import `MarkdownTableSerializer` from docling-core
- **AND** the RichSerializerProvider SHALL configure both picture and table serializers

### Requirement: Deprecation Path for Legacy Scripts

The system SHALL maintain backward compatibility by preserving original scripts with clear deprecation notices during the transition period.

#### Scenario: Legacy script warns and delegates

- **WHEN** a user invokes the original `run_docling_html_to_doctags_parallel.py`
- **THEN** the script SHALL print a deprecation warning recommending the unified CLI
- **AND** SHALL delegate to the new implementation
- **AND** SHALL maintain identical CLI interface for compatibility

#### Scenario: Legacy scripts removed after transition

- **WHEN** one release cycle has passed since the unified CLI launch
- **AND** no user issues have been reported with the new implementation
- **THEN** the legacy scripts MAY be removed from the codebase
- **AND** documentation SHALL be updated to reflect the removal

### Requirement: Error Handling and Recovery
The system SHALL implement comprehensive error handling with specific error codes, graceful degradation, and actionable error messages for operators.

#### Scenario: Validation errors provide actionable context
- **WHEN** a Pydantic validation error occurs during JSONL writing
- **THEN** the system SHALL log the error with document ID, row number, and field name
- **AND** SHALL include the invalid value in the error message
- **AND** SHALL continue processing other documents unless fatal error

#### Scenario: Malformed DocTags files are skipped with warning
- **WHEN** a DocTags file cannot be parsed by Docling
- **THEN** the system SHALL log a warning with file path and error details
- **AND** SHALL write a manifest entry with status "failure"
- **AND** SHALL continue processing remaining files

#### Scenario: GPU out-of-memory triggers batch size recommendation
- **WHEN** a CUDA out-of-memory error occurs during embedding
- **THEN** the system SHALL catch the error
- **AND** SHALL log current batch sizes (SPLADE, Qwen)
- **AND** SHALL recommend reducing batch size by 50%
- **AND** SHALL exit with error code 137 (OOM-specific)

#### Scenario: Network errors during vLLM communication are retried
- **WHEN** HTTP request to vLLM server fails with connection error
- **THEN** the system SHALL retry up to 3 times with exponential backoff
- **AND** SHALL log each retry attempt
- **AND** SHALL fail document processing if all retries exhausted

### Requirement: Performance Monitoring and Benchmarks
The system SHALL track and report performance metrics at each processing stage to enable optimization and capacity planning.

#### Scenario: Per-stage timing is recorded in manifest
- **WHEN** a document completes processing at any stage
- **THEN** the system SHALL record duration_s with millisecond precision
- **AND** SHALL include start_time and end_time timestamps in manifest
- **AND** SHALL compute and log throughput (documents/minute)

#### Scenario: Memory usage is monitored during embeddings
- **WHEN** Pass B begins encoding vectors
- **THEN** the system SHALL start memory profiling with tracemalloc
- **AND** SHALL log peak memory usage at the end
- **AND** SHALL warn if peak exceeds 80% of available GPU memory

#### Scenario: Batch processing throughput meets target
- **WHEN** processing a corpus of 1000 documents
- **THEN** chunking SHALL complete at ≥10 documents/minute on CPU
- **AND** embeddings SHALL complete at ≥5 documents/minute with single GPU
- **AND** failures SHALL not exceed 1% of documents

#### Scenario: Performance regression is detected
- **WHEN** running the test suite with golden fixtures
- **THEN** processing time SHALL not exceed 110% of baseline
- **AND** memory usage SHALL not exceed 120% of baseline
- **AND** test SHALL fail if regression thresholds are exceeded

### Requirement: Backward Compatibility for JSONL Formats
The system SHALL read older schema versions and provide migration paths for consumers to handle format evolution.

#### Scenario: Old chunk files without schema_version are handled
- **WHEN** loading a chunk JSONL file without schema_version field
- **THEN** the system SHALL assume schema version "docparse/1.0.0"
- **AND** SHALL successfully parse rows with compatible fields
- **AND** SHALL add schema_version field if re-writing file

#### Scenario: Old vector files without provenance are compatible
- **WHEN** loading a vector JSONL file from pre-refactor pipeline
- **THEN** the system SHALL successfully parse rows missing provenance metadata
- **AND** SHALL NOT require provenance field for reads
- **AND** SHALL only require provenance for new writes

#### Scenario: Schema version mismatch triggers clear error
- **WHEN** a consumer encounters an unsupported schema_version (e.g., "docparse/2.0.0")
- **THEN** the system SHALL raise ValueError with message: "Unsupported schema version: docparse/2.0.0. Compatible versions: [list]"
- **AND** SHALL direct user to migration documentation

### Requirement: Configuration Management and Precedence
The system SHALL apply configuration from multiple sources with clearly defined precedence rules.

#### Scenario: Configuration precedence order
- **WHEN** multiple configuration sources provide the same parameter
- **THEN** CLI flags SHALL take highest precedence
- **AND** environment variables SHALL override defaults
- **AND** default values SHALL be used only if no override

#### Scenario: Configuration validation on startup
- **WHEN** a processing script initializes
- **THEN** it SHALL validate all configuration parameters
- **AND** SHALL reject invalid combinations (e.g., min_tokens > max_tokens)
- **AND** SHALL log final effective configuration before processing

#### Scenario: Sensitive configuration is not logged
- **WHEN** logging effective configuration
- **THEN** API keys, tokens, and credentials SHALL be redacted
- **AND** SHALL be replaced with "***REDACTED***" in logs
- **AND** SHALL still be usable internally

### Requirement: Data Quality Validation
The system SHALL implement multi-level validation to detect and prevent silent data quality issues.

#### Scenario: Empty documents are rejected
- **WHEN** a DocTags file converts to a document with zero pages
- **THEN** the system SHALL log error: "Empty document: <doc_id>"
- **AND** SHALL write manifest entry with status "failure" and error "empty-document"
- **AND** SHALL NOT create output files for this document

#### Scenario: Chunks with excessive token counts are flagged
- **WHEN** a chunk exceeds max_tokens by more than 10%
- **THEN** the system SHALL log warning with chunk_id and token count
- **AND** SHALL include "oversized_chunk" in manifest warnings list
- **AND** SHALL still write the chunk but mark for review

#### Scenario: Vector dimension mismatches are fatal
- **WHEN** an embedding vector has incorrect dimension
- **THEN** the system SHALL raise ValueError immediately
- **AND** SHALL NOT write partial vector file
- **AND** SHALL log expected vs actual dimensions

### Requirement: Monitoring and Alerting Support
The system SHALL expose metrics and structured logs to enable external monitoring and alerting systems.

#### Scenario: Manifest entries enable failure alerting
- **WHEN** an external monitoring system queries the manifest
- **THEN** it SHALL filter entries by status="failure"
- **AND** SHALL group failures by stage and error type
- **AND** SHALL alert if failure rate exceeds threshold (e.g., >5% in 1 hour)

#### Scenario: Progress tracking via manifest
- **WHEN** a long-running pipeline is executing
- **THEN** operators SHALL query manifest for latest timestamp per stage
- **AND** SHALL compute documents_remaining = total - processed
- **AND** SHALL estimate completion time based on recent throughput

#### Scenario: Structured logs enable log aggregation
- **WHEN** logs are ingested by log aggregation system (e.g., ELK, Loki)
- **THEN** JSON format SHALL be parseable without regex
- **AND** extra_fields SHALL be indexed as structured data
- **AND** operators SHALL query by doc_id, stage, or error type

### Requirement: Calibration and Tuning Tools
The system SHALL provide tools to calibrate parameters for specific document collections and hardware configurations.

#### Scenario: Tokenizer calibration recommends adjustments
- **WHEN** running calibration script on a corpus
- **THEN** it SHALL sample at least 100 documents
- **AND** SHALL compute mean and standard deviation of token count ratios
- **AND** SHALL recommend min_tokens adjustment if ratio differs by >10%

#### Scenario: Batch size tuning for GPU capacity
- **WHEN** user runs embedding script with too-large batch size
- **AND** OOM error occurs
- **THEN** error message SHALL recommend: "Try --batch-size-qwen 32" (50% of current)
- **AND** SHALL include current GPU memory utilization in message

### Requirement: Deterministic Processing for Reproducibility
The system SHALL produce bit-identical outputs when processing the same inputs with the same configuration.

#### Scenario: Chunk ordering is deterministic
- **WHEN** processing the same DocTags file twice
- **THEN** chunk_id assignment SHALL be identical
- **AND** chunk text SHALL be identical
- **AND** source_chunk_idxs SHALL be identical

#### Scenario: UUIDs are stable across runs
- **WHEN** a chunk file already has UUIDs assigned
- **THEN** Pass A SHALL NOT regenerate UUIDs
- **AND** existing UUIDs SHALL be preserved
- **AND** only chunks without UUIDs SHALL be assigned new ones

#### Scenario: BM25 statistics are reproducible
- **WHEN** computing BM25 statistics for the same corpus twice
- **THEN** N (document count) SHALL be identical
- **AND** avgdl (average length) SHALL be identical to 3 decimal places
- **AND** document frequency dict SHALL have identical term→count mappings

### Requirement: Dry Run and Validation Modes
The system SHALL support dry-run execution and validation-only modes for testing and auditing.

#### Scenario: Dry run shows planned operations without execution
- **WHEN** invoking any processing script with `--dry-run` flag
- **THEN** the system SHALL enumerate input files
- **AND** SHALL log planned operations (convert, chunk, embed)
- **AND** SHALL NOT write any output files
- **AND** SHALL exit with code 0

#### Scenario: Validation mode checks existing outputs
- **WHEN** invoking with `--validate` flag
- **THEN** the system SHALL load all output JSONL files
- **AND** SHALL validate every row against schemas
- **AND** SHALL report count of valid vs invalid rows
- **AND** SHALL exit with code 1 if any validation failures

### Requirement: Graceful Shutdown and Signal Handling
The system SHALL handle interruption signals gracefully to prevent data corruption.

#### Scenario: SIGINT triggers graceful shutdown
- **WHEN** user sends SIGINT (Ctrl+C) during processing
- **THEN** the system SHALL catch the signal
- **AND** SHALL complete current document processing
- **AND** SHALL write manifest entry for in-progress document
- **AND** SHALL remove any lock files before exiting

#### Scenario: SIGTERM allows cleanup before exit
- **WHEN** orchestration system sends SIGTERM
- **THEN** the system SHALL flush all buffers
- **AND** SHALL finalize manifest writes
- **AND** SHALL stop vLLM server if it owns the process
- **AND** SHALL exit with code 143 (128 + 15)

### Requirement: Help Text and Documentation Completeness
The system SHALL provide comprehensive help text and examples for all CLI tools.

#### Scenario: --help displays usage examples
- **WHEN** invoking any CLI script with `--help`
- **THEN** it SHALL display at least 3 usage examples
- **AND** SHALL document all required and optional flags
- **AND** SHALL indicate default values for optional flags
- **AND** SHALL reference full documentation URL

#### Scenario: Error messages link to troubleshooting docs
- **WHEN** a known error condition occurs
- **THEN** the error message SHALL include a documentation URL
- **AND** SHALL use format: "See: <url>#section-name"
- **AND** URL SHALL point to relevant troubleshooting section
