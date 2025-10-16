# Document Parsing Pipeline Specification

## ADDED Requirements

### Requirement: Document Identifier Canonicalization

The document parsing pipeline SHALL assign document identifiers based on relative file paths from the input directory root to ensure global uniqueness and traceability across all processing stages. The system SHALL compute the relative path by subtracting the input directory base from each discovered file path and converting the result to POSIX-style forward-slash notation. Document identifiers SHALL remain stable across chunk generation, embedding generation, and manifest recording to enable reliable cross-referencing and resume logic.

The pipeline SHALL preserve file stem values independently from document identifiers for use in local filename generation, progress logging, and human-readable output contexts where full path information would introduce unnecessary verbosity. All manifest entries SHALL record document identifiers using the same relative path representation to maintain perfect alignment between input tracking, intermediate artifacts, and output products.

#### Scenario: Single directory without collisions

- **WHEN** processing files in a flat input directory structure
- **THEN** document identifiers equal relative paths which match basenames
- **AND** chunk rows, manifest entries, and output filenames reference consistent identifiers

#### Scenario: Subdirectory structure with basename collisions

- **WHEN** processing input directory containing subdirectories with identically-named files such as teamA/report.pdf and teamB/report.pdf
- **THEN** document identifiers equal teamA/report.pdf and teamB/report.pdf respectively
- **AND** chunk rows for each file contain distinct doc_id values preventing collision
- **AND** manifest entries record separate processing outcomes for each unique relative path

#### Scenario: Resume after partial run

- **WHEN** resuming a processing run that was interrupted or skipped files
- **THEN** manifest index lookups use relative path identifiers to match previous processing records
- **AND** input hash comparison succeeds for unchanged files enabling skip optimization
- **AND** no false positive matches occur due to basename ambiguity

### Requirement: Output Path Structure Mirroring

The document parsing pipeline SHALL preserve input directory hierarchy in output locations by mirroring relative path structures from source to destination directories. For PDF conversion operations, the system SHALL compute output paths by appending the source file's relative path to the output directory root and replacing the file suffix with the appropriate output extension. The pipeline SHALL create all necessary parent directories in the output hierarchy before writing conversion results to prevent file system errors.

This mirroring behavior SHALL apply uniformly across HTML and PDF conversion paths to maintain operational consistency and operator expectations. Output path construction SHALL occur during task creation and SHALL embed the complete target path in task metadata for worker process consumption.

#### Scenario: Flat input structure

- **WHEN** converting PDFs from a flat input directory without subdirectories
- **THEN** output files appear directly under the output directory root
- **AND** behavior matches previous implementation maintaining backward compatibility

#### Scenario: Nested input structure

- **WHEN** converting PDFs from input directory containing nested subdirectories
- **THEN** output directory structure mirrors input directory structure preserving folder hierarchy
- **AND** output files reside in corresponding subdirectories matching source locations
- **AND** worker processes create parent directories before writing output artifacts

#### Scenario: Basename collision prevention

- **WHEN** processing multiple source files with identical basenames in different subdirectories
- **THEN** each output file occupies a unique path reflecting its source subdirectory
- **AND** no output files overwrite each other due to path collision
- **AND** manifest records distinct output paths for each source file

#### Scenario: HTML pipeline consistency

- **WHEN** comparing output path construction between HTML and PDF pipelines
- **THEN** both pipelines apply identical directory mirroring logic
- **AND** operators observe uniform behavior across conversion types

### Requirement: Schema Field Uniqueness

Document parsing schema definitions SHALL declare each model field exactly once to ensure deterministic validation behavior and predictable serialization results. The ChunkRow schema SHALL contain single definitions for convenience fields including has_image_captions, has_image_classification, and num_images that mirror provenance metadata for efficient filtering operations.

The system SHALL reject schema definitions containing duplicate field declarations during model construction to prevent field override ambiguities that arise from multiple declarations with potentially conflicting constraints or defaults. Field validators and constraints SHALL remain attached to the canonical field definition retaining all validation logic and boundary checks.

#### Scenario: Model construction with unique fields

- **WHEN** constructing a ChunkRow instance with image metadata fields
- **THEN** model contains exactly one attribute for each declared field name
- **AND** field validators execute once per field during validation
- **AND** serialization produces one key per field in output dictionaries

#### Scenario: Schema validation enforcement

- **WHEN** Pydantic processes ChunkRow class definition
- **THEN** no field override warnings or errors occur during import
- **AND** model_config extra='forbid' constraint remains effective
- **AND** introspection reveals expected field count matching declaration count

#### Scenario: Round-trip serialization consistency

- **WHEN** serializing a ChunkRow to JSON and deserializing back to model
- **THEN** field values persist exactly without duplication or loss
- **AND** convenience fields maintain alignment with provenance metadata
- **AND** validation constraints apply consistently across serialization boundary

### Requirement: Model Instance Caching for Performance

The embedding generation pipeline SHALL cache expensive model instances across multiple invocation contexts to eliminate redundant initialization overhead and reduce overall processing time. The system SHALL implement a module-level cache keyed by configuration parameters including model directory path, data type, tensor parallelism degree, GPU memory utilization fraction, and quantization mode.

On function entry, the pipeline SHALL compute a cache key from the supplied configuration and probe the cache for an existing instance matching those parameters. On cache miss, the system SHALL instantiate a new model object, insert it into the cache under the computed key, and return the instance for use. On cache hit, the system SHALL return the cached instance directly bypassing initialization routines.

Cache lifetime SHALL extend for the duration of the process execution context, with cleanup occurring automatically during process termination through garbage collection and object destructor invocation. The caching mechanism SHALL operate safely in single-process execution contexts and SHALL produce identical output vectors regardless of cache state.

#### Scenario: First invocation creates instance

- **WHEN** calling qwen_embed for the first time in a process with specific configuration
- **THEN** system instantiates new LLM object with supplied parameters
- **AND** stores instance in cache under configuration key
- **AND** returns instance for immediate use in embedding generation

#### Scenario: Subsequent invocation reuses instance

- **WHEN** calling qwen_embed again with identical configuration parameters
- **THEN** system retrieves cached LLM instance without instantiation
- **AND** skips expensive model loading and initialization routines
- **AND** embedding generation completes faster due to eliminated overhead

#### Scenario: Different configuration creates separate instance

- **WHEN** calling qwen_embed with modified configuration such as different data type or tensor parallelism
- **THEN** cache key differs from previous invocations
- **AND** system instantiates new LLM object for the new configuration
- **AND** both instances coexist in cache independently

#### Scenario: Output consistency with caching

- **WHEN** generating embeddings for identical input texts across cached and uncached invocations
- **THEN** output vectors remain identical within floating point precision tolerances
- **AND** vector dimensions match expected values
- **AND** normalization properties hold across cache states

### Requirement: Iterator and Variable Consolidation

The document parsing pipeline SHALL utilize shared iterator implementations from the common utilities module to maintain consistent file discovery behavior across all processing stages. The system SHALL eliminate local reimplementations of directory traversal logic that duplicate functionality already present in centralized helpers.

Processing functions SHALL avoid redundant variable assignments that shadow or overwrite earlier declarations within the same scope. The system SHALL initialize collection variables once at their point of use with values derived from the authoritative data source.

#### Scenario: Chunk file discovery consistency

- **WHEN** embedding pipeline enumerates chunk files for processing
- **THEN** system uses iter_chunks function from common module
- **AND** discovers identical file set as chunking pipeline output
- **AND** respects consistent file naming pattern across stages

#### Scenario: Variable assignment clarity

- **WHEN** processing chunk rows to extract UUIDs and text content
- **THEN** system initializes collections once with values from row iteration
- **AND** no shadowed or overwritten variables exist in scope
- **AND** linters report no unused assignment warnings

#### Scenario: Shared utility maintenance

- **WHEN** updating file discovery logic for new requirements
- **THEN** changes apply uniformly across all consumers through common module
- **AND** no divergent implementations require separate updates
- **AND** testing validates single authoritative implementation

### Requirement: Configuration-Driven Model Discovery

The document parsing pipeline SHALL resolve model file system locations through environment variable configuration to enable portable deployment across development workstations, continuous integration environments, and production infrastructure. The system SHALL check environment variables in order of increasing specificity with model-specific overrides taking precedence over general cache location settings.

For PDF conversion models, the resolution order SHALL be: DOCLING_PDF_MODEL environment variable if set, otherwise DOCSTOKG_MODEL_ROOT combined with conventional subdirectory path, otherwise HF_HOME combined with conventional subdirectory path, otherwise user home directory with HuggingFace cache subdirectory and conventional model path. The system SHALL preserve explicit command-line flag overrides as highest precedence to support operator-directed model selection.

Model path resolution SHALL occur during pipeline initialization and SHALL emit resolved paths to structured logs for operational observability. The system SHALL reject hard-coded absolute paths specific to individual developer workstations.

#### Scenario: Standard HuggingFace cache layout

- **WHEN** operator sets HF_HOME environment variable pointing to cache directory
- **THEN** PDF pipeline resolves model path under HF_HOME with conventional subdirectory
- **AND** pipeline starts successfully without additional configuration
- **AND** logs record resolved model path for verification

#### Scenario: Custom model root

- **WHEN** operator sets DOCSTOKG_MODEL_ROOT to alternative cache location
- **THEN** system uses model root as base for model resolution
- **AND** conventional subdirectory path appends to custom root
- **AND** resolved path points to operator-specified location

#### Scenario: Model-specific override

- **WHEN** operator sets DOCLING_PDF_MODEL to explicit model directory
- **THEN** system uses exact path from environment variable
- **AND** ignores general cache location variables for this model
- **AND** enables experimentation with alternative model versions

#### Scenario: Command-line precedence

- **WHEN** operator provides --model flag with explicit path
- **THEN** command-line value overrides all environment variable defaults
- **AND** enables per-invocation model selection without environment changes
- **AND** supports rapid iteration and debugging workflows

#### Scenario: Missing model detection

- **WHEN** offline mode enabled and resolved model path does not exist
- **THEN** system raises informative error with missing path details
- **AND** prompts operator to pre-download model or adjust configuration
- **AND** prevents execution with incomplete model cache

### Requirement: Structured Logging Uniformity

The document parsing pipeline SHALL emit all diagnostic, progress, and error information through the structured JSON logging system to enable machine-parseable observability and monitoring integration. The system SHALL reject unstructured output methods including print statements that produce inconsistent formatting and complicate automated log analysis.

All log messages SHALL conform to the JSON schema defined by the common logging formatter including timestamp, level, logger name, message text, and optional extra_fields dictionary for structured context. The pipeline SHALL use appropriate log levels to distinguish informational progress updates, warning conditions, and error states.

#### Scenario: Force mode notification

- **WHEN** HTML pipeline executes with force flag enabled
- **THEN** system emits logger.info message describing force mode activation
- **AND** log message includes structured JSON format
- **AND** no print statement output appears on stdout

#### Scenario: Resume mode notification

- **WHEN** HTML pipeline executes with resume flag enabled
- **THEN** system emits logger.info message describing resume mode activation
- **AND** log message includes structured JSON format
- **AND** no print statement output appears on stdout

#### Scenario: Log parsing automation

- **WHEN** monitoring system ingests pipeline log output
- **THEN** every log line parses successfully as JSON object
- **AND** structured fields enable filtering and aggregation
- **AND** no unstructured text disrupts parsing logic

#### Scenario: Observability dashboard integration

- **WHEN** logs stream to observability platform
- **THEN** platform extracts structured fields for visualization
- **AND** operators query logs using structured field predicates
- **AND** dashboard widgets display metrics derived from extra_fields

### Requirement: Directory Naming Consistency

The document parsing system SHALL maintain consistent naming between code-defined output directories and documentation references to prevent operator confusion and deployment errors. The system SHALL select a canonical directory name for embedding vector outputs and SHALL apply this name uniformly across function implementations, manifest entries, CLI help text, README examples, and configuration templates.

If code defines directory name as "Vectors", documentation SHALL reference "Vectors" throughout. If documentation standardizes on "Embeddings", code SHALL implement data_vectors function to return path with "Embeddings" subdirectory. The system SHALL reject mixed naming where code and documentation refer to different directory names for the same artifact category.

#### Scenario: Documentation matches code implementation

- **WHEN** operator reads README to understand output directory structure
- **THEN** documented directory names match paths created by pipeline execution
- **AND** example commands reference actual output locations
- **AND** troubleshooting guides describe correct directory layout

#### Scenario: Code matches documentation standard

- **WHEN** documentation establishes "Vectors" as canonical name
- **THEN** data_vectors function returns path ending in "Vectors" subdirectory
- **AND** manifest entries record output paths using "Vectors" directory
- **AND** CLI help text references "Vectors" in descriptions

#### Scenario: Migration path for name changes

- **WHEN** project selects directory name different from current code
- **THEN** change proposal includes migration script or instructions
- **AND** deployment documentation notes directory name transition
- **AND** operators receive clear guidance on handling existing outputs

### Requirement: Multiprocessing Safety Standardization

The document parsing pipeline SHALL enforce spawn start method for multiprocessing contexts to guarantee CUDA operation safety in worker processes. The system SHALL centralize spawn configuration logic in a shared utility function to prevent divergent implementations and ensure uniform error handling across pipeline entry points.

The spawn configuration helper SHALL attempt to set the start method with force flag enabled to override any previously configured method. On RuntimeError indicating the method was already set, the helper SHALL check current method and emit warning if different from spawn. The helper SHALL accept optional logger parameter for diagnostic output integration.

Both PDF and HTML pipeline main functions SHALL invoke the spawn configuration helper during initialization before creating process pools or executors. This ensures all child processes inherit the spawn start method regardless of entry point or execution context.

#### Scenario: Fresh process spawn configuration

- **WHEN** pipeline initializes in process with no prior multiprocessing configuration
- **THEN** set_spawn_or_warn successfully sets spawn as start method
- **AND** logs confirmation message if logger provided
- **AND** subsequent worker processes use spawn semantics

#### Scenario: Spawn method already configured correctly

- **WHEN** pipeline initializes after external code sets spawn method
- **THEN** set_spawn_or_warn handles RuntimeError gracefully
- **AND** verifies current method is spawn
- **AND** continues execution without warnings

#### Scenario: Incompatible method previously set

- **WHEN** pipeline initializes after external code sets fork or forkserver method
- **THEN** set_spawn_or_warn detects method mismatch
- **AND** emits warning about CUDA safety risk
- **AND** includes current method name in warning message
- **AND** allows execution to continue with degraded safety guarantees

#### Scenario: Uniform behavior across pipelines

- **WHEN** comparing PDF and HTML pipeline initialization
- **THEN** both pipelines invoke identical spawn configuration helper
- **AND** both produce consistent diagnostic output
- **AND** both provide equivalent CUDA safety guarantees

### Requirement: Legacy Module Deprecation Path

The document parsing system SHALL phase out legacy module shim mechanisms that create synthetic submodules for backward compatibility. The system SHALL emit deprecation warnings when legacy import paths are accessed to notify users of impending removal while maintaining temporary compatibility.

Import statements targeting DocsToKG.DocParsing.pdf_pipeline or DocsToKG.DocParsing.html_pipeline SHALL trigger DeprecationWarning with message indicating the preferred import path and timeline for removal. The warning SHALL include specific guidance directing users to import from DocsToKG.DocParsing.pipelines instead.

Documentation SHALL reference only the canonical pipelines module in examples, API guides, and tutorials. Test suites SHALL migrate to direct pipelines module usage or SHALL use monkeypatching on pipelines module instead of relying on legacy shims.

#### Scenario: Legacy PDF pipeline import

- **WHEN** user code imports from DocsToKG.DocParsing.pdf_pipeline
- **THEN** system emits DeprecationWarning during import
- **AND** warning message identifies legacy module name
- **AND** warning message specifies replacement import path
- **AND** warning message states removal timeline
- **AND** imported symbols function correctly maintaining temporary compatibility

#### Scenario: Legacy HTML pipeline import

- **WHEN** user code imports from DocsToKG.DocParsing.html_pipeline
- **THEN** system emits DeprecationWarning during import
- **AND** warning content parallels PDF pipeline deprecation message
- **AND** imported symbols function correctly maintaining temporary compatibility

#### Scenario: Canonical import path

- **WHEN** user code imports from DocsToKG.DocParsing.pipelines
- **THEN** no deprecation warnings emit
- **AND** all pipeline functions available directly
- **AND** import represents future-proof supported path

#### Scenario: Documentation guidance

- **WHEN** developers consult README or API documentation
- **THEN** examples demonstrate pipelines module imports exclusively
- **AND** no legacy module references appear in current guidance
- **AND** migration notes explain deprecated paths for existing code

#### Scenario: Test suite modernization

- **WHEN** test suite executes against current codebase
- **THEN** tests import from pipelines module directly
- **AND** monkeypatching targets functions in pipelines namespace
- **AND** no test failures occur due to deprecation warnings
- **AND** test fixtures demonstrate preferred patterns for users
