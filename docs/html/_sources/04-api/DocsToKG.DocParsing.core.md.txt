# 1. Module: core

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core``.

## 1. Overview

DocParsing Core Utilities

This module centralises lightweight helpers that power multiple DocParsing
pipeline stages. Utilities span path discovery, atomic file writes, JSONL
parsing, manifest bookkeeping, CLI glue, and structured logging so that
chunking, embedding, and conversion scripts can share consistent behaviour
without an additional dependency layer.

Key Features:
- Resolve DocsToKG data directories with environment and ancestor discovery
- Stream JSONL inputs and outputs with validation and error tolerance
- Emit structured JSON logs suited for machine ingestion and dashboards
- Manage pipeline manifests, batching helpers, and advisory file locks

Usage:
    from DocsToKG.DocParsing import core

    chunks_dir = core.data_chunks()
    with core.atomic_write(chunks_dir / "example.jsonl") as handle:
        handle.write("{}")

Dependencies:
- json, pathlib, logging: Provide standard I/O and diagnostics primitives.
- typing: Supply type hints consumed by Sphinx documentation tooling and API generators.
- pydantic (optional): Some helpers integrate with schema validation routines.

All helpers are safe to import in multiprocessing contexts and avoid heavy
third-party dependencies beyond the standard library.

## 2. Functions

### `dedupe_preserve_order(markers)`

Return ``markers`` without duplicates while preserving input order.

### `_ensure_str_sequence(value, label)`

Normalise structural marker entries into string lists.

### `_load_yaml_markers(raw)`

Deserialize YAML marker overrides, raising when PyYAML is unavailable.

### `_load_toml_markers(raw)`

Deserialize TOML marker definitions with compatibility fallbacks.

### `load_structural_marker_profile(path)`

Load heading/caption marker overrides from JSON, YAML, or TOML files.

### `load_structural_marker_config(path)`

Backward compatible alias for :func:`load_structural_marker_profile`.

### `build_subcommand(parser, options)`

Attach CLI options described by ``options`` to ``parser``.

### `_coerce_path(value, base_dir)`

Convert ``value`` into an absolute :class:`Path`.

### `_coerce_optional_path(value, base_dir)`

Convert optional path-like values.

### `_coerce_bool(value, _base_dir)`

Convert truthy strings or numbers to boolean.

### `_coerce_int(value, _base_dir)`

Convert ``value`` to ``int``.

### `_coerce_float(value, _base_dir)`

Convert ``value`` to ``float``.

### `_coerce_str(value, _base_dir)`

Return ``value`` coerced to string.

### `_coerce_str_tuple(value, _base_dir)`

Return ``value`` as a tuple of strings.

### `normalize_http_timeout(timeout)`

Normalize timeout inputs into a ``(connect, read)`` tuple of floats.

### `get_http_session()`

Return a shared :class:`requests.Session` configured with retries.

Args:
timeout: Optional override for the ``(connect, read)`` timeout tuple. Scalars
override only the read timeout while preserving the default connect value.
base_headers: Headers merged into the shared session.
retry_total: Maximum retry attempts applied for connect/read failures.
retry_backoff: Exponential backoff factor between retries.
status_forcelist: HTTP status codes that trigger retries.
allowed_methods: HTTP verbs eligible for retries.

Returns:
Tuple containing the shared session and the effective timeout tuple.

### `_manifest_value(value)`

Convert values to manifest-friendly representations.

### `load_config_mapping(path)`

Load a configuration mapping from JSON, YAML, or TOML.

### `expand_path(path)`

Return ``path`` expanded to an absolute :class:`Path`.

Args:
path: Candidate filesystem path supplied as string or :class:`Path`.

Returns:
Absolute path with user home components resolved.

### `resolve_hf_home()`

Resolve the HuggingFace cache directory respecting ``HF_HOME``.

Args:
None

Returns:
Path: Absolute path to the HuggingFace cache directory.

### `resolve_model_root(hf_home)`

Resolve the DocsToKG model root honoring ``DOCSTOKG_MODEL_ROOT``.

Args:
hf_home: Optional HuggingFace cache directory to treat as the base path.

Returns:
Path: Absolute directory where DocsToKG models should be stored.

### `looks_like_filesystem_path(candidate)`

Return ``True`` when ``candidate`` appears to reference a local path.

### `resolve_pdf_model_path(cli_value)`

Determine PDF model path using CLI and environment precedence.

### `init_hf_env(hf_home, model_root)`

Initialise Hugging Face and transformer cache environment variables.

Args:
hf_home: Optional explicit HF cache directory.
model_root: Optional DocsToKG model root override.

Returns:
Tuple of ``(hf_home, model_root)`` paths after normalisation.

### `_detect_cuda_device()`

Best-effort detection of CUDA availability to choose a default device.

### `ensure_model_environment(hf_home, model_root)`

Initialise and cache the HuggingFace/model-root environment settings.

### `_ensure_optional_dependency(module_name, message)`

Import ``module_name`` or raise with ``message``.

### `ensure_splade_dependencies(import_error)`

Validate that SPLADE optional dependencies are importable.

### `ensure_qwen_dependencies(import_error)`

Validate that Qwen/vLLM optional dependencies are importable.

### `ensure_splade_environment()`

Bootstrap SPLADE-related environment defaults and return resolved settings.

### `ensure_qwen_environment()`

Bootstrap Qwen/vLLM environment defaults and return resolved settings.

### `detect_data_root(start)`

Locate the DocsToKG Data directory via env var or ancestor scan.

Checks the ``DOCSTOKG_DATA_ROOT`` environment variable first. If not set,
scans ancestor directories for a ``Data`` folder containing expected
subdirectories (``PDFs``, ``HTML``, ``DocTagsFiles``, or
``ChunkedDocTagFiles``).

Args:
start: Starting directory for the ancestor scan. Defaults to the
current working directory when ``None``.

Returns:
Absolute path to the resolved ``Data`` directory. When
``DOCSTOKG_DATA_ROOT`` is set but the directory does not yet exist,
it is created automatically.

Examples:
>>> os.environ["DOCSTOKG_DATA_ROOT"] = "/tmp/data"
>>> (Path("/tmp/data")).mkdir(parents=True, exist_ok=True)
>>> detect_data_root()
PosixPath('/tmp/data')

>>> os.environ.pop("DOCSTOKG_DATA_ROOT")
>>> detect_data_root(Path("/workspace/DocsToKG/src"))
PosixPath('/workspace/DocsToKG/Data')

### `_ensure_dir(path)`

Create ``path`` if needed and return its absolute form.

Args:
path: Directory to create when missing.

Returns:
Absolute path to the created directory.

Examples:
>>> _ensure_dir(Path("./tmp_dir"))
PosixPath('tmp_dir')

### `data_doctags(root)`

Return the DocTags directory and ensure it exists.

Args:
root: Optional override for the starting directory used when
resolving the DocsToKG data root.

Returns:
Absolute path to the DocTags directory.

Examples:
>>> isinstance(data_doctags(), Path)
True

### `data_chunks(root)`

Return the chunk directory and ensure it exists.

Args:
root: Optional override for the starting directory used when
resolving the DocsToKG data root.

Returns:
Absolute path to the chunk directory.

Examples:
>>> isinstance(data_chunks(), Path)
True

### `data_vectors(root)`

Return the vectors directory and ensure it exists.

Args:
root: Optional override for the starting directory used when
resolving the DocsToKG data root.

Returns:
Absolute path to the vectors directory.

Examples:
>>> isinstance(data_vectors(), Path)
True

### `data_manifests(root)`

Return the manifests directory and ensure it exists.

Args:
root: Optional override for the starting directory used when
resolving the DocsToKG data root.

Returns:
Absolute path to the manifests directory.

Examples:
>>> isinstance(data_manifests(), Path)
True

### `prepare_data_root(data_root_arg, default_root)`

Resolve and prepare the DocsToKG data root for a pipeline invocation.

### `resolve_pipeline_path()`

Derive a pipeline directory path respecting data-root overrides.

### `data_pdfs(root)`

Return the PDFs directory and ensure it exists.

Args:
root: Optional override for the starting directory used when
resolving the DocsToKG data root.

Returns:
Absolute path to the PDFs directory.

Examples:
>>> isinstance(data_pdfs(), Path)
True

### `data_html(root)`

Return the HTML directory and ensure it exists.

Args:
root: Optional override for the starting directory used when
resolving the DocsToKG data root.

Returns:
Absolute path to the HTML directory.

Examples:
>>> isinstance(data_html(), Path)
True

### `derive_doc_id_and_doctags_path(source_pdf, pdfs_root, doctags_root)`

Return manifest doc identifier and DocTags output path for ``source_pdf``.

### `derive_doc_id_and_chunks_path(doctags_file, doctags_root, chunks_root)`

Return manifest doc identifier and chunk output path for ``doctags_file``.

### `derive_doc_id_and_vectors_path(chunk_file, chunks_root, vectors_root)`

Return manifest doc identifier and vectors output path for ``chunk_file``.

Args:
chunk_file: Path to the chunk JSONL artefact.
chunks_root: Root directory containing chunk artefacts.
vectors_root: Root directory where vector outputs should be written.

Returns:
Tuple containing the manifest ``doc_id`` and the full vectors output path.

### `compute_relative_doc_id(path, root)`

Return POSIX-style relative identifier for a document path.

Args:
path: Absolute path to the document on disk.
root: Root directory that anchors relative identifiers.

Returns:
str: POSIX-style relative path suitable for manifest IDs.

### `compute_stable_shard(identifier, shard_count)`

Deterministically map ``identifier`` to a shard in ``[0, shard_count)``.

### `should_skip_output(output_path, manifest_entry, input_hash, resume, force)`

Return ``True`` when resume/skip conditions indicate work can be skipped.

### `_stringify_path(value)`

Return a string representation for path-like values used in manifests.

### `manifest_log_skip()`

Record a manifest entry indicating the pipeline skipped work.

Args:
stage: Logical pipeline phase originating the log entry.
doc_id: Identifier of the document being processed.
input_path: Source artefact that would have been processed.
input_hash: Content hash associated with ``input_path``.
output_path: Destination artefact that remained unchanged.
duration_s: Elapsed seconds for the short-circuited step.
schema_version: Manifest schema version for downstream readers.
hash_alg: Hash algorithm used to compute ``input_hash``.
**extra: Additional metadata to merge into the manifest row.

### `manifest_log_success()`

Record a manifest entry marking successful pipeline output.

Args:
stage: Logical pipeline phase originating the log entry.
doc_id: Identifier of the document being processed.
duration_s: Elapsed seconds for the successful step.
schema_version: Manifest schema version for downstream readers.
input_path: Source artefact that produced ``output_path``.
input_hash: Content hash associated with ``input_path``.
output_path: Destination artefact written by the pipeline.
hash_alg: Hash algorithm used to compute ``input_hash``.
**extra: Additional metadata to merge into the manifest row.

### `manifest_log_failure()`

Record a manifest entry describing a failed pipeline attempt.

Args:
stage: Logical pipeline phase originating the log entry.
doc_id: Identifier of the document being processed.
duration_s: Elapsed seconds before the failure occurred.
schema_version: Manifest schema version for downstream readers.
input_path: Source artefact that triggered the failure.
input_hash: Content hash associated with ``input_path``.
output_path: Destination artefact that may be incomplete.
error: Human-readable description of the failure condition.
hash_alg: Hash algorithm used to compute ``input_hash``.
**extra: Additional metadata to merge into the manifest row.

### `get_logger(name, level)`

Get a structured JSON logger configured for console output.

Args:
name: Name of the logger to create or retrieve.
level: Logging level (case insensitive). Defaults to ``"INFO"``.

Returns:
Configured :class:`logging.Logger` instance.

Examples:
>>> logger = get_logger("docparse")
>>> logger.level == logging.INFO
True

### `log_event(logger, level, message)`

Emit a structured log record using the ``extra_fields`` convention.

### `find_free_port(start, span)`

Locate an available TCP port on localhost within a range.

Args:
start: Starting port for the scan. Defaults to ``8000``.
span: Number of sequential ports to check. Defaults to ``32``.

Returns:
The first free port number. Falls back to an OS-assigned ephemeral port
if the requested range is exhausted.

Examples:
>>> port = find_free_port(8500, 1)
>>> isinstance(port, int)
True

### `atomic_write(path)`

Write to a temporary file and atomically replace the destination.

Pattern: open a sibling ``*.tmp`` file, write the payload, flush and
``fsync`` the descriptor, then ``rename`` it over the original path. This
guarantees that readers never observe a partially written file even if the
process crashes mid-write.

Args:
path: Target path to write.

Returns:
Context manager yielding a writable text handle.

Yields:
Writable text file handle. Caller must write data before context exit.

Raises:
Any exception raised while writing or replacing the file is propagated
after the temporary file is cleaned up.

Examples:
>>> target = Path("/tmp/example.txt")
>>> with atomic_write(target) as handle:
...     _ = handle.write("hello")

### `iter_doctags(directory)`

Yield DocTags files within ``directory`` and subdirectories.

Args:
directory: Root directory to scan for DocTags artifacts.

Returns:
Iterator over absolute ``Path`` objects.

Yields:
Absolute paths to discovered ``.doctags`` or ``.doctag`` files sorted
lexicographically.

Examples:
>>> next(iter_doctags(Path(".")), None) is None
True

### `iter_chunks(directory)`

Yield chunk JSONL files from ``directory`` and all descendants.

Args:
directory: Directory containing chunk artifacts.

Returns:
Iterator over absolute ``Path`` objects.

Yields:
Absolute paths to files matching ``*.chunks.jsonl`` sorted
lexicographically.

Examples:
>>> next(iter_chunks(Path(".")), None) is None
True

### `jsonl_load(path, skip_invalid, max_errors)`

Load a JSONL file into memory with optional error tolerance.

### `jsonl_save(path, rows, validate)`

Persist dictionaries to a JSONL file atomically.

Args:
path: Destination JSONL file.
rows: Sequence of dictionaries to serialize.
validate: Optional callback invoked per row before serialization.

Returns:
None: This function performs I/O side effects only.

Raises:
ValueError: If ``validate`` raises an exception for any row.

Examples:
>>> tmp = Path("/tmp/example.jsonl")
>>> jsonl_save(tmp, [{"a": 1}])
>>> tmp.read_text(encoding="utf-8").strip()
'{"a": 1}'

### `jsonl_append_iter(target, rows)`

Append JSON-serialisable rows to a JSONL file.

Args:
target: Destination path or writable handle for the JSONL file.
rows: Iterable of JSON-serialisable mappings.
atomic: When True, writes occur via :func:`atomic_write` (ignored when
``target`` is already an open handle).

Returns:
The number of rows written.

### `build_jsonl_split_map(path)`

Return newline-aligned byte ranges that partition ``path``.

### `_iter_jsonl_records(path)`

*No documentation available.*

### `_manifest_filename(stage)`

Return manifest filename for a given stage.

### `manifest_append(stage, doc_id, status)`

Append a structured entry to the processing manifest.

Args:
stage: Pipeline stage emitting the entry.
doc_id: Identifier of the document being processed.
status: Outcome status (``success``, ``failure``, or ``skip``).
duration_s: Optional duration in seconds.
warnings: Optional list of warning labels.
error: Optional error description.
schema_version: Schema identifier recorded for the output.
**metadata: Arbitrary additional fields to include.

Returns:
``None``.

Raises:
ValueError: If ``status`` is not recognised.

Examples:
>>> manifest_append("chunk", "doc1", "success")
>>> (data_manifests() / "docparse.chunk.manifest.jsonl").exists()
True

### `resolve_hash_algorithm(default)`

Return the active content hash algorithm, honoring env overrides.

Args:
default: Fallback algorithm name to use when no override is present.

Returns:
Hash algorithm identifier resolved from ``DOCSTOKG_HASH_ALG`` or ``default``.

### `compute_chunk_uuid(doc_id, start_offset, text)`

Derive a deterministic UUID for a chunk using doc ID, offset, and text content.

Args:
doc_id: Identifier for the source document (used as a namespace component).
start_offset: Character offset of the chunk text within the document.
text: Chunk text used for content-based stability.
algorithm: Hash algorithm name; defaults to ``sha1`` but honours
:envvar:`DOCSTOKG_HASH_ALG` overrides.

Returns:
UUID string derived from the hash digest while enforcing RFC4122 metadata bits.

### `relative_path(path, root)`

Return ``path`` rendered relative to ``root`` when feasible.

### `quarantine_artifact(path, reason)`

Move ``path`` to a ``.quarantine`` sibling for operator review.

Args:
path: Artefact to quarantine.
reason: Explanation describing why the artefact was quarantined.
logger: Optional logger used to emit structured diagnostics.
create_placeholder: When ``True`` a placeholder file is created even if
``path`` does not presently exist (useful for failed writes).

Returns:
Path to the quarantined artefact.

### `compute_content_hash(path, algorithm)`

Compute a content hash for ``path`` using the requested algorithm.

Args:
path: File whose contents should be hashed.
algorithm: Hash algorithm name supported by :mod:`hashlib`.

Notes:
The ``DOCSTOKG_HASH_ALG`` environment variable overrides ``algorithm``
when set, enabling fleet-wide hash changes without code edits.

Returns:
Hex digest string.

Examples:
>>> tmp = Path("/tmp/hash.txt")
>>> _ = tmp.write_text("hello", encoding="utf-8")
>>> compute_content_hash(tmp) == hashlib.sha1(b"hello").hexdigest()
True

### `load_manifest_index(stage, root)`

Load the latest manifest entries for a specific pipeline stage.

Args:
stage: Manifest stage identifier to filter entries by.
root: Optional DocsToKG data root used to resolve the manifest path.

Returns:
Mapping of ``doc_id`` to the most recent manifest entry for that stage.

Raises:
None: Manifest rows that fail to parse are skipped to keep processing resilient.

Examples:
>>> index = load_manifest_index("embeddings")  # doctest: +SKIP
>>> isinstance(index, dict)
True

### `iter_manifest_entries(stages, root)`

Yield manifest entries for the requested ``stages`` sorted by timestamp.

### `summarize_manifest(entries)`

Compute status counts and durations for manifest ``entries``.

### `acquire_lock(path, timeout)`

Acquire an advisory lock using ``.lock`` sentinel files.

Args:
path: Target file path whose lock should be acquired.
timeout: Maximum time in seconds to wait for the lock.

Returns:
Iterator yielding a boolean when the lock is acquired.

Yields:
``True`` once the lock is acquired.

Raises:
TimeoutError: If the lock cannot be obtained within ``timeout``.

Examples:
>>> target = Path("/tmp/lock.txt")
>>> with acquire_lock(target):
...     pass

### `_pid_is_running(pid)`

Return ``True`` if a process with the given PID appears to be alive.

### `set_spawn_or_warn(logger)`

Ensure the multiprocessing start method is set to ``spawn``.

Args:
logger: Optional logger that receives diagnostic messages about the start
method configuration.

Returns:
None: The function mutates global multiprocessing state and logs warnings.

This helper attempts to set the start method to ``spawn`` with ``force=True``.
If a ``RuntimeError`` occurs (meaning the method was already set), it checks
if the current method is ``spawn``. If not, it emits a warning about the
potential CUDA safety risk, logging the current method so callers understand
the degraded safety state.

### `_run_chunk(argv)`

Execute the Docling chunker subcommand.

### `_run_embed(argv)`

Execute the embedding pipeline subcommand.

### `_run_token_profiles(argv)`

Execute the tokenizer profiling subcommand.

### `_run_plan(argv)`

Display the doctags → chunk → embed plan without executing.

### `_run_manifest(argv)`

Inspect pipeline manifest artifacts via CLI.

### `_build_doctags_parser(prog)`

Create an :mod:`argparse` parser configured for DocTags conversion.

### `_scan_pdf_html(input_dir)`

Return booleans indicating whether PDFs or HTML files exist beneath ``input_dir``.

### `_directory_contains_suffixes(directory, suffixes)`

Return True when ``directory`` contains at least one file ending with ``suffixes``.

### `_detect_mode(input_dir)`

Infer conversion mode based on the contents of ``input_dir``.

### `_merge_args(parser, overrides)`

Merge override values into the default parser namespace.

### `_run_doctags(argv)`

Execute the DocTags conversion subcommand.

### `_preview_list(items, limit)`

Return a truncated preview list with remainder hint.

### `_plan_doctags(argv)`

Compute which DocTags inputs would be processed.

### `_plan_chunk(argv)`

Compute which DocTags files the chunk stage would touch.

### `_plan_embed(argv)`

Compute which chunk files the embed stage would process or validate.

### `_display_plan(plans)`

Pretty-print plan summaries to stdout.

### `_run_all(argv)`

Execute DocTags conversion, chunking, and embedding sequentially.

### `main(argv)`

Dispatch to one of the DocParsing subcommands.

### `run_all(argv)`

Public wrapper for the ``all`` subcommand.

### `chunk(argv)`

Public wrapper for the ``chunk`` subcommand.

### `embed(argv)`

Public wrapper for the ``embed`` subcommand.

### `doctags(argv)`

Public wrapper for the ``doctags`` subcommand.

### `token_profiles(argv)`

Public wrapper for the ``token-profiles`` subcommand.

### `plan(argv)`

Public wrapper for the ``plan`` subcommand.

### `manifest(argv)`

Public wrapper for the ``manifest`` subcommand.

### `_coerce_pair(values)`

*No documentation available.*

### `apply_env(self)`

Overlay configuration from environment variables.

### `update_from_file(self, cfg_path)`

Overlay configuration from ``cfg_path``.

### `apply_args(self, args)`

Overlay configuration from an argparse namespace.

### `from_env(cls)`

Instantiate a configuration populated solely from environment variables.

### `finalize(self)`

Hook allowing subclasses to normalise derived fields.

### `to_manifest(self)`

Return a manifest-friendly snapshot of the configuration.

### `_coerce_field(self, name, value, base_dir)`

Run field-specific coercion logic before manifest serialization.

### `is_overridden(self, field_name)`

Return ``True`` when ``field_name`` was explicitly overridden.

### `entry(self, doc_id)`

Return the manifest entry associated with ``doc_id`` when available.

### `should_skip(self, doc_id, output_path, input_hash)`

Return ``True`` when work for ``doc_id`` can be safely skipped.

### `should_process(self, doc_id, output_path, input_hash)`

Return ``True`` when ``doc_id`` requires processing.

### `process(self, msg, kwargs)`

Merge adapter context into ``extra`` metadata for structured output.

### `bind(self)`

Attach additional persistent fields to the adapter and return ``self``.

### `child(self)`

Create a new adapter inheriting context with optional overrides.

### `_length_bucket(length)`

Return the power-of-two bucket for ``length``.

### `_ordered_indices(self)`

*No documentation available.*

### `__iter__(self)`

*No documentation available.*

## 3. Classes

### `CLIOption`

Declarative CLI argument specification used by ``build_subcommand``.

### `StageConfigBase`

Base dataclass for stage configuration objects.

### `BM25Stats`

Corpus-level statistics required for BM25 weighting.

### `SpladeCfg`

Runtime configuration for SPLADE sparse encoding.

### `QwenCfg`

Configuration for generating dense embeddings with Qwen via vLLM.

### `ChunkWorkerConfig`

Lightweight configuration shared across chunker worker processes.

### `ChunkTask`

Work unit describing a single DocTags file to chunk.

### `ChunkResult`

Result envelope emitted by chunker workers.

### `ResumeController`

Centralize resume/force decisions using manifest metadata.

### `StructuredLogger`

Logger adapter that enriches structured logs with shared context.

### `Batcher`

Yield fixed-size batches from an iterable with optional policies.

Args:
iterable: Source iterable providing items to batch.
batch_size: Maximum number of elements per yielded batch.
policy: Optional batching policy. When ``"length"`` the iterable is
bucketed by ``lengths`` before batching.
lengths: Sequence of integer lengths aligned with ``iterable`` used for
length-aware batching policies.

Examples:
>>> list(Batcher([1, 2, 3, 4, 5], 2))
[[1, 2], [3, 4], [5]]

### `_Command`

Callable wrapper storing handler metadata for subcommands.
