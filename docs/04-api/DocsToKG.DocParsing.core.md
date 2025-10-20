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

The package previously exposed thin wrappers such as ``core.doctags()`` and
``core.embed()`` for direct CLI invocation. Those helpers have been removed as
part of the Typer migration—callers should now import
``DocsToKG.DocParsing.core.cli`` and execute the ``Typer`` application (or the
private ``_execute_*`` helpers) instead.

## 2. Functions

### `_ensure_str_sequence(value, label)`

Normalise structural marker entries into string lists.

### `load_structural_marker_profile(path)`

Load heading/caption marker overrides from JSON, YAML, or TOML files.

### `load_structural_marker_config(path)`

Backward compatible alias for :func:`load_structural_marker_profile`.

### `build_subcommand(parser, options)`

Attach CLI options described by ``options`` to ``parser``.

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

### `_coerce_pair(values)`

Coerce timeout sequences to ``(connect, read)`` floats.

### `entry(self, doc_id)`

Return the manifest entry associated with ``doc_id`` when available.

### `should_skip(self, doc_id, output_path, input_hash)`

Return ``True`` when work for ``doc_id`` can be safely skipped.

### `should_process(self, doc_id, output_path, input_hash)`

Return ``True`` when ``doc_id`` requires processing.

### `_walk(current)`

*No documentation available.*

### `_length_bucket(length)`

Return the power-of-two bucket for ``length``.

### `_ordered_indices(self)`

*No documentation available.*

### `__iter__(self)`

*No documentation available.*

## 3. Classes

### `CLIOption`

Declarative CLI argument specification used by ``build_subcommand``.

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
