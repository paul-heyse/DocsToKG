# 1. Module: io

This reference documents the DocsToKG module ``DocsToKG.DocParsing.io``.

## 1. Overview

Low-level I/O helpers shared across DocParsing stages.

This module houses JSONL streaming utilities, atomic write helpers, and manifest
bookkeeping routines. It deliberately avoids importing the CLI-facing modules so
that other packages can depend on these primitives without pulling in heavy
dependencies.

## 2. Functions

### `_partition_normalisation_buffer(buffer)`

Split ``buffer`` into a flushable prefix and a retained suffix.

The suffix preserves the trailing grapheme cluster so that Unicode
normalisation remains stable when additional combining marks are read from
subsequent chunks.

### `_iter_normalised_text_chunks(handle)`

Yield UTF-8 encoded NFKC-normalised chunks from ``handle``.

### `atomic_write(path)`

Write to a temporary file and atomically replace the destination.

### `iter_jsonl(path)`

Stream JSONL records from ``path`` without materialising the full file.

### `iter_jsonl_batches(paths, batch_size)`

Yield JSONL rows from ``paths`` in batches of ``batch_size`` records.

### `dedupe_preserve_order(items)`

Return ``items`` without duplicates while preserving encounter order.

### `jsonl_load(path, skip_invalid, max_errors)`

Load a JSONL file into memory with optional error tolerance.

.. deprecated:: 0.2.0
Use :func:`iter_jsonl` or :func:`iter_jsonl_batches` for streaming access.

### `jsonl_save(path, rows, validate)`

Persist dictionaries to a JSONL file atomically.

### `jsonl_append_iter(target, rows)`

Append JSON-serialisable rows to a JSONL file.

### `build_jsonl_split_map(path)`

Return newline-aligned byte ranges that partition ``path``.

### `iter_doctags(directory)`

Yield DocTags files within ``directory`` and subdirectories.

The returned paths retain their logical location beneath ``directory`` even
when they are symbolic links. Files that resolve to the same on-disk target
are emitted once, preferring concrete files over symlinks and ordering the
results lexicographically by their logical (relative) path.

### `_iter_jsonl_records(path)`

Yield JSON-decoded records between optional byte offsets.

### `_sanitise_stage(stage)`

Return a filesystem-friendly identifier for ``stage``.

### `_telemetry_filename(stage, kind)`

Return a telemetry filename for ``stage`` and ``kind``.

### `_manifest_filename(stage)`

Return manifest filename for a given stage.

### `manifest_append(stage, doc_id, status)`

Append a structured entry to the processing manifest.

### `resolve_manifest_path(stage, root)`

Return the manifest path for ``stage`` relative to ``root``.

### `resolve_attempts_path(stage, root)`

Return the attempts log path for ``stage`` relative to ``root``.

### `_normalise_hash_name(candidate)`

Normalise algorithm names for comparison against hashlib.

### `_hash_algorithms_available()`

Return the cached set of available hashlib algorithms.

### `_select_hash_algorithm(requested, default)`

Return a supported hash algorithm honouring env overrides and defaults.

### `_select_hash_algorithm_uncached(requested, default, env_override)`

Resolve a hash algorithm without consulting the selection cache.

### `_clear_hash_algorithm_cache()`

Reset memoized hash algorithm selections (intended for testing).

### `resolve_hash_algorithm(default)`

Return the active content hash algorithm, guarding invalid overrides.

### `make_hasher(name)`

Return a configured hashlib object with guarded algorithm resolution.

### `compute_chunk_uuid(doc_id, start_offset, text)`

Compute a deterministic UUID for a chunk of text.

### `relative_path(path, root)`

Return ``path`` rendered relative to ``root`` when feasible.

### `quarantine_artifact(path, reason)`

Move ``path`` to a ``.quarantine`` sibling for operator review.

### `compute_content_hash(path, algorithm)`

Compute a content hash for ``path`` using the requested algorithm.

### `load_manifest_index(stage, root)`

Load the latest manifest entries for a specific pipeline stage.

### `_manifest_timestamp_key(entry)`

Return a sortable timestamp key or ``None`` when unavailable.

### `_iter_manifest_tail_lines(path, limit)`

Yield the newest ``limit`` JSONL lines from ``path`` in chronological order.

### `_iter_manifest_file(path, stage)`

Yield manifest entries for a single stage file.

### `iter_manifest_entries(stages, root)`

Yield manifest entries for ``stages`` sorted by timestamp.

When ``limit`` is provided, only the newest ``limit`` rows per manifest file are
read using bounded tail windows, reducing the amount of history that needs to
be scanned.

### `update(self, text)`

Ingest ``text`` into the hash while preserving Unicode normalisation semantics.

### `hexdigest(self)`

Finalize the digest and return the hexadecimal representation.

### `__lt__(self, other)`

*No documentation available.*

### `__eq__(self, other)`

*No documentation available.*

### `_push_entry(entry, stream)`

*No documentation available.*

## 3. Classes

### `StreamingContentHasher`

Incrementally compute a content hash that mirrors :func:`compute_content_hash`.

### `_ManifestHeapKey`

Comparable key that preserves manifest order when timestamps are missing.
