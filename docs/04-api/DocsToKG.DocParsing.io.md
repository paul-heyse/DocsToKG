# 1. Module: io

This reference documents the DocsToKG module ``DocsToKG.DocParsing.io``.

## 1. Overview

Low-level I/O helpers shared across DocParsing stages.

This module houses JSONL streaming utilities, atomic write helpers, and manifest
bookkeeping routines. It deliberately avoids importing the CLI-facing modules so
that other packages can depend on these primitives without pulling in heavy
dependencies.

## 2. Functions

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

Yield DocTags files within ``directory`` and subdirectories while preserving
their logical locations beneath the provided root. Entries that resolve to the
same filesystem target are emitted once, preferring real files over symbolic
links and returning results sorted by their logical path.

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

### `_select_hash_algorithm(requested, default)`

Return a supported hash algorithm honouring env overrides and defaults.

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

### `iter_manifest_entries(stages, root, *, limit=None)`

Yield manifest entries for the requested ``stages`` sorted by timestamp.

When ``limit`` is provided the reader only loads the newest ``limit`` rows from
each manifest file using bounded tail windows, which avoids scanning the full
history for tail-only inspections.
