# 1. Module: discovery

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core.discovery``.

## 1. Overview

Path and artifact discovery helpers for DocParsing pipelines.

## 2. Functions

### `_ensure_str_sequence(value, label)`

Normalise structural marker entries into string lists.

### `load_structural_marker_profile(path)`

Load heading/caption marker overrides from JSON, YAML, or TOML files.

### `load_structural_marker_config(path)`

Backward compatible alias for :func:`load_structural_marker_profile`.

### `derive_doc_id_and_doctags_path(source_pdf, pdfs_root, doctags_root)`

Return manifest doc identifier and DocTags output path for ``source_pdf``.

### `derive_doc_id_and_chunks_path(doctags_file, doctags_root, chunks_root)`

Return manifest doc identifier and chunk output path for ``doctags_file``.

### `derive_doc_id_and_vectors_path(chunk_file, chunks_root, vectors_root)`

Return manifest doc identifier and vectors output path for ``chunk_file``.

### `compute_relative_doc_id(path, root)`

Return POSIX-style relative identifier for a document path.

### `compute_stable_shard(identifier, shard_count)`

Deterministically map ``identifier`` to a shard in ``[0, shard_count)``.

### `iter_chunks(directory)`

Yield :class:`ChunkDiscovery` records for chunk files under ``directory``.

### `__fspath__(self)`

Return the resolved path for :func:`os.fspath` compatibility.

### `_walk(current)`

Yield chunk files beneath ``current`` depth-first with symlink guards.

## 3. Classes

### `ChunkDiscovery`

Discovery record that retains logical and resolved chunk paths.
