Amazing—here’s a **production-grade, AI-agent-ready implementation plan** for **PR-8: Storage & Dataset Layout (Arrow/Parquet first; partitions; footers)**. It’s narrative-only (no code), but specific enough for an agent to implement without guesswork.

---

# Scope & non-goals

**Scope**

* Make **Parquet the default** for Chunks and Vectors (dense, sparse/SPLADE, lexical/BM25).
* Establish **dataset layout**, **partitioning**, **schemas**, **compression/encoding**, **writer/reader contracts**, **metadata/footers**, **CLI/config**, **migration**, **tests**, and **performance tuning**.
* Keep **DocTags** as JSONL (small, line-oriented), at least in this PR.

**Non-goals**

* No new retrieval features or vector quantization (leave PQ/IVF/HNSW for later PRs).
* No remote object store changes (local-first; but design remains fsspec-friendly).

---

# Directory layout (canonical)

Under the active `Data/` root:

```
Data/
  Doctags/{yyyy}/{mm}/{rel_id}.jsonl

  Chunks/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet

  Vectors/{family=dense|sparse|lexical}/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet

  Manifests/...
```

**Rules**

* `rel_id` is the *stable, relative identifier* (usually relative path without extension).
* **Partition keys** live in the path segments (`family=…`, `fmt=…`, `yyyy`, `mm`) for easy dataset discovery with DuckDB/Polars/Arrow.
* No flat directories with thousands of files; year/month partitions keep file count per directory healthy.
* Preserve the existing Manifests directory. We only add extras in manifest rows (see below).

**Config keys (new)**

* `chunks.format`: `parquet` (default) | `jsonl` (escape hatch)
* `vectors.format`: `parquet` (default) | `jsonl`
* `vectors.partitioning`: fixed to `{family}/{fmt}/{yyyy}/{mm}` for now
* `storage.root`: existing data root; just ensure Parquet paths derive from it

---

# Arrow/Parquet schemas (explicit and versioned)

## 1) Chunk rows (Parquet)

**Table schema (columns)**

* `doc_id: string` — relative id
* `chunk_id: int64` — monotonically increasing within `doc_id` (0-based)
* `text: large_string` — the chunk body (UTF-8)
* `tokens: int32` — token count for the model tokenizer used
* `span: struct<start:int32, end:int32>` — character offsets into the source (inclusive/exclusive)
* `section: string` (optional) — high-level structural hint (title, body, table, abstract, etc.)
* `meta: map<string, string>` (optional) — small, flat metadata (keep total < 4 KB per row)
* `created_at: timestamp[us, tz=UTC]` — write time (stage time)
* `schema_version: string` — e.g., `docparse/chunks/1.0.0`

**Constraints & notes**

* **Dictionary encoding** enabled for `section`, any low-cardinality string fields.
* `text` is **LargeString** to avoid 2 GB limits.
* Keep `meta` *small*; heavy JSON belongs in DocTags source, not here.

## 2) Vector rows (Parquet)

Vectors are written **per family** into separate datasets under `Vectors/family=…`.

### 2a) Dense (e.g., Qwen/ST)

* `doc_id: string`
* `chunk_id: int64`
* `dim: int32` — embedding dimension for the row (constant across a run; present for safety)
* `vec: fixed_size_list<float32>[dim]` **or** `list<float32>` (see “Dense layout choice” below)
* `normalize_l2: boolean` — whether vectors were L2-normalized (true by default)
* `created_at: timestamp[us, UTC]`
* `schema_version: string` — `docparse/vectors/dense/1.0.0`
* **File-level metadata** (see “Footers & metadata”): `provider_name`, `model_id@rev`, `dim`, `dtype`, `device`, `cfg_hash`

**Dense layout choice**

* **Primary**: `fixed_size_list<float32>[dim]` if `dim` is constant across a run (almost always). Fastest scan, best compression with `byte_stream_split`.
* **Fallback**: `list<float32>` if dim can differ (rare). The writer decides based on run config; include `dim` per row either way.

### 2b) Sparse (SPLADE)

* `doc_id: string`
* `chunk_id: int64`
* `nnz: int32` — number of non-zeros
* `indices: list<int32>` — vocabulary ids or hashed ids (consistent across run)
* `weights: list<float32>` — same length as `indices`
* `created_at: timestamp[us, UTC]`
* `schema_version: string` — `docparse/vectors/sparse/1.0.0`
* File-level metadata: `sparse_model_id@rev`, `vocab_id` or `hash_scheme`, `postproc.prune_below`, `topk_per_doc`, `normalize_doclen`

**Notes**

* Keep `indices` and `weights` as **parallel lists** (simpler for Polars/DuckDB).
* If you prefer, `values: list<struct<idx:int32, w:float32>>` is also viable; test tool support first.
* Record the **vocab definition** (or hashing scheme) in file metadata.

### 2c) Lexical (BM25)

Two layout options (choose one; keep the other as experimental):

* **Option A (id-space)**: `indices: list<int32>`, `weights: list<float32>`, `nnz: int32`, plus metadata describing the tokenizer and df stats used (vocab local to this run). Best for compactness and speed.
* **Option B (term-space)**: `terms: list<string>` (dictionary-encoded), `weights: list<float32>`, `nnz: int32`. Friendlier for ad-hoc human queries, but larger.

Columns mirror Sparse. Use `schema_version: docparse/vectors/lexical/1.0.0`.

---

# Compression, encoding, and row-group policy

**Defaults (per file unless noted)**

* `compression = "zstd"`, `compression_level = 5` (tuneable via config)
* `write_statistics = true` (per column)
* `use_dictionary = true` for low-cardinality strings; **off** for floats
* **Dense floats**: enable **byte-stream split** encoding (improves Zstd compression and scan speed)
* **Bloom filters**: `doc_id`, `chunk_id` (optional; useful on very large datasets)
* **Row group size** (target size on disk):

  * Chunks: 16–64 MB per row group (e.g., ~5k–20k rows depending on `text` size)
  * Dense: 32–64 MB (e.g., 10k–25k vectors of 768–1536 dims)
  * Sparse/Lexical: group by average `nnz`; aim for ~32 MB per group

**Page size**: keep defaults (mostly fine), but allow advanced tuning flag later.

**File rolling**

* Roll Parquet file when **either**:

  * row count exceeds a threshold (e.g., 100k rows), **or**
  * estimated file size > 256–512 MB
* **File naming**: maintain per-document write (one file per `rel_id`) where feasible; if rolling is needed for jumbo docs, add a suffix `.part-000NN.parquet` and reflect it in the manifest extras for full traceability.

---

# Footers & dataset metadata (critical for provenance)

For **every Parquet file**, populate key-value metadata in the file footer:

* `docparse.schema_version` — same as column value, e.g., `docparse/vectors/dense/1.0.0`
* `docparse.family` — `dense|sparse|lexical` (for Vectors)
* `docparse.model_id` — embedding or SPLADE/BM25 model id (and `@rev` if pinned)
* `docparse.provider` — `dense.qwen_vllm`, `dense.tei`, `dense.sentence_transformers`, `sparse.splade_st`, `lexical.local_bm25`
* `docparse.dim` — for dense
* `docparse.dtype` — `float32` (dense), `float32` (weights), `int32` (indices)
* `docparse.device` — `cpu|cuda:N` (if relevant for dense)
* `docparse.cfg_hash` — stage config hash
* `docparse.created_by` — package name/version
* `docparse.created_at` — ISO UTC

Also write (once per dataset root) **`_common_metadata`**/**`_metadata`** files with Arrow schema and partition discovery hints to accelerate readers.

---

# Writers (stage integration)

**Global rules**

* **Parquet is the default** (configurable). JSONL is an escape hatch for debugging.
* Writers **never** hand-roll file IO; they call a small `ParquetWriter` utility that:

  * enforces schemas,
  * sets compression/encodings/row-group sizing,
  * writes footer metadata,
  * rolls files when thresholds hit,
  * ensures atomicity (write temp → fsync → move), and
  * returns the **actual** path(s) written for manifest logging.

**Chunks stage**

* Build Arrow table from chunk rows; validate schema.
* `Chunks/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet`
* Manifest extras: `chunk_rows`, `row_group_count`, `parquet_bytes`.

**Embedding stage**

* For each enabled family, write a separate file:

  * `Vectors/family=dense/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet`, etc.
* For **dense**:

  * choose **fixed_size_list** if `dim` constant (record `dim` in footer); otherwise list<float32>.
  * set `byte_stream_split` on the float column(s).
* For **sparse/lexical**:

  * ensure **parallel list lengths** (`indices`/`weights`), maintain `nnz` column.
* Manifest extras (per file): `vector_format="parquet"`, `family`, `rows_written`, `row_group_count`, `dim` (dense), `avg_nnz` (sparse), `parquet_bytes`.

**JSONL escape hatch**

* Keep existing JSONL writers behind `--vector-format=jsonl` or `--chunks-format=jsonl`.
* Preserve PR-2 **atomic append** semantics.

---

# Readers (DatasetView helpers)

Create a small **DatasetView** utility with three entrypoints:

* `open_chunks(root, filters=None)` → returns a lazy scan (Polars or DuckDB or Arrow dataset) over `Chunks/{fmt=parquet}/**/*.parquet`
* `open_vectors(root, family, filters=None)` → scan over `Vectors/family=.../{fmt=parquet}/**/*.parquet`
* `open_vectors_all(root, filters=None)` → union across families (with a virtual `family` column) for quick QA

**Features**

* Accept **fsspec URLs** transparently (`file://`, `s3://`, etc.) so future object stores work with no changes.
* Optional `columns` projection, `filters` pruning (pushdown).
* For **dense** fixed-size vectors, expose a helper to materialize a NumPy matrix for a slice (`doc_id == X`) without full table load.

**Inspect CLI**

* `docparse inspect <dataset>`:

  * prints dataset type (Chunks or Vectors + family),
  * Arrow schema,
  * row count (approximate via statistics, or exact if requested),
  * sample partitions (yyyy/mm),
  * min/max timestamps,
  * file count & total bytes.

---

# Manifests & provenance (traceability)

* **Config row** for embeddings already records `vector_format`; ensure it’s **always** present and matches the writer.
* **Per-file success rows** include:

  * `vector_format="parquet"` (or `jsonl`),
  * `family` (for Vectors),
  * `parquet_bytes`, `row_group_count`,
  * `dim` (dense), `avg_nnz` (sparse),
  * write timing.
* If rolling produces multiple parts for one `rel_id`, record an array of `output_paths` (or emit multiple success rows—choose one policy and document it).

---

# Migration & backfill (from JSONL)

**One-time utilities**

* `docparse convert chunks --from jsonl --to parquet --in <dir> --out <dir>` (stream JSONL → Parquet writer)
* `docparse convert vectors --family dense|sparse|lexical --from jsonl --to parquet ...`
* Keep **idempotent**: converters skip rows/files if Parquet already exists **and** fingerprints/bytes match.

**Validation**

* Converter performs row-count parity checks and (for dense) verifies Euclidean norms if normalize=true.
* After conversion, write a manifest success row for the Parquet artifact(s).

**Rollback**

* CLI flags always allow `--vector-format=jsonl` and `--chunks-format=jsonl`.
* Make format selection visually clear in `config show` and manifests.

---

# Acceptance criteria

* **Default** runs write Chunks and all enabled Vectors as Parquet with the layout above.
* `docparse inspect` can summarize datasets quickly.
* DuckDB/Polars scans are **fast and memory-stable** on a mini corpus (e.g., 50–100k chunks / vectors).
* Manifests accurately reflect file type (`vector_format`), family, and partitioning.

---

# Test plan (CI & local)

## Schema validation

* For each dataset, open Arrow schema; assert required columns & logical types match spec.
* For dense (fixed_size_list), verify `list_size == dim` from metadata.

## Round-trip

* Write Parquet → read with Arrow/Polars/DuckDB → sample queries:

  * Chunks: average tokens by `section`; search `doc_id` prefix; regex on text.
  * Dense: compute cosine between two chunks from one doc (read a slice only).
  * Sparse: reconstruct CSR for a doc; check `sum(nnz) == total nnz`.
  * Lexical: sample few rows and confirm `nnz == len(terms) == len(weights)`.

## Performance sanity

* Time scan of 100k dense vectors in DuckDB; confirm sub-second schema inspect, and <few seconds simple aggregates, within reference hardware expectations.
* Verify memory headroom (no unexpected blowups with Polars lazy scans).

## File rolling (optional now, required later)

* Create a synthetic jumbo doc to trigger rolling; assert file name suffixing and manifest rows list all parts.

## Backward compatibility

* Run with `--vector-format=jsonl` and assert outputs match old JSONL shape.

---

# Performance & space tuning (knobs & best practices)

* **Zstd level**: default 5; allow 3 (faster) to 9 (smaller). Provide a config key `parquet.compression_level`.
* **Row group targeting**: expose `parquet.target_rowgroup_mb` per dataset (Chunks/Vectors).
* **Dense floats**: **byte-stream split** is the single most impactful encoding; keep it on for dense vectors.
* **Dictionary**: turn on for `section`, low-cardinality strings; off for `text`.
* **Bloom filters**: optional, behind `parquet.bloom_filters = ["doc_id", "chunk_id"]`.
* **Polars lazy**: recommend `scan_parquet` and projection pushdown (document this in the developer docs).
* **DuckDB**: encourage `parquet_scan('Vectors/family=dense/fmt=parquet/**')`.

---

# Risks & mitigations

* **Tool support for fixed_size_list**: some older readers have rough edges. Mitigation: allow fallback to `list<float32>` via `vectors.dense.fixed_size=false`.
* **Large strings in chunks**: keep row groups modest; avoid excessive coalescing; consider optional `text_present=false` profile for ultra-lean chunk catalogs that store only hashes (future).
* **Very large docs**: enable rolling; document suffix policy.
* **Sparse schema churn**: cement `indices+weights+nnz` now; any shift to `list<struct{…}>` should be additive and guarded by schema version bump.

---

# Work breakdown (reviewable commits)

1. **Commit A — Schemas & layout**

   * Write schema declarations (constants/types), new directory/path builders, and config keys (format defaults).
   * Add footer metadata builder.

2. **Commit B — ParquetWriter utility**

   * Encapsulate compression/encoding/row-group sizing, rolling, atomic write, and metadata population.

3. **Commit C — Chunks stage**

   * Switch default writer to Parquet; keep JSONL flag.
   * Update manifest extras.

4. **Commit D — Embedding stage**

   * Dense writer (fixed_size_list, with fallback).
   * Sparse/Lexical writer (parallel lists).
   * Manifest extras (`vector_format`, `family`, `dim`, `avg_nnz`, sizes).

5. **Commit E — DatasetView & inspect CLI**

   * Lazy scans for Chunks/Vectors; `docparse inspect` summary.

6. **Commit F — Tests & docs**

   * Add schema, round-trip, and performance sanity tests.
   * Update README/dev docs with layout, CLI examples, and performance knobs.

7. **Commit G — Converters (optional in this PR)**

   * `docparse convert chunks|vectors` JSONL→Parquet utilities.

---

# Operator & developer docs (ship with this PR)

* **Data layout & schemas** (one page): directory map, partition keys, schemas, and examples.
* **Performance guide**: Zstd levels, row-group sizing, byte-stream split, bloom filters.
* **How-to**: `docparse inspect`, DuckDB and Polars quick recipes.
* **Migration notes**: JSONL escape hatches, converters, and manifest fields to watch (`vector_format`, `family`).

---

Delivering PR-8 to this spec will give you **fast, compact, analytics-friendly** artifacts with strong provenance and predictable performance, while keeping the escape hatch to JSONL if any downstream dependency lags.
