Here’s a **production-grade “Data Layout & Schemas” spec** for your DocParsing artifacts. Treat it as the single source of truth your writers/readers, tests, and docs all conform to.

---

# 0) Purpose & scope

* **Goal:** Make Parquet the **default** for `Chunks` and `Vectors` with explicit, documented schemas; keep `DocTags` as JSONL (line-oriented, small).
* **Why:** Columnar storage (Arrow/Parquet) + partitioned layout enables fast scans (DuckDB/Polars), compact on-disk size (Zstd + encodings), and smooth schema evolution.

Non-goals for this spec: retrieval indexes (FAISS/OpenSearch), vector quantization, or remote stores (fsspec-ready but out of scope here).

---

# 1) Directory layout (canonical)

All paths are under a single **data root** (e.g., `Data/`). Use **path partitions** for discovery & pruning.

```
Data/
  Doctags/{yyyy}/{mm}/{rel_id}.jsonl

  Chunks/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet

  Vectors/{family=dense|sparse|lexical}/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet

  Manifests/...
```

**Rules**

* `rel_id` = **relative identifier** derived from the source path (see §2). It is stable, deterministic, and filesystem-safe.
* Partitions `{yyyy}/{mm}` are based on the **write time** of the artifact (UTC).
* `{family}` and `{fmt}` segments are **literal partition keys** to make dataset discovery trivial.

---

# 2) Identifiers & normalization

## 2.1 `rel_id` (relative identifier)

* Derived from the source’s relative path **without extension**.
* Unicode normalized to **NFC**.
* Keep case (case-preserving). Do **not** downcase automatically.
* Replace path separators with `/` (already the case on POSIX); do not embed `..` or leading `/`.

### 2.1.1 Character safety

* Allowed: `[A-Za-z0-9._~/ -]`.
* Replace any other character with `_` **unless** you choose to percent-encode (RFC 3986). Pick one approach and keep it consistent.
* Maximum length: 512 code points (enforced by writers).

## 2.2 `doc_id`

* Stored **in rows** (columns) and equals `rel_id` for a given artifact.
* Uniqueness per dataset: `(doc_id, chunk_id)` is a **primary key** in `Chunks` and `Vectors`.

---

# 3) Dataset: **Doctags** (still JSONL)

### Path

```
Data/Doctags/{yyyy}/{mm}/{rel_id}.jsonl
```

### Line schema (conceptual)

* Each line is a DocTag block (body text, table, figure, caption, etc.) with its own metadata.
* Keep Doctags small and line-oriented; do not encode large binary tangles. Heavy layout images/figures are referenced externally (paths/URIs).

### Manifests

* Per-file manifest rows already include `input_hash`, `output_path`, and parse engine/model info; keep as is.

---

# 4) Dataset: **Chunks** (Parquet, default)

### 4.1 Path

```
Data/Chunks/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet
```

### 4.2 Arrow/Parquet schema (required columns)

| Column           | Type                            | Required | Notes                                              |
| ---------------- | ------------------------------- | -------: | -------------------------------------------------- |
| `doc_id`         | `string`                        |        ✓ | equals `rel_id` of the source doctags              |
| `chunk_id`       | `int64`                         |        ✓ | 0-based, continuous within `doc_id`                |
| `text`           | `large_string`                  |        ✓ | UTF-8; large string for safety                     |
| `tokens`         | `int32`                         |        ✓ | tokenizer-specific length                          |
| `span`           | `struct<start:int32,end:int32>` |        ✓ | character offsets into source text (end exclusive) |
| `created_at`     | `timestamp[us, tz=UTC]`         |        ✓ | writer’s UTC timestamp                             |
| `schema_version` | `string`                        |        ✓ | e.g., `docparse/chunks/1.0.0`                      |

### 4.3 Optional columns (additive)

| Column    | Type                 | Notes                                                        |
| --------- | -------------------- | ------------------------------------------------------------ |
| `section` | `string`             | low-cardinality (e.g., `title`, `abstract`, `body`, `table`) |
| `meta`    | `map<string,string>` | small, flat, ≤ 4 KB per row                                  |

### 4.4 Invariants

* `(doc_id, chunk_id)` unique.
* `0 ≤ span.start ≤ span.end ≤ len(source_text)`. If not verifiable, guarantee `0 ≤ start ≤ end`.
* `tokens ≥ 0`.
* `text` is **non-empty** for materialized chunks; if you later support textless catalogs, set a file-level flag (not in scope here).

### 4.5 Encoding & compression (defaults)

* `compression = zstd`, level `5`.
* `write_statistics = true`.
* Dictionary encoding **on** for `section`, **off** for `text`.
* Target row-group: ~**16–64 MB** per group (configurable).

---

# 5) Dataset: **Vectors** (Parquet, default)

Vectors are split **by family** under `Vectors/family=...`. Each family file contains one row per `(doc_id, chunk_id)` present in the corresponding Chunks file.

## 5.1 Common columns (all families)

| Column           | Type                 | Required | Notes                             |
| ---------------- | -------------------- | -------: | --------------------------------- |
| `doc_id`         | `string`             |        ✓ |                                   |
| `chunk_id`       | `int64`              |        ✓ |                                   |
| `created_at`     | `timestamp[us, UTC]` |        ✓ | write time                        |
| `schema_version` | `string`             |        ✓ | `docparse/vectors/{family}/1.0.0` |

File-footer metadata (see §7) carries model/provider provenance.

---

## 5.2 Family: **Dense**

### Path

```
Data/Vectors/family=dense/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet
```

### Columns (dense)

| Column         | Type                                                   | Required | Notes                                                                |
| -------------- | ------------------------------------------------------ | -------: | -------------------------------------------------------------------- |
| `doc_id`       | `string`                                               |        ✓ |                                                                      |
| `chunk_id`     | `int64`                                                |        ✓ |                                                                      |
| `dim`          | `int32`                                                |        ✓ | dimension for this row (constant across run; keep column for safety) |
| `vec`          | `fixed_size_list<float32>[dim]` **or** `list<float32>` |        ✓ | prefer fixed-size if `dim` is constant                               |
| `normalize_l2` | `boolean`                                              |        ✓ | usually `true`                                                       |

**Preferred layout:** `fixed_size_list<float32>` with **byte-stream split** encoding on the float column for compression and scan speed.

**Fallback:** `list<float32>` if you truly have heterogeneous dims.

**Invariants**

* If fixed-size: length(vec) == dim for every row (enforced by schema).
* If variable: writer must assert `len(vec) == dim` per row.
* `(doc_id, chunk_id)` unique.

---

## 5.3 Family: **Sparse** (SPLADE)

### Path

```
Data/Vectors/family=sparse/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet
```

### Columns (sparse)

| Column     | Type            | Required | Notes                    |
| ---------- | --------------- | -------: | ------------------------ |
| `doc_id`   | `string`        |        ✓ |                          |
| `chunk_id` | `int64`         |        ✓ |                          |
| `nnz`      | `int32`         |        ✓ | non-zeros in this row    |
| `indices`  | `list<int32>`   |        ✓ | vocabulary or hashed ids |
| `weights`  | `list<float32>` |        ✓ | same length as `indices` |

**Invariants**

* `nnz == len(indices) == len(weights)`.
* Weights are **non-negative** (SPLADE activations).
* `(doc_id, chunk_id)` unique.

**Notes**

* Keep index space consistent within a run (vocab id space or hash scheme); record details in the file footer (see §7).

---

## 5.4 Family: **Lexical** (BM25)

Two representational forms; pick **one** as the default and document it in metadata.

### Option A (default): **ID-space**

```
Data/Vectors/family=lexical/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet
```

| Column     | Type            | Required | Notes        |
| ---------- | --------------- | -------: | ------------ |
| `doc_id`   | `string`        |        ✓ |              |
| `chunk_id` | `int64`         |        ✓ |              |
| `nnz`      | `int32`         |        ✓ |              |
| `indices`  | `list<int32>`   |        ✓ | term ids     |
| `weights`  | `list<float32>` |        ✓ | BM25 weights |

**Metadata**: `lexical.representation = "indices"`, `tokenizer_id`, `vocab_id` (if applicable), `k1`, `b`, `stopwords_policy`, `min_df`, `max_df_ratio`.

### Option B (optional): **Term-space** (friendlier for ad-hoc exploration; bigger)

| Column    | Type            | Required | Notes              |
| --------- | --------------- | -------: | ------------------ |
| `terms`   | `list<string>`  |        ✓ | dictionary-encoded |
| `weights` | `list<float32>` |        ✓ |                    |
| `nnz`     | `int32`         |        ✓ |                    |

**Metadata**: `lexical.representation = "terms"` and the same BM25 parameters.

**Invariants**

* `nnz == len(indices|terms) == len(weights)`.
* `(doc_id, chunk_id)` unique.

---

# 6) Partitioning & discovery

* Dataset discovery is purely **path-partitioned** using the `{family}` and `{fmt}` segments and `{yyyy}/{mm}` time partitions.
* Readers (Polars/DuckDB/Arrow) should scan glob patterns:

  * Dense: `Vectors/family=dense/fmt=parquet/*/*/*.parquet`
  * Sparse: `Vectors/family=sparse/fmt=parquet/*/*/*.parquet`
  * Lexical: `Vectors/family=lexical/fmt=parquet/*/*/*.parquet`
  * Chunks: `Chunks/fmt=parquet/*/*/*.parquet`
* Provide a tiny `DatasetView` that exposes **lazy scans** with optional `columns` and `filters` (e.g., `doc_id == X`, month ranges).

---

# 7) Parquet **file-footer metadata** (provenance)

Every Parquet file written by the pipeline must include the following **key-value metadata**:

* `docparse.schema_version` = `docparse/chunks/1.0.0` or `docparse/vectors/{family}/1.0.0`
* `docparse.family` = `dense|sparse|lexical` (Vectors only)
* `docparse.provider` = e.g., `dense.qwen_vllm`, `dense.tei`, `dense.sentence_transformers`, `sparse.splade_st`, `lexical.local_bm25`
* `docparse.model_id` = model or tokenizer id (include `@rev` if pinned)
* `docparse.dim` = integer (Dense only)
* `docparse.dtype` = `float32` (dense), `float32` (weights), `int32` (indices)
* `docparse.device` = `cpu|cuda:N` (Dense only if relevant)
* `docparse.cfg_hash` = stage config hash for reproducibility
* `docparse.created_by` = package name/version
* `docparse.created_at` = ISO-8601 UTC timestamp
* **Lexical/Sparse specific:**

  * `docparse.lexical.representation` = `indices|terms`
  * `docparse.bm25.k1`, `docparse.bm25.b`, `docparse.stopwords_policy`, `docparse.min_df`, `docparse.max_df_ratio`
  * `docparse.sparse.vocab_id` or `docparse.sparse.hash_scheme` (for SPLADE)

Additionally, at the dataset root, optionally write Arrow `_metadata` / `_common_metadata` files to accelerate schema discovery.

---

# 8) Writing policy (applies to all Parquet writers)

* **Atomicity:** write to a temp path, finalize (fsync), then **atomic rename**; never leave partial files behind.
* **Row-group targeting:** default to ~32 MB (dense/sparse) and 16–64 MB (chunks); configurable.
* **Compression:** Zstd default (level 5); configurable 3–9.
* **Float encoding (Dense):** enable **byte-stream split** on the float column(s).
* **Bloom filters:** optional for `doc_id`, `chunk_id` (config key).
* **Rolling:** If a file would exceed 256–512 MB, roll with `.part-000NN.parquet`. Log all parts in manifest extras.

---

# 9) JSONL escape hatch (back-compat & debugging)

* Keep `--chunks-format=jsonl` and `--vector-format=jsonl`.
* JSONL schemas map 1:1 to Parquet columns:

  * Dense: `{doc_id, chunk_id, dim, vec:[float], normalize_l2, created_at, schema_version}`
  * Sparse/Lexical: `{doc_id, chunk_id, nnz, indices|terms:[...], weights:[...], created_at, schema_version}`

---

# 10) Manifests & traceability

For each artifact written:

* **Config row** (stage start) includes `vector_format`/`chunks_format`, partitions, and key settings (you already emit this).
* **Success row** (per file) includes:

  * `vector_format="parquet"` (or `jsonl`)
  * `family` (for vectors)
  * `parquet_bytes` (on-disk size)
  * `row_group_count`
  * `dim` (dense), `avg_nnz` (sparse/lexical)
  * `output_path` (or list of parts, if rolled)

This makes downstream audits and conversions easy.

---

# 11) Validation & invariants (CI-enforced)

* **Chunks**: schema columns present; `(doc_id, chunk_id)` unique; `tokens ≥ 0`; `span` bounds consistent (if checked).
* **Dense**: `len(vec) == dim`; `normalize_l2 ∈ {true,false}`; `(doc_id, chunk_id)` unique.
* **Sparse/Lexical**: `nnz == len(indices|terms) == len(weights)`; weights finite; `(doc_id, chunk_id)` unique.
* **Footers**: required `docparse.*` metadata present.
* **Partition sanity**: file path matches footer (e.g., `family=dense` ↔ `docparse.family=dense`).

---

# 12) Quick-use recipes (operators & analysts)

**DuckDB**

```sql
-- Dense
SELECT COUNT(*) FROM parquet_scan('Data/Vectors/family=dense/fmt=parquet/*/*/*.parquet');

-- Lexical top terms (term-space option)
SELECT terms, weights FROM parquet_scan('Data/Vectors/family=lexical/fmt=parquet/*/*/*.parquet')
WHERE doc_id='papers/physics/123' AND chunk_id=0;
```

**Polars (lazy)**

```python
import polars as pl
ds = pl.scan_parquet("Data/Chunks/fmt=parquet/*/*/*.parquet")
ds.filter(pl.col("section") == "abstract").select(pl.len()).collect()
```

---

# 13) Schema evolution policy

* **SemVer per dataset** in `schema_version` and footer:

  * **Patch** (`x.y.z`): bug fixes, no column changes.
  * **Minor** (`x.y+1.0`): **additive** columns only; old readers safe.
  * **Major** (`x+1.0.0`): breaking changes (retype, rename, representation changes).
* **Allowed without major bump**:

  * Add optional columns.
  * Add new file-footer keys.
  * Tighten column **semantics** (documented).
* **Requires major bump**:

  * Rename/remove columns.
  * Change column types or vector representation (e.g., switch dense from fixed-size to variable list **without fallback**).
  * Switch lexical default representation (indices→terms) **without** dual support.

---

# 14) Performance knobs (documented config keys)

* `parquet.compression` (default `zstd`)
* `parquet.compression_level` (default `5`)
* `parquet.target_rowgroup_mb` (dense/sparse defaults ~32, chunks 16–64)
* `parquet.bloom_filters` (list of column names; default off)
* `vectors.dense.fixed_size` (bool; default true)
* `vectors.partitioning` (fixed `{family}/{fmt}/{yyyy}/{mm}` in this phase)

---

# 15) Examples (end-to-end)

**Example source → rel_id**

* Source: `Papers/Nature/2025_10_gene-editing.pdf`
* `rel_id`: `Papers/Nature/2025_10_gene-editing`

**Generated artifacts**

```
Data/Doctags/2025/10/Papers/Nature/2025_10_gene-editing.jsonl

Data/Chunks/fmt=parquet/2025/10/Papers/Nature/2025_10_gene-editing.parquet

Data/Vectors/family=dense/fmt=parquet/2025/10/Papers/Nature/2025_10_gene-editing.parquet
Data/Vectors/family=sparse/fmt=parquet/2025/10/Papers/Nature/2025_10_gene-editing.parquet
Data/Vectors/family=lexical/fmt=parquet/2025/10/Papers/Nature/2025_10_gene-editing.parquet
```

**Dense row (conceptual)**

```
doc_id="Papers/Nature/2025_10_gene-editing"
chunk_id=42
dim=1024
vec=[...]  # 1024 float32
normalize_l2=true
created_at=2025-10-20T03:14:15Z
schema_version="docparse/vectors/dense/1.0.0"
```

**Sparse row (conceptual)**

```
doc_id, chunk_id, nnz=27, indices=[...27...], weights=[...27...], created_at, schema_version
```

**Lexical row (id-space)**

```
doc_id, chunk_id, nnz=19, indices=[...19...], weights=[...19...], created_at, schema_version
```

---

This document is intentionally **exacting** so your writers, readers, runner, and CI can all align without ambiguity. If you want, I can turn §§4–5 into **executable Arrow schema declarations** and a **Parquet footer contract** your tests enforce automatically (e.g., a tiny “schema check” utility and a “footer check” that fails CI if required `docparse.*` keys are missing).
