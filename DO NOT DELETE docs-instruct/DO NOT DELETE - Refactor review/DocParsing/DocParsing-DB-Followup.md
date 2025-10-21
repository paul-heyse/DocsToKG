Absolutely—here’s a **surgical, repo-aware implementation plan** to finish **Item 1: “Parquet everywhere for Chunks + Dataset layout & Inspect”**. It’s written so an AI agent can land it as a sequence of small, low-risk PRs. No code, just precise directions (file paths, functions to add, invariants to enforce, tests, and rollout).

---

# Scope (what changes / what doesn’t)

**We will:**

* Make **Parquet the default** output for **Chunks** (DocTags → Chunks).
* Adopt a **partitioned dataset layout** for Chunks mirroring what you already have for Vectors.
* Add a **DatasetView** utility and a **`docparse inspect`** CLI to introspect Parquet datasets quickly.
* Annotate **manifests** with format/partitioning/provenance for traceability.

**We will not:**

* Change DocTags output (still JSONL) in this PR.
* Change any retrieval/index logic, or Parquet for Vectors (already implemented).
* Introduce Prefect or runner/providers changes here (keep this PR self-contained).

---

# Current state delta (why this is easy now)

* Vectors already support **Parquet** (format negotiation, lazy Arrow import, manifest extras).
* JSONL atomic append is done (we’ll keep it for debugging/escape hatch).
* The repo already centralizes IO helpers and manifest logging; we’ll reuse those seams.

---

# Target layout & schemas (authoritative)

## Directory layout (under `Data/`)

```
Data/
  Doctags/{yyyy}/{mm}/{rel_id}.jsonl              # unchanged (line-oriented)
  Chunks/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet
  Vectors/{family=dense|sparse|lexical}/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet
  Manifests/...
```

* `rel_id` = relative, normalized identifier (stable, derived from source path; no extension).
* `{yyyy}/{mm}` partitions are based on **write time (UTC)**.
* Keep JSONL as **escape hatch** via `--format jsonl`.

## Arrow schema for **Chunks**

Required columns:

* `doc_id: string` (== `rel_id`)
* `chunk_id: int64` (0-based, contiguous per doc)
* `text: large_string` (UTF-8)
* `tokens: int32`
* `span: struct<start:int32, end:int32>` (end exclusive)
* `created_at: timestamp[us, UTC]`
* `schema_version: string` (e.g., `docparse/chunks/1.0.0`)

Optional (additive) columns:

* `section: string` (low-cardinality)
* `meta: map<string,string>` (flat, ≤4KB per row)

**Encoding defaults**

* `compression=zstd`, `compression_level=5`
* `write_statistics=true`
* Dict encoding **on** for `section`, **off** for `text`
* Row-group size target: **16–64 MB** (configurable)

**File-footer metadata** (Parquet key/value):

* `docparse.schema_version=docparse/chunks/1.0.0`
* `docparse.cfg_hash=<stage config hash>`
* `docparse.created_by=<package@version>`
* `docparse.created_at=<ISO-UTC>`

*(Vectors’ footers are already done; no change here.)*

---

# Work items (by commit)

## Commit A — **Storage paths & partition helpers**

**New**: `src/DocsToKG/DocParsing/storage/paths.py`

* `chunks_parquet_path(data_root: Path, rel_id: str, dt: datetime) -> Path`
  Returns `Data/Chunks/fmt=parquet/{yyyy}/{mm}/{rel_id}.parquet`.
* `ensure_parent(path: Path) -> None`
  Create parents; no write-time race (use existing atomic writer’s parent creation where possible).

**Acceptance**

* Unit test: given `(Data, docs/chem/abc)`, month 2025-10 → correct path.

---

## Commit B — **Arrow schema & footer contract for Chunks**

Leverage your existing vector Parquet contract pattern.

**New**: `src/DocsToKG/DocParsing/storage/chunk_schemas.py`

* `chunks_schema(include_optional: bool = True) -> pa.Schema`
* `build_footer_chunks(cfg_hash: str, created_by="DocsToKG-DocParsing", created_at=None, extra=None) -> Dict[str,str>`
* `attach_footer_metadata(table: pa.Table, meta: Mapping[str,str]) -> pa.Table`
* `validate_parquet_file(path: str) -> ValidationResult` (check required footer keys & semver)

**Acceptance**

* Unit test: schema fields match spec; footer contains required keys; validator passes on a tiny temp file.

---

## Commit C — **ParquetChunksWriter** (atomic, rolled, provenance)

**New**: `src/DocsToKG/DocParsing/storage/chunks_writer.py`

* Input: iterable of **validated** chunk rows (`doc_id, chunk_id, text, tokens, span, section?, meta?`), stage `cfg_hash`, `data_root`, `rel_id`, current time (`dt_utc`), and writer options.
* Build a `pyarrow.Table` in **batches** to control memory.

  * If many rows: write in chunks with `pq.write_to_dataset` or use `ParquetWriter` and call `writer.write_table(batch)`.
* Apply **schema** and call `attach_footer_metadata` with `build_footer_chunks(...)`.
* **Atomic write**: write to a temp path (same dir), fsync, then `rename()` to final Parquet path.
* **Row group sizing**: accumulate rows to hit ~16–64 MB per row group; expose a `target_rowgroup_mb` option with a conservative default.
* **Rolling** (optional now; add hooks): if estimated file size exceeds 256–512 MB, roll into `.part-000NN.parquet` and return **all** part paths.
* Return: `WriteResult{paths: List[Path], rows_written: int, row_group_count: int, parquet_bytes: int}`.

**Acceptance**

* Unit test: write 1k rows, validate columns, footer, and `rows_written`, `parquet_bytes > 0`.
* Crash safety: simulate exception mid-write → no final file present.

---

## Commit D — **Chunk runtime integration**

Touch the chunk stage only (keep JSONL path intact as escape hatch).

* Add config key **`chunk.format`** with default **`parquet`** (CLI `--format` already exists or add it; back-compat `jsonl` maintained).
* In chunk runtime:

  * If `format=parquet`: route to **ParquetChunksWriter**.
  * If `format=jsonl`: keep current JSONL writer (unchanged, PR-2 atomic append).
* Build `rel_id` from input doctags path (normalize as elsewhere).
* Compute **`cfg_hash`** for the chunk stage (hash of relevant config subset).
* **Manifests**: log per-file `success` with extras:

  * `chunks_format="parquet"`, `rows_written`, `row_group_count`, `parquet_bytes`, `output_path` (or list).
  * For JSONL, keep current extras; set `chunks_format="jsonl"`.

**Acceptance**

* CLI default run writes Parquet; `--format jsonl` still works.
* Manifest row contains `chunks_format` and size counts.

---

## Commit E — **DatasetView & Inspect CLI**

**New**: `src/DocsToKG/DocParsing/storage/dataset_view.py`

* `open_chunks(data_root: Path, columns=None, filters=None) -> pa.dataset.Dataset | polars.LazyFrame`
* `open_vectors(data_root: Path, family: str, columns=None, filters=None) -> ...`
* `summary(ds) -> DatasetSummary{schema, approx_rows, files, total_bytes, partitions}`

  * Use Arrow’s `dataset` + `fragment` metadata to avoid full scans for row counts (approximate), with an option to force exact scan.

**Wire CLI**: fill the `inspect dataset` command (you already have the Typer skeleton):

* `--dataset {chunks, vectors-dense, vectors-sparse, vectors-lexical}`
* `--root`, `--columns`, `--filters`, `--limit`, `--stats`
* Print: dataset type, projected schema, **rows** (approximate fast path), file count, bytes, partitions (yyyy/mm), and 3 sample `doc_id`s.

**Acceptance**

* On a mini corpus, `docparse inspect --dataset chunks` returns schema + counts in <1s.
* `--filters 'doc_id == "X"'` narrows summary correctly.

---

## Commit F — **Docs & examples**

* Update internal docs:

  * “**Data layout & schemas**” (Chunks Parquet; already done for Vectors).
  * “**Inspect**” quick-start (DuckDB/Polars recipes).
  * “**Config rows**” show `chunks_format` and partitions.

**Acceptance**

* Examples run snappily against the mini corpus.

---

## Commit G — **Tests & CI**

### Unit

* **Paths**: partition path builder outputs correct path for multiple months and deep `rel_id`s.
* **Schema**: exact fields & types, including nested `span`.
* **Footer**: validator passes; missing key triggers failure.

### Integration

* **Write/read round-trip**:

  * Build a tiny doctags → chunks pipeline; write Parquet.
  * Open with Arrow/Polars; compute: total rows, average tokens, check `(doc_id, chunk_id)` uniqueness, and sample `span` consistency.
* **Manifest parity**: assert `chunks_format="parquet"`, `rows_written` equals table rows, and `output_path` exists.
* **JSONL escape hatch**: run with `--format jsonl`; outputs and manifests unchanged from pre-PR behavior.

### Performance sanity (non-assertive)

* Time `inspect` and a simple DuckDB scan on 50–100k rows; log throughput to CI output for reference.

---

# Validation & invariants

* `(doc_id, chunk_id)` unique within file and across rolled parts.
* `tokens ≥ 0`; `0 ≤ span.start ≤ span.end`; `text` non-empty.
* Footer must include `docparse.schema_version`, `docparse.cfg_hash`, `docparse.created_at`, `docparse.created_by`.
* Manifest `success` rows must include `chunks_format` and either a single `output_path` or a list `output_paths` if rolled.

---

# Config & CLI (minimal deltas)

* **Chunk**: add `--format {parquet,jsonl}` (default **parquet**). If this already exists for vectors, mirror flag names & help strings for muscle memory.
* **Global** (optional now): `parquet.compression`, `parquet.compression_level`, `parquet.target_rowgroup_mb` (defaults as above). You can defer exposing these until a later “perf knobs” PR if you want.

---

# Rollout plan

1. **Ship Parquet Chunks** gated by CLI default; JSONL remains available.
2. **Enable `docparse inspect`** (read-only, zero risk).
3. Dogfood on a small corpus; compare disk size and scan speed (Polars/DuckDB).
4. Announce **Parquet default** for Chunks in CHANGELOG; highlight `--format jsonl` escape hatch.
5. After one minor, consider removing JSONL from docs as “default”, keeping only as a debug path.

---

# Risks & mitigations

* **pyarrow availability**: keep import **lazy**; on ImportError, print an actionable message and suggest `--format jsonl`.
* **Very large `text`** rows: keep row-group target modest (16–64MB) to limit memory; stream batches to the writer.
* **Downstream expectations**: some tools might still expect JSONL Chunks → keep JSONL escape hatch and add a **converter** later if you need to backfill.

---

# Definition of Done (for this item)

* `chunk` stage writes **Parquet** by default and **JSONL** on request.
* `Data/Chunks/fmt=parquet/{yyyy}/{mm}/{rel_id}.parquet` exists with correct schema & footer.
* Manifests include `chunks_format`, sizes, row groups, and paths.
* `docparse inspect` summarizes **Chunks** and **Vectors** datasets fast.
* Tests (schema/round-trip/manifest) pass; CI time is stable.

---

## Optional follow-ups (not required for this PR)

* **JSONL → Parquet converter**: `docparse convert chunks --from jsonl --to parquet` (idempotent; parity checks).
* **Bloom filters** for `doc_id`/`chunk_id` (config flag).
* **File rolling** heuristics & manifest array output when a single doc generates very large Parquet.

---

If you want, I can turn the above into a **commit checklist** with exact file diffs to touch in the current repo (e.g., where to call the writer in your chunk runtime, which manifest helper to extend with `chunks_format`, and the CLI option tuple to add under the “I/O” help panel).

Perfect — below is the **commit-by-commit checklist** to land **Item #1 (Parquet Chunks + Dataset layout + Inspect)** on top of `main@113930b`. I’ve kept it surgical: files to touch/create, what to add, where to route calls, manifest fields to add, Typer options to expose (or confirm), validation invariants, tests, and manual QA steps.

---

# PR: Chunks → Parquet by default + DatasetView + `inspect`

## Commit A — Storage paths & partitions

**Create**

* `src/DocsToKG/DocParsing/storage/paths.py`

**Add**

* `chunks_parquet_path(data_root: Path, rel_id: str, dt_utc: datetime) -> Path`
  Returns: `Data/Chunks/fmt=parquet/{yyyy}/{mm}/{rel_id}.parquet`
* `vectors_parquet_glob(data_root: Path, family: str) -> str`
  Returns: `Data/Vectors/family={family}/fmt=parquet/*/*/*.parquet`
* `chunks_parquet_glob(data_root: Path) -> str`
  Returns: `Data/Chunks/fmt=parquet/*/*/*.parquet`
* `ensure_parent(path: Path) -> None` (wrapper calling `.mkdir(parents=True, exist_ok=True)`)

**Acceptance**

* Unit: given `data_root="Data"` and `rel_id="papers/xyz/abc"`, month Oct-2025 → `Data/Chunks/fmt=parquet/2025/10/papers/xyz/abc.parquet`.

---

## Commit B — Arrow schema & footer contract (Chunks)

> **If you already added** `docparse_parquet_schemas.py` with `chunks_schema(...)` and footer helpers from our earlier step, **reuse it**. Otherwise, create the chunk subset now.

**Create (if missing)**

* `src/DocsToKG/DocParsing/storage/docparse_parquet_schemas.py`
  Ensure it exports:

  * `chunks_schema(include_optional: bool = True) -> pa.Schema`
  * `attach_footer_metadata(table: pa.Table, meta: Mapping[str,str]) -> pa.Table`
  * `build_footer_common(...)`, **or** a chunk-specific `build_footer_chunks(cfg_hash, created_by, created_at, extra)` that sets:

    * `docparse.schema_version="docparse/chunks/1.0.0"`
    * `docparse.cfg_hash`, `docparse.created_by`, `docparse.created_at`

**Schema (required columns)**

* `doc_id: string`
* `chunk_id: int64`
* `text: large_string`
* `tokens: int32`
* `span: struct<start:int32,end:int32>`
* `created_at: timestamp[us, tz=UTC]`
* `schema_version: string` (`docparse/chunks/1.0.0`)

**Optional**

* `section: string`
* `meta: map<string,string>`

**Acceptance**

* Unit: a temp Parquet written with this schema passes `schema.field_names` & `field.types` checks and footer contains keys above.

---

## Commit C — ParquetChunksWriter (atomic, batched, rolled)

**Create**

* `src/DocsToKG/DocParsing/storage/chunks_writer.py`

**Add**

* `ParquetChunksWriter.write(rows_iter, *, data_root: Path, rel_id: str, cfg_hash: str, created_by: str, dt_utc: datetime, target_rowgroup_mb: int = 32, compression: str = "zstd", compression_level: int = 5) -> WriteResult`

  * Build batches (`pyarrow.Table` or `RecordBatch`) from `rows_iter` to bound memory.
  * Apply `chunks_schema()` exactly; `attach_footer_metadata` with `docparse.*` keys.
  * Use `pyarrow.parquet.ParquetWriter` to control row-group boundaries (~16–64MB on disk).
  * **Atomic finalize**: write to `*.parquet.tmp`, fsync file + directory, then `rename()` to final path from `chunks_parquet_path(...)`.
  * **Return** `WriteResult(paths=[final_path], rows_written, row_group_count, parquet_bytes)`.

**Invariants enforced here**

* `(doc_id, chunk_id)` unique within write.
* `len(text) > 0`; `tokens ≥ 0`.
* `0 ≤ span.start ≤ span.end` (best-effort; if you don’t have source length, still assert ordering).

**Acceptance**

* Unit: write ~1k rows, assert returned counts; reading back shows the same row count and footer keys.

---

## Commit D — Chunk runtime integration

**Touch**

* `src/DocsToKG/DocParsing/chunking/runtime.py` (or your current chunk stage entrypoint)
* `src/DocsToKG/DocParsing/cli.py` (or the Typer module you made) — **only if** `--format` isn’t exposed yet

**Add/Confirm**

* **Settings key**: `chunk.format` with default **`parquet`** (keep `jsonl` as escape hatch).
* **CLI flag** (if not already present):
  `--format {parquet,jsonl}` under the **“I/O”** help panel for the `chunk` command.
* Compute **`rel_id`** from input doctags path once per item (normalize NFC, strip extension, keep directory structure).
* Compute **`cfg_hash`** for chunk stage (stable hash over `min_tokens`, `max_tokens`, tokenizer id, format, etc.).
* **Route by format**:

  * `parquet` → call `ParquetChunksWriter.write(...)`, get `WriteResult`.
  * `jsonl` → existing JSONL writer (unchanged), still using the PR-2 atomic append.
* **Manifest success row**: extend payload with:

  * `chunks_format: "parquet" | "jsonl"`
  * For `parquet`: `rows_written`, `row_group_count`, `parquet_bytes`
  * `output_path` (string) or **`output_paths`** (array) if you later roll large files
* **Resume/force semantics**: keep your existing checks; if you already added fingerprints later, honor them (exact match on `input_sha256` + `cfg_hash`).

**Acceptance**

* Default run uses Parquet; `--format jsonl` preserves old behavior.
* Manifest lines show `chunks_format` and size metadata.

---

## Commit E — DatasetView + `inspect` CLI

**Create**

* `src/DocsToKG/DocParsing/storage/dataset_view.py`

**Add**

* `open_chunks(data_root: Path, *, columns: list[str]|None = None, filters: str|None = None) -> ArrowDatasetOrPolars`
  Uses `pyarrow.dataset.dataset(chunks_parquet_glob(...))` or Polars `scan_parquet`.
* `open_vectors(data_root: Path, family: str, *, columns=None, filters=None)`
  Uses `vectors_parquet_glob(...)`.
* `summarize(dataset) -> DatasetSummary`

  * `schema`
  * **fast** file/byte counts from fragments
  * **approx rows** from statistics (exact count optional)
  * partition values seen for `{yyyy}/{mm}`
  * sample `doc_id` values

**Wire CLI** (your skeleton already exists):

* In `inspect dataset` handler, implement:

  * Translate `DatasetKind` → glob
  * Build dataset via `DatasetView`
  * Print schema (field names/types), #files, total bytes, partitions, and (optionally) approx rows
  * Honor `--columns`, `--filters`, `--limit`, `--stats`

**Acceptance**

* `docparse inspect --dataset chunks` returns schema + file & byte counts in < 1s on your mini corpus.
* `--dataset vectors-dense` finds the glob and reports correctly.

---

## Commit F — Docs

**Touch**

* `docs/` (or your README/API pages if you keep them there)

**Add**

* “**Data layout & schemas**” additions for **Chunks Parquet** (we already finalized the spec), including:

  * Directory layout with `{fmt=parquet}/{yyyy}/{mm}`
  * Arrow schema table
  * Footer keys and meaning
* “**Inspect** quickstart” with mini examples:

  * DuckDB: `parquet_scan('Data/Chunks/fmt=parquet/*/*/*.parquet')`
  * Polars: `scan_parquet(...)` and `.select(pl.len()).collect()`

---

## Commit G — Tests & CI

**Create**

* `tests/storage/test_chunks_schema.py`

  * Build a table with required columns; validate schema equality; write temp Parquet; validate footer.
* `tests/storage/test_chunks_writer.py`

  * Happy path: write 1k rows → read back row count matches; footer validated; `(doc_id, chunk_id)` unique.
  * Crash safety: simulate exception → no final file.
* `tests/storage/test_dataset_view.py`

  * On a tiny corpus, `inspect` summary fields are non-empty and schema includes expected fields.
* `tests/chunk/test_chunk_runtime_parquet.py`

  * End-to-end: doctags fixture → chunk run `--format parquet` → assert manifest has `chunks_format="parquet"` and counts; verify Parquet exists and schema OK.
* (Keep) JSONL tests unchanged for back-compat and add one “escape hatch” run `--format jsonl`.

**CI**

* Ensure `pyarrow` is present in the pipeline environment used by these tests.
* For optional performance sanity, log time to inspect (not an assertion).

---

# Exact places to update **manifest fields**

**File**

* `src/DocsToKG/DocParsing/logging.py` (or wherever `manifest_log_success` composes extras)

**Add (for chunk success rows)**

```json
{
  "chunks_format": "parquet",
  "rows_written": <int>,
  "row_group_count": <int>,
  "parquet_bytes": <int>
}
```

* Keep `input_path`, `output_path` (string) as you already do.
* If you later roll one doc into multiple files, either:

  * emit multiple success rows (one per part), **or**
  * add `output_paths: [..]` and set `rows_written`/`parquet_bytes` as totals.
    (Pick one and document — simplest now is **one row per file**.)

---

# Confirm / add the **Typer** option tuple (Chunk → I/O)

If not already present in your Typer command:

```python
fmt: Annotated[
    Format,  # Enum(parquet, jsonl)
    typer.Option("--format", help="parquet (default) | jsonl.", rich_help_panel="I/O")
] = Format.parquet
```

* Keep `--in-dir`, `--out-dir` under the same “I/O” panel.
* In the handler, merge this CLI value into `ChunkCfg.format` before invoking the stage.

---

# Validation invariants to enforce (writer or runtime)

* `(doc_id, chunk_id)` unique within file.
* `tokens ≥ 0`.
* `0 ≤ span.start ≤ span.end` (and ideally `end ≤ len(source)` if available).
* `schema_version == "docparse/chunks/1.0.0"` in column **and** footer key `docparse.schema_version`.
* Footer keys exist: `docparse.cfg_hash`, `docparse.created_by`, `docparse.created_at`.

---

# Manual QA script (copy/paste)

1. **Write Parquet Chunks**

```bash
python -m DocsToKG.DocParsing.cli chunk run \
  --in-dir Data/Doctags \
  --out-dir Data/Chunks \
  --format parquet \
  --workers 4
```

2. **Inspect**

```bash
python -m DocsToKG.DocParsing.cli inspect dataset --dataset chunks --root Data --stats
```

3. **Open in DuckDB**

```sql
SELECT COUNT(*) FROM parquet_scan('Data/Chunks/fmt=parquet/*/*/*.parquet');
```

4. **Manifest check**

```bash
tail -n 5 Data/Manifests/docparse.chunk.manifest.jsonl | jq .
# Confirm: chunks_format="parquet", rows_written, row_group_count, parquet_bytes
```

5. **Escape hatch**

```bash
python -m DocsToKG.DocParsing.cli chunk run --format jsonl --workers 2
# Verify legacy JSONL path is intact
```

---

# Rollback / safety

* If `pyarrow` import fails at runtime: catch and instruct the user to run with `--format jsonl` (keep JSONL code path intact).
* All new code is additive; JSONL path and manifests remain unchanged except for the new `chunks_format` key (which you can omit for JSONL if you prefer no change there).

---
