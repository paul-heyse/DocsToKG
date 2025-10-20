Here’s the **field-accurate manifest spec** and a **runner behaviors matrix** that line up with what’s in your `main` branch today, plus a few optional niceties. I’m using the symbols and file shapes that exist in `src/DocsToKG/DocParsing` right now.

---

# Manifest file model (what is written, where, and how)

## File locations & naming

* **Directory**: `Data/Manifests/` under the active data root (computed via `data_manifests(root, ensure=…)`). The helpers *avoid* creating this directory when *reading* (CLI / tailers); they only ensure it exists on *write*.
* **Per-stage filenames** use a canonical pattern:
  `docparse.{stage_sanitised}.{kind}.jsonl`
  where `{kind}` is `"manifest"` for the manifest stream and `"attempts"` for the attempts stream; the sanitiser replaces non [A-Za-z0-9._-] chars with `-`.

  * Helpers: `_telemetry_filename(stage, kind)`, `_manifest_filename(stage)`, `resolve_manifest_path(stage, root)`, `resolve_attempts_path(stage, root)`.
* **Common stage ids** in the tree today:

  * PDF DocTags: `MANIFEST_STAGE` (the code binds a logger with `stage=MANIFEST_STAGE` and writes a `__config__` row at start).
  * HTML DocTags: `HTML_MANIFEST_STAGE = "doctags-html"`.
  * Embeddings stage: `EMBED_STAGE` (used for event telemetry and summary rows).

## Atomicity & append discipline

* **Append API**: all manifest writes go through `jsonl_append_iter` with **atomic single-shot** append; directory fsync is performed. `manifest_append` defaults to `atomic=True`.
* **Telemetry sink** (`StageTelemetry`) also appends attempts and manifest rows and (in the current code) calls `jsonl_append_iter(..., atomic=True)` under its private lock wrapper.

## Entry shapes (row schema)

Entries are JSON Lines (one JSON object per line). Writer functions guarantee a **UTC** timestamp at write-time (via `manifest_append`). Required / optional fields are enforced by composition in `logging.py` and `io.py`.

### Common payload for **success** entries

```json
{
  "timestamp": "...",          // generated in io.manifest_append
  "stage": "<stage-id>",       // e.g., "doctags", "doctags-html", "embeddings"
  "doc_id": "<relative-id>",   // relative path id is the project-wide norm
  "status": "success",
  "duration_s": <float>,
  "schema_version": "<semver>",// e.g., "docparse/1.1.0"
  "input_path": "<path|string>",
  "input_hash": "<hex>",
  "hash_alg": "<sha1|sha256|…>",// auto-filled when absent
  "output_path": "<path|string>",

  "...extra": "free-form extras"
}
```

* The high-level helpers in `logging.py` merge **extras** (e.g., chunk counts, vector totals, SPLADE stats, etc.) before calling `manifest_append`. When present, they also infer an integer `tokens` to send into the telemetry sink (using `tokens | chunk_count | vector_count`).
* `hash_alg` is auto-populated if omitted; the code path sets `hash_alg or resolve_hash_algorithm()`.

### Common payload for **skip** entries

```json
{
  "timestamp": "...",
  "stage": "<stage-id>",
  "doc_id": "<relative-id>",
  "status": "skip",
  "duration_s": <float>,
  "schema_version": "<semver|null>",
  "input_path": "<path|string>",
  "input_hash": "<hex>",
  "hash_alg": "<sha1|sha256|…>",
  "output_path": "<path|string>",
  "reason": "<optional-human-string>",

  "...extra": "free-form extras"
}
```

* Composition and auto-fields mirror the *success* case; the logging helper records `reason` if provided.

### Common payload for **failure** entries

```json
{
  "timestamp": "...",
  "stage": "<stage-id>",
  "doc_id": "<relative-id>",
  "status": "failure",
  "duration_s": <float>,
  "schema_version": "<semver>",
  "input_path": "<path|string>",
  "input_hash": "<hex>",
  "hash_alg": "<sha1|sha256|…>",
  "output_path": "<path|string|null>",
  "error": "<string>",

  "...extra": "free-form extras"
}
```

* The helper includes `error` and merges extra metadata (e.g., parse engine, model info).

### Stage-specific examples you already emit

* **Doctags config row** at start (PDF):
  `doc_id="__config__"`, carries resolved run config and I/O dirs.
* **Doctags per-file rows**: success/skip/failure all include `parse_engine`, `model_name`, `served_models`, `vllm_version` extras.
* **Embeddings config row** at start (vector format included now): extra `vector_format`, `files_parallel`, sparsity thresholds, etc.
* **Embeddings corpus summary** at end: `doc_id="__corpus__"` with SPLADE/Qwen stats, totals, warnings.

### Data types and guards

* `input_path` / `output_path` are stringified with a small helper; absent/`None` becomes `null` in JSON.
* The manifest append function injects the **timestamp** and validates that `status` ∈ {`"success"`, `"skip"`, `"failure"`}.
* **Relative IDs**: the codebase has moved toward relative identifiers for manifest keys; the planning and migration notes reflect this.

---

# Attempts stream (who tried to do what, when)

Your **attempts** file (`docparse.{stage}.attempts.jsonl`) records lifecycle events independent from the success/failure rows:

* **Dataclasses** in `telemetry.py` define the wire shape:

  * `Attempt(run_id, file_id, stage, status, reason, started_at, finished_at, bytes, metadata)`
  * `ManifestEntry(run_id, file_id, stage, output_path, tokens, schema_version, duration_s, metadata)`
    The sink flattens `metadata` into the top-level JSON per line.
* **Paths**: use `resolve_attempts_path(stage, root)` and `resolve_manifest_path(stage, root)`; both are under `Data/Manifests/` with the telemetry naming scheme.

---

# Reading & indexing

* **Tail iteration**: `iter_manifest_entries(stages, root, *, limit=None)` merges the newest lines across stage files, optionally using **bounded tail windows** per file to avoid scanning full histories (controlled by `_MANIFEST_TAIL_MIN_WINDOW` and `_MANIFEST_TAIL_BYTES_PER_ENTRY`).
* **Read-only**: readers resolve `data_manifests(root, ensure=False)` when tailing; they do not create directories during a simple CLI read.

---

# Runner behaviors matrix (PR-5 target semantics that match today’s logging)

> Rows below assume the unified runner invokes stage workers and the logging/telemetry helpers above. “Writes” refers to manifest & attempts streams.

### Legend

* **Result**: what the worker returns (`success|skip|failure`)
* **Writes**: which rows appear in which files
* **Notes**: normalization, counters, and payload highlights

| Scenario                        | Precondition & Flags                                                          | Result           | Writes                                              | Notes                                                                                 |
| ------------------------------- | ----------------------------------------------------------------------------- | ---------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Doctags **config** preflight    | Stage start                                                                   | success          | `manifest[__config__]`                              | Emits resolved run config (I/O dirs, model, vLLM info).                               |
| Doctags **skip** (resume)       | Output exists **and** input hash unchanged; `--resume` true and not `--force` | skip             | `manifest[skip]`                                    | Records `reason` and carries parse metadata; `hash_alg` auto-filled.                  |
| Doctags **success**             | Work done                                                                     | success          | `manifest[success]`                                 | Includes `duration_s`, `schema_version`, engine/model extras.                         |
| Doctags **failure**             | Exception in worker                                                           | failure          | `manifest[failure]`                                 | Carries `error`, engine/model extras; continues run unless budget cancels (PR-5).     |
| HTML Doctags **no input files** | `list_htmls/iter_htmls` empty                                                 | skip (aggregate) | `log_event` + (often) no per-doc manifest rows      | Logged via `log_event` with `error_code="NO_INPUT_FILES"`.                            |
| Embeddings **config** preflight | Stage start                                                                   | success          | `manifest[__config__]`                              | Emits vector format, shard, sparsity thresholds, env devices.                         |
| Embeddings **plan-only**        | `--plan-only`                                                                 | success          | `log_event(plan summary)`                           | Runner logs process/skip counts and previews; no vector writes.                       |
| Embeddings **skip** (resume)    | Output exists, input hash unchanged; `--resume` true and not `--force`        | skip             | `manifest[skip]`                                    | Row per skipped chunk file; also an info log with relpaths.                           |
| Embeddings **per-file success** | Vectors written                                                               | success          | `log_event("Embedding file written")` *(telemetry)* | Includes `vectors`, average SPLADE nnz & Qwen norms; emits per-file timing.           |
| Embeddings **corpus summary**   | End of stage                                                                  | success          | `manifest[__corpus__]`                              | Aggregates `total_vectors`, p95/99, zero-nnz %, peak memory, skipped/quarantined.     |
| Embeddings **failure**          | Vector write fails for a file                                                 | failure          | `manifest[failure]`                                 | Includes `vector_format`, input hash, error; runner continues unless budget cancels.  |

### Attempt rows (if you enable them in the runner around each item)

| Scenario               | Writes               | Notes                                                                                                                        |         |                                                                         |
| ---------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------------------------------------------------------------- |
| Before invoking worker | `attempts[started]`  | `Attempt(run_id, file_id, stage, status="started", started_at, bytes=_input_bytes(input_path))`. Path handling accepts `Path | str`.   |                                                                         |
| On completion          | `attempts[finished]` | `status="success                                                                                                             | skip    | failure"`,`finished_at`,`reason`if failure, and flattened`metadata`.  |

> **Tip**: keep `doc_id` stable and relative. Your planning notes already call out the transition to relative ids for manifest stability.

---

# Content of extras (by stage)

These are already present in your rows and will continue to appear through the PR-5 runner since logging stays identical:

* **Doctags PDF/HTML**:
  `parse_engine`, `model_name`, `served_models`, `vllm_version`, and for HTML you also tag input sanitiser profiles as part of worker context.
* **Embeddings per-file**:
  `vectors`, `splade_avg_nnz`, `qwen_avg_norm`, plus relpaths for input/output, elapsed ms, `vector_format`.
* **Embeddings corpus**:
  `total_vectors`, SPLADE p95/p99 nnz, zero-pct, Qwen norm distribution (avg/std/p95/p99), `skipped_files`, `quarantined_files`, `files_parallel`, `splade_attn_backend_used`, `sparsity_warn_threshold_pct`.

---

# Edge-case semantics (recommended & partially present)

* **Vector format field**: You now pass `vector_format` (`jsonl|parquet`) through config and row extras. Keep emitting it in config, per-file events, and failures for *traceability*.
* **Tokens field**: The sink’s `write_manifest` treats `tokens=None` as `0`. Only supply an integer when it adds clarity (e.g., `chunk_count`, `vector_count`), otherwise omit.
* **Input/Output stringification**: paths are coerced to strings before writing JSON; missing paths remain `null`.

---

# Runner → manifests: exact “when we write what”

Below is a **timing flow** you can wire straight into your `run_stage` (PR-5), preserving today’s shapes:

1. **before_stage** hook

   * Write `__config__` success entry for the stage (includes full config snapshot). (Doctags, Embeddings).
   * (Embeddings) emit a `log_event(status="config")` with vector format, devices, etc., for operator UX.

2. **per item**

   * (Optional) `attempts[started]` → then run worker. See *Attempts stream*.
   * On **skip** → `manifest_log_skip(...)` with `reason` (if known).
   * On **success** → `manifest_log_success(...)`.
   * On **failure** → `manifest_log_failure(...)`.
   * (Optional) `attempts[finished]`.

3. **after_stage** hook

   * (Embeddings) `__corpus__` success summary with totals and distribution stats.

All three helpers route through `manifest_append` → `jsonl_append_iter(..., atomic=True)`.

---

# Extras: Stage ids & filenames cheat-sheet

* **HTML Doctags**: stage `"doctags-html"` → file `docparse.doctags-html.manifest.jsonl`.
* **PDF Doctags**: stage `MANIFEST_STAGE` (commonly `"doctags"` in your CLI contexts) → `docparse.doctags.manifest.jsonl`. (Bound in logs at stage start; same helpers resolve filename.)
* **Embeddings**: stage `EMBED_STAGE` → `docparse.embeddings.manifest.jsonl` (plus attempts file). (Used throughout embedding runtime log/summary.)

---

# Optional polish (low risk)

* **Manifest tail on CLI**: your `limit` impl is in place; keep it wired to `manifest --tail N` so SREs can tail recent entries without full scans.
* **Relative IDs everywhere**: planning code and migration notes already call for relative `doc_id`; enforce that in workers (you’re already doing it in doctags and embedding via `relative_path`).
* **Attempts “bytes”**: you broadened `_input_bytes` to accept `Path | str`; that’s perfect for embedding/HTML where string paths are common.

---

## TL;DR checklists

**Writer invariants**

* `timestamp` always present (UTC).
* `{stage, doc_id, status}` always present.
* `hash_alg` always present after write (explicit or auto-resolved).
* Atomic append used for all manifest writes.

**Reader invariants**

* `iter_manifest_entries(stages, root, limit=None)` merges and sorts across files; `limit` bounds I/O by reading from the tail per file.

**Stage semantics hints**

* Doctags writes: per-doc rows + `__config__`.
* Embeddings writes: per-file telemetry, `__corpus__` summary, `__config__`, and per-file skip rows under resume.

---
