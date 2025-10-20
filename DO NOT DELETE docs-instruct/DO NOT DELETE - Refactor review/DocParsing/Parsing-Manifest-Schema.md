Awesome — below is a **lintable JSON Schema** for DocParsing **manifest rows** (single JSONL line), plus an optional schema for the **attempts** stream. I’ve aligned field names, types, and invariants with what your code actually writes today:

* `manifest_append(...)` injects `timestamp`, enforces `status ∈ {"success","failure","skip"}`, and carries `duration_s`, `warnings`, and `schema_version`, merging any extra metadata you pass (e.g., `input_path`, `input_hash`, `hash_alg`, `output_path`, engine/Model info).
* The manifest shard filename pattern is `docparse.{stage}.manifest.jsonl` (stage sanitized).
* The manifest helpers construct payloads for **success/skip/failure** with the same base keys and add `hash_alg` via `resolve_hash_algorithm()` when not supplied.
* `jsonl_append_iter(..., atomic=True)` is the default now for manifests and telemetry (single-shot append + fsync of file and directory).
* The “manifest tail” CLI uses a bounded tail-window reader; schema below doesn’t depend on that but it’s relevant for tooling.

---

## 1) JSON Schema — **DocParsing manifest entry** (one JSON object per line)

```json
{
  "$id": "https://docs.docsToKG/DocParsing/manifest-entry.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "DocParsing Manifest Entry",
  "type": "object",
  "additionalProperties": true,
  "properties": {
    "timestamp": { "type": "string", "format": "date-time", "description": "UTC ISO-8601, injected by manifest_append()" },
    "stage":      { "type": "string", "minLength": 1, "pattern": "^[A-Za-z0-9._-]+$", "description": "e.g., doctags-pdf, doctags-html, chunks, embeddings" },
    "doc_id":     { "type": "string", "minLength": 1, "description": "Stable, relative identifier (often a relative path-like id)" },
    "status":     { "type": "string", "enum": ["success","failure","skip"] },
    "duration_s": { "type": "number", "minimum": 0 },
    "warnings":   { "type": "array", "items": { "type": "string" }, "default": [] },
    "schema_version": {
      "type": "string",
      "description": "Row schema/version tag",
      "pattern": "^docparse/[0-9]+\\.[0-9]+\\.[0-9]+$"
    },

    "input_path":  { "type": ["string","null"] },
    "input_hash":  { "type": ["string","null"], "description": "Hex digest of content hash; algorithm in hash_alg" },
    "hash_alg":    { "type": ["string","null"], "description": "Name of the hashing algorithm, e.g., sha256" },
    "output_path": { "type": ["string","null"] },

    "error": { "type": "string", "description": "Failure reason (only present for status=failure)" }
  },

  "required": ["timestamp","stage","doc_id","status","duration_s"],

  "allOf": [
    {
      "if": { "properties": { "status": { "const": "success" } } },
      "then": { "required": ["input_path","input_hash","output_path"] }
    },
    {
      "if": { "properties": { "status": { "const": "skip" } } },
      "then": { "required": ["input_path","input_hash","output_path"] }
    },
    {
      "if": { "properties": { "status": { "const": "failure" } } },
      "then": { "required": ["error","input_path","input_hash","output_path"] }
    }
  ],

  "examples": [
    {
      "timestamp": "2025-10-19T08:10:21Z",
      "stage": "doctags-html",
      "doc_id": "teamA/report.html",
      "status": "success",
      "duration_s": 0.734,
      "schema_version": "docparse/1.1.0",
      "input_path": "/data/html/teamA/report.html",
      "input_hash": "1f3870be274f6c49b3e31a0c6728957f",
      "hash_alg": "sha256",
      "output_path": "/data/doctags/teamA/report.doctags",
      "parse_engine": "docling-html",
      "html_sanitizer": "balanced"
    },
    {
      "timestamp": "2025-10-19T08:11:08Z",
      "stage": "doctags-pdf",
      "doc_id": "teamB/brief.pdf",
      "status": "skip",
      "duration_s": 0.002,
      "schema_version": "docparse/1.1.0",
      "input_path": "/data/pdf/teamB/brief.pdf",
      "input_hash": "0cc175b9c0f1b6a831c399e269772661",
      "hash_alg": "sha256",
      "output_path": "/data/doctags/teamB/brief.doctags",
      "parse_engine": "docling-vlm",
      "model_name": "granite-docling-258M"
    },
    {
      "timestamp": "2025-10-19T08:12:49Z",
      "stage": "doctags-pdf",
      "doc_id": "teamC/broken.pdf",
      "status": "failure",
      "duration_s": 0.287,
      "schema_version": "docparse/1.1.0",
      "input_path": "/data/pdf/teamC/broken.pdf",
      "input_hash": "751d3eab9c1c2f11d2f0a3ecb71a4c41",
      "hash_alg": "sha256",
      "output_path": "/data/doctags/teamC/broken.doctags",
      "error": "PDF_CONVERSION_FAILED",
      "parse_engine": "docling-vlm"
    },
    {
      "timestamp": "2025-10-19T08:00:00Z",
      "stage": "embeddings",
      "doc_id": "__config__",
      "status": "success",
      "duration_s": 0.0,
      "schema_version": "docparse/1.1.0",
      "input_path": "/data/chunks",
      "input_hash": "",
      "output_path": "/data/vectors",
      "vector_format": "parquet",
      "files_parallel": 8
    },
    {
      "timestamp": "2025-10-19T08:30:00Z",
      "stage": "embeddings",
      "doc_id": "__corpus__",
      "status": "success",
      "duration_s": 10.214,
      "schema_version": "docparse/1.1.0",
      "input_path": "/data/chunks",
      "input_hash": "",
      "output_path": "/data/vectors",
      "total_vectors": 5342,
      "splade_avg_nnz": 27.4,
      "qwen_avg_norm": 0.997
    }
  ]
}
```

**Why these fields & constraints?**

* `status` and `timestamp` come directly from `manifest_append`, which injects a UTC ISO-8601 timestamp and validates the status set.
* Success/skip/failure **payload shapes** mirror your logging helpers that add `input_path`, `input_hash`, `hash_alg`, and `output_path` (with `hash_alg` defaulted if not provided).
* The schema is **extensible** (`additionalProperties: true`) to allow provider- and stage-specific extras like `parse_engine`, `model_name`, `served_models`, `vector_format`, etc., which you already include.

---

## 2) JSON Schema — **Attempts entry** (optional, if you validate attempts JSONL)

This matches the dataclass + sink behavior: attempts rows are written with `run_id`, `file_id`, `stage`, `status`, `reason?`, start/finish times as epoch seconds, `bytes`, and **any flattened metadata** you included.

```json
{
  "$id": "https://docs.docsToKG/DocParsing/attempt-entry.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "DocParsing Attempt Entry",
  "type": "object",
  "additionalProperties": true,
  "properties": {
    "run_id":      { "type": "string", "minLength": 1 },
    "file_id":     { "type": "string", "minLength": 1 },
    "stage":       { "type": "string", "minLength": 1, "pattern": "^[A-Za-z0-9._-]+$" },
    "status":      { "type": "string", "minLength": 1 },
    "reason":      { "type": ["string","null"] },
    "started_at":  { "type": "number", "minimum": 0, "description": "Epoch seconds" },
    "finished_at": { "type": "number", "minimum": 0, "description": "Epoch seconds" },
    "bytes":       { "type": "integer", "minimum": 0 }
  },
  "required": ["run_id","file_id","stage","status","started_at","finished_at","bytes"],
  "examples": [
    {
      "run_id": "2025-10-19T08:00:00Z#abcd",
      "file_id": "teamA/report.doctags",
      "stage": "doctags-html",
      "status": "success",
      "reason": null,
      "started_at": 1697702400.0,
      "finished_at": 1697702400.734,
      "bytes": 133742
    }
  ]
}
```

> Note: The sink flattens any `metadata` dict into top-level fields before writing attempts/manifest entries — keep `additionalProperties: true`.

---

## Placement & filenames (recommended)

* **Manifest rows**: `Data/Manifests/docparse.{stage}.manifest.jsonl` (e.g., `docparse.doctags-html.manifest.jsonl`).
* **Attempts rows**: `Data/Manifests/docparse.{stage}.attempts.jsonl` (the same stage shard family used by `StageTelemetry`).

---

## Validation policy & linter rules (drop straight into CI)

1. **Per-line validation**: run a JSON Schema validator against each JSONL line (streaming). Fail on first invalid line; report `line_no`, `path`, `error`.
2. **Severity**

   * **Error**: missing required base fields; status not in enum; negative `duration_s`; malformed `timestamp` (non-ISO); non-string `stage`/`doc_id`.
   * **Warn**: missing `schema_version` (allowed but discouraged); `hash_alg` missing (allowed, your logger back-fills via `resolve_hash_algorithm()`).
3. **Status-specific checks** (enforced by schema `if/then`)

   * `success`/`skip`: must include `input_path`, `input_hash`, `output_path`.
   * `failure`: must include `error` plus the same IO fields.
4. **Atomicity & order-agnostic**: trust that writes were atomic single-shot (your writer guarantees this) and that readers can handle interleaved rows across stages.
5. **Tail safety**: tools that tail the last N entries should call `iter_manifest_entries(..., limit=N)` to avoid scanning full history.

---

## Evolution & compatibility guidelines

* **Add, don’t break**: you can add new **optional** top-level fields freely — schema allows extras (`additionalProperties: true`).
* **Keep base keys stable**: `timestamp`, `stage`, `doc_id`, `status`, `duration_s` must remain (your writer populates/injects these).
* **Version tagging**: keep `schema_version` of rows aligned to `docparse/x.y.z`; reject malformed tags at ingest time (linter warning → optional hard error later).
* **Stage ids**: sanitized and lowercased are fine; you already normalize to filesystem-safe form.
* **Config & summary rows**: use special `doc_id`s (`"__config__"`, `"__corpus__"`) with `status="success"`; allow arbitrary extras for snapshots/summaries (schema already permits). (These are emitted today in doctags/embedding flows.)

---

## Operator notes & best practices

* **Hash provenance**: always include `hash_alg` for forensics; your logger will backfill if omitted, but recording it explicitly is clearer.
* **Path canonicalization**: store **absolute** `input_path`/`output_path` or a consistent relative root; your schema is permissive, but consistency simplifies triage.
* **Consistent IDs**: prefer **relative** `doc_id` (path-like) across all stages — your planning/PRs already moved in that direction and readers treat `doc_id` as opaque.
* **One shard per stage**: keep manifests sharded per stage (filenames above) rather than monolithic; your reader already merges per-stage shards and can tail efficiently.

---

### Where this spec comes from (key callouts)

* **Status/shape injected by writer** and **field names** (`input_path`, `input_hash`, `hash_alg`, `output_path`, `error`) as used in manifest helpers.
* **Shard naming** (`docparse.{stage}.manifest.jsonl`).
* **Atomic append** default.
