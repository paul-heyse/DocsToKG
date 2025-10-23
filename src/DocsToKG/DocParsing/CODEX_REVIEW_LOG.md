# Codex Review Log — `src/DocsToKG/DocParsing`
This log is maintained by `scripts/codex-review-parallel.sh`.

<!-- 2025-10-22 16:55:34Z UTC -->
No eligible files after excludes.

<!-- 2025-10-22 17:01:20Z UTC -->
No eligible files after excludes.

<!-- 2025-10-22 17:13:46Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 01:39:11Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 02:48:00Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 03:05:49Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 03:55:00Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 04:04:17Z UTC -->
## Pass 1 — find and fix real bugs

### Batch 0 (Pass 1)
- Broken: `SettingsBuilder.add_env_overrides` split every underscore into nested keys, so overrides like `DOCSTOKG_APP_LOG_LEVEL` or `DOCSTOKG_EMBED_VECTOR_FORMAT` were ignored because they landed under non-existent paths (`app.log.level`, etc.).
- Fixed: normalise env names with a helper that honours `__` nesting, preserves snake_case field names, and falls back to known root sections before applying overrides.
- TODO: backfill coverage that exercises snake_case and double-underscore environment overrides for the unified settings builder.

### Batch 0 (Pass 1)
- Broken: `core.http.normalize_http_timeout` treated mapping inputs as iterables of keys, triggering `ValueError` when callers passed httpx-style timeout dicts so `get_http_session(timeout={...})` crashed.
- Fixed:
  - detect `httpx.Timeout` instances and coerce their connect/read timeouts safely
  - accept mapping inputs by coercing `connect`/`read`/`timeout` fields with sane defaults
- TODO: add focused coverage for mapping/Timeout inputs to `normalize_http_timeout`.

<!-- 2025-10-23 04:06:49Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 04:06:56Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 0 (Pass 2)
- Broken: `core.http.normalize_http_timeout` treated `(None, <read>)` tuples as if both slots were provided, so the connect timeout jumped to the read value instead of keeping the default.
- Fixed:
  - preserve positional defaults by recognising `None` entries while still coercing numeric/string values
  - reuse validation that surfaces bad literals early rather than silently misconfiguring retries
- TODO: backfill coverage that exercises tuple inputs with `None` placeholders for connect/read.

<!-- 2025-10-23 04:24:41Z UTC -->
## Pass 1 — find and fix real bugs

### Batch 0 (Pass 1)
- Broken: `storage.dataset_view.summarize` tried `Path.stat()` on fragment paths, throwing `FileNotFoundError` for remote or non-local filesystems so summaries could not run outside local disk.
- Fixed: resolve fragment metadata via the Arrow filesystem (fall back to local `Path.stat`), keep partition/doc-id extraction in place, and suppress filesystem lookup errors.
- TODO: add coverage that simulates non-local filesystems to catch regressions.

<!-- 2025-10-23 04:29:09Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 0 (Pass 2)
- Broken: `storage.dataset_view._extract_partition_from_path` only matched POSIX-style separators, so Windows/UNC fragment paths never surfaced partition buckets during dataset summaries.
- Fixed: Accept both `/` and `\` when parsing partition components so summaries stay accurate on Windows and shared-network mounts.
- TODO: Add regression coverage with Windows-style fragment paths once a cross-platform fixture harness lands.

<!-- 2025-10-23 04:46:08Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 04:50:55Z UTC -->
## Pass 2 — find and fix real bugs

<!-- 2025-10-23 05:59:19Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 05:59:23Z UTC -->
## Pass 2 — find and fix real bugs

<!-- 2025-10-23 05:59:29Z UTC -->
## Pass 3 — find and fix real bugs

<!-- 2025-10-23 06:00:14Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 06:45:07Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 06:45:11Z UTC -->
## Pass 2 — find and fix real bugs

<!-- 2025-10-23 06:52:53Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 07:04:13Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 0 (Pass 2)
- Broken: `embedding.backends.dense.tei` rejected valid TEI responses that wrap embeddings in an object, so inference runs crashed with a validation error instead of returning vectors.
- Fixed:
  - accept TEI payloads exposing an `embeddings` list or OpenAI-style `data[*].embedding` entries before coercing to floats.
- TODO: Cover TEI response-shape variants in unit tests.

<!-- 2025-10-23 07:12:42Z UTC -->
## Pass 3 — find and fix real bugs

### Batch 0 (Pass 3)
- Broken: empty-token corpora yielded `avgdl=0`, causing `bm25_vector` to raise `ZeroDivisionError` when normalising lengths.
- Fixed:
  - clamp the accumulated average doc length to a positive floor during finalisation.
  - reuse the guarded average length inside `bm25_vector` to keep per-chunk weighting safe.
- TODO: Add coverage for empty-token chunk inputs to ensure BM25 stays stable.

<!-- 2025-10-23 07:17:58Z UTC -->
## Pass 4 — find and fix real bugs

### Batch 0 (Pass 4)
- Broken: `embedding/backends/lexical/local_bm25.py` tokenised inputs using the class `[\\w]`, so every document collapsed to runs of the letter `w`, producing meaningless BM25 stats and vectors.
- Fixed:
  - correct the regex to `\w+` so `_tokenize` returns real word tokens under both `regex` and `re`.
- TODO: add a regression test that exercises LocalBM25Provider tokenisation end-to-end.

<!-- 2025-10-23 07:21:26Z UTC -->
## Pass 5 — find and fix real bugs
