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

<!-- 2025-10-23 04:06:49Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 04:06:56Z UTC -->
## Pass 2 — find and fix real bugs

<!-- 2025-10-23 04:24:41Z UTC -->
## Pass 1 — find and fix real bugs

### Batch 0 (Pass 1)
- Broken: `storage.dataset_view.summarize` tried `Path.stat()` on fragment paths, throwing `FileNotFoundError` for remote or non-local filesystems so summaries could not run outside local disk.
- Fixed: resolve fragment metadata via the Arrow filesystem (fall back to local `Path.stat`), keep partition/doc-id extraction in place, and suppress filesystem lookup errors.
- TODO: add coverage that simulates non-local filesystems to catch regressions.
