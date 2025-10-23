# Codex Review Log — `src/DocsToKG/HybridSearch`
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

<!-- 2025-10-23 04:06:55Z UTC -->
## Pass 2 — find and fix real bugs

<!-- 2025-10-23 04:24:41Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 04:25:01Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 0 (Pass 2)
- Broken: Dense planner signatures iterated set-valued filters in arbitrary order, preventing cache hits and destabilising dense oversampling.
- Fix:
  - Normalise set inputs by sorting their normalised elements with a deterministic key before tuple conversion.
  - Added a helper to generate stable sort keys for signature elements.
- TODO: None.

<!-- 2025-10-23 04:46:08Z UTC -->
## Pass 1 — find and fix real bugs

### Batch 0 (Pass 1)
- Broken: List-valued filters stopped matching numeric metadata because request normalization coerced expected values to strings while registry values stayed numeric.
- Fix:
  - Add cross-type comparison helpers in `devtools/opensearch_simulator.py` so filter checks compare both native and string/number representations.
  - Reuse the flexible matcher for multi-valued metadata to keep list filters consistent.
- TODO: None.

<!-- 2025-10-23 04:58:47Z UTC -->
## Pass 2 — find and fix real bugs
### Batch 0 (Pass 2)
- Broken: `FaissRouter.restore_all` dropped cached namespace snapshots even when `store.restore` raised, permanently discarding the fallback image.
- Fix:
  - Track restore success and only clear `_snapshots` once the restore completes without error.
- TODO: None.

<!-- 2025-10-23 05:59:19Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 05:59:23Z UTC -->
## Pass 2 — find and fix real bugs

<!-- 2025-10-23 05:59:29Z UTC -->
## Pass 3 — find and fix real bugs

<!-- 2025-10-23 05:59:39Z UTC -->
## Pass 4 — find and fix real bugs

<!-- 2025-10-23 06:00:14Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 06:45:07Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 06:45:11Z UTC -->
## Pass 2 — find and fix real bugs

<!-- 2025-10-23 06:52:53Z UTC -->
## Pass 1 — find and fix real bugs

### Batch 0 (Pass 1)
- Broken: `FaissRouter.restore_all` always forwarded snapshot metadata to adapter restores, raising `TypeError` for stores that omit a `meta` parameter and blocking namespace recovery.
- Fix:
  - Normalise legacy snapshot payloads with the existing helper before restore attempts.
  - Reuse the reflective `restore_store` wrapper to only pass metadata when the adapter supports it.
- TODO: None.

<!-- 2025-10-23 06:57:19Z UTC -->
## Pass 2 — find and fix real bugs
### Batch 0 (Pass 2)
- Broken: `HybridSearchConfig.from_dict` crashed when configs set sections to `null`, raising `TypeError` during reload and blocking defaults from applying.
- Fix:
  - Coerce per-section payloads so `None` maps to an empty dict while non-mapping inputs raise a clear `ValueError`.
  - Reuse the coercion for all config blocks to keep alias translation working with validated data.
- TODO: None.

<!-- 2025-10-23 07:04:16Z UTC -->
## Pass 3 — find and fix real bugs

### Batch 0 (Pass 3)
- Broken: Ingestion accepted blank or malformed chunk UUIDs until FAISS tried to coerce them, raising `ValueError` and misclassifying the data issue as retryable.
- Fix:
  - Normalise and validate chunk/vector UUID fields while loading artifacts so bad identifiers raise `IngestError` early.
  - Canonicalise cached vector IDs and continue flagging entries that omit UUIDs.
- TODO: None.

<!-- 2025-10-23 07:17:04Z UTC -->
## Pass 4 — find and fix real bugs
