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
