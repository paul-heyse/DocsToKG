# 1. Hybrid Search Operations Runbook

## 2. Overview
Hybrid search combines BM25, SPLADE, and FAISS dense retrieval. This runbook summarizes
daily operations, calibration routines, and contingency procedures for the ingestion and
retrieval subsystems.

## 3. Calibration Sweeps
1. Launch the validation harness:
   `python -m DocsToKG.HybridSearch.validation --run-real-ci --pytest-args "-q --real-vectors"`
   or invoke `python -m DocsToKG.HybridSearch.validation --dataset
   tests/data/hybrid_dataset.jsonl` for ad-hoc checks.
2. Review `calibration.json` in the generated report directory. Confirm that self-hit accuracy is ≥0.95 for oversample ≥2.
3. If accuracy drops below threshold:
   - Increase `dense.oversample` in `hybrid_config.json`.
   - Reload the configuration (`HybridSearchConfigManager.reload`).
   - Re-run the validation harness and compare the new report.

## 4. Namespace Onboarding
1. Create a namespace entry in the configuration with chunk window overrides if needed.
2. Run the ingestion pipeline with a smoke dataset for the new namespace.
3. Verify OpenSearch stats using `build_stats_snapshot` and ensure `document_count` > 0 for the namespace.
4. Execute `verify_pagination` for a representative query to confirm cursor stability.

## 5. Failover and Rollback
1. Serialize the FAISS index with `serialize_state` and persist the payload alongside OpenSearch snapshots.
2. During failover, restore OpenSearch from snapshot and call `restore_state` on the FAISS
   manager. Confirm startup logs include a `faiss-index-config` event with the expected
   `index_type`, `nlist`, `nprobe`, and device metadata before serving traffic.
3. Warm the cache by issuing `HybridSearchRequest` probes for top queries.
4. Use `verify_pagination` to ensure cursor continuity after failover.
5. If errors persist, revert to the previous config file and call `HybridSearchConfigManager.reload`.

## 6. Backup and Restore Drills
1. Schedule weekly jobs to run `serialize_state` and OpenSearch snapshots.
2. Store artifacts in durable object storage with retention >=30 days.
3. Quarterly, perform a restore into a staging environment and execute the validation harness to confirm parity.
4. Record results in the operational log and escalate if any validation report fails.

## 7. Delete Churn Monitoring
- Track `delete_chunks` and `ingest_chunks` counters from `Observability.metrics_snapshot`.
- When `should_rebuild_index` returns `True`, schedule a full rebuild:
  1. Drain ingestion traffic.
  2. Reconstruct FAISS from the authoritative registry list.
  3. Re-enable ingestion and confirm `ntotal` aligns with registry count.

## 8. Module Consolidation Migration
- Legacy modules (`results`, `similarity`, `retrieval`, `schema`, `operations`, `tools`) now emit
  `DeprecationWarning`. Update automation and notebooks to import from the consolidated
  modules documented in `docs/hybrid_search_module_migration.md` before the v0.6.0 release.
