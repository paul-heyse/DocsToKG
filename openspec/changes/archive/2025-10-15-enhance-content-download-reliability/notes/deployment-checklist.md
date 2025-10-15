# Deployment & Monitoring Checklist

Use this checklist when promoting the content download reliability changes to
staging and production environments.

## Pre-Deployment

- `pytest -k "download or resolver or pipeline"` on the release branch.
- `pytest tests/test_full_pipeline_integration.py` to verify manifest + metrics flow.
- Run `scripts/mock_download_job.py --concurrent-resolvers 4 --domain-min-interval example.org=0.5`
  against staging metadata to confirm CLI wiring.
- Confirm `.metrics.json` files land alongside manifests in staging storage.
- Smoke-test resume mode: rerun a staging job with `--resume` and ensure previously
  downloaded PDFs are skipped without re-fetching.

## Deployment Plan

1. Merge implementation PRs to `main`; deploy to canary worker pool only.
2. Clear resolver caches on canary nodes (`python -m DocsToKG.ContentDownload.resolvers.cache clear`).
3. Monitor resolver error rate and retry budget exhaustion for 24 hours.
4. If stable, roll out to remaining workers; no configuration migration required.
5. Post-rollout, archive validator screenshots/metrics to `reports/content-download/2025-10-16/`.

## Post-Deployment Monitoring

- Dashboard widgets:
  - Success rate vs baseline (`resolver_hit_rate_delta`).
  - Retry budget exhaustion count and median backoff delay.
  - Duplicate URL skip rate (global dedupe).
  - Domain rate limiting wait time percentiles.
- Alert thresholds:
  - Success rate drops by >5 percentage points for any resolver.
  - Retry budget exhaustion exceeds 2 % of downloads.
  - `.metrics.json` ingestion lag >10 minutes.
- Operator feedback: gather notes in the weekly Ops sync; file follow-up issues
  for any regressions encountered.
