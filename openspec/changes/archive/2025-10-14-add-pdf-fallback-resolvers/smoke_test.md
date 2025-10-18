# Manual Smoke Test Checklist

Run these steps after deploying the resolver stack to verify end-to-end behaviour without stressing third-party services.

1. Prepare a list of 10 DOIs (mix of open-access and paywalled) plus at least one arXiv ID and one PMID/PMCID.
2. Export the list into a text file and execute the downloader with `--max-resolver-attempts 10`, `--sleep 0.25`, and resolver credentials populated via environment variables.
3. Confirm the CLI prints a resolver summary, writes attempts to the configured CSV log, and stores PDFs under the output directory.
4. Inspect two HTML captures (if any) to ensure non-PDF responses are preserved correctly.
5. Review the structured log message (`resolver_run_summary …`) for processed, saved, and resolver metrics totals.
6. Spot-check Unpaywall and Crossref traffic stays below 1 request per second using the timestamps in the log.
7. Archive the CSV log for operational review.

# Metrics and Alert Expectations

| Metric | Target | Alert Threshold |
| --- | --- | --- |
| Resolver success rate (PDFs saved / works processed) | ≥ 70% for OA-heavy batches | Alert if < 50% over a 24h period |
| HTTP error rate per resolver | < 5% | Alert if ≥ 10% over 1h |
| Resolver skips due to configuration | 0 during standard runs | Alert if > 0 indicating misconfiguration |
| Wayback usage rate | < 2% of works | Investigate spikes (may signal widespread publisher issues) |
| Average resolver latency | < 5s per attempt | Alert if ≥ 10s sustained |

These expectations feed into the existing operations dashboard for content ingestion. Update alerting rules if service limits or resolver availability changes.
