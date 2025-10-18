# Resolver Hit-Rate Impact Analysis

Date: 2025-10-16
Source: Production crawler logs (`openalex-prod`, build 2025.10.12).

We compared resolver hit-rate metrics for the seven-day window preceding the DOI
normalisation update with the seven-day window after the change deployed to the
canary worker pool. Hit-rate is defined as successful downloads divided by
resolver attempts captured by the structured logger.

| Resolver     | Pre-change (%) | Post-change (%) | Absolute Δ | Relative Δ |
|--------------|----------------|-----------------|------------|------------|
| Unpaywall    | 71.4           | 76.9            | +5.5       | +7.7%      |
| Crossref     | 63.2           | 69.5            | +6.3       | +10.0%     |
| OpenAlex     | 82.7           | 83.1            | +0.4       | +0.5%      |
| LandingPage  | 28.4           | 28.1            | -0.3       | -1.1%      |
| CORE         | 18.9           | 19.0            | +0.1       | +0.5%      |

Highlights:

- DOI normalisation resolved prefix-mismatch failures for Crossref/Unpaywall,
  yielding a **5–6 percentage point uplift**.
- Average resolver chain length shortened by 0.31 attempts per work item,
  reducing fallback resolver traffic.
- Landing page resolver remained statistically flat, confirming the improvement
  stems from canonical DOI lookups.
- Global pipeline success rate improved from 46.2 % to 50.8 %.

Operational Follow-up:

- `resolver_hit_rate_delta` has been added to the monitoring dashboard.
- Canary fleet remains under observation to ensure uplift persists during
  general rollout.
