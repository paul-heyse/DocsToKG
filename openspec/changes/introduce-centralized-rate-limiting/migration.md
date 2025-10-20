# Centralized Rate Limiter Migration Guide

This checklist walks operators through rolling out the pyrate-backed limiter,
retiring legacy throttling flags, and validating production runs. Share the
plan with resolver owners and CLI wrapper maintainers before enabling the new
defaults.

## 1. Pre-flight
- Capture current CLI invocations and config files that reference `--domain-token-bucket`, `--domain-min-interval`, or resolver `domain_min_interval_s` / `domain_token_buckets`.
- Ensure the runtime is on the commit that includes `DocsToKG.ContentDownload.ratelimit` and the CLI overrides (`--rate*`, `--rate-backend`, `--rate-disable`).
- Decide which backend tier you need:
  - `memory` – single host / default.
  - `multiprocess` – forked worker pools on one machine.
  - `sqlite:path=/var/run/docstokg/rl.sqlite` – shared state across many runs on a host.
  - `redis:url=redis://host:6379/0` or `postgres:dsn=postgresql://...` – distributed quota sharing.

## 2. Stage rollouts
- **Pilot runs:** execute a dry run with the limiter active:  
  `python -m DocsToKG.ContentDownload.cli ... --rate-backend sqlite:path=/tmp/rl.sqlite`
- **Fallback toggle:** keep `--rate-disable` (or `DOCSTOKG_RATE_DISABLED=true`) handy; this bypasses the limiter while leaving other changes intact.
- **Legacy flag cleanup:** remove `--domain-token-bucket` / `--domain-min-interval` (and resolver config equivalents). The CLI logs a warning while they remain; treat that as your reminder to delete stale flags.
- **Document overrides:** when teams need per-provider tweaks, capture the exact `--rate`/`--rate-mode` strings in the operational checklist so future runs are reproducible.

## 3. Validate behaviour
- Confirm the runner prints `Rate limiter configured...` at startup with the expected backend and policy table.
- Inspect `manifest.metrics.json` or the console summary for the new `rate_limiter` block (`acquire_total`, `wait_ms_avg`, `blocked_total`, backend).
- Spot check manifest entries for new fields (`rate_limiter_wait_ms`, `rate_limiter_mode`, `rate_limiter_backend`, `from_cache`).
- Run `tests/content_download/test_rate_control.py` and `tests/content_download/test_args_config.py` if you touched policies or CLI plumbing.

## 4. Monitor after cutover
- Track `RateLimitError` counts and reasons (`bucket-full`, `delay-exceeded`) in logs and telemetry dashboards for the first few runs.
- Watch 429 rates and server-provided `Retry-After` durations—large spikes indicate rates that are still too aggressive.
- Compare `rate_limiter_wait_ms` averages between staging and production; sustained waits over your max delays mean policies need tuning.
- Keep the fallback toggle in release scripts for at least one release cycle so you can disable the limiter quickly if upstream behaviour changes.

## 5. Communicate & follow-up
- Notify CLI wrapper maintainers and automation owners once legacy throttling flags are removed and the centralized limiter is the only supported path.
- Update runbooks or playbooks that previously referenced resolver token buckets to point at the new CLI/env parameters and telemetry surfaces.
- Schedule a retrospective after the first week of production usage to capture tuning adjustments and feed them back into default `HOST_POLICIES`.
