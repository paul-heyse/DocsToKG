## Why
- ContentDownload currently layers bespoke token buckets, semaphores, and ad-hoc sleeps across multiple modules, which makes politeness policy drift likely and complicates maintenance.
- Introducing `pyrate-limiter` as the single source of rate control consolidates behavior under one transport, ensures cache hits are free, and unlocks multi-window quotas plus backend portability called for in the pyrate-limiter design notes.

## What Changes
- Create a centralized rate-limiting subsystem in `DocsToKG.ContentDownload.networking` (or an extracted `ratelimit` helper) that builds `pyrate_limiter.Rate`, `Limiter`, and bucket instances per `(host, role)` tuple with canonical roles `metadata`, `landing`, and `artifact`.
- Wrap the existing HTTPX/Hishel stack in a new `RateLimitedTransport` that acquires limiter tokens only on cache misses, differentiates HEAD vs GET weighting, and enforces per-role `max_delay` behavior before delegating to the wire transport.
- Extend CLI/env configuration (augment `DocsToKG.ContentDownload.args.build_parser` plus downstream config wiring) to declare host policies (`--rate host=limit/interval,…`), backend selection (`--rate-backend {memory,multiprocess,sqlite,redis,postgres}`), limiter mode (`--rate-mode host=wait:ms|raise`), and per-run overrides while emitting a startup policy table.
- Reconcile existing throttling flags (`--sleep`, `--domain-token-bucket`, `--max-concurrent-per-host`, semaphores) so networking is authoritative: deprecated flags map into RatePolicy defaults, and duplicate throttling paths are removed.
- Replace pipeline-level `TokenBucket` and host semaphore usage with role-aware limiter acquisition: update `DocsToKG.ContentDownload.pipeline` helpers to set `request.extensions["role"]` before delegating to `networking.request_with_retries` and delete the custom token-bucket implementation.
- Extend `DocsToKG.ContentDownload.errors.RateLimitError` to carry host, role, waited_ms, and next_allowed metadata so manifests and telemetry have the necessary context.
- Add structured telemetry counters, delay histograms, and limiter exception handling so Tenacity does not retry internal BucketFull conditions yet continues to obey upstream 429/Retry-After semantics; surface metrics through `telemetry`, `statistics`, and `summary` modules.
- Provide regression and unit coverage that cached responses do not consume tokens, multi-window limits behave, and limiter errors surface as `RateLimitError` with host/role metadata.

## Impact
- Affected specs: docstokg-content-download (ContentDownload politeness and networking contracts).
- Affected code: `src/DocsToKG/ContentDownload/networking.py`, `httpx_transport.py`, `pipeline.py`, resolver modules, CLI plumbing (`args.py`, `cli.py`, `runner.py`), telemetry/statistics/summary modules, and tests under `tests/ContentDownload/`.
- Operational impact: runs gain configurable, observable rate governance; default policies reflect documented provider quotas and can be tuned without code changes.
