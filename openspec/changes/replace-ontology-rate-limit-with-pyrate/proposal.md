## Why
The OntologyDownload package still depends on bespoke token-bucket classes that persist JSON files and hand-roll retry alignment. That code is brittle, diverges from the documented pyrate-limiter transition plan, and makes it hard for junior engineers to reason about shared quotas. Adopting pyrate-limiter gives us a tested limiter implementation, first-class SQLite persistence, and clearer hooks for Retry-After handling while keeping the existing CLI, planner, and resolver behaviour intact.

## What Changes
- Replace `DocsToKG.OntologyDownload.io.rate_limit` internals with a pyrate-limiter powered manager that caches `Limiter` instances by `(service|_, host|default)` and exposes the same `get_bucket(...).consume()` contract for networking, planning, resolver, and checksum call sites.
- Introduce a `DownloadConfiguration.rate_limiter` toggle (default `"pyrate"`) plus doctor/telemetry output so operators can fall back to the legacy bucket during rollout and confirm which backend is active at runtime.
- Honour HTTP `Retry-After` by returning parsed delays from `apply_retry_after` and letting the retry loops in `io.network`, `planning`, `checksums`, and `resolvers` sleep before the next `consume()` instead of mutating limiter state directly.
- Persist shared quotas with `pyrate_limiter.SQLiteBucket` when `shared_rate_limit_dir` is configured; otherwise use `InMemoryBucket`, and log the backend selection the first time each limiter is created.
- Update testing utilities, fixtures, and documentation (README + pyrate-limiter reference) to explain the new manager, illustrate how string limits map onto `Rate/Duration`, and remove references to the old `TokenBucket`/`SharedTokenBucket` classes.

## Impact
- Affected specs: ontology-download-rate-limiting
- Affected code: `src/DocsToKG/OntologyDownload/io/rate_limit.py`, `src/DocsToKG/OntologyDownload/io/network.py`, `src/DocsToKG/OntologyDownload/resolvers.py`, `src/DocsToKG/OntologyDownload/planning.py`, `src/DocsToKG/OntologyDownload/checksums.py`, `src/DocsToKG/OntologyDownload/settings.py`, `src/DocsToKG/OntologyDownload/testing/__init__.py`, `src/DocsToKG/OntologyDownload/io/__init__.py`, dependency manifests, and related documentation.
