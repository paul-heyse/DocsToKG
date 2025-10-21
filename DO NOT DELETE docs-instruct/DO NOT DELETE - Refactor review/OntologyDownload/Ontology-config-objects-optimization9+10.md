Absolutely — here are **two deep, code-free, agent-ready implementation plans** for the next pair of optimizations:

* **(9) Deterministic & fast test matrix**
* **(10) Performance playbook (budgets, benches, profiling)**

They plug cleanly into the architecture you already have (settings hash, HTTPX + Hishel, pyrate-limiter, libarchive extraction, DuckDB, Polars, events, and policy gates).

---

# 9) Deterministic & Fast Test Matrix

## 9.1 Objectives (what “done” means)

* Tests are **hermetic, reproducible, and fast**, with **clear strata** (unit → component → e2e).
* **No real network**; HTTP is mocked or in-process only.
* **Cross-platform** safety (Linux/macOS/Windows) and **seeded randomness**.
* A **single test vocabulary** (markers, fixtures, corpora, golden files) used across commands and subsystems.
* CI shows **stable runtimes**; flakes are treated as bugs, not “reruns”.

## 9.2 Files & layout

```
tests/
  conftest.py                      # global fixtures, seeds, policy, environment freezing
  fixtures/
    http.py                        # HTTPX MockTransport + in-process ASGI app
    fs.py                          # tmp root, encapsulation helpers, dirfd/openat probes
    duckdb.py                      # temp catalog, migrations, writer lock harness
    rate.py                        # limiter registry reset, sqlite bucket tempdir
    settings.py                    # traced settings loader with test overlays
    telemetry.py                   # event sink capture (list[dict])
  data/
    corpus/                        # adversarial archives & tiny fixtures (zip/tar.*)
      traversal.zip, link.tar, bomb.zip, longpaths.zip, unicode_nfd.zip, ...
    cli/                           # golden outputs (help pages), tiny audit JSONs
  unit/                            # pure units w/o I/O
  component/                       # HTTP client, extractor, catalog, ratelimiter
  e2e/                             # plan→pull→extract→validate→latest (tiny)
  property/                        # Hypothesis suites (paths, URLs, ratios)
```

## 9.3 Global determinism controls

* **Seeds**: set `PYTHONHASHSEED`, `random.seed`, `numpy` seed (if used), **Hypothesis profile** (deadline off for I/O, deterministic shrink).
* **Time & TZ**: `TZ=UTC`; use `freezegun` (or equivalent) in tests that depend on time.
* **Locale & FS**: fix locale to `C.UTF-8`; set process `umask` to 022 in tests; normalize paths to **POSIX** in golden outputs (Windows handled via helpers).
* **Env hygiene**: clear `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`; ensure `HOME` points to a temp dir; disable any telemetry “phone home”.
* **Network**: never open sockets in unit/component tests. Use **HTTPX MockTransport** or an **in-process ASGI/WSGI** app for streaming/chunked/redirect scenarios.

## 9.4 Markers & strata

* `@pytest.mark.unit` — no I/O, no DuckDB, no libarchive; < 50 ms each.
* `@pytest.mark.component` — touches one subsystem (HTTP, extraction, catalog); < 500 ms.
* `@pytest.mark.e2e` — small end-to-end (tiny archives); < 5 s.
* `@pytest.mark.property` — Hypothesis; cap examples/time; deterministic seed.
* `@pytest.mark.slow` — opt-in for heavy tests; not required in PR CI.
* `@pytest.mark.windows_only`, `@pytest.mark.posix_only` — platform splits.

**CI execution sets**

* PR: unit + component + e2e (no slow) on Linux; smoke on macOS/Windows.
* Nightly: full matrix with property+slow and macOS/Windows runners.

## 9.5 Core fixtures (deterministic)

* **settings_fixture**: traced settings with fixed overlay (cache dir, DB path, limits), returns settings + `config_hash`.
* **tmp_root**: per-test directory for storage root + encapsulation + dirfd helpers.
* **http_mock**: HTTPX MockTransport that serves fixtures: 200, 302 chain, 429 `Retry-After`, 5xx, content-type variants, streaming byte ranges.
* **limiter_reg**: ratelimiter registry reset; temp **SQLiteBucket** when multi-proc tests request it.
* **duckdb_catalog**: ephemeral DB + migration runner + writer lock; returns readers/writer connectors.
* **event_sink**: captures structured events; assertion helpers check presence/shape of `net.request`, `extract.*`, `policy.gate`, etc.

## 9.6 Property-based suites (Hypothesis)

* **URL gate**: generate host patterns (ASCII/IDN), ports, schemes; assert allow/deny invariants and idempotent normalization.
* **Path gate**: generate Unicode component trees (NFC/NFD, bidi, zero-width), random depths/lengths; assert no traversal, no escapes, collision detection works.
* **Extraction ratios**: generate per-entry sizes and compressed hints; assert global/per-entry ratio guards behave.

*Profiles:* fixed seeds, limited examples, deadline off (I/O mocked anyway).

## 9.7 Golden & snapshot testing

* **CLI help**: snapshot `--help` for each command (detects drift).
* **Delta outputs**: small fixed DB fixture; `delta summary A B` JSON golden (stable keys, order).
* **Audit JSON**: canonicalize and compare (sorted keys, path order per configured mode).

## 9.8 Coverage & flake policy

* **Coverage**: branch coverage thresholds per layer (e.g., 95% unit, 85% component, 70% e2e); exclude compat shims; include error paths.
* **Flakes**: no auto rerun; failing test quarantined with CI annotation and a **bug** is filed—goal is **0** flaky tests.

## 9.9 Cross-platform specifics

* **Windows**: reserved names and trailing dot/space tests; long path support under `\\?\` opt-in toggle; path separator normalization in outputs.
* **macOS**: NFD/NFC normalization checks; ensure collision detection catches NFD/NFC dupes.

## 9.10 Acceptance checklist

* [ ] No real network used; all HTTP tests use MockTransport/ASGI.
* [ ] Seeds/TZ/locale/umask fixed; help/delta/audit golden files stable.
* [ ] Property tests cover URL + path + ratio gates with deterministic seeds.
* [ ] Cross-platform suites green (Linux/macOS/Windows smoke in PR; full nightly).
* [ ] Event sink assertions exist in key tests (at least one per subsystem).
* [ ] CI wall-time for PR lane under target (e.g., < 8 min for Linux matrix).

---

# 10) Performance Playbook (budgets, benches, profiling)

## 10.1 Objectives

* **Budgets** for hot paths with CI regression detection (relative to a baseline).
* **Micro-benches** (units of work) + **macro e2e** (observed wall time/throughput).
* **Profiling hooks** (CPU/time/memory + plan explainers) that developers can toggle locally.
* **Repeatable, network-free** conditions.

## 10.2 Tooling & artifacts

* `pytest-benchmark` for microbenches (calibrated rounds, percentiles).
* `psutil` for memory/file-descriptor high-water marks; resource leak checks.
* `pyinstrument` (sampling) or `py-spy` (optional) for CPU flame graphs.
* DuckDB `EXPLAIN ANALYZE` and Polars `.explain()` plans under `--profile`.
* Baselines stored in repo under `.ci/perf/baselines/<runner-class>.json` (per CI machine class).

## 10.3 Budgets (examples; tune to your CI class)

* **HTTPX policy path** (mock):

  * p95 `net.request.elapsed_ms` (200 JSON, 128 KiB body) < **5 ms**
  * Redirect audit (1 hop) < **8 ms**
* **Ratelimiter acquire**:

  * fail-fast path < **0.1 ms**; block path accuracy ± **5%** of expected sleep
* **Extraction**:

  * Pre-scan 10k headers (no writes) < **250 ms**
  * Streaming write throughput (1 GiB highly compressible zip) ≥ **150 MiB/s** on CI class
  * Hashing overhead < **10%** of streaming time
* **DuckDB**:

  * Bulk insert 50k `extracted_files` rows via Arrow appender < **1.5 s**
  * `v_version_stats` on 200k rows < **200 ms**
* **Delta macros**:

  * `version_delta_summary(A,B)` on 200k rows < **250 ms**
* **Polars pipelines** (scan_ndjson of audit with 1M entries):

  * Lazy + streaming `collect` (latest summary) < **2.0 s**

*(If your CI runners differ, calibrate once and commit baseline per class.)*

## 10.4 Micro-bench suites (pytest-benchmark)

Create `benchmarks/`:

* **bench_httpx.py**

  * GET 200 (128 KiB body) → p95 elapsed, CPU time.
  * 302→200 redirect audit (1 hop).
  * 429 + Retry-After (2 s) → ensure backoff doesn’t busy-spin; measure overhead (excluding sleep).

* **bench_ratelimit.py**

  * `acquire(mode="fail")` under N/sec window, repeated.
  * `acquire(mode="block")` expected wait; accuracy and overhead.

* **bench_extract.py**

  * Pre-scan 10k entries (synthetic table-of-contents).
  * Stream extraction of a 500 MiB archive (generated locally with compressible content) to tmpfs; measure throughput and CPU profile (optional).
  * Per-file hashing inline vs parallel (if you add it later): compare overhead.

* **bench_duckdb.py**

  * Arrow appender insert 50k rows; DB size growth, write time.
  * `EXPLAIN ANALYZE` of `v_version_stats` and delta macros (capture total runtime).

* **bench_polars.py**

  * `scan_ndjson` (1M entries) → latest summary with groupby; `collect(streaming=True)`.

Each bench records:

* time percentiles, ops/sec,
* RSS delta (memory), fd counts (before/after),
* environment metadata (CPU model, cores, Python, DuckDB version).

## 10.5 Macro e2e perf (smoke + nightly)

* **smoke_perf** (PR lane, fast):

  * dataset: 20 small archives (~50 MiB total).
  * run: plan → pull → extract → validate → set latest.
  * budget: **< 25 s** wall time; emit event counts (extract.done bytes, entries).
* **nightly_perf** (full):

  * dataset: ~1–2 GiB compressible archives; validations on subset.
  * record: wall time, throughput (MiB/s), DuckDB growth, #events; compare to baseline JSON; **fail** if regression > threshold (e.g., +20%).

## 10.6 Regression detection (CI)

* On PR: run **micro-benches** with `--benchmark-autosave`; compare to baseline for the runner class; warn (not fail) if delta > 15%.
* On nightly: **fail** if > 20% regression for any critical benchmark; artifacts include flame graphs (if enabled) and `EXPLAIN` plans for slow queries.

**Baseline discipline**

* When intentional changes improve/worsen perf, update the baseline with a documented reason in the PR description.

## 10.7 Profiling toggles

* CLI `--profile` flag:

  * For DB queries: prefix with `EXPLAIN ANALYZE`.
  * For Polars: print pipeline `.explain()`; enable `POLARS_VERBOSE=1`.
  * For Python CPU: wrap command with `pyinstrument` if `PYINSTRUMENT=1`.
  * Emit a `perf.profile.emitted` event that points to saved artifacts.

## 10.8 Resource & leak checks

* Wrap benches and long-running tests with a fixture that records:

  * RSS (psutil), fd count (Linux `/proc/self/fd`), thread count.
  * Assert **no growth** beyond threshold across repeated calls (e.g., 5 runs).
* HTTP client close checks: ensure no open sockets or leaked transports after tests.

## 10.9 Data generation (repeatable, local)

* Script `tools/make_perf_archives.py`: generate deterministic tar/zip datasets:

  * compressible (repeated patterns), semi-random, Unicode-rich paths, deeply nested.
  * write seed into archive comment/metadata; save manifest (file list with sha256, sizes).

## 10.10 Reporting & docs

* `ontofetch perf report` CLI to read latest benchmark JSON(s) and render:

  * top regressions, top improvements, environment summary.
* Doc page: budgets per layer + “How to profile locally” (pyinstrument/duckdb/polars).

## 10.11 Acceptance checklist

* [ ] Bench harnesses exist for HTTPX, ratelimiter, extraction, DuckDB, Polars; **no network**.
* [ ] Baselines per runner are committed; PR lane warns on >15% deltas; nightly fails on >20%.
* [ ] Macro smoke & nightly perf suites produce wall-time, throughput, and resource metrics.
* [ ] `--profile` produces actionable plans/graphs; artifacts attached to CI runs.
* [ ] No resource leaks across benches; fd/thread counts steady.

---

## Suggested PR sequence (low risk)

**PR-T1 — Test substrate**
Global fixtures (seeds/TZ/locale/env), corpus, HTTPX MockTransport, event sink, markers, coverage config.

**PR-T2 — Unit/Component suites**
Gates, HTTP client, ratelimiter, extractor pre-scan, catalog migrations.

**PR-T3 — Property & Cross-platform**
Hypothesis suites; Windows/macOS runners; Unicode & long path tests.

**PR-P1 — Bench harness + baselines**
Micro-benches, smoke perf, baseline JSON, CI wiring for warnings.

**PR-P2 — Nightly perf & profiling**
Nightly pipeline with larger dataset; fail on >20% regressions; profiling toggles & artifact upload.

---

**What you get:** a **repeatable**, **auditable** test and performance framework. Failures are **diagnosable in minutes**, perf regressions are **caught early**, and every optimization you make shows up as a clear, measured win.
