Absolutely — here are two “pop”-level architectural overviews you can drop into your docs. Each one shows how parts fit together, what talks to what, and the exact artifacts (hashes, schemas, events) that make the system auditable and self-documenting.

---

# A) Configuration You Can Trust (and Audit)

## A1. Big-picture component map

```
┌──────────────────────────────┐
│            CLI               │  e.g. `ontofetch extract ...`
│  (Typer command entrypoint)  │
└──────────────┬───────────────┘
               │ builds overlay (--config, flags)
               ▼
┌─────────────────────────────────────────────────────────────┐
│           Traced Settings Loader (Pydantic v2)              │
│  precedence & source tracing:                               │
│  CLI overlay ▶ CONFIG file ▶ .env.ontofetch ▶ .env ▶ ENV ▶ defaults
│                                                             │
│  • Validates & normalizes into submodels                    │
│  • Records source_fingerprint[field] = {cli|config|env|...} │
│  • Emits JSON Schema (for agents/CI)                        │
└───────┬───────────────┬───────────────┬───────────────┬─────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌────────────┐   ┌──────────────┐  ┌─────────────┐  ┌───────────┐
│HttpSettings│   │SecurityPolicy│  │Extraction   │  │RateLimit  │
│timeouts/UA │   │allowlist/DNS │  │limits/globs │  │per service│
└────────────┘   └──────────────┘  └─────────────┘  └───────────┘
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                        │ normalized, immutable config
                        ▼
                 ┌──────────────┐
                 │ Config Hash  │  SHA-256 of normalized, non-secret fields
                 │  (provable)  │  (sorted JSON, enum→str, sets→sorted lists)
                 └──────┬───────┘
                        │ used everywhere for provenance
                        ▼
┌─────────────────────────────────────┐
│  Observability Emitters (events)    │  net.request / extract.* / cli.*
│  + Audit JSON + DuckDB events       │  include {run_id, config_hash, ...}
└─────────────────────────────────────┘
```

## A2. Runtime sequence (every command)

```
User → CLI parse → Build CLI overlay
              ↓
       Load+Trace settings
              ↓  (validate/normalize; record sources)
         Compute config_hash
              ↓
     Build context (settings, run_id)
              ↓
  Command logic (planner/downloader/…)
              ↓
 Emit events (include config_hash) + write audit JSON
              ↓
             Exit (status codes)
```

## A3. Contracts that make it auditable

**Traced sources contract**

* `source_fingerprint: Map[str field → str source]`
  Sources: `cli`, `config:/abs/path.toml`, `.env.ontofetch`, `.env`, `env`, `default`.

**Normalization contract (examples)**

* `SecurityPolicy.normalized_allowed_hosts() -> (exacts, suffixes, per_host_ports, ip_literals)`
  – hosts punycoded + lowercased; per-host ports validated; global ports merged.
* `Extraction.include_filters() -> Callable[path->bool]`
  – NFC normalization then include/exclude glob match.

**Config hash contract**

* Input: full normalized dict (minus volatile/secrets).
* Output: 64-hex SHA-256.
* Must be stable for the same effective config irrespective of source order.

**JSON Schema artifacts**

* `docs/schemas/settings.schema.json` (top-level)
* `docs/schemas/settings.*.subschema.json` (HTTP, Security, Extraction…)
* CI check: regenerate & diff in PRs.

## A4. Error & telemetry shape

* Validation failures: always include **env key**, **expected format**, and **schema path**.
* Event envelope (all events):
  `{ ts, type, level, run_id, config_hash, context{app_version, os, py, libarchive}, payload{…} }`

## A5. Testing lenses

* **Determinism**: same inputs across sources → same `config_hash`.
* **Precedence**: CLI overrides config file > env; `.env.ontofetch` before `.env`.
* **Alias safety**: legacy env names populate new fields; “deprecation used” note appears in `settings show`.

---

# B) CLI as a First-Class Product (fast, safe, self-documenting)

## B1. Command system layout (as components)

```
┌──────────────────────────────────────────┐
│         Typer App (main.py)              │
│  Global opts: --config -v/-vv --format   │
│  --dry-run --version                     │
└───────────────┬───────────────┬──────────┘
                │               │
                │               └─────────────► Autocompletions
                │                                 (DuckDB readers)
                ▼
         Context Factory
   (Settings+Logger+Printer+RunId)
                │
                ├──────────► Settings Cmds: show | schema | validate
                │
┌───────────┬───────────┬────────────┬───────────┬───────────┐
│  plan     │  pull     │  extract   │ validate  │   delta   │  … subcommands
└───────────┴───────────┴────────────┴───────────┴───────────┘
      │            │         │             │            │
      │            │         │             │            │
      ▼            ▼         ▼             ▼            ▼
  Network   Downloader  Extractor     Validators     DuckDB macros
 (HTTPX)   (rate-limit) (policy)        (SHACL…)        (A↔B)
```

## B2. Lifecycle of any command (state machine)

```
START
  │
  ├─ Parse args (Typer; typed & bounded)
  │
  ├─ Build CLI overlay & load settings (see A2)
  │     └─ print --version (eager) → EXIT 0
  │
  ├─ Preflight:
  │     • build context (logger/printer/run_id)
  │     • install completions (opt)
  │     • confirm destructive ops (or require --yes)
  │
  ├─ Execute subcommand
  │     • honors --dry-run (no writes, show plan)
  │     • emits cli.command.start/done/error
  │
  ├─ Render output via Printer
  │     • table (Rich) | json | yaml
  │
  └─ Map errors → exit codes (2:args, 3:policy, 4:IO, 5:net, 6:db, 7:unknown)
END
```

## B3. Option & I/O contracts (make it safe & predictable)

**Global options**

* `--config PATH` (envvar `ONTOFETCH_CONFIG`) → feeds traced loader.
* `-v/-vv` → sets log level INFO/DEBUG across app.
* `--format [table|json|yaml]` → one rendering pipeline.
* `--dry-run` → *no writes*; show planned actions & targets.
* `--version` (eager) → prints `{app_version, git_sha, build_date, python, os, libarchive_version}`.

**Destructive commands**

* `prune --apply` and `latest set` require:
  `--yes` **or** interactive confirmation (`type PRUNE to continue`).
  On TTY absence, require `--yes`.

**Autocompletions**

* `--version-id` completes from `SELECT version_id FROM versions ORDER BY created_at DESC LIMIT 50`.
* `--service` completes from distinct `artifacts.service`.
* `--format` completes from distinct `extracted_files.format`.

**Exit-code mapping (uniform)**

* 0 OK • 2 Usage/validation • 3 Policy gate reject • 4 Storage/FS
* 5 Network/TLS • 6 DB/Migration • 7 Unknown

## B4. Evented CLI (answers, not just logs)

Every command emits:

* `cli.command.start` `{cmd, args_redacted, run_id, config_hash}`
* `cli.command.done`  `{cmd, elapsed_ms, rows|bytes|files metrics}`
* `cli.command.error` `{cmd, error_code, message}`

…and all downstream events (net.request, extract.*, db.*) include the same `run_id` for correlation.

## B5. Docs without drift

* `typer` introspection generates a Markdown page per command: synopsis, options (with envvar bridges), examples, exit codes.
* CI regenerates docs & diffs. Build fails if docs are stale.

## B6. Testing grid (CliRunner)

* **Golden help** snapshots per command (detects accidental interface drift).
* **Happy path**: table and JSON outputs agree in row counts/keys.
* **Error path**: missing args → exit 2; policy hit → exit 3 with `E_*` code surfaced; destructive without `--yes` → prompt (or exit non-0 in CI).
* **E2E smoke**: `settings schema`, `settings validate`, `db versions`, `delta summary A B` against a tiny fixture DB.

---

## C) How these two “click” together (integration view)

```
[Typer CLI] --(global opts/env)--> [Traced Settings Loader]
        │                               │
        │                     source_fingerprint + normalized submodels
        │                               │
        ├────────── emits cli.command.start/run_id/config_hash ───────► [Events]
        │                               │
        ▼                               ▼
   [Command Logic]  ───────────→ uses →  Http/Security/Extraction/RateLimit submodels
        │                               │
        └─────── structured results ────┘
                    render (table/json/yaml)
        │
        └─ emits cli.command.done + metrics → [Events + DuckDB]
```

**Key invariants**

* The **exact** effective configuration that produced any output is provable (`config_hash`), reproducible (JSON Schema), and attributable (source_fingerprint).
* The CLI never bypasses the loader: *every* command consumes the same normalized, immutable submodels.
* Help/docs/completions are generated from **live types**, so UX and code never drift.

---

If you’d like, I can now produce similar “at-a-glance” diagrams for the next pairs (HTTPX & Rate-limit; DuckDB & Polars; Observability & Safety) so the whole system reads like a cohesive platform.
