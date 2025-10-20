Awesome — here’s an **AI-agent–ready, production-grade implementation plan** for **PR-7: Configuration & Profiles (Pydantic Settings; CLI/ENV layering)**. It’s narrative-only (no code), but precise enough to land as a sequence of small commits. I’ve aligned it with your current tree so you can see exactly where to plug things in and what to retire, and I’ve called out patterns you already use elsewhere in the repo.

---

# Scope & intent

**What we’re doing**

* Introduce **typed, reproducible settings** for DocParsing using **Pydantic v2 `BaseSettings`** with `SettingsConfigDict(env_prefix="DOCSTOKG_", case_sensitive=False, extra="ignore")`.
* Provide **profiles** (`local`, `gpu`, `airgapped`, `dev`) loaded from `docstokg.toml`/`.yaml`.
* Enforce **one precedence** everywhere: **CLI > ENV > profile file > defaults**.
* Expose a **root Typer callback** that materializes an **AppContext** (effective config, telemetry, writer). Subcommands receive only their slice (doctags/chunk/embed + runner).

**What we’re not doing (this PR)**

* No orchestrator (Prefect) wiring.
* No behavior change in stage logic beyond using the unified config.
* No breaking removal of legacy flags; we’ll shim and deprecate for one minor.

---

# Current state to leverage

* You already have stage-scoped config surfaces for **chunking** (`ChunkerCfg`, env/args loaders) and **embedding** (`EmbedCfg`, env/args merge), but they are not unified into a single Pydantic Settings stack, and layering semantics vary.
* You have **config file** loading helpers (YAML/TOML) under DocParsing config loaders — reuse them for profile ingestion.
* In **OntologyDownload**, you already use `SettingsConfigDict(env_prefix="ONTOFETCH_")` with a robust ENV reader & graceful fallback when Pydantic isn’t available. Use that pattern here (minus the fallback; PR-1 made Pydantic mandatory in DocParsing).
* Telemetry and manifest helpers are centralized (you just moved to `telemetry_scope`), so the root context can own the sink and remove per-command duplication.

---

# Design blueprint

## 1) Settings models (single source of truth)

Create **five** Pydantic models under `src/DocsToKG/DocParsing/settings.py` (or `config/settings.py` if you prefer). Each must be *data only*, with field validators (no side effects):

* **AppCfg** (global):
  `data_root`, `manifests_root`, `models_root`, `log_level`, `log_format=json|console`, `metrics_enabled`, `metrics_port`, `tracing_enabled`, `profile`, `strict_config`, `random_seed`.
* **RunnerCfg** (shared execution):
  `policy=io|cpu|gpu`, `workers`, `schedule=fifo|sjf`, `retries`, `retry_backoff_s`, `per_item_timeout_s`, `error_budget`, `max_queue`, `adaptive={off|conservative|aggressive}`, `fingerprinting=bool`.
* **DocTagsCfg** (stage):
  inputs/outputs, pdf/html mode selection & sanitizers, served model hints, vLLM readiness timeout, resume/force/verify-hash flags (keep names matching your CLI). (Your doctags docs already describe these knobs.)
* **ChunkCfg** (stage):
  `min_tokens`, `max_tokens`, tokenizer id, serializer provider, outputs (default Parquet per PR-8), resume/force flags. You already expose `ChunkerCfg` — we’ll migrate it to `BaseSettings`.
* **EmbedCfg** (stage):
  family toggles (dense/sparse/lexical), vector format (default Parquet), **provider** subtree (from PR-4 spec) for dense/tei/qwen_vllm/sparse/lexical params. Your current embedding config surface exists; unify it as a `BaseSettings` with env prefix mapping.

**All** models:
`model_config = SettingsConfigDict(env_prefix="DOCSTOKG_", case_sensitive=False, extra="ignore")`.

**Validators** (deep, semantic):

* TEI URL required when `dense.backend=tei`.
* BM25 constraints `k1>0`, `0≤b≤1` (you already document BM25 shape in embedding runtime docs).
* Resume rules mutually exclusive with force (guard at CLI, but re-validate here).
* Paths normalized to absolute, created lazily by writers only.

## 2) Profile files (docstokg.toml / .yaml)

* Location: project root or `config/`. Naming: `docstokg.toml` by default; support `.yaml` via your existing loader helpers.
* Structure:

```toml
[profile.local.app]
data_root = "Data"
log_level = "INFO"

[profile.local.runner]
policy = "io"
workers = 8

[profile.gpu.app]
data_root = "/mnt/data"

[profile.gpu.runner]
policy = "gpu"
workers = 8
schedule = "sjf"

[profile.gpu.embed.dense]
backend = "qwen_vllm"
model_id = "Qwen2-7B-Embedding"
batch_size = 64
```

* Support `--profile <name>` and ENV `DOCSTOKG_PROFILE=<name>` (CLI takes precedence).

## 3) Precedence & layering (canonical algorithm)

At the **root Typer callback**, materialize the effective config with this algorithm:

1. **Defaults**: `AppCfg()`, `RunnerCfg()`, `DocTagsCfg()`, `ChunkCfg()`, `EmbedCfg()` (zero ENV influence here).
2. **Profile file**: if `--profile`/ENV set and file present, **deep-merge** the `profile.<name>` mapping onto defaults field-for-field.
3. **ENV**: build **BaseSettings** instances so `DOCSTOKG_*` values override merged values. (Because we already merged profile onto defaults, Pydantic ENV application is a natural “layer 3”.)
4. **CLI**: parse options → apply on top (final layer).
5. Compute **cfg_hash** per stage = stable hash of the *stage-relevant* fields (stringified & key-sorted). Persist in context; used by resume/fingerprints in PR-10.

**Hard rule:** **CLI > ENV > profile > defaults**. This becomes contractually true by the order above.

## 4) Root Typer context (`AppContext`)

* Create a tiny `AppContext`: `{ app: AppCfg, runner: RunnerCfg, doctags: DocTagsCfg, chunk: ChunkCfg, embed: EmbedCfg, writer, telemetry }`.
* The root callback:

  * Parses `--profile` and global flags (`--verbose`, `--log-format`, `--metrics`, `--tracing`).
  * Loads profile file & merges.
  * Constructs Pydantic Settings for each model (ENV applied).
  * Applies CLI overrides.
  * Configures logging once (structlog level + console/json), **installs telemetry scope** once, and returns the context via `ctx.obj`. (You already centralized `telemetry_scope`.)

Subcommands now **only** read `ctx.obj` and pass the right config slice to the **runner** and **stage worker**.

## 5) Back-compat layer (one minor release)

* Implement a **legacy mapping table**: map deprecated flags/env to new fields (e.g., `--bm25-k1` → `embed.lexical.local_bm25.k1`, `DOCSTOKG_QWEN_MODEL_DIR` → `embed.dense.qwen_vllm.download_dir`).
* If both a legacy source and the new key are set, **new wins**; emit a one-line **deprecation warning** tagged with the stage.
* Keep the existing stage loaders (`from_args`, `from_env`) as thin shims that call the new settings builder, so tests/docs referring to them keep passing during the transition.

## 6) Validation surface (shallow vs deep)

* **Shallow (Typer) callbacks**: fast shape checks (paths exist *if required*, integers ≥1, enums). Errors raise `BadParameter` mapped to your `CLIValidationError` types — you already have a CLI error taxonomy.
* **Deep (Pydantic)** validators: semantic checks (TEI URL required, BM25 ranges, dense/sparse provider coherence). These raise `ValidationError` → catch in the root callback and convert to a single, readable message for the CLI.

## 7) Manifest config provenance

* At stage start, write `doc_id="__config__"` with the **redacted** config snapshot (`cfg_hash`, `profile`, roots, format). Your current stages already emit config rows; attach `profile` and `cfg_hash` consistently here. (Embedding config rows are already present; extend similarly for doctags/chunk.)

---

# Step-by-step implementation

## Commit A — Settings scaffolding & profiles

* Add `DocParsing/settings.py` with `AppCfg`, `RunnerCfg`, `DocTagsCfg`, `ChunkCfg`, `EmbedCfg`, **all** with `SettingsConfigDict(env_prefix="DOCSTOKG_")`. (Mirror the successful pattern from OntologyDownload’s `EnvironmentOverrides` for ENV handling.)
* Add `DocParsing/profile_loader.py` that opens `docstokg.toml`/`.yaml` and returns a nested mapping `{app,runner,doctags,chunk,embed}` for a profile name. Reuse the existing YAML/TOML loaders.
* Add a small `config_hash.py` that deterministically hashes a model’s `model_dump(exclude_none=True, by_alias=True)`.

**Acceptance (A):** `from DocsToKG.DocParsing.settings import AppCfg` instantiates with ENV overrides; profile loader returns a mapping for `local` when file exists.

## Commit B — Root Typer callback & context

* In your Typer root (you’ve just moved to Typer; PR-3), add `@app.callback()` that:

  * Parses `--profile`, `--verbose` (counting), `--log-format`, and observability toggles.
  * Loads profile mapping; applies **BaseSettings** (ENV) for each model; applies CLI overrides.
  * Computes `cfg_hash` per stage; builds `AppContext`; configures logging/telemetry once using your `telemetry_scope`.
* Subcommands (`doctags`, `chunk`, `embed`, `all`) pull their configs from `ctx.obj`.

**Acceptance (B):** Running `docparse --profile local chunk --help` shows grouped options; running with only `--profile gpu` produces the same behavior as providing the equivalent flags/env manually.

## Commit C — Stage shims & deprecation

* In `DocParsing/chunking/config.py` and `DocParsing/embedding/config.py`, keep `from_env`/`from_args` but rewrite them to call the **new** settings builder and return a dataclass compatible with the old return type (or a thin wrapper). Mark as **deprecated** in docstrings and logs.
* Implement the **legacy mapping** table; log a one-line deprecation when a legacy knob is used.

**Acceptance (C):** Legacy code paths & tests importing stage config helpers continue to pass; CLI shows deprecation lines once per run if old flags/env are used.

## Commit D — Validation hardening

* Add Typer **option callbacks** for shallow checks (dirs, ints, enums).
* Add Pydantic **validators** for semantic checks (TEI URL, BM25 limits, provider coherence). Bubble Pydantic `ValidationError` to a single `CLIValidationError` so CLI output is clean. (You already have CLI error helpers.)

**Acceptance (D):** Bad TEI URL fails before any work runs; errors are crisp and point to the exact option.

## Commit E — Manifests & config rows

* Ensure the **config row** includes `profile`, `cfg_hash`, and `vector_format/chunks_format`, mirroring your existing embedding config row behavior.
* Keep per-file rows unchanged; we’re only annotating start rows and keeping provenance unified.

**Acceptance (E):** Manifest rows show `profile` and `cfg_hash` for doctags/chunk/embed `__config__` entries.

## Commit F — Docs & help panels

* Add **docs page** “How configuration is layered” with a precedence diagram and ENV prefix table (`DOCSTOKG_*`).
* Update Typer `--help` to show grouped options (rich help panels), and include `--profile`. (You already expose CLI docs per module; add this page to `docs/04-api` index.)

**Acceptance (F):** `docparse config show --profile gpu` prints merged effective config; docs render ENV prefix + precedence clearly.

---

# Acceptance criteria (“done”)

* Running with **only** `--profile gpu` reproduces baseline behavior; no additional flags required.
* **Any** CLI flag overrides ENV and profile **every time**.
* Legacy flags/env still work but emit deprecation lines; new keys win.
* Config snapshot in manifests (`__config__`) includes `profile`, `cfg_hash`, and format.
* No behavior regression in doctags/chunk/embed with default settings.

---

# Tests to add

## 1) Precedence matrix

* Matrix of 8 scenarios (defaults, profile only, env only, CLI only, profile+env, profile+CLI, env+CLI, all 3). Assert the effective config matches the expected winner (CLI > ENV > profile).

## 2) Validation clarity

* TEI backend without `url` → deep (Pydantic) error text includes key name and remediation.
* BM25 `b=-0.1` → fails with range message; `k1=0` → fails with “> 0” message.

## 3) Snapshot help

* Capture `docparse --help` and each subcommand’s `--help` with Typer/Click’s `CliRunner` snapshot test so UX changes are deliberate.

## 4) Manifest config row

* Verify `__config__` entries have `profile`, `cfg_hash`, and (for embeddings) `vector_format`. (Embedding already wrote config rows; doctags/chunk must match.)

---

# Risks & rollback

* **Risk:** A downstream script depended on stage `from_args` semantics.
  **Mitigation:** Keep shims for one minor; new settings under the hood.
* **Risk:** Operators mixing old env names with new keys get confused.
  **Mitigation:** One-line deprecation **per key**, pointing to the new key; add a doc table “Legacy → New”.
* **Temporary relax:** If friction appears, allow `DOCSTOKG_STRICT_CONFIG=false` to treat unknown keys as warnings (default strict=true in CI).

---

# Documentation updates (ship in this PR)

* “**Configuration Layering & Profiles**” (precedence, example `docstokg.toml`, ENV table).
* “**Profile cookbook**” (examples for `local`, `gpu`, `airgapped`, `dev`).
* “**Legacy knobs**” (mapping table and removal schedule).

---

# Why this fits your codebase today

* Stage configs and loaders already exist; we’re **standardizing** them under Pydantic Settings and a root **Typer context** while preserving legacy surfaces.
* Profile file loading reuses your **DocParsing config loaders**; no new dependency needed.
* The **OntologyDownload** package is a working example of Pydantic Settings + ENV handling that we can mirror (minus the “no Pydantic” fallback you used there).

---

# Work breakdown (PR checklist)

1. **Settings + Profile loader** (models, loader, cfg_hash) — small, isolated commit.
2. **Root callback + context** (logging/telemetry once) — subcommands untouched.
3. **Stage shims** (deprecated `from_env/from_args`; legacy mapping).
4. **Validation** (Typer shallow + Pydantic deep).
5. **Manifest config rows** (add `profile`, `cfg_hash`).
6. **Docs + help panels** (precedence page, ENV table).
7. **Tests** (matrix, validation, snapshots, manifests).

---

If you want, I can follow this with (a) a **legacy-to-new mapping table** enumerating every current DocParsing flag/env and its new `Settings` path, and (b) a **“config diff”** Typer subcommand spec (`docparse config show/diff`) so operators can print or compare the effective config live.
