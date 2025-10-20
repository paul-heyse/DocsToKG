Absolutely—here’s a **deep, code-free, agent-ready plan** to implement the two novel pillars:

---

# 1) Configuration you can trust (and audit)

## 1.1 Goals (non-negotiables)

* **Strict, typed, validated** settings for every knob that affects security, safety, or I/O.
* **Deterministic “effective config”** with a **provenance trail** (where each field came from).
* A **stable, canonical JSON Schema** (and sub-schemas) so agents and CI can **validate before running**.
* A **config hash** embedded in every run/audit/event for **reproducibility**.
* **Aliasing without drift** so you can refactor internals without breaking existing env files and ops scripts.

## 1.2 Files & boundaries

* `src/DocsToKG/OntologyDownload/settings.py` — Pydantic v2 `BaseSettings` + sub-models.
* `src/DocsToKG/OntologyDownload/settings_sources.py` — custom **traced sources** + precedence.
* `src/DocsToKG/OntologyDownload/settings_schema.py` — schema generation utilities.
* `src/DocsToKG/OntologyDownload/cli/settings_cmd.py` — `settings show|schema|validate`.
* `docs/schemas/settings.schema.json` (and per-submodel schemas).
* Pre-commit hook / CI job to verify schemas & hashes haven’t drifted.

## 1.3 Source precedence (and how we’ll trace it)

**Order (highest → lowest)**

1. **CLI overlay** (a dict your Typer layer passes in; e.g., `--http-timeout-read 15`)
2. **Config file** (if `ONTOFETCH_CONFIG` or `--config` points to TOML/YAML/JSON)
3. **`.env.ontofetch`** (if present)
4. **`.env`** (if present)
5. **Environment** (`ONTOFETCH_*`)
6. **Defaults** baked into models

**Tracing (“who set this?”):**
Wrap each Pydantic `SettingsSource` in a **`TracingSettingsSource`** that:

* On `get_field_value(field_name)`, if it returns a value, appends `(field_name, source_name)` into a mutable **`source_map`** held on the loader context.
* After construction, the `Settings` instance exposes `settings.source_fingerprint: dict[field, source_name]`.
  We **never** log values—only the **source name** (`cli`, `config:<path>`, `.env.ontofetch`, `.env`, `env`, `default`).

## 1.4 Strict typing & validation rules (selected examples)

Use **Pydantic v2** constrained types / annotated validators:

* **Hosts/allowlist**: a `HostPattern` union type (exact domain, `*.suffix`, IP literal, CIDR).
* **Ports**: `conint(ge=1, le=65535)`; for lists, `set[int]` validated + normalized.
* **Paths**: resolve to **absolute POSIX**; forbid traversal (`..`), Windows drive letters, and NUL; assert parent exists (or create it for cache/db).
* **Rates**: `RateSpec` type (“`N/second|minute|hour`”) parsed to a tuple `(limit:int, unit:Enum, rps:float)`.
* **Ratios**: `confloat(ge=1.0)` for total/entry compression ratios.
* **Timeouts/buffers/wall-time**: positive numeric bounds via `confloat(gt=0)` / `conint(ge=0)`.

**Normalization contracts** (must be true post-init):

* `security.normalized_allowed_hosts()` returns **(exacts, suffixes, per_host_ports, ip_literals)** with punycoded ASCII labels and **lowercased** host parts.
* `security.allowed_port_set()` merges global ports with per-host overrides; includes 80/443 if none given.
* `extraction.include_filters()` returns a compiled predicate that applies normalization (NFC) before matching globs.
* `ratelimit.parse_service_rate_limit(service)` returns a **canonical** `RateSpec` with **rps** derived.

## 1.5 Aliasing without drift (compatibility + stable docs)

* For any renamed field, add:

  * `validation_alias="OLD_ENV_NAME"` so **legacy envs** still populate the new field.
  * `serialization_alias="NEW_PUBLIC_NAME"` so generated schema/docs **stay stable** even if internal names evolve.
* Maintain a **deprecation map** so `settings show` can warn when a legacy alias was used.
* For compact maps (e.g., `"ols:4/second;bioportal:2/second"`), define a `TypeAdapter[dict[str, RateSpec]]` that accepts:

  * **String** shorthand (semicolon-separated pairs)
  * **JSON** map
  * **TOML/YAML** native maps
    and always outputs a canonical dict for downstream use.

## 1.6 Config hash (deterministic & redaction-safe)

* Build a **canonical dict** of **normalized** settings:

  * Convert Enums to their string values.
  * Turn sets into **sorted lists**.
  * Paths to **absolute POSIX strings**.
  * Exclude volatile fields (e.g., run_id) and secrets (none in this scope; if added later, use a redaction set).
* Serialize with a **canonical JSON dumper** (sorted keys, no whitespace differences).
* Hash with SHA-256 → `config_hash`.
  Emit this:

  * in `extract.start` / `pre_scan.done` / `extract.done` events,
  * into the **audit JSON**,
  * into DuckDB `events` (if enabled).

## 1.7 JSON Schema (for agents & CI)

* Generate **one top-level schema** for `Settings` via `model_json_schema()`; also generate **sub-schemas** for each domain (HTTP, Security, Extraction, …).
* Post-process:

  * Add `"examples"` and `"description"` from model field metadata.
  * Ensure enums list all canonical string values.
* Write to `docs/schemas/settings.schema.json` (and `*.subschema.json`).
* CLI: `ontofetch settings schema --out docs/schemas/`
  CI: compare generated schema to repo copy; **fail** on drift unless deliberately updated.

## 1.8 “Source fingerprinting” in practice

* `settings show` prints a **Rich table** with columns: `field`, `value(redacted if sensitive)`, `source`, `notes(deprecated alias used?)`.
* `settings audit` attaches `source_fingerprint` (field→source) to the **events payload** (not persisted to the main audit JSON unless requested), so postmortems show why a bad value happened (e.g., `.env` overrode config file).

## 1.9 Error messages (human-oriented)

Every validation error must:

* include the **env var name** that failed (not just field path),
* print the **expected format** (e.g., “use `N/second|minute|hour`”),
* suggest the **nearest valid enum** on typos (case-insensitive matches),
* point to the **schema** path in `docs/schemas/…`.

## 1.10 Tests (unit + property + integration)

* **Unit**: per-field parsing (good and bad), alias resolution, source precedence, normalized outputs, config hash determinism (same input order different? → same hash).
* **Property**: randomized host patterns & ports; guarantee `normalized_allowed_hosts()` round-trips and rejects invalid combos.
* **Integration**:

  * `.env` + config + CLI overlay precedence,
  * `settings schema` regenerates identical schema (unless intentionally changed),
  * events contain `config_hash`, and `settings show` redacts sensitive fields correctly (if added later).

## 1.11 Deliverables (acceptance)

* `Settings` + sub-models implemented with strict types & validators.
* Custom **traced sources** produce a `source_fingerprint`.
* CLI: `settings show|schema|validate` present and tested.
* Stable **JSON Schema** files under `docs/schemas/`.
* `config_hash` embedded in events and audit JSON.
* Migration doc: legacy aliases recognized; deprecation warnings printed.

---

# 2) CLI as a first-class product (fast, safe, self-documenting)

## 2.1 Goals

* Zero-surprise UX: **typed options**, safe defaults, consistent output, **–format** switch (json|yaml|table).
* **Discoverable**: rich help, examples, autocompletions, `—version` eager.
* **Safe**: confirmation for destructive ops; `--dry-run`; clear exit codes.
* **Tested**: `CliRunner` coverage for every command and major error path.

## 2.2 Files & structure

* `src/DocsToKG/OntologyDownload/cli/__init__.py` — Typer app factory.
* `src/DocsToKG/OntologyDownload/cli/main.py` — `app = Typer(...)` with common options and callbacks.
* Command modules:

  * `settings_cmd.py` (show|schema|validate),
  * `plan_cmd.py`, `pull_cmd.py`, `extract_cmd.py`, `validate_cmd.py`,
  * `latest_cmd.py` (get/set),
  * `prune_cmd.py` (dry-run|apply),
  * `db_cmd.py` (latest|versions|files|stats|doctor),
  * `delta_cmd.py` (use the macros: summary|files|renames|formats|validation).
* `src/DocsToKG/OntologyDownload/cli/_common.py` — shared types, formatting, confirmation, context.
* Tests: `tests/cli/test_*.py`.

## 2.3 Global behavior (Typer patterns)

* **Context object** with:

  * effective `Settings` (constructed once),
  * `logger` (structlog),
  * `printer` (table/json/yaml),
  * global `verbosity` flags (‐v/‐vv),
  * `dry_run` boolean.
* Global options (in `@app.callback()`):

  * `--config PATH` (envvar: `ONTOFETCH_CONFIG`) to point at a config file,
  * `-v / -vv` → map to `INFO`/`DEBUG`,
  * `--format [table|json|yaml]` → unify output,
  * `--dry-run / -n` → do not change state,
  * `--version` (eager): prints version + git sha + build date + Python + OS + **libarchive version**.
* **envvar bridges**: For frequently used knobs, set `envvar=...` on options so ops can choose flags or env.

## 2.4 Typed, bounded, and safe options

* Use Typer’s type system to enforce constraints:

  * `Path` (exists vs writeable), `Annotated[int, ValueRange(1, 65535)]` for ports, `Literal[...]` or `Enum` for mode switches, `confloat` wrappers for ratios, etc.
* **Autocompletions**:

  * `version_id` options complete from DuckDB (`SELECT version_id FROM versions ORDER BY created_at DESC LIMIT 50`).
  * `service` names complete from `artifacts.service`.
  * `format` completes from distinct `extracted_files.format`.
* **Confirmation prompts**:

  * Destructive commands (e.g., `prune --apply`, `latest set`) require `--yes` or interactive confirm; for prune, require typing the exact string `PRUNE`.
* **Exit codes**:

  * `0` success, `2` bad arguments/validation, `3` policy violation (e.g., traversal), `4` storage/IO, `5` network, `6` db/migration, `7` unknown.

## 2.5 Output & doc quality

* **Rich** tables with consistent columns; **json/yaml** for machine consumption (and round-trippable).
* **Examples**: each command includes **3 short examples** in the help epilog (Typer supports help text).
* **Auto-docs**: ship a task `scripts/gen_cli_docs.py` that introspects the Typer app and emits Markdown (name, synopsis, options, envvars, examples). Regenerate in CI to catch drift and publish to docs.

## 2.6 Commands (scope & expectations)

**`ontofetch settings show|schema|validate`**

* `show`: prints effective config (with **source** column), optional `--json`.
* `schema`: writes JSON Schema files (top-level + subs).
* `validate --file config.yaml`: validates against JSON Schema and prints errors with JSON pointer paths.

**`ontofetch plan` / `pull` / `extract` / `validate`**

* All read settings from the shared context; support `--jobs`, `--service`, `--version`.
* `extract` supports `--dry-run` to produce intended outputs without writing.

**`ontofetch latest get|set`**

* `get`: prints current latest (DB + file marker, noting mismatches).
* `set <version>`: confirmation required; updates DB and marker atomically.

**`ontofetch prune --dry-run|--apply`**

* Lists orphans (counts/bytes); `--apply` requires typing `PRUNE`.
* `--exclude-glob` to keep certain paths (typed list).

**`ontofetch db` subcommands**

* `versions` (with counts/bytes), `files --version`, `stats --version`, `doctor --fix` (reconcile FS↔DB).
* All support `--format` and autocompletions.

**`ontofetch delta` subcommands** (backed by your macros)

* `summary A B`, `files A B`, `renames A B`, `formats A B`, `validation A B`.

## 2.7 Completions & per-user app dir

* Ship `app install-completion` and `app show-completion` commands (Typer built-ins).
* Use `platformdirs` to store:

  * per-user RC file (`~/.config/ontofetch/cli.toml`) that can inject default flags,
  * cache dirs (`~/.cache/ontofetch/http`), and
  * local DB default location if not explicitly set.

## 2.8 Testing strategy (CliRunner)

* **Golden help**: capture `--help` output for each command; snapshot to detect drift.
* **Happy paths**: one test per command producing table/json output; assert exit code `0`.
* **Error paths**: invalid version id, schema validation failure, destructive commands without confirmation → correct exit codes and messages.
* **End-to-end smoke** (fast): `settings schema`, `settings validate`, `db latest`, `delta summary A B` with a small fixture DB.

## 2.9 Acceptance criteria

* Invoking `ontofetch` shows a **fast** global help (under ~100ms cold).
* `--version` prints software + build + libarchive version and exits with `0`.
* Commands honor `--format`, `--dry-run`, and `-v/-vv`.
* Completions work on bash/zsh/fish (document the one-liner to install).
* Each destructive command enforces confirmation; `--yes` bypass works and is test-covered.
* Docs generated from the CLI reflect the current flag set; CI fails if auto-docs are out of date.

---

## Implementation sequence (short PRs, minimal risk)

**PR-A: Settings trust & tracing**

* Add strict Pydantic models + validators + normalization contracts.
* Implement `TracingSettingsSource` & precedence; expose `source_fingerprint`.
* Embed `config_hash` in events & audits.
* Add `settings show|schema|validate` commands & tests.
* Commit JSON Schema under `docs/schemas/`.

**PR-B: Typer app foundation**

* Create app skeleton (`main.py`, context, global flags, logger).
* Add `--version`, `--format`, `-v/-vv`, `--dry-run`, `--config`.
* Implement `settings` subcommands & golden-help tests.

**PR-C: Completions + confirmations + exit codes**

* Add autocompletions backed by DuckDB.
* Implement confirmation prompts & `--yes`.
* Unify exit codes; add tests.

**PR-D: CLI docs & examples**

* Add script to generate docs; wire to CI.
* Fill examples for each command; verify drift detection.

---

### What you get when this lands

* A **provably correct config layer** (traced, hashed, schema’d) that agents and humans can **trust**.
* A **professional CLI** that’s fast, discoverable, safe, and self-documenting—ready for both ops and automation.
