Here’s a **surgical, narrative-only implementation plan** for **PR-3: Unify CLI on Typer; delete custom argparse scaffolding**. It’s written so an AI programming agent can execute it step-by-step—no guessing, no Python code.

---

# Scope & intent

**Goals**

1. Replace the DocParsing command-line entry point based on `argparse` with a **single Typer app** exposing subcommands: `doctags`, `chunk`, `embed`, and `all`.
2. **Delete** the bespoke dispatcher (`_Command`, `COMMANDS`, manual `argparse` parsing) in `src/DocsToKG/DocParsing/cli.py` and any thin per-command wrappers that only forward to stage parsers. Today, `cli.py` constructs an `ArgumentParser`, builds a `_Command` registry, and dispatches based on `command`/`args`.
3. Preserve **backward-compatible Python entry points** (`DocParsing.cli.chunk()`, `embed()`, `doctags()`), but make them simple one-liners that call the new Typer handlers (or underlying stage functions), not `argparse`. The current helpers return the result of `_run_*` functions and live in the same module.
4. Keep the **stage parser functions** available for programmatic use (e.g., `pipelines.pdf_parse_args`, `html_parse_args`), but **stop using them from the top-level CLI**—they can remain for integration tests and legacy programmatic callers. The repo already has stage-local `argparse` parsers in `pdf_pipeline.py` and `pipelines.py`.
5. Ensure the **public package facade** continues to re-export stage entry points from `__init__.py` (pdf/html) unchanged; this file is already wiring those names.
6. Update docs/README to show **Typer usage** (already moving toward `python -m DocsToKG.DocParsing.cli <subcmd>`). The README currently demonstrates the unified CLI shape; we’ll make help/completions consistent with Typer.

**Non-goals**

* No behavioral changes to pipeline algorithms or I/O layout.
* No change to stage-local flags beyond mapping them into Typer options (names & defaults stay the same).

---

# Why this is needed (current duplication)

* `src/DocsToKG/DocParsing/cli.py` implements its **own dispatcher** with `_Command`, `COMMANDS`, and an `ArgumentParser` for a “meta-CLI,” then forwards into stage parsers. This scaffolding is entirely replaced by Typer’s subcommand model.
* Stage modules still define their own `argparse` surfaces (`pdf_pipeline`, `html_pipeline`, `pipelines`)—we’ll leave those for programmatic use but **stop duplicating argument wiring at the top level**.
* The repository already carries Typer’s dependency stack (note `shellingham` “via typer” entries), so adopting Typer is consistent with project deps.

---

# Target design (the “after” picture)

1. **Single Typer app** in `src/DocsToKG/DocParsing/cli.py`:

   * `app = Typer(help="Unified DocParsing CLI…")`
   * Subcommands: `doctags`, `chunk`, `embed`, and `all`.
   * Each subcommand exposes **the same flags** users see today (names/defaults/semantics), but defined as Typer options.
   * Global options kept minimal (e.g., `--data-root`) and inherited where sensible.

2. **Handlers delegate to stage functions**:

   * `doctags`: orchestrates HTML/PDF conversion by calling `pipelines.html_main(...)` or `pipelines.pdf_main(...)` depending on `--mode` (or auto-detect), not by re-parsing with `argparse`. (Stage parsers remain for programmatic use.)
   * `chunk`: calls the Docling hybrid chunker driver exposed via the current chunk runtime.
   * `embed`: calls the embedding runtime.
   * `all`: runs `doctags → chunk → embed` sequentially, honoring `--resume/--force` and propagating dirs.

3. **Backward-compat Python helpers**:

   * Keep `def chunk(argv: Sequence[str]|None=None) -> int` etc., but have them call the **Typer subcommand function** (or underlying stage main) directly. Today they call `_run_*`; after the change, they remain shims for importers.

4. **Entry point**:

   * Keep module execution (`python -m DocsToKG.DocParsing.cli`) working by exporting a `main()` that runs the Typer app.
   * Optionally define a console script (`docparse`) in `pyproject.toml` to improve UX (not mandatory for this PR).

5. **Docs**:

   * Ensure `docs/04-api/DocsToKG.DocParsing.cli.md` reflects Typer subcommands and help. (File already documents a unified CLI module.)
   * README examples already show `python -m DocsToKG.DocParsing.cli <subcmd>`; leave as-is / tighten help text.

---

# Step-by-step implementation

## 0) Pre-flight

* Branch: `codex/pr3-typer-cli`.
* Confirm Typer deps present (we already see `shellingham` via Typer).
* Inventory current CLI surfaces and flags to **preserve**:

  * Top-level docstring and dispatcher in `DocParsing/cli.py`.
  * The README examples for `doctags`, `chunk`, `embed`.
  * Stage parsers in `pdf_pipeline.py` and `html_pipeline` docs (parse via argparse).

## 1) Replace the bespoke dispatcher in `DocParsing/cli.py` with Typer

**Files**: `src/DocsToKG/DocParsing/cli.py`

**Actions**:

1. **Remove** `_Command`, `COMMANDS`, and the `argparse.ArgumentParser` construction from this module. Keep the docstring and examples up top (refresh wording to match Typer help). The current dispatcher builds an `ArgumentParser`, then selects a handler; delete that path entirely.

2. **Create a Typer app** with four subcommands:

   * `doctags`: Options:

     * `--mode [auto|pdf|html]` (default: `auto`, preserving current behavior described in docs and README),
     * `--input DIR`, `--output DIR`, `--workers INT`, `--force/--no-force`, `--resume/--no-resume`, `--served-model-name` (pdf path), etc. (mirror existing flags demonstrated in current CLI docs and README).
   * `chunk`: Options: `--in-dir`, `--out-dir`, token bounds (`--min-tokens`, `--max-tokens`), tokenizer, resume/force.
   * `embed`: Options: `--chunks-dir`, `--out-dir`, `--resume`, BM25 params, SPLADE/Qwen batches/paths, offline switches.
   * `all`: Combines the three stages; accept a subset of shared flags and pass through the rest to sub-stages (document pass-through behavior).

3. **Handlers delegate** to stage orchestrators:

   * `doctags`: implement mode resolution without calling stage `parse_args`; call stage `*_main()` functions (e.g., `pipelines.html_main(...)` / `pipelines.pdf_main(...)`). The repo centralizes pdf/html orchestration in `pipelines.py`.
   * `chunk` and `embed`: call the unified runtime drivers you already expose via the CLI module today (they were previously used by `_run_chunk` and `_run_embed`).

4. **`main()` compatibility**:

   * Keep a `main(argv: Sequence[str]|None=None) -> int` that invokes the Typer app and returns an exit code; maintain the `if __name__ == "__main__"` guard to raise `SystemExit(main())`. (This exists today with bespoke dispatch.)

**Acceptance (Step 1)**

* Running `python -m DocsToKG.DocParsing.cli --help` shows Typer help; running `… doctags --help` etc. present subcommand-specific help.

## 2) Keep stage parsers for programmatic use; stop wiring them at top-level

**Files**: `src/DocsToKG/DocParsing/pdf_pipeline.py`, `src/DocsToKG/DocParsing/html_pipeline.py`, `src/DocsToKG/DocParsing/pipelines.py`

**Actions**:

1. Do **not** remove `parse_args()`/`build_parser()` within stage modules—leave them for tests and specialized integration (you’ve invested in these already). `pdf_pipeline.py` explicitly carries an `argparse` parser with docstrings.

2. Ensure Typer handlers **construct argument objects directly** (not by calling stage `parse_args()`), then call the stage’s `*_main()`.

3. Keep the `__init__.py` **re-exports** intact (pdf/html parse/main). They’re already wired to `pipelines`.

**Acceptance (Step 2)**

* Importers can still `from DocsToKG.DocParsing import pdf_parse_args` and friends. Typer CLI does not import stage parsers at runtime.

## 3) Maintain Python helpers in `DocParsing/cli.py` as shims

**Files**: `src/DocsToKG/DocParsing/cli.py`

**Actions**:

* Keep `def chunk(argv=None)`, `def embed(argv=None)`, `def doctags(argv=None)` as stable helpers, but **change their body** to call the Typer handlers (or invoke the underlying stage functions) rather than bespoke `_run_*` functions. Today these shims call `_run_chunk/_run_embed/_run_doctags`; post-change they should be thin wrappers with identical semantics for tests/importers.

**Acceptance (Step 3)**

* Existing tests importing `DocParsing.cli.chunk()` still pass and return an `int` status.

## 4) Wiring for completions & entry point (optional but recommended)

**Files**: `pyproject.toml`

**Actions**:

* Add an **optional** console script: `docparse = DocsToKG.DocParsing.cli:main` so users can call `docparse <subcmd>`.
* (Optional) Document shell completion installation instructions in README (Typer exposes Click’s completion capability via the CLI command itself).

**Acceptance (Step 4)**

* `docparse --help` works if console-script is configured. Module execution (`python -m …`) continues to work.

## 5) Documentation pass

**Files**:

* `docs/04-api/DocsToKG.DocParsing.cli.md` (already present),
* `src/DocsToKG/DocParsing/README.md`.

**Actions**:

1. Update CLI docs to **show Typer help** and subcommands; confirm the “Unified DocParsing CLI” page points to `python -m DocsToKG.DocParsing.cli <command>`. (This page already documents the module—tighten it with Typer semantics.)
2. Keep README examples (they already call the unified CLI subcommands).

**Acceptance (Step 5)**

* Docs build shows Typer subcommands & help with updated examples.

## 6) Clean out leftover bespoke scaffolding

**Files**: `src/DocsToKG/DocParsing/cli.py`

**Actions**:

* **Delete** `_Command` class and `COMMANDS` dict, plus the “choices=COMMANDS.keys()” parser construction and the manual `REMAINDER` forwarding. These are entirely superseded by Typer. (See current bespoke construction: parser, `choices`, and handler call.)

**Acceptance (Step 6)**

* No occurrences of `_Command`, `COMMANDS`, or `argparse.REMAINDER` in the module.

---

# Flag mapping (preserve user-visible behavior)

> The goal is **parity**: same option names & defaults users already learned.

**`doctags`**

* `--mode`: `auto|pdf|html` (default `auto`), current CLI/docs already reference auto-detection. README examples show `--mode pdf` or just `doctags`.
* `--input`, `--output`: directory paths (default to project Data layout).
* `--workers`, `--resume/--no-resume`, `--force/--no-force`.
* pdf-only options (served VLM names/port) remain available when `mode=pdf`, but Typer will list them under `doctags` for discoverability.

**`chunk`**

* `--in-dir`, `--out-dir`, `--min-tokens`, `--max-tokens`, tokenizer path/name, `--resume/--force`.
* These reflect the Docling hybrid chunker arguments surfaced by the current CLI. (The CLI and README already document min/max tokens & directories.)

**`embed`**

* `--chunks-dir`, `--out-dir`, `--resume`, BM25 (`--bm25-k1`, `--bm25-b`), SPLADE and Qwen batch sizes / model dirs, offline flags.
* README shows the `--out-dir Data/Embeddings` and batch options.

**`all`**

* Accepts a concise set of shared flags (`--data-root`, `--resume/--force`, `--workers`) and forwards stage-specific options as needed (document pass-through convention in `--help`). The CLI docstring already lists “Available commands” and previously referenced an `all` entry; make that official.

---

# Tests (what to add/update)

1. **Smoke help**: `cli --help` and each subcommand’s `--help` render without error and show expected options.
2. **Dispatch parity**:

   * `cli doctags --mode html --input <dir>` calls the HTML converter path (assert it reaches `pipelines.html_main`).
   * `cli doctags --mode pdf` reaches PDF path.
   * `cli chunk` and `cli embed` reach their respective runtimes.
3. **Return code semantics**: subcommands propagate stage exit codes as before (0 on success).
4. **Backward-compat helpers**:

   * `from DocsToKG.DocParsing.cli import chunk, embed, doctags` still callable, return `int`.
5. **Module exec**: `python -m DocsToKG.DocParsing.cli doctags --help` exits 0.
6. **No bespoke scaffolding**: test imports fail for `_Command`/`COMMANDS` (i.e., confirm they’re gone).

---

# Documentation tasks

* Replace any remaining mentions of `cli.doctags_convert`, `cli.chunk_and_coalesce`, `cli.embed_vectors` with `cli doctags|chunk|embed`. The docs refactor commits already moved in this direction; complete the sweep.
* Ensure the API ref page for `DocsToKG.DocParsing.cli` reflects the Typer subcommands (it already exists; update headings/sections).

---

# Deletions / moves in this PR

* **Delete** bespoke dispatch constructs in `DocParsing/cli.py`: `_Command`, `COMMANDS`, parser build, manual `REMAINDER` splitting.
* **Keep** stage `argparse` parsers in `pdf_pipeline.py` and `pipelines.py`; do not wire them at top level anymore.
* **Retain** package re-exports in `DocParsing/__init__.py` (they already point to `pipelines`).

---

# Risks & mitigations

* **CLI flag drift**: If Typer option names diverge from historical flags, users break.
  **Mitigation**: Mirror existing names exactly; add hidden aliases for any legacy spellings if they exist.
* **Behavioral mismatches in `--mode auto`**: Ensure the same auto-detect logic the old CLI used (docs already describe mode inference) is preserved. **Test** with mixed/empty directories.
* **Docs desynchronization**: The repository already has a unified CLI doc page; **rebuild docs** and visually inspect the CLI reference & README examples.

---

# Work breakdown (reviewable commits)

1. **Commit A — Introduce Typer app**

   * Replace bespoke dispatcher in `DocParsing/cli.py` with `app = Typer()` and four subcommands.
   * Keep `main()` and `if __name__ == "__main__"` module exec path.
     *Removes* `_Command`/`COMMANDS` usage.

2. **Commit B — Wire subcommands to stage orchestrators**

   * Implement handlers for `doctags|chunk|embed|all` that call stage `*_main()` functions (no stage `parse_args()`).

3. **Commit C — Back-compat Python shims**

   * Keep `chunk()/embed()/doctags()` helpers but forward to the new handlers/underlying mains.

4. **Commit D — Docs update**

   * Update `docs/04-api/DocsToKG.DocParsing.cli.md` to show Typer help & subcommands.
   * Verify README examples remain valid (unified CLI already shown).

5. **Commit E — Optional console script**

   * Add `docparse` console entry to `pyproject.toml` (if desired).

6. **Commit F — Tests**

   * Add Typer `CliRunner` tests for help, dispatch, and return codes.
   * Verify legacy Python helpers return int and reach the correct handlers.

---

# Acceptance criteria (“done”)

* `python -m DocsToKG.DocParsing.cli --help` and each subcommand’s `--help` display Typer help successfully.
* `doctags`, `chunk`, `embed`, and `all` execute the same underlying flows as before (no change in outputs/exit codes).
* No references to bespoke `_Command`, `COMMANDS`, `argparse.REMAINDER` remain in `DocParsing/cli.py`.
* Package re-exports in `DocParsing/__init__.py` remain stable for programmatic callers.
* Docs & README accurately show the Typer-based interface.

---

# Rollback plan

If Typer introduces unforeseen regressions, re-enable the old dispatcher by restoring the previous `cli.py` from Git history (keep the Typer app in a side module for a later re-attempt). Because we preserved stage `argparse` parsers in pdf/html pipelines, the old behavior is trivially recoverable.

---

# Why this shrinks code & complexity immediately

* **Deletes** the in-house command registry and meta-parser in `cli.py`; Typer’s subcommands become the source of truth for the CLI.
* Keeps stage parsers for tests/internals but **stops re-wiring them twice** (stage and top-level).
* Aligns with the project’s dependency trajectory (Typer & shellingham already present).

If you’d like next, I can enumerate the **exact option list** for each subcommand (names, types, defaults, help strings) by walking the stage parsers and producing a one-to-one Typer mapping table.
