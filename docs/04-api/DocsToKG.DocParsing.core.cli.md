# 1. Module: cli

Reference for ``DocsToKG.DocParsing.core.cli`` – the Typer-based command line
suite that now drives the DocParsing stages through typed wrappers while
preserving the legacy helpers for downstream automation.

## 1. Overview

The module exposes a single `typer.Typer` application (`app`) plus a collection
of helper utilities that translate typed Typer options into the `argv` lists the
existing stage helpers expect. Every subcommand keeps the underlying argparse
parsers intact so `doctags()`, `chunk()`, `embed()`, `run_all()` and friends
continue to accept `Sequence[str] | None` and return integer exit codes exactly
as before. Typer supplies the surface UX (help output, type conversion, flag
aliases) while the helpers ensure parity with the legacy argparse behaviour.

## 2. Functions

### `_append_option(argv, flag, value, *, formatter=str, default=_DEFAULT_SENTINEL)`
Append an option/value pair to ``argv`` when ``value`` is set and differs from
``default``. Used by the subcommand builders to avoid altering legacy defaults.

### `_append_flag(argv, flag, enabled)`
Append ``flag`` to ``argv`` when ``enabled`` is ``True``.

### `_append_multi_values(argv, flag, values, *, formatter=str)`
Append ``flag`` for each entry in ``values`` (used for repeatable options).

### `_build_doctags_cli_args(...)`
Return the legacy ``docparse doctags`` argv list based on the typed Typer
parameters (mode, directories, vLLM options, resume flags, etc.).

### `_build_chunk_cli_args(...)`
Translate Typer inputs for ``docparse chunk`` into the argv list consumed by the
chunker helper.

### `_build_embed_cli_args(...)`
Translate Typer inputs for ``docparse embed`` into the argv list consumed by the
embedding helper (including planning, cache, and sharding flags).

### `_build_token_profiles_cli_args(...)`
Compose the argv list for ``docparse token-profiles`` while honouring optional
repeatable ``--tokenizer`` values.

### `_build_manifest_cli_args(...)`
Build the argv list for ``docparse manifest`` (stages, tail, summary flags).

### `_build_run_all_cli_args(...)`
Build the shared argv list for ``docparse all`` / ``docparse plan`` so the
orchestrator helper continues to orchestrate doctags → chunk → embed.

### `_run_stage(handler, argv)`
Execute a stage helper while normalising DocParsing ``CLIValidationError``
exceptions to match legacy error formatting.

### `build_doctags_parser(prog)`
Return the argparse parser for the DocTags conversion command (still used by the
legacy helper).

### `doctags(argv)`, `chunk(argv)`, `embed(argv)`, `token_profiles(argv)`,
`plan(argv)`, `manifest(argv)`, `run_all(argv)`
Unchanged legacy entry points – now shared by the Typer commands and downstream
automation.

### `_import_chunk_module()` / `_chunk_import_error_messages(exc)`
Retained helpers that lazily import the optional chunker module and render
friendly dependency errors.

## 3. Typer Commands

Each command accepts typed parameters via `typing_extensions.Annotated` +
`typer.Option(...)`, maps them to the legacy argv list via the helper functions
above, and then calls the existing stage helper so behaviour (exit codes,
validation, optional dependency handling) remains unchanged.

### `_doctags_cli(...)`
Typer surface for ``docparse doctags`` (includes hidden ``--pdf`` / ``--html``
aliases for the old mode shortcuts).

### `_chunk_cli(...)`
Typer surface for ``docparse chunk`` with typed options for presets, directories
and resume flags.

### `_embed_cli(...)`
Typer surface for ``docparse embed`` covering cache controls, sharding options,
format selection, and planning flags.

### `_token_profiles_cli(...)`
Typer surface for ``docparse token-profiles`` with repeatable ``--tokenizer``
options.

### `_manifest_cli(...)`
Typer surface for ``docparse manifest`` – mirrors the legacy parser flags for
stage filtering, tailing, and formatting.

### `_plan_cli(...)`
Typer surface for ``docparse plan`` that injects the ``--plan`` flag while
respecting all orchestrator options.

### `_all_cli(...)`
Typer surface for ``docparse all`` including the ``--plan/--plan-only`` flag to
mimic legacy behaviour.

### `main(argv)`
Entry point used by ``python -m DocsToKG.DocParsing.core.cli``; forwards the
argument list to the Typer app so help output triggers ``SystemExit`` the same as
argparse while keeping programmatic imports intact.

## 4. Variables

### `CLI_DESCRIPTION`
Extended summary (with usage examples) for the unified DocParsing CLI.

### `app`
The root `typer.Typer` application that exposes the DocParsing subcommands.
