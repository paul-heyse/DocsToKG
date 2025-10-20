# 1. Module: cli

Reference for ``DocsToKG.DocParsing.core.cli`` – the Typer-powered entry point
that orchestrates the DocParsing stages.

## 1. Overview

The module exposes a single `typer.Typer` application (`app`) plus a set of
helpers that translate typed Typer options into the arguments expected by the
DocTags, chunking, embedding, planning, and manifest orchestrators. Legacy
module-level shims (`doctags()`, `chunk()`, `run_all()`, …) have been retired;
the Typer commands now call the stage orchestrators directly via private
``_execute_*`` helpers, keeping behaviour and exit codes identical to the former
dispatcher while presenting a cleaner API surface.

## 2. Functions

### `_append_option(argv, flag, value, *, formatter=str, default=_DEFAULT_SENTINEL)`
Append an option/value pair to ``argv`` when ``value`` is set and differs from
``default``. Used by the subcommand builders to avoid altering legacy defaults.

### `_append_flag(argv, flag, enabled)`
Append ``flag`` to ``argv`` when ``enabled`` is ``True``.

### `_append_multi_values(argv, flag, values, *, formatter=str)`
Append ``flag`` for each entry in ``values`` (used for repeatable options).

### `_build_doctags_cli_args(...)`
Return the argument vector for the DocTags command (mode selection, vLLM
options, resume flags).

### `_build_chunk_cli_args(...)`
Return the argument vector for the chunking command (profile presets, token
windows, resume/force controls).

### `_build_embed_cli_args(...)`
Return the argument vector for the embedding command (file locations, batch
sizes, sharding, validation/plan flags).

### `_build_token_profiles_cli_args(...)`
Compose arguments for the tokenizer profiling command.

### `_build_manifest_cli_args(...)`
Compose arguments for manifest inspection (stage filters, tail/summarise flags).

### `_build_run_all_cli_args(...)`
Compose arguments for the `all` / `plan` orchestration command so the three
stages share consistent defaults.

### `_execute_doctags(argv)`
Run the DocTags orchestrator with the supplied CLI arguments and return its exit
code.

### `_execute_chunk(argv)`
Run the chunking orchestrator with the supplied CLI arguments and return its exit
code.

### `_execute_embed(argv)`
Run the embedding orchestrator with the supplied CLI arguments and return its
exit code.

### `_execute_token_profiles(argv)`
Run the tokenizer profiling command with the supplied CLI arguments and return
its exit code.

### `_execute_manifest(argv)`
Inspect manifest artefacts using the supplied CLI arguments.

### `_execute_plan(argv)`
Generate the doctags → chunk → embed plan without executing the stages.

### `_execute_run_all(argv)`
Run doctags → chunk → embed sequentially with the supplied CLI arguments,
propagating exit codes from each stage.

### `_run_stage(handler, argv)`
Execute a stage helper while normalising DocParsing ``CLIValidationError``
exceptions into user-facing hints.

### `build_doctags_parser(prog)`
Return the argparse parser used internally to configure DocTags runs (still
consumed by the private `_execute_doctags` helper).

## 3. Typer Commands

Each command accepts typed parameters via `typing_extensions.Annotated` +
`typer.Option(...)`, validates/normalises the inputs (path checks, numeric
bounds, hidden aliases for legacy flags), builds the argv list via the helpers
above, and then calls the corresponding ``_execute_*`` helper. This keeps the
end-user UX aligned with Typer best practices while maintaining behaviour
parity with the retired dispatcher.

### `_doctags_cli(...)`
Typer surface for ``docparse doctags`` (includes hidden ``--pdf`` and ``--html``
aliases for the old mode shortcuts).

### `_chunk_cli(...)`
Typer surface for ``docparse chunk`` with typed options for presets, directories
and resume/force controls.

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
retain plan-preview behaviour.

### `main(argv)`
Entry point used by ``python -m DocsToKG.DocParsing.core.cli``; forwards the
argument list to ``app`` so help output and exits match Click/Typer semantics.

## 4. Variables

### `CLI_DESCRIPTION`
Extended summary (with usage examples) for the unified DocParsing CLI.

### `app`
The root `typer.Typer` application that exposes the DocParsing subcommands.
