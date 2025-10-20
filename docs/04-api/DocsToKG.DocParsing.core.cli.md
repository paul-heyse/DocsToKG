# 1. Module: cli

Reference for ``DocsToKG.DocParsing.core.cli`` – the Typer-based command line
suite that fronts the DocParsing stages and preserves the legacy argparse
surfaces for downstream automation.

## 1. Overview

The module now exports a single `typer.Typer` application (`app`) that routes
subcommands (`doctags`, `chunk`, `embed`, `token-profiles`, `plan`, `manifest`,
`all`) through the existing stage orchestrators. Each subcommand keeps the
argparse parser it used before the refactor so `--help` output, defaults, and
validation remain unchanged.

## 2. Functions

### `_run_stage(handler, argv)`
Execute a stage handler while normalising DocParsing `CLIValidationError`
exceptions.

### `build_doctags_parser(prog)`
Return the argparse parser for the DocTags conversion command.

### `_doctags_help_text()`
Render the legacy `docparse doctags` help text used by the Typer wrapper.

### `_resolve_doctags_paths(args)`
Resolve mode, input, and output directories for the DocTags stage.

### `doctags(argv)`
Invoke the DocTags conversion workflow, returning the exit code.

### `_chunk_import_error_messages(exc)`
Produce the user-facing messages shown when optional chunking dependencies are
missing.

### `_import_chunk_module()`
Reload and return the chunking module, refreshing the package cache.

### `_chunk_help_text()`
Render the legacy `docparse chunk` help output (or dependency guidance).

### `chunk(argv)`
Execute the Docling hybrid chunker subcommand.

### `embed(argv)`
Run the embedding pipeline subcommand.

### `_embed_help_text()`
Render the legacy `docparse embed` help output.

### `token_profiles(argv)`
Execute the tokenizer profiling subcommand.

### `_token_profiles_help_text()`
Render the legacy `docparse token-profiles` help output (or dependency guidance).

### `plan(argv)`
Display the doctags → chunk → embed plan without executing any stages.

### `manifest(argv)`
Inspect pipeline manifest artifacts via CLI.

### `_build_manifest_parser()`
Construct the argparse parser shared by the manifest command and help wrapper.

### `_manifest_help_text()`
Render the legacy `docparse manifest` help output.

### `_manifest_main(argv)`
Implementation backing the `docparse manifest` subcommand.

### `_build_run_all_parser()`
Construct the argparse parser used by both `docparse all` and `docparse plan`.

### `_run_all_help_text()`
Render the legacy `docparse all` help output.

### `_plan_help_text()`
Render the legacy `docparse plan` help output.

### `_build_stage_args(args)`
Construct doctags/chunk/embed argument lists for the `docparse all` orchestrator.

### `run_all(argv)`
Run doctags → chunk → embed sequentially, respecting resume/force flags.

### `_forward_with_context(ctx, handler)`
Bridge Typer context arguments into the legacy argparse handlers and exit with
their status codes.

### `_doctags_cli(ctx)`
Typer callback for the `docparse doctags` command.

### `_chunk_cli(ctx)`
Typer callback for the `docparse chunk` command.

### `_embed_cli(ctx)`
Typer callback for the `docparse embed` command.

### `_token_profiles_cli(ctx)`
Typer callback for the `docparse token-profiles` command.

### `_plan_cli(ctx)`
Typer callback for the `docparse plan` command.

### `_manifest_cli(ctx)`
Typer callback for the `docparse manifest` command.

### `_all_cli(ctx)`
Typer callback for the `docparse all` command.

### `main(argv)`
Entry point used by ``python -m DocsToKG.DocParsing.core.cli``; runs the Typer
application with the provided argument list and returns the resulting exit code.

## 3. Classes

### `_ParserHelpCommand`
Custom Typer command subclass that appends the legacy argparse help text to the
Typer-generated help output.

### `_ManifestHelpFormatter`
ArgumentDefaultsHelpFormatter variant that preserves hyphenated aliases on a
single line.

## 4. Variables

### `CLI_DESCRIPTION`
Extended summary (with usage examples) for the unified DocParsing CLI.

### `app`
The root `typer.Typer` application that exposes the DocParsing subcommands.
