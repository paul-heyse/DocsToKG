## Why
The current DocParsing CLI (`DocsToKG.DocParsing.core.cli`) builds and dispatches commands by hand using a custom `_Command` registry plus an `argparse` meta-parser. Every time we touch a flag we have to update that registry, the manual dispatcher, the per-command parser, and the documentation. This duplication has already led to skew (for example, flag help text that appears in the README but not in `--help`). The project already vendors Typer for other tools, so moving the DocParsing CLI onto Typer lets us define each command once, get structured `--help` output for free, and slim down the amount of code we have to keep in sync.

Think of the refactor in three pieces:
- **Surface parity** — Every command (`doctags`, `chunk`, `embed`, `all`, `plan`, `manifest`, `token-profiles`) must keep the exact same option names, default values, and behaviour. Typer just becomes the parser/dispatcher.
- **Python helpers remain stable** — Downstream automation imports `chunk()`, `embed()`, `doctags()`, or `run_all()` directly. Those helpers still need to accept the same `Sequence[str]` style arguments and return the same exit codes.
- **Documentation becomes truthful** — Once Typer is in place we can render the new `--help` text in README/API docs so newcomers (and automation) see the same information everywhere.

## What Changes
- **Replace the dispatcher**: Delete `_Command`, `COMMANDS`, and the top-level `argparse` wiring in `DocsToKG.DocParsing.core.cli`. Create a `typer.Typer()` instance (`app`) and port each existing command across. For every Typer command:
  1. Copy the current function that performs the work (for example `_run_doctags`), keep the body, and wrap it in a Typer command definition (`@app.command("doctags")`).
  2. Annotate parameters with types so Typer can parse them (e.g. `mode: Annotated[str, typer.Option(..., help="...")] = "auto"`).
  3. Preserve default values, env-vars, and required flags exactly as they exist today. If a flag was optional before, give it the same default in the Typer option.
  4. Return an `int` exit code. When a command already returns `int`, return that. When it previously called `sys.exit`, change it to return the value so the helper functions can still capture it.
- **Add a Typer entry point**: Expose a `main(argv: Optional[Sequence[str]] = None) -> int` that calls `app()` and keep `if __name__ == "__main__": raise SystemExit(main())`. This mimics the previous `main` but routes through Typer.
- **Re-implement helpers in terms of Typer**: Update `doctags()`, `chunk()`, `embed()`, and `run_all()` so they simply call the corresponding Typer handler (or an internal utility the Typer handler also calls). They should no longer recreate argument parsers. Each helper should:
  1. Convert `argv` to a `List[str]`.
  2. Call the same internal function that the Typer command uses.
  3. Wrap calls in `_run_stage` so existing error formatting stays intact.
- **Keep stage parsers untouched**: Modules like `DocsToKG.DocParsing.doctags` still export functions that construct `argparse` parsers for programmatic callers and tests. Do not delete them. The Typer command should call their stage entry points (`main()` or equivalent), not the parser builder.
- **Document the behaviour**: Update `docs/04-api/DocsToKG.DocParsing.cli.md` and any README snippets so the examples use Typer. Capture the new `python -m DocsToKG.DocParsing.core.cli --help` output and highlight subcommands, options, and shell completion instructions. Make it explicit that the CLI now supports Typer’s features (completion, env var hints, etc.).
- **Testing expectations**: Add new tests that use Typer’s `CliRunner` to execute:
  - `--help` (ensures commands register correctly).
  - Representative invocations for `doctags`, `chunk`, `embed`, and `all`, asserting the exit codes match the old behaviour.
  - One negative path (bad flag value) to make sure Typer surfaces validation errors similar to the previous implementation.
- **Housekeeping**: Remove any imports that belonged to the old `_Command` infrastructure and keep `__all__` up to date so consumers continue to import the same names.

## Impact
- Affected specs: docparsing-cli
- Affected code: `src/DocsToKG/DocParsing/core/cli.py`, `src/DocsToKG/DocParsing/__init__.py`, `src/DocsToKG/DocParsing/core/tests`, documentation under `docs/`
