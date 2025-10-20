# CLI Surface Inventory (Typer Migration)

This note captures the legacy command surfaces prior to the Typer refactor and documents the artefacts produced for parity checks.

- **Commands discovered**: `doctags`, `chunk`, `embed`, `all`, `plan`, `manifest`, `token-profiles`.
- **Legacy help output**: snapshots captured in `notes/help/*.txt` using `python -m DocsToKG.DocParsing.core.cli <command> --help` before and after the Typer conversion. These serve as the reference for flag names, aliases, defaults, and descriptions.
- **Exit/validation behaviour**: Each helper (`doctags()`, `chunk()`, `embed()`, `run_all()`, `manifest()`, `token_profiles()`) still funnels through `_run_stage` where applicable, preserving `CLIValidationError` formatting (exit code `2`) and propagating underlying stage exit codes. Optional dependency failures remain surfaced via the existing ImportError guards in `chunk()` and `token_profiles()`.
- **Backward-compatible shims**: Module-level helpers continue to accept `Sequence[str] | None` so downstream automation can call them directly and receive integer exit codes identical to invoking the CLI.

These artefacts satisfy tasks 1.1.1–1.1.3 of the change plan and provide the baseline for regressions while the Typer commands evolve.
