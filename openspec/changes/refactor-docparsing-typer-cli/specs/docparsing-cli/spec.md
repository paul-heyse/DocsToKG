## ADDED Requirements
### Requirement: Typer-Based DocParsing CLI
The system SHALL expose the DocParsing workflows through a Typer multi-command application defined in `DocsToKG.DocParsing.core.cli`, replacing the bespoke `_Command`/`argparse` dispatcher while preserving the existing subcommand set (`doctags`, `chunk`, `embed`, `all`, `plan`, `manifest`, `token-profiles`) with the same option names and defaults. Each Typer command SHALL map one-to-one with the legacy `_run_*` implementation so behaviour and exit codes remain unchanged.

#### Scenario: CLI help renders via Typer
- **WHEN** an operator runs `python -m DocsToKG.DocParsing.core.cli --help`
- **THEN** the command exits with status `0`
- **AND** the help output lists the Typer application name and each available subcommand with its help text.

#### Scenario: Subcommand help mirrors legacy options
- **WHEN** an operator runs `python -m DocsToKG.DocParsing.core.cli chunk --help`
- **THEN** the help output lists every option that existed in the legacy CLI (`--data-root`, `--manifest`, `--workers`, and so on)
- **AND** the default values shown match the values exposed by the legacy `_Command` implementation.

#### Scenario: Subcommands reuse stage orchestrators
- **WHEN** an operator runs `python -m DocsToKG.DocParsing.core.cli doctags --mode pdf --data-root Data`
- **THEN** the CLI invokes the existing DocTags stage orchestrator with the provided arguments
- **AND** the exit code matches the return value of the stage orchestrator.

### Requirement: Stable Programmatic Shims
The system SHALL retain the module-level helper functions `doctags()`, `chunk()`, `embed()`, and `run_all()` in `DocsToKG.DocParsing.core.cli`, and SHALL implement them as thin wrappers over the Typer handlers so that downstream automation continues to receive integer exit codes consistent with invoking the CLI subcommands. Each helper SHALL continue to accept an optional `Sequence[str]` argument representing CLI-like flags.

#### Scenario: Programmatic doctags invocation remains supported
- **WHEN** an automation script imports `DocsToKG.DocParsing.core.cli.doctags` and calls it with arguments analogous to the CLI (for example `["--mode", "html"]`)
- **THEN** the helper delegates to the Typer `doctags` handler
- **AND** the helper returns the same exit code that the CLI subcommand would emit for the provided arguments.

#### Scenario: Programmatic run_all delegates sequentially
- **WHEN** an automation script calls `DocsToKG.DocParsing.core.cli.run_all(["--data-root", "Data"])`
- **THEN** the helper runs the same sequence of stage orchestrators as the Typer `all` command (`doctags`, then `chunk`, then `embed`)
- **AND** the helper returns the exit code produced by the orchestrator chain without raising unexpected exceptions.
