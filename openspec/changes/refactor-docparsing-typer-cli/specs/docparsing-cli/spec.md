## ADDED Requirements
### Requirement: Typer-Based DocParsing CLI
The system SHALL expose the DocParsing workflows through a Typer multi-command application defined in `DocsToKG.DocParsing.core.cli`, replacing the bespoke `_Command`/`argparse` dispatcher while preserving the existing subcommand set (`doctags`, `chunk`, `embed`, `all`, `plan`, `manifest`, `token-profiles`) with the same option names and defaults. Each Typer command SHALL invoke the corresponding stage orchestrator directly so behaviour and exit codes remain unchanged without relying on legacy helper functions.

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
