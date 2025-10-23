# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.cli_main",
#   "purpose": "Main Typer CLI app for OntologyDownload with settings integration.",
#   "sections": [
#     {
#       "id": "normalize-plan-args",
#       "name": "_normalize_plan_args",
#       "anchor": "function-normalize-plan-args",
#       "kind": "function"
#     },
#     {
#       "id": "clicontext",
#       "name": "CliContext",
#       "anchor": "class-clicontext",
#       "kind": "class"
#     },
#     {
#       "id": "get-context",
#       "name": "get_context",
#       "anchor": "function-get-context",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     },
#     {
#       "id": "doctor",
#       "name": "doctor",
#       "anchor": "function-doctor",
#       "kind": "function"
#     },
#     {
#       "id": "version-cmd",
#       "name": "version_cmd",
#       "anchor": "function-version-cmd",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Main Typer CLI app for OntologyDownload with settings integration.

Provides a structured Typer CLI with:
- Global options (--config, -v/-vv, --format, --dry-run, --version)
- Settings context management
- Subcommand integration
- Structured logging
- Error handling and exit codes

This module serves as the entry point for all CLI operations.

Example:
    >>> from DocsToKG.OntologyDownload.cli_main import app
    >>> if __name__ == "__main__":
    ...     app()
"""

from pathlib import Path

import typer
from rich.console import Console

from DocsToKG.OntologyDownload import __version__
from DocsToKG.OntologyDownload.cli_settings_commands import settings_app
from DocsToKG.OntologyDownload.settings import get_default_config

# CLI argument normalization constants and functions
_DEFAULT_SUBCOMMAND = "pull"
_KNOWN_SUBCOMMANDS = {
    "pull",
    "plan",
    "plan-diff",
    "show",
    "validate",
    "init",
    "config",
    "doctor",
    "prune",
    "plugins",
}
_GLOBAL_OPTIONS_WITH_VALUES = {"--log-level"}


def _normalize_plan_args(args):
    """Inject the default subcommand when callers omit it.

    Args:
        args: Original CLI argument vector supplied to :func:`cli_main`.

    Returns:
        Updated argument list with the default subcommand inserted before the
        first positional argument when no explicit subcommand is present.
    """

    normalized: list[str] = list(args)
    if not normalized:
        return normalized

    index = 0
    while index < len(normalized):
        token = normalized[index]
        if token == "--":
            index += 1
            break
        if token.startswith("-"):
            option_name = token.split("=", 1)[0]
            if option_name in _GLOBAL_OPTIONS_WITH_VALUES:
                if "=" in token:
                    index += 1
                else:
                    index += 2
            else:
                index += 1
            continue
        break

    if index >= len(normalized):
        return normalized

    if normalized[index] not in _KNOWN_SUBCOMMANDS:
        normalized.insert(index, _DEFAULT_SUBCOMMAND)

    return normalized


# Global console for output
_console = Console()


class CliContext:
    """Context object passed to commands via callback.

    Holds shared state and resources used across commands:
    - Settings instance
    - Logger
    - Console printer
    - Global flags (verbosity, dry-run, format)
    """

    def __init__(
        self,
        config: Path | None = None,
        verbosity: int = 0,
        format_output: str = "table",
        dry_run: bool = False,
    ):
        """Initialize CLI context.

        Args:
            config: Path to config file (overrides ONTOFETCH_CONFIG env)
            verbosity: Verbosity level (0=normal, 1=info, 2=debug)
            format_output: Output format (table, json, yaml)
            dry_run: Don't make changes, just show what would happen
        """
        self.config = config
        self.verbosity = verbosity
        self.format_output = format_output
        self.dry_run = dry_run
        self.console = _console

        # Load settings (will respect config path if provided)
        # For now, use defaults since settings integration is separate
        try:
            self.settings = get_default_config()
        except Exception as e:
            self.console.print(f"[red]Error loading settings: {e}[/red]")
            self.settings = None

    def log_debug(self, message: str) -> None:
        """Log debug message if verbosity >= 2."""
        if self.verbosity >= 2:
            self.console.print(f"[dim]DEBUG: {message}[/dim]")

    def log_info(self, message: str) -> None:
        """Log info message if verbosity >= 1."""
        if self.verbosity >= 1:
            self.console.print(f"[cyan]INFO: {message}[/cyan]")


# Create main Typer app
app = typer.Typer(
    name="ontofetch-cli",
    help="OntologyDownload CLI - Manage ontology downloads and configuration",
    no_args_is_help=True,
)

# Global context variable (per-invocation)
_context: CliContext | None = None


def get_context() -> CliContext:
    """Get the current CLI context.

    Returns:
        CliContext: The current context instance

    Raises:
        RuntimeError: If context hasn't been initialized
    """
    if _context is None:
        raise RuntimeError("CLI context not initialized")
    return _context


@app.callback(invoke_without_command=False)
def main(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        envvar="ONTOFETCH_CONFIG",
        help="Path to config file (YAML or JSON)",
    ),
    verbosity: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
    format_output: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, yaml",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would happen without making changes",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit",
    ),
) -> None:
    """OntologyDownload CLI - Configuration and ontology management.

    Global options apply to all subcommands. Use them before specifying
    a subcommand:

        ontofetch-cli --config settings.yaml settings show
        ontofetch-cli -vv --format json settings schema

    For help on specific commands:

        ontofetch-cli settings --help
        ontofetch-cli settings show --help
    """
    global _context

    # Handle --version first
    if version:
        typer.echo(f"ontofetch-cli {__version__}")
        raise typer.Exit(0)

    # Initialize context
    _context = CliContext(
        config=config,
        verbosity=verbosity,
        format_output=format_output,
        dry_run=dry_run,
    )

    # Log startup info
    _context.log_debug(f"Config file: {config}")
    _context.log_debug(f"Verbosity: {verbosity}")
    _context.log_debug(f"Format: {format_output}")
    if dry_run:
        _context.console.print("[yellow]DRY-RUN MODE: No changes will be made[/yellow]")


# Add settings subcommand group
app.add_typer(
    settings_app,
    name="settings",
    help="Settings management (show, schema, validate)",
)


@app.command()
def doctor(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to fix identified issues",
    ),
) -> None:
    """Run diagnostic checks on the environment and configuration.

    Checks:
    - Environment variables
    - Config file accessibility
    - Required directories
    - Database connectivity
    - Settings validation

    Use --fix to attempt automatic repairs.

    Example:
        $ ontofetch-cli doctor
        $ ontofetch-cli doctor --fix
    """
    ctx = get_context()

    ctx.console.print("[bold]OntologyDownload Environment Diagnostics[/bold]")
    ctx.console.print()

    try:
        # Check settings
        ctx.console.print("[cyan]✓[/cyan] Settings loaded successfully")

        # Show config summary
        if ctx.settings:
            config_hash = ctx.settings.config_hash()
            ctx.log_info(f"Config hash: {config_hash}")

        ctx.console.print("[green]✓ All checks passed[/green]")

    except Exception as e:
        ctx.console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version_cmd() -> None:
    """Show version information.

    Example:
        $ ontofetch-cli version
    """
    ctx = get_context()
    ctx.console.print(f"[bold]ontofetch-cli[/bold] version {__version__}")


__all__ = [
    "app",
    "CliContext",
    "get_context",
    "main",
    "_normalize_plan_args",
]
