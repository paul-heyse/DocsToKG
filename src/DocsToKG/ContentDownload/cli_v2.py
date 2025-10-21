"""Typer-based CLI for ContentDownload with Pydantic v2 configuration."""

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from DocsToKG.ContentDownload.config import (
    export_config_schema,
    load_config,
    validate_config_file,
)
from DocsToKG.ContentDownload.resolvers.registry_v2 import (
    build_resolvers,
    get_registry,
)

# Feature flags support
try:
    from DocsToKG.ContentDownload.config.feature_flags import (
        get_feature_flags,
        FeatureFlag,
    )

    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False

# Optional CLI config commands (when feature enabled)
if FEATURE_FLAGS_AVAILABLE:
    try:
        from DocsToKG.ContentDownload.cli_config import register_config_commands

        CLI_CONFIG_AVAILABLE = True
    except ImportError:
        CLI_CONFIG_AVAILABLE = False
else:
    CLI_CONFIG_AVAILABLE = False

console = Console()
app = typer.Typer(help="DocsToKG ContentDownload")

# ============================================================================
# Setup
# ============================================================================


def _setup_logging(verbose: bool) -> None:
    """Setup logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# ============================================================================
# Commands
# ============================================================================


@app.command()
def run(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
        envvar="DTKG_CONFIG",
    ),
    resolver_order: Optional[str] = typer.Option(
        None,
        "--resolver-order",
        help="Comma-separated resolver order",
    ),
    max_workers: Optional[int] = typer.Option(None, "--workers", help="Number of parallel workers"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose"),
) -> None:
    """Run ContentDownload with artifact resolution."""
    _setup_logging(verbose)

    try:
        cli_overrides: dict = {"dry_run": dry_run}
        if resolver_order:
            cli_overrides["resolvers"] = {"order": [r.strip() for r in resolver_order.split(",")]}
        if max_workers:
            cli_overrides["max_workers"] = max_workers

        cfg = load_config(path=config, cli_overrides=cli_overrides)

        console.print(
            Panel(
                f"[bold green]✓ Config loaded[/bold green]\n"
                f"Hash: {cfg.config_hash()[:8]}...\n"
                f"Resolvers: {len(cfg.resolvers.order)}",
                title="ContentDownload",
            )
        )

        resolvers = build_resolvers(cfg)
        console.print(f"[green]✓ Built {len(resolvers)} resolvers[/green]")

        if dry_run:
            console.print("[yellow]Dry run mode[/yellow]")

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        if verbose:
            raise
        raise typer.Exit(code=1)


@app.command()
def print_config(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
        envvar="DTKG_CONFIG",
    ),
    raw: bool = typer.Option(False, "--raw", help="Raw JSON"),
) -> None:
    """Print merged effective config."""
    try:
        cfg = load_config(path=config)

        if raw:
            data = cfg.model_dump(mode="json")
            typer.echo(json.dumps(data, indent=2))
        else:
            data = cfg.model_dump(mode="json")
            console.print(
                Panel(json.dumps(data, indent=2), title="ContentDownload Config", expand=False)
            )

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def validate_config(
    config: str = typer.Argument(..., help="Path to config file"),
) -> None:
    """Validate a config file."""
    try:
        validate_config_file(config)
        console.print("[green]✓ Config valid[/green]")
    except Exception as e:
        console.print(f"[red]✗ Invalid: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def explain(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
        envvar="DTKG_CONFIG",
    ),
) -> None:
    """Explain resolver configuration and ordering."""
    try:
        cfg = load_config(path=config)
        registry = get_registry()

        table = Table(title="Resolver Configuration")
        table.add_column("Order", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Enabled", style="yellow")
        table.add_column("Status", style="magenta")

        for idx, resolver_name in enumerate(cfg.resolvers.order, 1):
            resolver_cfg = getattr(cfg.resolvers, resolver_name, None)
            if resolver_cfg is None:
                continue

            in_registry = resolver_name in registry
            status = "[green]✓ Registered[/green]" if in_registry else "[red]✗ Missing[/red]"

            table.add_row(
                str(idx),
                resolver_name,
                "[green]Yes[/green]" if resolver_cfg.enabled else "[red]No[/red]",
                status,
            )

        console.print(table)

        enabled_count = sum(
            1 for name in cfg.resolvers.order if getattr(cfg.resolvers, name).enabled
        )
        console.print(f"\n[cyan]Enabled: {enabled_count}/{len(cfg.resolvers.order)}[/cyan]")

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def schema(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save schema to file"),
) -> None:
    """Export JSON Schema for ContentDownloadConfig."""
    try:
        schema_data = export_config_schema()

        if output:
            output.write_text(json.dumps(schema_data, indent=2))
            console.print(f"[green]✓ Schema written to {output}[/green]")
        else:
            console.print(
                Panel(json.dumps(schema_data, indent=2), title="JSON Schema", expand=False)
            )

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(code=1)


# ============================================================================
# Optional Feature: Config Commands (when enabled via DTKG_FEATURE_CLI_CONFIG_COMMANDS)
# ============================================================================


def _register_optional_commands() -> None:
    """Register optional commands based on feature flags.

    When DTKG_FEATURE_CLI_CONFIG_COMMANDS=1, registers config inspection
    subcommands. When disabled, this has no effect.
    """
    if not FEATURE_FLAGS_AVAILABLE or not CLI_CONFIG_AVAILABLE:
        return

    try:
        flags = get_feature_flags()
        if flags.is_enabled(FeatureFlag.CLI_CONFIG_COMMANDS):
            register_config_commands(app)
            console.print(
                "[yellow]ℹ Config commands registered (DTKG_FEATURE_CLI_CONFIG_COMMANDS=1)[/yellow]"
            )
    except Exception as e:
        console.print(f"[yellow]⚠ Could not register config commands: {e}[/yellow]")


# Register optional commands when CLI is loaded
_register_optional_commands()


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
