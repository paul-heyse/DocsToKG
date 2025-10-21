"""
Modern ContentDownload CLI with Pydantic v2 Config

Uses Typer for clean, composable command structure with Pydantic v2 config.
Subcommands: run, print-config, validate-config, explain

Environment: $DTKG_CONFIG or --config for file path
             $DTKG_* environment variables for config overrides
             CLI flags for additional overrides
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON as RichJSON
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

_LOGGER = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="ContentDownload — Resolver-driven artifact acquisition")


# ============================================================================
# Utilities
# ============================================================================


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
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
        help="Path to config file (YAML/JSON)",
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
                Panel(
                    RichJSON.from_data(data),
                    title="Effective Config",
                )
            )

        console.print(f"[green]Config hash: {cfg.config_hash()}[/green]")

    except Exception as e:
        console.print(f"[red]✗ Config error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def validate_config(
    config: str = typer.Argument(..., help="Path to config file"),
) -> None:
    """Validate a config file."""
    try:
        validate_config_file(config)
        console.print(f"[green]✓ {config} is valid[/green]")
    except Exception as e:
        console.print(f"[red]✗ Invalid config: {e}[/red]")
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
        table.add_column("Resolver", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Registered", style="yellow")

        for idx, resolver_name in enumerate(cfg.resolvers.order, 1):
            resolver_cfg = getattr(cfg.resolvers, resolver_name)
            status = "✓ Enabled" if resolver_cfg.enabled else "✗ Disabled"
            registered = "✓" if resolver_name in registry else "✗"

            table.add_row(str(idx), resolver_name, status, registered)

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
            console.print(f"[green]✓ Schema saved to {output}[/green]")
        else:
            console.print(Panel(RichJSON.from_data(schema_data), title="Schema"))

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for ContentDownload CLI."""
    app()


if __name__ == "__main__":
    main()
