"""Typer-based CLI for ContentDownload with Pydantic v2 configuration."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from DocsToKG.ContentDownload.bootstrap import BootstrapConfig, run_from_config
from DocsToKG.ContentDownload.config import (
    ContentDownloadConfig,
    export_config_schema,
    load_config,
    validate_config_file,
)
from DocsToKG.ContentDownload.http_session import HttpConfig
from DocsToKG.ContentDownload.resolvers.registry_v2 import (
    build_resolvers,
    get_registry,
)

# Feature flags support
try:
    from DocsToKG.ContentDownload.config.feature_flags import (
        FeatureFlag,
        get_feature_flags,
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

        bootstrap_cfg = _build_bootstrap_config(cfg, resolvers)
        run_result = run_from_config(
            bootstrap_cfg,
            artifacts=None,
            dry_run=dry_run,
        )

        console.print(
            Panel(
                f"[bold green]Pipeline ready[/bold green]\n"
                f"Success: {run_result.success_count}\n"
                f"Skipped: {run_result.skip_count}\n"
                f"Errors: {run_result.error_count}",
                title="Execution Summary",
            )
        )

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
# Helpers
# ============================================================================


def _build_bootstrap_config(
    cfg: ContentDownloadConfig,
    resolvers: list[Any],
) -> BootstrapConfig:
    """Bridge ContentDownloadConfig to runtime BootstrapConfig."""

    telemetry_paths: dict[str, Path] = {}
    tele_cfg = cfg.telemetry

    if "csv" in tele_cfg.sinks:
        csv_path = Path(tele_cfg.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_paths["csv"] = csv_path
        telemetry_paths.setdefault("last_attempt", csv_path.with_name("last.csv"))

    if "jsonl" in tele_cfg.sinks:
        manifest_path = Path(tele_cfg.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_paths["manifest_index"] = manifest_path.with_name("index.json")
        telemetry_paths["summary"] = manifest_path.with_name("summary.json")
        telemetry_paths["sqlite"] = manifest_path.with_name("manifest.sqlite")

    http_cfg = HttpConfig(
        user_agent=cfg.http.user_agent,
        mailto=cfg.http.mailto,
        timeout_connect_s=cfg.http.timeout_connect_s,
        timeout_read_s=cfg.http.timeout_read_s,
        pool_connections=cfg.http.max_keepalive_connections,
        pool_maxsize=cfg.http.max_connections,
        verify_tls=cfg.http.verify_tls,
        proxies=dict(cfg.http.proxies) if cfg.http.proxies else None,
    )

    resolver_registry: dict[str, Any] = {}
    for resolver in resolvers:
        name = getattr(resolver, "name", None) or getattr(resolver, "_registry_name", None)
        if not name:
            name = f"resolver_{len(resolver_registry) + 1}"
        resolver_registry[name] = resolver

    return BootstrapConfig(
        http=http_cfg,
        telemetry_paths=telemetry_paths or None,
        resolver_registry=resolver_registry,
        resolver_retry_configs={},
        policy_knobs={},
    )


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
