"""CLI commands for configuration inspection and validation.

Provides high-level commands to:
- Print merged configuration after precedence application
- Validate configuration files
- Export JSON Schema for tooling integration
- Show default values
- Display specific config sections
"""

from __future__ import annotations

import json
import logging
from typing import Optional

try:
    import typer
    from typing_extensions import Literal
except ImportError:
    typer = None  # type: ignore
    Literal = None  # type: ignore

from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
from DocsToKG.ContentDownload.config.loader import load_config
from DocsToKG.ContentDownload.config.schema import export_config_schema

LOGGER = logging.getLogger(__name__)


def cmd_config_print_merged(
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (YAML/JSON)",
    ),
) -> None:
    """
    Print merged configuration after precedence application.
    
    Shows the final configuration after file → environment → CLI precedence.
    Useful for debugging configuration issues.
    
    Example:
        contentdownload config print-merged -c config.yaml
    """
    try:
        cfg = load_config(config_file)
        output = cfg.model_dump(mode="json")
        typer.echo(json.dumps(output, indent=2))
    except Exception as e:
        typer.secho(f"❌ Error loading config: {e}", fg="red", err=True)
        raise typer.Exit(1)


def cmd_config_validate(
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path to validate",
    ),
) -> None:
    """
    Validate configuration file.
    
    Checks config against Pydantic v2 models and reports any validation errors.
    Exit code 0 if valid, 1 if invalid.
    
    Example:
        contentdownload config validate -c config.yaml
    """
    try:
        cfg = load_config(config_file)
        typer.secho(f"✅ Config is valid", fg="green")
        typer.echo(f"   Schema version: 1")
        typer.echo(f"   Config hash: {cfg.config_hash()[:16]}...")
    except Exception as e:
        typer.secho(f"❌ Config validation failed:", fg="red", err=True)
        typer.secho(f"   {e}", fg="red", err=True)
        raise typer.Exit(1)


def cmd_config_export_schema(
    output_file: Optional[str] = typer.Option(
        "config-schema.json",
        "--output",
        "-o",
        help="Output file path for JSON Schema",
    ),
) -> None:
    """
    Export JSON Schema for IDE/tooling integration.
    
    Generates JSON Schema that can be used by:
    - IDE/editors for autocomplete and validation
    - JSON Schema validators
    - Documentation generators
    
    Example:
        contentdownload config export-schema -o schema.json
    """
    try:
        export_config_schema(output_file)
        typer.secho(f"✅ Schema exported to {output_file}", fg="green")
    except Exception as e:
        typer.secho(f"❌ Error exporting schema: {e}", fg="red", err=True)
        raise typer.Exit(1)


def cmd_config_defaults() -> None:
    """
    Show default configuration values.
    
    Displays the default ContentDownloadConfig with all subsystem defaults.
    
    Example:
        contentdownload config defaults
    """
    try:
        cfg = ContentDownloadConfig()
        output = cfg.model_dump(mode="json")
        typer.echo(json.dumps(output, indent=2))
    except Exception as e:
        typer.secho(f"❌ Error generating defaults: {e}", fg="red", err=True)
        raise typer.Exit(1)


def cmd_config_show(
    section: str = typer.Argument(
        ...,
        help="Config section to display (http, hishel, robots, download, telemetry, queue, orchestrator, storage, catalog, resolvers)",
    ),
) -> None:
    """
    Display specific configuration section.
    
    Shows detailed configuration for a single subsystem.
    
    Example:
        contentdownload config show hishel
        contentdownload config show resolvers
    """
    try:
        cfg = ContentDownloadConfig()
        output = cfg.model_dump(mode="json")
        
        if section not in output:
            valid_sections = list(output.keys())
            typer.secho(
                f"❌ Unknown section: {section}",
                fg="red",
                err=True,
            )
            typer.secho(f"   Valid sections: {', '.join(valid_sections)}", err=True)
            raise typer.Exit(1)
        
        section_data = output[section]
        typer.echo(json.dumps(section_data, indent=2))
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(1)


def register_config_commands(app: typer.Typer) -> None:
    """
    Register config commands with Typer application.
    
    Args:
        app: Typer application instance
        
    Example:
        app = typer.Typer()
        register_config_commands(app)
    """
    if typer is None:
        LOGGER.warning("Typer not installed; config commands unavailable")
        return
    
    config_app = typer.Typer(help="Configuration inspection and validation")
    config_app.command()(cmd_config_print_merged)
    config_app.command()(cmd_config_validate)
    config_app.command()(cmd_config_export_schema)
    config_app.command()(cmd_config_defaults)
    config_app.command()(cmd_config_show)
    
    app.add_typer(config_app, name="config", help="Config commands")


__all__ = [
    "cmd_config_print_merged",
    "cmd_config_validate",
    "cmd_config_export_schema",
    "cmd_config_defaults",
    "cmd_config_show",
    "register_config_commands",
]
