"""
Unified Typer CLI for DocParsing with Pydantic Settings integration.

Implements the complete command-line interface orchestrating all DocParsing stages
with typed options, precedence-based configuration (CLI > ENV > profile > defaults),
rich help panels, and built-in introspection commands (config show/diff, inspect).

NAVMAP:
- CLI_ROOT: Root Typer app with global callback
- ENUMS: Validated choice enums for options
- GLOBAL_COMMANDS: config, inspect, all subcommands
- STAGE_COMMANDS: doctags, chunk, embed subcommands
- HELPERS: Settings extraction, error handling, rich output
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import typer
from typing_extensions import Literal

from DocsToKG.DocParsing.app_context import (
    build_app_context,
    AppContext,
)
from DocsToKG.DocParsing.settings import (
    LogLevel,
    LogFormat,
    RunnerPolicy,
    RunnerSchedule,
    RunnerAdaptive,
    DoctagsMode,
    Format,
    DenseBackend,
    TeiCompression,
    AttnBackend,
)

# ============================================================================
# CLI Application Setup
# ============================================================================

app = typer.Typer(
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
    help="[bold]DocParsing[/bold] — Convert documents to chunked embeddings with reproducible configuration.",
)

config_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")
app.add_typer(config_app, name="config", help="Introspect and manage configuration")


# ============================================================================
# Root Callback (Global Options)
# ============================================================================


@app.callback()
def root_callback(
    ctx: typer.Context,
    profile: Annotated[
        Optional[str], typer.Option("--profile", help="Profile name to load from docstokg.toml")
    ] = None,
    data_root: Annotated[
        Optional[Path], typer.Option("--data-root", help="Override base data directory")
    ] = None,
    log_level: Annotated[
        Optional[str], typer.Option("--log-level", help="Logging level (DEBUG|INFO|WARNING|ERROR)")
    ] = None,
    log_format: Annotated[
        Optional[str], typer.Option("--log-format", help="Logging format (console|json)")
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase verbosity (-v=DEBUG, -vv=TRACE)",
        ),
    ] = 0,
    metrics: Annotated[
        bool, typer.Option("--metrics/--no-metrics", help="Enable Prometheus metrics")
    ] = False,
    metrics_port: Annotated[
        int, typer.Option("--metrics-port", help="Prometheus metrics port")
    ] = 9108,
    strict_config: Annotated[
        bool,
        typer.Option(
            "--strict-config/--no-strict-config", help="Treat unknown config keys as errors"
        ),
    ] = True,
) -> None:
    """
    [bold]DocParsing Configuration & Workflow Orchestration[/bold]

    Configure and run document parsing pipelines with reproducible settings.
    Use profiles, environment variables, or CLI flags to customize behavior.

    [bold yellow]Precedence:[/bold yellow] CLI args > ENV vars > profiles > defaults

    [bold yellow]Examples:[/bold yellow]

    [cyan]docparse --profile gpu chunk --min-tokens 256[/cyan]
    Run chunking with GPU profile and override min_tokens

    [cyan]docparse config show --stage embed --format yaml[/cyan]
    Show effective embedding config

    [cyan]docparse --profile local embed --resume[/cyan]
    Resume embedding from local profile
    """
    try:
        # Map verbose count to log level
        if verbose >= 2:
            effective_log_level = "DEBUG"
        elif verbose == 1:
            effective_log_level = "DEBUG"
        else:
            effective_log_level = log_level

        # Build application context with all settings layered
        ctx.obj = build_app_context(
            profile=profile,
            strict_config=strict_config,
            track_sources=False,  # Can be enabled for debugging
            data_root=data_root,
            log_level=effective_log_level,
            log_format=log_format,
            metrics_enabled=metrics,
            metrics_port=metrics_port,
        )

    except Exception as e:
        typer.secho(f"[red]✗ Configuration Error:[/red] {e}", err=True)
        raise typer.Exit(code=1)


# ============================================================================
# Config Commands (Introspection)
# ============================================================================


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    stage: Annotated[
        str,
        typer.Option("--stage", help="Show specific stage (app|runner|doctags|chunk|embed|all)"),
    ] = "all",
    fmt: Annotated[
        str, typer.Option("--format", help="Output format (yaml|json|toml|env)")
    ] = "yaml",
    annotate_source: Annotated[
        bool,
        typer.Option("--annotate-source/--no-annotate-source", help="Show config layer origin"),
    ] = False,
    redact: Annotated[
        bool, typer.Option("--redact/--no-redact", help="Redact sensitive fields")
    ] = True,
) -> None:
    """
    Display effective configuration after layering (profile + ENV + CLI).

    Shows the actual configuration that will be used for execution,
    including the source (default/profile/env/cli) of each setting if requested.
    """
    app_ctx: AppContext = ctx.obj
    if not app_ctx:
        typer.secho("[red]✗ Configuration not initialized[/red]", err=True)
        raise typer.Exit(code=1)

    try:
        settings = app_ctx.settings

        # Redact if requested
        if redact:
            config_dict = settings.model_dump_redacted()
        else:
            config_dict = settings.model_dump()

        # Filter by stage if requested
        if stage != "all":
            if stage not in config_dict:
                typer.secho(f"[red]✗ Unknown stage: {stage}[/red]", err=True)
                raise typer.Exit(code=1)
            config_dict = {stage: config_dict[stage]}

        # Format output
        if fmt == "yaml":
            try:
                import yaml

                output = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
            except ImportError:
                typer.secho("[yellow]⚠ PyYAML not available, falling back to JSON[/yellow]")
                import json

                output = json.dumps(config_dict, indent=2, default=str)
        elif fmt == "json":
            import json

            output = json.dumps(config_dict, indent=2, default=str)
        else:
            typer.secho(f"[red]✗ Unsupported format: {fmt}[/red]", err=True)
            raise typer.Exit(code=1)

        typer.echo(output)

        # Append cfg_hash and profile info
        typer.echo("\n[dim]# Configuration metadata[/dim]")
        typer.echo(f"profile: {app_ctx.profile or 'none'}")
        typer.echo(f"cfg_hashes:")
        for stage_name, hash_val in app_ctx.cfg_hashes.items():
            typer.echo(f"  {stage_name}: {hash_val}")

    except Exception as e:
        typer.secho(f"[red]✗ Error showing config:[/red] {e}", err=True)
        raise typer.Exit(code=1)


@config_app.command("diff")
def config_diff(
    ctx: typer.Context,
    lhs_profile: Annotated[
        str, typer.Option("--lhs-profile", help="Left profile (or 'none')")
    ] = "none",
    rhs_profile: Annotated[
        str, typer.Option("--rhs-profile", help="Right profile (or 'none')")
    ] = "gpu",
    show_hash: Annotated[
        bool, typer.Option("--show-hash/--no-show-hash", help="Show config hashes")
    ] = True,
) -> None:
    """
    Compare two configuration profiles.

    Shows differences between two profiles to help plan configuration changes
    before deploying new settings.

    [bold yellow]Example:[/bold yellow]
    [cyan]docparse config diff --lhs-profile local --rhs-profile gpu[/cyan]
    """
    try:
        # Build two contexts
        lhs_ctx = build_app_context(profile=lhs_profile if lhs_profile != "none" else None)
        rhs_ctx = build_app_context(profile=rhs_profile if rhs_profile != "none" else None)

        # Simple diff: show both configs side by side
        typer.echo(f"[bold]LHS Profile:[/bold] {lhs_profile}")
        typer.echo(lhs_ctx.settings.model_dump_redacted())

        typer.echo(f"\n[bold]RHS Profile:[/bold] {rhs_profile}")
        typer.echo(rhs_ctx.settings.model_dump_redacted())

        if show_hash:
            typer.echo("\n[bold]Config Hashes:[/bold]")
            for stage in ["app", "runner", "doctags", "chunk", "embed"]:
                lhs_hash = lhs_ctx.cfg_hashes.get(stage, "N/A")
                rhs_hash = rhs_ctx.cfg_hashes.get(stage, "N/A")
                match = "✓" if lhs_hash == rhs_hash else "✗"
                typer.echo(f"  {stage}: {lhs_hash} {match} {rhs_hash}")

    except Exception as e:
        typer.secho(f"[red]✗ Error comparing configs:[/red] {e}", err=True)
        raise typer.Exit(code=1)


# ============================================================================
# Placeholder Stage Commands
# ============================================================================


@app.command()
def doctags(
    ctx: typer.Context,
    input_dir: Annotated[
        Optional[Path], typer.Option("--input-dir", help="Input directory for PDF/HTML files")
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option("--output-dir", help="Output directory for DocTags")
    ] = None,
    mode: Annotated[
        Optional[str], typer.Option("--mode", help="Conversion mode (auto|pdf|html)")
    ] = None,
    model_id: Annotated[Optional[str], typer.Option("--model-id", help="DocTags model ID")] = None,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Resume from manifest")
    ] = True,
    force: Annotated[bool, typer.Option("--force/--no-force", help="Force recomputation")] = False,
    # Runner options
    workers: Annotated[
        Optional[int], typer.Option("--workers", help="Max parallel workers")
    ] = None,
    policy: Annotated[
        Optional[str], typer.Option("--policy", help="Execution policy (io|cpu|gpu)")
    ] = None,
) -> None:
    """
    Convert PDF/HTML documents to DocTags.

    Parses raw documents and extracts structured content using
    the Granite DocTags model via vLLM or direct inference.

    [bold yellow]Example:[/bold yellow]
    [cyan]docparse doctags --mode pdf --input-dir Data/PDFs --output-dir Data/DocTags[/cyan]
    """
    app_ctx: AppContext = ctx.obj
    if not app_ctx:
        typer.secho("[red]✗ Configuration not initialized[/red]", err=True)
        raise typer.Exit(code=1)

    typer.secho("[yellow]⚠ doctags stage execution not yet implemented in Phase 2[/yellow]")
    typer.echo(f"[dim]Profile: {app_ctx.profile or 'none'}[/dim]")
    typer.echo(f"[dim]Config hash: {app_ctx.cfg_hashes['doctags']}[/dim]")


@app.command()
def chunk(
    ctx: typer.Context,
    input_dir: Annotated[
        Optional[Path], typer.Option("--in-dir", help="DocTags input directory")
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option("--out-dir", help="Chunks output directory")
    ] = None,
    fmt: Annotated[
        Optional[str], typer.Option("--format", help="Output format (parquet|jsonl)")
    ] = None,
    min_tokens: Annotated[
        Optional[int], typer.Option("--min-tokens", help="Minimum tokens per chunk")
    ] = None,
    max_tokens: Annotated[
        Optional[int], typer.Option("--max-tokens", help="Maximum tokens per chunk")
    ] = None,
    tokenizer: Annotated[
        Optional[str], typer.Option("--tokenizer", help="Tokenizer model ID")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Resume from manifest")
    ] = True,
    force: Annotated[bool, typer.Option("--force/--no-force", help="Force recomputation")] = False,
    # Runner options
    workers: Annotated[
        Optional[int], typer.Option("--workers", help="Max parallel workers")
    ] = None,
    policy: Annotated[
        Optional[str], typer.Option("--policy", help="Execution policy (io|cpu|gpu)")
    ] = None,
) -> None:
    """
    Chunk DocTags into token-aware units.

    Performs structural and token-aware coalescence with deterministic
    span hashing for reproducible chunking across runs.

    [bold yellow]Example:[/bold yellow]
    [cyan]docparse chunk --in-dir Data/DocTags --out-dir Data/Chunks --min-tokens 256[/cyan]
    """
    app_ctx: AppContext = ctx.obj
    if not app_ctx:
        typer.secho("[red]✗ Configuration not initialized[/red]", err=True)
        raise typer.Exit(code=1)

    typer.secho("[yellow]⚠ chunk stage execution not yet implemented in Phase 2[/yellow]")
    typer.echo(f"[dim]Profile: {app_ctx.profile or 'none'}[/dim]")
    typer.echo(f"[dim]Config hash: {app_ctx.cfg_hashes['chunk']}[/dim]")


@app.command()
def embed(
    ctx: typer.Context,
    chunks_dir: Annotated[
        Optional[Path], typer.Option("--chunks-dir", help="Chunks input directory")
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option("--out-dir", help="Vectors output directory")
    ] = None,
    vector_format: Annotated[
        Optional[str], typer.Option("--format", help="Vector format (parquet|jsonl)")
    ] = None,
    dense_backend: Annotated[
        Optional[str],
        typer.Option("--dense-backend", help="Dense backend (qwen_vllm|tei|sentence_transformers)"),
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Resume from manifest")
    ] = True,
    force: Annotated[bool, typer.Option("--force/--no-force", help="Force recomputation")] = False,
    # Runner options
    workers: Annotated[
        Optional[int], typer.Option("--workers", help="Max parallel workers")
    ] = None,
    policy: Annotated[
        Optional[str], typer.Option("--policy", help="Execution policy (io|cpu|gpu)")
    ] = None,
) -> None:
    """
    Generate embeddings for chunks.

    Produces dense (Qwen/vLLM/TEI/ST), sparse (SPLADE),
    and lexical (BM25) embeddings with configurable backends and batching.

    [bold yellow]Example:[/bold yellow]
    [cyan]docparse --profile gpu embed --chunks-dir Data/Chunks --out-dir Data/Vectors[/cyan]
    """
    app_ctx: AppContext = ctx.obj
    if not app_ctx:
        typer.secho("[red]✗ Configuration not initialized[/red]", err=True)
        raise typer.Exit(code=1)

    typer.secho("[yellow]⚠ embed stage execution not yet implemented in Phase 2[/yellow]")
    typer.echo(f"[dim]Profile: {app_ctx.profile or 'none'}[/dim]")
    typer.echo(f"[dim]Config hash: {app_ctx.cfg_hashes['embed']}[/dim]")


@app.command()
def all(
    ctx: typer.Context,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Resume all stages from manifests")
    ] = True,
    force: Annotated[
        bool, typer.Option("--force/--no-force", help="Force recompute all stages")
    ] = False,
    stop_on_fail: Annotated[
        bool, typer.Option("--stop-on-fail/--keep-going", help="Stop on first failure")
    ] = True,
) -> None:
    """
    Run the full pipeline: DocTags → Chunk → Embed.

    Orchestrates all stages sequentially with shared configuration
    and coordinated resume/force behavior.

    [bold yellow]Example:[/bold yellow]
    [cyan]docparse --profile gpu all --resume[/cyan]
    """
    app_ctx: AppContext = ctx.obj
    if not app_ctx:
        typer.secho("[red]✗ Configuration not initialized[/red]", err=True)
        raise typer.Exit(code=1)

    typer.secho("[yellow]⚠ Full pipeline execution not yet implemented in Phase 2[/yellow]")
    typer.echo(f"[dim]Profile: {app_ctx.profile or 'none'}[/dim]")
    typer.echo(f"[dim]Stages: doctags → chunk → embed[/dim]")


@app.command()
def inspect(
    ctx: typer.Context,
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help="Dataset to inspect (chunks|vectors-dense|vectors-sparse|vectors-lexical)",
        ),
    ] = "chunks",
    root: Annotated[Optional[Path], typer.Option("--root", help="Dataset base directory")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows to show (0=all)")] = 0,
) -> None:
    """
    Quickly inspect dataset schema and statistics.

    Shows row counts, file counts, total bytes, and sample records
    without loading entire datasets.

    [bold yellow]Example:[/bold yellow]
    [cyan]docparse inspect --dataset chunks --limit 10[/cyan]
    """
    app_ctx: AppContext = ctx.obj
    if not app_ctx:
        typer.secho("[red]✗ Configuration not initialized[/red]", err=True)
        raise typer.Exit(code=1)

    typer.secho("[yellow]⚠ Dataset inspection not yet implemented in Phase 2[/yellow]")
    typer.echo(f"[dim]Dataset: {dataset}[/dim]")
    typer.echo(f"[dim]Limit: {limit} rows[/dim]")


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        typer.secho("\n[yellow]⚠ Interrupted by user[/yellow]", err=True)
        raise typer.Exit(code=130)
    except Exception as e:
        typer.secho(f"[red]✗ Unexpected error:[/red] {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    main()

__all__ = ["app", "main"]
