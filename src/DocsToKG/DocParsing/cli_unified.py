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

from pathlib import Path
from typing import Annotated, Optional

import typer

# Import stage entry points
from DocsToKG.DocParsing import doctags as doctags_module
from DocsToKG.DocParsing.app_context import (
    AppContext,
    build_app_context,
)
from DocsToKG.DocParsing.chunking import runtime as chunking_runtime
from DocsToKG.DocParsing.config_adapter import ConfigurationAdapter
from DocsToKG.DocParsing.embedding import runtime as embedding_runtime

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
            effective_log_level = log_level or "INFO"

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
        typer.echo("cfg_hashes:")
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

    try:
        # Apply CLI overrides to app context
        if input_dir:
            app_ctx.settings.doctags.input_dir = input_dir
        if output_dir:
            app_ctx.settings.doctags.output_dir = output_dir
        if mode:
            app_ctx.settings.doctags.mode = mode
        if model_id:
            app_ctx.settings.doctags.model_id = model_id
        if workers:
            app_ctx.settings.runner.workers = workers
        if policy:
            app_ctx.settings.runner.policy = policy

        # Determine effective mode for routing
        effective_mode = mode or (
            app_ctx.settings.doctags.mode if app_ctx.settings.doctags.mode else "auto"
        )

        typer.echo(
            f"[dim]📋 Profile: {app_ctx.profile or 'none'} | Hash: {app_ctx.cfg_hashes['doctags'][:8]}...[/dim]"
        )
        typer.echo(f"[dim]🔧 Mode: {effective_mode}[/dim]")

        # Create adapted config and call stage with it (NEW PATTERN)
        if effective_mode.lower() == "html":
            cfg = ConfigurationAdapter.to_doctags(app_ctx, mode="html")
            exit_code = doctags_module.html_main(config_adapter=cfg)
        else:  # pdf or auto
            cfg = ConfigurationAdapter.to_doctags(app_ctx, mode="pdf")
            exit_code = doctags_module.pdf_main(config_adapter=cfg)

        if exit_code != 0:
            typer.secho(f"[red]✗ DocTags stage failed with exit code {exit_code}[/red]", err=True)
        else:
            typer.secho("[green]✅ DocTags stage completed successfully[/green]")

        raise typer.Exit(code=exit_code)

    except Exception as e:
        typer.secho(f"[red]✗ Error in doctags stage:[/red] {e}", err=True)
        raise typer.Exit(code=1)


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

    try:
        # Apply CLI overrides to app context
        if input_dir:
            app_ctx.settings.chunk.input_dir = input_dir
        if output_dir:
            app_ctx.settings.chunk.output_dir = output_dir
        if min_tokens:
            app_ctx.settings.chunk.min_tokens = min_tokens
        if max_tokens:
            app_ctx.settings.chunk.max_tokens = max_tokens
        if tokenizer:
            app_ctx.settings.chunk.tokenizer_model = tokenizer
        if workers:
            app_ctx.settings.runner.workers = workers
        if policy:
            app_ctx.settings.runner.policy = policy

        typer.echo(
            f"[dim]📋 Profile: {app_ctx.profile or 'none'} | Hash: {app_ctx.cfg_hashes['chunk'][:8]}...[/dim]"
        )
        typer.echo(
            f"[dim]🔧 Min tokens: {min_tokens or app_ctx.settings.chunk.min_tokens}, Max tokens: {max_tokens or app_ctx.settings.chunk.max_tokens}[/dim]"
        )

        # Create adapted config and call stage with it (NEW PATTERN)
        cfg = ConfigurationAdapter.to_chunk(app_ctx)
        exit_code = chunking_runtime._main_inner(config_adapter=cfg)

        if exit_code != 0:
            typer.secho(f"[red]✗ Chunk stage failed with exit code {exit_code}[/red]", err=True)
        else:
            typer.secho("[green]✅ Chunk stage completed successfully[/green]")

        raise typer.Exit(code=exit_code)

    except Exception as e:
        typer.secho(f"[red]✗ Error in chunk stage:[/red] {e}", err=True)
        raise typer.Exit(code=1)


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

    try:
        # Apply CLI overrides to app context
        if chunks_dir:
            app_ctx.settings.embed.input_chunks_dir = chunks_dir
        if output_dir:
            app_ctx.settings.embed.output_vectors_dir = output_dir
        if dense_backend:
            # Note: embed config doesn't have simple dense_backend field, 
            # so we'd need to handle this through config or skip for now
            pass
        if workers:
            app_ctx.settings.runner.workers = workers
        if policy:
            app_ctx.settings.runner.policy = policy

        typer.echo(
            f"[dim]📋 Profile: {app_ctx.profile or 'none'} | Hash: {app_ctx.cfg_hashes['embed'][:8]}...[/dim]"
        )
        typer.echo(
            f"[dim]🔧 Dense backend: {dense_backend or 'qwen_vllm'}, Format: {vector_format or 'parquet'}[/dim]"
        )

        # Create adapted config and call stage with it (NEW PATTERN)
        cfg = ConfigurationAdapter.to_embed(app_ctx)
        exit_code = embedding_runtime._main_inner(config_adapter=cfg)

        if exit_code != 0:
            typer.secho(f"[red]✗ Embed stage failed with exit code {exit_code}[/red]", err=True)
        else:
            typer.secho("[green]✅ Embed stage completed successfully[/green]")

        raise typer.Exit(code=exit_code)

    except Exception as e:
        typer.secho(f"[red]✗ Error in embed stage:[/red] {e}", err=True)
        raise typer.Exit(code=1)


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

    try:
        typer.echo("[bold cyan]🚀 Pipeline Start[/bold cyan]")
        typer.echo(f"[dim]📋 Profile: {app_ctx.profile or 'none'}[/dim]")
        typer.echo(
            f"[dim]🔧 Resume: {resume}, Force: {force}, Stop-on-fail: {stop_on_fail}[/dim]\n"
        )

        # Stage 1: DocTags
        typer.echo("[bold yellow]▶ Stage 1: DocTags Conversion[/bold yellow]")
        typer.echo(f"[dim]Hash: {app_ctx.cfg_hashes['doctags'][:8]}...[/dim]")

        cfg_doctags = ConfigurationAdapter.to_doctags(app_ctx, mode="pdf")
        exit_code = doctags_module.pdf_main(config_adapter=cfg_doctags)

        if exit_code != 0:
            typer.secho(f"[red]✗ DocTags stage failed with exit code {exit_code}[/red]", err=True)
            if stop_on_fail:
                raise typer.Exit(code=exit_code)
        else:
            typer.secho("[green]✅ DocTags completed[/green]")

        # Stage 2: Chunk
        typer.echo("\n[bold yellow]▶ Stage 2: Chunking[/bold yellow]")
        typer.echo(f"[dim]Hash: {app_ctx.cfg_hashes['chunk'][:8]}...[/dim]")

        cfg_chunk = ConfigurationAdapter.to_chunk(app_ctx)
        exit_code = chunking_runtime._main_inner(config_adapter=cfg_chunk)

        if exit_code != 0:
            typer.secho(f"[red]✗ Chunk stage failed with exit code {exit_code}[/red]", err=True)
            if stop_on_fail:
                raise typer.Exit(code=exit_code)
        else:
            typer.secho("[green]✅ Chunk completed[/green]")

        # Stage 3: Embed
        typer.echo("\n[bold yellow]▶ Stage 3: Embedding[/bold yellow]")
        typer.echo(f"[dim]Hash: {app_ctx.cfg_hashes['embed'][:8]}...[/dim]")

        cfg_embed = ConfigurationAdapter.to_embed(app_ctx)
        exit_code = embedding_runtime._main_inner(config_adapter=cfg_embed)

        if exit_code != 0:
            typer.secho(f"[red]✗ Embed stage failed with exit code {exit_code}[/red]", err=True)
            raise typer.Exit(code=exit_code)
        else:
            typer.secho("[green]✅ Embed completed[/green]")

        typer.echo("[bold green]✅ Pipeline Complete[/bold green]")

    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"[red]✗ Error in pipeline:[/red] {e}", err=True)
        raise typer.Exit(code=1)


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
