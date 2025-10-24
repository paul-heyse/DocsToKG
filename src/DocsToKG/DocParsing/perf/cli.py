# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.perf.cli",
#   "purpose": "Typer CLI for DocParsing performance monitoring.",
#   "sections": [
#     {
#       "id": "app",
#       "name": "app",
#       "anchor": "variable-app",
#       "kind": "data"
#     },
#     {
#       "id": "run-command",
#       "name": "run",
#       "anchor": "function-run",
#       "kind": "function"
#     },
#     {
#       "id": "compare-command",
#       "name": "compare",
#       "anchor": "function-compare",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Typer CLI for DocParsing performance monitoring."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from DocsToKG.DocParsing.env import detect_data_root
from DocsToKG.DocParsing.perf.baseline import compare_metrics, load_baseline
from DocsToKG.DocParsing.perf.fixtures import FixturePaths, prepare_synthetic_html_fixture
from DocsToKG.DocParsing.perf.runner import StageMetrics, run_stage

DEFAULT_STAGES = ("doctags", "chunk", "embed")

app = typer.Typer(
    no_args_is_help=True,
    help="Performance monitoring helpers for DocParsing stages.",
    rich_markup_mode="rich",
)


def _stage_commands(fixture: FixturePaths, *, workers: int | None = None) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {
        "doctags": [
            "doctags",
            "--mode",
            "html",
            "--input-dir",
            str(fixture.html_dir),
            "--output-dir",
            str(fixture.doctags_dir),
            "--resume",
            "--no-force",
        ],
        "chunk": [
            "chunk",
            "--input-dir",
            str(fixture.doctags_dir),
            "--output-dir",
            str(fixture.chunks_dir),
            "--resume",
            "--no-force",
        ],
        "embed": [
            "embed",
            "--chunks-dir",
            str(fixture.chunks_dir),
            "--out-dir",
            str(fixture.vectors_dir),
            "--backend",
            "bm25",
            "--format",
            "jsonl",
            "--resume",
            "--no-force",
        ],
    }

    if workers:
        for stage in ("doctags", "chunk", "embed"):
            commands[stage].extend(["--workers", str(workers)])
    return commands


@app.command()
def run(
    stages: Annotated[
        list[str] | None,
        typer.Option(
            "--stage",
            help="Stages to execute (repeat for multiple).",
        ),
    ] = None,
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Override the DocsToKG data root."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Directory receiving profiling artifacts."),
    ] = None,
    documents: Annotated[
        int,
        typer.Option("--documents", min=1, max=1000, help="Number of fixture documents."),
    ] = 5,
    profile: Annotated[
        bool,
        typer.Option("--profile/--no-profile", help="Enable cProfile for each stage."),
    ] = True,
    workers: Annotated[
        int | None,
        typer.Option("--workers", help="Override worker count for each stage."),
    ] = None,
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Optional baseline JSON file for regression checks."),
    ] = None,
    wall_threshold: Annotated[
        float,
        typer.Option("--wall-threshold", help="Allowed wall-time regression as fraction."),
    ] = 0.15,
    cpu_threshold: Annotated[
        float,
        typer.Option("--cpu-threshold", help="Allowed CPU regression as fraction."),
    ] = 0.15,
    rss_threshold: Annotated[
        float,
        typer.Option("--rss-threshold", help="Allowed RSS regression as fraction."),
    ] = 0.20,
) -> None:
    """Execute DocParsing stages and record profiling artifacts."""

    selected_stages = stages or list(DEFAULT_STAGES)
    if not selected_stages:
        typer.secho("[red]No stages selected for profiling[/red]", err=True)
        raise typer.Exit(code=1)

    resolved_data_root = data_root or detect_data_root()
    fixture = prepare_synthetic_html_fixture(data_root=resolved_data_root, documents=documents)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    profiles_root = output_dir or (fixture.root / "runs")
    run_dir = profiles_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    commands = _stage_commands(fixture, workers=workers)

    typer.echo(f"[dim]Profiling output: {run_dir}[/dim]")

    results: list[StageMetrics] = []
    for stage in selected_stages:
        stage_command = commands.get(stage)
        if not stage_command:
            typer.secho(f"[yellow]Skipping unknown stage {stage}[/yellow]")
            continue
        typer.echo(f"[cyan]→ Executing {stage}: {' '.join(stage_command)}[/cyan]")
        metrics = run_stage(
            stage=stage,
            command=stage_command,
            output_dir=run_dir,
            profile=profile,
            env={"DOCSTOKG_DATA_ROOT": str(resolved_data_root)},
            extra={"documents": documents},
        )
        results.append(metrics)

    summary_path = run_dir / "summary.json"
    summary_payload = {
        "timestamp": timestamp,
        "data_root": str(resolved_data_root),
        "stages": [metric.to_json() for metric in results],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if baseline:
        baseline_metrics = load_baseline(baseline)
        comparison = compare_metrics(
            current=results,
            baseline=baseline_metrics,
            wall_threshold=wall_threshold,
            cpu_threshold=cpu_threshold,
            rss_threshold=rss_threshold,
        )
        comparison_path = run_dir / "comparison.json"
        comparison_path.write_text(json.dumps(comparison.to_json(), indent=2), encoding="utf-8")
        if comparison.regressions:
            typer.secho("[red]Performance regressions detected:[/red]", err=True)
            for regression in comparison.regressions:
                typer.secho(f"  • {regression}", err=True)
            raise typer.Exit(code=2)
        if comparison.improvements:
            typer.secho("[green]Improvements detected:[/green]")
            for improvement in comparison.improvements:
                typer.secho(f"  • {improvement}")
        if comparison.unchanged:
            typer.echo(
                "[dim]No significant changes for stages: "
                + ", ".join(sorted(comparison.unchanged))
            )


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="Baseline JSON file.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate run summary JSON.")],
    wall_threshold: Annotated[
        float,
        typer.Option("--wall-threshold", help="Allowed wall-time regression as fraction."),
    ] = 0.15,
    cpu_threshold: Annotated[
        float,
        typer.Option("--cpu-threshold", help="Allowed CPU regression as fraction."),
    ] = 0.15,
    rss_threshold: Annotated[
        float,
        typer.Option("--rss-threshold", help="Allowed RSS regression as fraction."),
    ] = 0.20,
) -> None:
    """Compare two profiling runs and report regressions."""

    baseline_metrics = load_baseline(baseline)
    candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
    stages_payload = candidate_payload.get("stages", [])
    current_metrics: list[StageMetrics] = []
    for stage_payload in stages_payload:
        metric = StageMetrics(
            stage=stage_payload["stage"],
            command=stage_payload.get("command", []),
            wall_time_s=float(stage_payload["wall_time_s"]),
            cpu_time_s=float(stage_payload["cpu_time_s"]),
            max_rss_bytes=(
                int(stage_payload["max_rss_bytes"])
                if stage_payload.get("max_rss_bytes") is not None
                else None
            ),
            exit_code=int(stage_payload["exit_code"]),
            timestamp=datetime.fromisoformat(stage_payload["timestamp"]),
            stdout_path=Path(stage_payload["stdout_path"]),
            stderr_path=Path(stage_payload["stderr_path"]),
            profile_path=(
                Path(stage_payload["profile_path"]) if stage_payload.get("profile_path") else None
            ),
            collapsed_profile_path=(
                Path(stage_payload["collapsed_profile_path"])
                if stage_payload.get("collapsed_profile_path")
                else None
            ),
            extra=stage_payload.get("extra", {}),
        )
        current_metrics.append(metric)

    comparison = compare_metrics(
        current=current_metrics,
        baseline=baseline_metrics,
        wall_threshold=wall_threshold,
        cpu_threshold=cpu_threshold,
        rss_threshold=rss_threshold,
    )

    typer.echo(json.dumps(comparison.to_json(), indent=2))
    if comparison.regressions:
        typer.secho("[red]Regressions detected[/red]", err=True)
        raise typer.Exit(code=2)


__all__ = ["app", "run", "compare"]
