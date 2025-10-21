"""Observability CLI commands: tail, stats, export.

Provides:
- obs tail - Stream recent events in real-time
- obs stats - Show aggregated statistics
- obs export - Export events to various formats
"""

import logging
from pathlib import Path
from typing import Optional

import typer

from DocsToKG.OntologyDownload.observability.queries import (
    get_query,
    list_queries,
    query_summary,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="obs",
    help="Observability commands: view and analyze events",
    short_help="View events and stats",
)


# ============================================================================
# Helper Functions
# ============================================================================


def _get_duckdb_connection():
    """Get DuckDB connection (stub for now).

    In production, this would connect to the configured DuckDB instance.
    """
    try:
        import duckdb

        return duckdb.connect()
    except ImportError:
        typer.echo(
            "❌ DuckDB not installed. Install with: pip install duckdb",
            err=True,
        )
        raise typer.Exit(code=1)


def _format_table(query_result, headers: list[str]) -> str:
    """Format query result as table (simple version).

    Args:
        query_result: Iterator of result tuples
        headers: Column names

    Returns:
        Formatted table string
    """
    rows = list(query_result)
    if not rows:
        return "(no results)"

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    # Build table
    lines = []

    # Header
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Rows
    for row in rows:
        row_line = " | ".join(f"{str(v):<{w}}" for v, w in zip(row, widths))
        lines.append(row_line)

    return "\n".join(lines)


# ============================================================================
# tail command
# ============================================================================


@app.command(name="tail")
def obs_tail(
    count: int = typer.Option(
        20,
        "--count",
        "-n",
        help="Number of recent events to show",
    ),
    level: Optional[str] = typer.Option(
        None,
        "--level",
        "-l",
        help="Filter by level (INFO, WARN, ERROR)",
    ),
    event_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by event type (e.g., 'net.request')",
    ),
    service: Optional[str] = typer.Option(
        None,
        "--service",
        "-s",
        help="Filter by service",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Stream recent events (tail -f equivalent).

    Shows the N most recent events with optional filtering.
    """
    try:
        con = _get_duckdb_connection()

        # Build query
        query = "SELECT ts, type, level, service, run_id FROM events"
        conditions = []

        if level:
            conditions.append(f"level = '{level}'")
        if event_type:
            conditions.append(f"type LIKE '{event_type}%'")
        if service:
            conditions.append(f"service = '{service}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY ts DESC LIMIT {count}"

        result = con.execute(query).fetchall()

        if json_output:
            # Return as JSON
            rows = con.execute(query).df()
            typer.echo(rows.to_json(orient="records", date_format="iso"))
        else:
            # Return as table
            headers = ["ts", "type", "level", "service", "run_id"]
            table = _format_table(result, headers)
            typer.echo(table)

        con.close()

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


# ============================================================================
# stats command
# ============================================================================


@app.command(name="stats")
def obs_stats(
    query_name: Optional[str] = typer.Argument(
        None,
        help="Name of stock query to run (e.g., 'net_request_p95_latency')",
    ),
    list_all: bool = typer.Option(
        False,
        "--list",
        help="List all available queries",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Show operational statistics using stock queries.

    Run a stock query to answer specific operational questions.
    Use --list to see available queries.
    """
    try:
        if list_all:
            # List all available queries
            summary = query_summary()
            typer.echo("📊 Available stock queries:\n")
            for name, description in sorted(summary.items()):
                typer.echo(f"  {name:<35} - {description}")
            return

        if not query_name:
            typer.echo(
                "❌ Please specify a query name or use --list to see available queries",
                err=True,
            )
            raise typer.Exit(code=1)

        # Run the specified query
        try:
            query = get_query(query_name)
        except KeyError:
            available = ", ".join(list_queries())
            typer.echo(
                f"❌ Query '{query_name}' not found.\nAvailable: {available}",
                err=True,
            )
            raise typer.Exit(code=1)

        con = _get_duckdb_connection()
        result = con.execute(query).fetchall()
        columns = con.execute(query).description

        if json_output:
            df = con.execute(query).df()
            typer.echo(df.to_json(orient="records"))
        else:
            headers = [col[0] for col in columns] if columns else []
            table = _format_table(result, headers)
            typer.echo(table)

        con.close()

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


# ============================================================================
# export command
# ============================================================================


@app.command(name="export")
def obs_export(
    output_path: Path = typer.Argument(
        ...,
        help="Path to export events to (.json, .jsonl, .parquet, .csv)",
    ),
    level: Optional[str] = typer.Option(
        None,
        "--level",
        "-l",
        help="Filter by level (INFO, WARN, ERROR)",
    ),
    event_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by event type",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Filter events since timestamp (ISO 8601)",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        help="Limit number of rows exported",
    ),
) -> None:
    """Export events to a file.

    Supports JSON, JSONL, Parquet, and CSV formats (inferred from extension).
    """
    try:
        # Validate output path
        if not output_path.suffix:
            typer.echo(
                "❌ Output path must have extension (.json, .jsonl, .parquet, .csv)",
                err=True,
            )
            raise typer.Exit(code=1)

        format_type = output_path.suffix.lower()
        if format_type not in [".json", ".jsonl", ".parquet", ".csv"]:
            typer.echo(
                f"❌ Unsupported format: {format_type}. Use .json, .jsonl, .parquet, or .csv",
                err=True,
            )
            raise typer.Exit(code=1)

        # Build query
        query = "SELECT * FROM events"
        conditions = []

        if level:
            conditions.append(f"level = '{level}'")
        if event_type:
            conditions.append(f"type LIKE '{event_type}%'")
        if since:
            conditions.append(f"ts >= '{since}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        if limit:
            query += f" LIMIT {limit}"

        # Export
        con = _get_duckdb_connection()
        df = con.execute(query).df()

        if format_type == ".json":
            df.to_json(output_path, orient="records", date_format="iso")
        elif format_type == ".jsonl":
            df.to_json(output_path, orient="records", lines=True)
        elif format_type == ".parquet":
            df.to_parquet(output_path)
        elif format_type == ".csv":
            df.to_csv(output_path, index=False)

        row_count = len(df)
        typer.echo(
            f"✅ Exported {row_count} events to {output_path}",
        )

        con.close()

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


__all__ = ["app", "obs_tail", "obs_stats", "obs_export"]
