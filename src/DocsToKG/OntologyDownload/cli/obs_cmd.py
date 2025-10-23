# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.cli.obs_cmd",
#   "purpose": "Observability CLI commands: tail, stats, export.",
#   "sections": [
#     {
#       "id": "get-duckdb-connection",
#       "name": "_get_duckdb_connection",
#       "anchor": "function-get-duckdb-connection",
#       "kind": "function"
#     },
#     {
#       "id": "format-table",
#       "name": "_format_table",
#       "anchor": "function-format-table",
#       "kind": "function"
#     },
#     {
#       "id": "obs-tail",
#       "name": "obs_tail",
#       "anchor": "function-obs-tail",
#       "kind": "function"
#     },
#     {
#       "id": "obs-stats",
#       "name": "obs_stats",
#       "anchor": "function-obs-stats",
#       "kind": "function"
#     },
#     {
#       "id": "obs-export",
#       "name": "obs_export",
#       "anchor": "function-obs-export",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Observability CLI commands: tail, stats, export.

Commands query the shared DuckDB catalog configured by the ontology
downloader so analytics operate on the same event store as event
emitters.

Provides:
- obs tail - Stream recent events in real-time
- obs stats - Show aggregated statistics
- obs export - Export events to various formats
"""

import logging
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

import typer

from DocsToKG.OntologyDownload.database import DatabaseConfiguration
from DocsToKG.OntologyDownload.settings import get_default_config

from DocsToKG.OntologyDownload.observability.queries import (
    get_query,
    list_queries,
    query_summary,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="obs",
    help="Observability commands backed by the shared DuckDB event catalog",
    short_help="View events and stats",
)


# ============================================================================
# Helper Functions
# ============================================================================


def _get_duckdb_connection():
    """Connect to the configured DuckDB catalog used by event emitters."""

    try:
        import duckdb
    except ImportError:  # pragma: no cover - dependency guard
        typer.echo(
            "❌ DuckDB not installed. Install with: pip install duckdb",
            err=True,
        )
        raise typer.Exit(code=1)

    config = get_default_config(copy=True)
    db_settings = config.defaults.db

    db_config = DatabaseConfiguration(
        db_path=db_settings.path,
        readonly=True,
        enable_locks=db_settings.writer_lock,
        threads=db_settings.threads,
    )

    db_path = db_config.db_path
    if db_path is None:
        typer.echo("❌ DuckDB catalog path is not configured", err=True)
        raise typer.Exit(code=1)

    if not db_path.exists():
        typer.echo(
            f"❌ DuckDB catalog not found at {db_path}. Emit events before querying.",
            err=True,
        )
        raise typer.Exit(code=1)

    connect_config: dict[str, object] = {}
    if db_config.threads is not None:
        connect_config["threads"] = db_config.threads
    if db_config.memory_limit is not None:
        connect_config["memory_limit"] = db_config.memory_limit

    try:
        connection = duckdb.connect(
            str(db_path),
            read_only=True,
            config=connect_config or None,
        )
    except Exception as exc:  # pragma: no cover - duckdb provides rich errors
        typer.echo(
            f"❌ Failed to open DuckDB catalog at {db_path}: {exc}",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    if db_config.enable_object_cache:
        try:
            connection.execute("PRAGMA enable_object_cache;")
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to enable DuckDB object cache", exc_info=True)

    return connection


def _format_table(query_result: Sequence[Sequence[Any]], headers: list[str]) -> str:
    """Format query result as table (simple version).

    Args:
        query_result: Sequence of result tuples
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
    level: str | None = typer.Option(
        None,
        "--level",
        "-l",
        help="Filter by level (INFO, WARN, ERROR)",
    ),
    event_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by event type (e.g., 'net.request')",
    ),
    service: str | None = typer.Option(
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
    con = None
    try:
        con = _get_duckdb_connection()

        # Build query
        query = "SELECT ts, type, level, service, run_id FROM events"
        conditions: list[str] = []
        params: list[Any] = []

        if level:
            conditions.append("level = ?")
            params.append(level)
        if event_type:
            conditions.append("type LIKE ?")
            params.append(f"{event_type}%")
        if service:
            conditions.append("service = ?")
            params.append(service)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY ts DESC LIMIT ?"
        params.append(count)

        cursor = con.execute(query, params)
        rows = cursor.fetchall()
        headers = ["ts", "type", "level", "service", "run_id"]

        if json_output:
            df = _rows_to_dataframe(rows, headers)
            typer.echo(df.to_json(orient="records", date_format="iso"))
        else:
            table = _format_table(rows, headers)
            typer.echo(table)

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)
    finally:
        if con is not None:
            con.close()


# ============================================================================
# stats command
# ============================================================================


@app.command(name="stats")
def obs_stats(
    query_name: str | None = typer.Argument(
        None,
        help=(
            "Name of stock query to run (e.g., 'net_latency_distribution' or "
            "'ratelimit_pressure')"
        ),
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
                typer.echo(f"  {name:<24} - {description}")
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

        with closing(_get_duckdb_connection()) as con:
            cursor = con.execute(query)
            columns = cursor.description or []

            if json_output:
                df = cursor.df()
                typer.echo(df.to_json(orient="records"))
            else:
                result = cursor.fetchall()
                headers = [col[0] for col in columns]
                table = _format_table(result, headers)
                typer.echo(table)

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
    level: str | None = typer.Option(
        None,
        "--level",
        "-l",
        help="Filter by level (INFO, WARN, ERROR)",
    ),
    event_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by event type",
    ),
    since: str | None = typer.Option(
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
    con = None
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
        conditions: list[str] = []
        params: list[Any] = []

        if level:
            conditions.append("level = ?")
            params.append(level)
        if event_type:
            conditions.append("type LIKE ?")
            params.append(f"{event_type}%")
        if since:
            conditions.append("ts >= ?")
            params.append(since)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        # Export
        con = _get_duckdb_connection()
        cursor = con.execute(query, params)
        description = cursor.description or []
        headers = [col[0] for col in description]
        rows = cursor.fetchall()
        df = _rows_to_dataframe(rows, headers)

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

        with closing(_get_duckdb_connection()) as con:
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

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)
    finally:
        if con is not None:
            con.close()


def _rows_to_dataframe(rows: Sequence[Sequence[Any]], headers: list[str]):
    """Return a pandas DataFrame for the given rows and headers."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - aligns with existing behaviour
        typer.echo(
            "❌ pandas is required for JSON and export formatting. Install with: pip install pandas",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    return pd.DataFrame(rows, columns=headers)


__all__ = ["app", "obs_tail", "obs_stats", "obs_export"]
