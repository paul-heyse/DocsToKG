"""CLI commands for monitoring infrastructure.

Provides commands to start/stop Prometheus metrics server and configure Grafana.

Usage:
    python -m DocsToKG.OntologyDownload.cli monitor start
    python -m DocsToKG.OntologyDownload.cli monitor status
    python -m DocsToKG.OntologyDownload.cli monitor grafana-config --output-dir /etc/grafana/provisioning
"""

import logging
from typing import Optional

import typer

from DocsToKG.OntologyDownload.policy.prometheus_endpoint import (
    start_metrics_server,
    stop_metrics_server,
    is_metrics_server_running,
    get_metrics,
)

try:
    from DocsToKG.OntologyDownload.policy.grafana_provisioning import (
        export_grafana_config,
        get_gates_dashboard_config,
    )
except ImportError:
    export_grafana_config = None
    get_gates_dashboard_config = None

logger = logging.getLogger(__name__)
monitor_app = typer.Typer(help="Monitoring infrastructure commands")


@monitor_app.command()
def start(
    host: str = typer.Option("0.0.0.0", help="Host to bind metrics server to"),
    port: int = typer.Option(8000, help="Port to bind metrics server to"),
) -> None:
    """Start Prometheus metrics server.

    Example:
        python -m DocsToKG.OntologyDownload.cli monitor start --port 8000
    """
    typer.echo(f"🚀 Starting Prometheus metrics server on {host}:{port}...")

    if start_metrics_server(host=host, port=port):
        typer.echo(f"✅ Metrics server started")
        typer.echo(f"📊 Access metrics at: http://{host}:{port}/metrics")
    else:
        typer.echo("❌ Failed to start metrics server", err=True)
        raise typer.Exit(1)


@monitor_app.command()
def stop() -> None:
    """Stop Prometheus metrics server.

    Example:
        python -m DocsToKG.OntologyDownload.cli monitor stop
    """
    typer.echo("🛑 Stopping Prometheus metrics server...")

    if stop_metrics_server():
        typer.echo("✅ Metrics server stopped")
    else:
        typer.echo("⚠️  Metrics server not running", err=True)


@monitor_app.command()
def status() -> None:
    """Check metrics server status.

    Example:
        python -m DocsToKG.OntologyDownload.cli monitor status
    """
    running = is_metrics_server_running()

    if running:
        typer.echo("✅ Prometheus metrics server is RUNNING")
        typer.echo("📊 Access metrics at: http://0.0.0.0:8000/metrics")
    else:
        typer.echo("⏸️  Prometheus metrics server is STOPPED")

    # Show sample metrics
    typer.echo("\n📈 Current Metrics (sample):")
    metrics = get_metrics()
    # Show first 20 lines of metrics
    lines = metrics.split('\n')[:20]
    for line in lines:
        if line and not line.startswith('#'):
            typer.echo(f"  {line}")


@monitor_app.command()
def grafana_config(
    output_dir: str = typer.Option(
        "/etc/grafana/provisioning",
        help="Output directory for Grafana provisioning files"
    ),
) -> None:
    """Export Grafana provisioning configuration.

    Creates datasource and dashboard configuration files for Grafana.

    Example:
        python -m DocsToKG.OntologyDownload.cli monitor grafana-config \\
            --output-dir ./grafana-provisioning
    """
    if export_grafana_config is None:
        typer.echo("❌ Grafana provisioning support not available", err=True)
        raise typer.Exit(1)

    typer.echo(f"📋 Exporting Grafana configuration to {output_dir}...")

    try:
        export_grafana_config(output_dir=output_dir)
        typer.echo("✅ Grafana configuration exported successfully")
        typer.echo(f"\n📂 Configuration files created in: {output_dir}")
        typer.echo("\nTo use with Grafana:")
        typer.echo("  1. Copy files to /etc/grafana/provisioning")
        typer.echo("  2. Restart Grafana: sudo systemctl restart grafana-server")
        typer.echo("  3. Access dashboard at: http://localhost:3000")
    except Exception as e:
        typer.echo(f"❌ Error exporting Grafana configuration: {e}", err=True)
        raise typer.Exit(1)


@monitor_app.command()
def dashboard() -> None:
    """Show Grafana dashboard configuration.

    Outputs the dashboard JSON configuration.

    Example:
        python -m DocsToKG.OntologyDownload.cli monitor dashboard > dashboard.json
    """
    if get_gates_dashboard_config is None:
        typer.echo("❌ Grafana support not available", err=True)
        raise typer.Exit(1)

    import json
    config = get_gates_dashboard_config()
    typer.echo(json.dumps(config, indent=2))


__all__ = ["monitor_app"]
