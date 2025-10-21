"""Prometheus metrics HTTP endpoint for security gates.

Provides a simple HTTP server that exposes gate metrics for Prometheus scraping.

Usage:
    from DocsToKG.OntologyDownload.policy.prometheus_endpoint import start_metrics_server

    # Start metrics server on port 8000
    start_metrics_server(port=8000)

    # Prometheus will scrape: http://localhost:8000/metrics
"""

import logging
import threading
from typing import Optional

try:
    from prometheus_client import generate_latest, REGISTRY
    from prometheus_client.exposition import HTTPServer, make_wsgi_app
except ImportError:
    # Fallback if prometheus_client not available
    REGISTRY = None  # type: ignore
    HTTPServer = None  # type: ignore
    make_wsgi_app = None  # type: ignore


logger = logging.getLogger(__name__)

# Global reference to the metrics server thread
_metrics_server_thread: Optional[threading.Thread] = None
_metrics_server_instance: Optional[HTTPServer] = None
_metrics_server_running = False


def start_metrics_server(host: str = "0.0.0.0", port: int = 8000) -> bool:
    """Start Prometheus metrics HTTP server in background thread.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)

    Returns:
        True if server started successfully, False otherwise
    """
    global _metrics_server_thread, _metrics_server_instance, _metrics_server_running

    if REGISTRY is None or HTTPServer is None:
        logger.warning("prometheus_client not available - metrics server disabled")
        return False

    if _metrics_server_running:
        logger.info(f"Metrics server already running on {host}:{port}")
        return True

    try:
        # Create metrics server
        app = make_wsgi_app(REGISTRY)
        _metrics_server_instance = HTTPServer((host, port), app)

        # Start in background thread
        _metrics_server_thread = threading.Thread(
            target=_metrics_server_instance.serve_forever,
            daemon=True,
            name="PrometheusMetricsServer"
        )
        _metrics_server_thread.start()
        _metrics_server_running = True

        logger.info(f"✅ Prometheus metrics server started on http://{host}:{port}/metrics")
        return True

    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        _metrics_server_running = False
        return False


def stop_metrics_server() -> bool:
    """Stop Prometheus metrics server.

    Returns:
        True if server was running and stopped, False otherwise
    """
    global _metrics_server_instance, _metrics_server_running

    if not _metrics_server_running or _metrics_server_instance is None:
        logger.info("Metrics server not running")
        return False

    try:
        _metrics_server_instance.shutdown()
        _metrics_server_running = False
        logger.info("✅ Prometheus metrics server stopped")
        return True
    except Exception as e:
        logger.error(f"Error stopping metrics server: {e}")
        return False


def get_metrics() -> str:
    """Get current metrics in Prometheus format.

    Returns:
        Metrics as string in Prometheus text exposition format
    """
    if REGISTRY is None:
        return "# Prometheus not available\n"

    try:
        return generate_latest(REGISTRY).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return f"# Error generating metrics: {e}\n"


def is_metrics_server_running() -> bool:
    """Check if metrics server is running.

    Returns:
        True if metrics server is running, False otherwise
    """
    return _metrics_server_running


__all__ = [
    "start_metrics_server",
    "stop_metrics_server",
    "get_metrics",
    "is_metrics_server_running",
]
