# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.policy.prometheus_endpoint",
#   "purpose": "Prometheus metrics HTTP endpoint for security gates.",
#   "sections": [
#     {
#       "id": "start-metrics-server",
#       "name": "start_metrics_server",
#       "anchor": "function-start-metrics-server",
#       "kind": "function"
#     },
#     {
#       "id": "stop-metrics-server",
#       "name": "stop_metrics_server",
#       "anchor": "function-stop-metrics-server",
#       "kind": "function"
#     },
#     {
#       "id": "get-metrics",
#       "name": "get_metrics",
#       "anchor": "function-get-metrics",
#       "kind": "function"
#     },
#     {
#       "id": "is-metrics-server-running",
#       "name": "is_metrics_server_running",
#       "anchor": "function-is-metrics-server-running",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Prometheus metrics HTTP endpoint for security gates.

Provides a simple HTTP server that exposes gate metrics for Prometheus scraping.

Usage:
    from DocsToKG.OntologyDownload.policy.prometheus_endpoint import start_metrics_server

    # Start metrics server on port 8000
    start_metrics_server(port=8000)

    # Prometheus will scrape: http://localhost:8000/metrics
"""

import logging

try:
    from prometheus_client import REGISTRY, generate_latest
    from prometheus_client.exposition import (
        ThreadingWSGIServer,
        WSGIRequestHandler,
        make_wsgi_app,
        start_http_server,
    )
except ImportError:
    # Fallback if prometheus_client not available
    REGISTRY = None  # type: ignore
    start_http_server = None  # type: ignore
    make_wsgi_app = None  # type: ignore
    ThreadingWSGIServer = None  # type: ignore
    WSGIRequestHandler = None  # type: ignore


logger = logging.getLogger(__name__)

# Global reference to the metrics server
_metrics_server_instance: ThreadingWSGIServer | None = None
_metrics_server_running = False


def start_metrics_server(host: str = "0.0.0.0", port: int = 8000) -> bool:
    """Start Prometheus metrics HTTP server in background thread.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)

    Returns:
        True if server started successfully, False otherwise
    """
    global _metrics_server_instance, _metrics_server_running

    if REGISTRY is None or start_http_server is None:
        logger.warning("prometheus_client not available - metrics server disabled")
        return False

    if _metrics_server_running:
        logger.info(f"Metrics server already running on {host}:{port}")
        return True

    try:
        # Use prometheus_client's built-in start_http_server
        start_http_server(port, addr=host, registry=REGISTRY)
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
    global _metrics_server_running

    if not _metrics_server_running:
        logger.info("Metrics server not running")
        return False

    # Note: prometheus_client's start_http_server doesn't provide easy shutdown
    # The server runs in a daemon thread and will stop when the main program exits
    logger.warning("Metrics server will stop when main program exits (daemon thread)")
    _metrics_server_running = False
    return True


def get_metrics() -> str:
    """Get current metrics in Prometheus format.

    Returns:
        Metrics as string in Prometheus text exposition format
    """
    if REGISTRY is None:
        return "# Prometheus not available\n"

    try:
        return generate_latest(REGISTRY).decode("utf-8")
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
