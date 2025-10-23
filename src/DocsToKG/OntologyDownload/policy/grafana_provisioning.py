# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.policy.grafana_provisioning",
#   "purpose": "Grafana provisioning configuration for security gates monitoring.",
#   "sections": [
#     {
#       "id": "get-prometheus-datasource-config",
#       "name": "get_prometheus_datasource_config",
#       "anchor": "function-get-prometheus-datasource-config",
#       "kind": "function"
#     },
#     {
#       "id": "get-gates-dashboard-config",
#       "name": "get_gates_dashboard_config",
#       "anchor": "function-get-gates-dashboard-config",
#       "kind": "function"
#     },
#     {
#       "id": "get-alert-rules-config",
#       "name": "get_alert_rules_config",
#       "anchor": "function-get-alert-rules-config",
#       "kind": "function"
#     },
#     {
#       "id": "export-grafana-config",
#       "name": "export_grafana_config",
#       "anchor": "function-export-grafana-config",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Grafana provisioning configuration for security gates monitoring.

Provides configuration objects for Grafana data sources and dashboards.
Can be used to programmatically provision Grafana instances.

Usage:
    from DocsToKG.OntologyDownload.policy.grafana_provisioning import (
        get_prometheus_datasource_config,
        get_gates_dashboard_config,
        export_grafana_config,
    )

    # Export configuration files for Grafana provisioning
    export_grafana_config(output_dir="/etc/grafana/provisioning")
"""

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def get_prometheus_datasource_config(
    prometheus_url: str = "http://localhost:9090",
) -> dict[str, Any]:
    """Get Prometheus data source configuration for Grafana.

    Args:
        prometheus_url: URL to Prometheus instance

    Returns:
        Data source configuration dictionary
    """
    return {
        "apiVersion": 1,
        "providers": [
            {
                "name": "Prometheus",
                "orgId": 1,
                "folder": "",
                "type": "file",
                "disableDeletion": False,
                "editable": True,
                "options": {"path": "/etc/grafana/provisioning/datasources"},
            }
        ],
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": prometheus_url,
                "isDefault": True,
                "editable": True,
                "jsonData": {"timeInterval": "5s"},
            }
        ],
    }


def get_gates_dashboard_config() -> dict[str, Any]:
    """Get Grafana dashboard configuration for security gates.

    Returns:
        Dashboard configuration dictionary
    """
    return {
        "dashboard": {
            "title": "Security Gates - OntologyDownload",
            "tags": ["gates", "security", "monitoring"],
            "timezone": "UTC",
            "schemaVersion": 27,
            "version": 1,
            "refresh": "5s",
            "panels": [
                {
                    "id": 1,
                    "title": "Gate Invocations (Pass/Reject)",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "rate(gate_invocations_total[5m])",
                            "legendFormat": "{{ gate }} - {{ outcome }}",
                        }
                    ],
                },
                {
                    "id": 2,
                    "title": "Gate Latency P95 (milliseconds)",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(gate_execution_ms_bucket[5m]))",
                            "legendFormat": "{{ gate }}",
                        }
                    ],
                },
                {
                    "id": 3,
                    "title": "Gate Pass Rate (%)",
                    "type": "stat",
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
                    "targets": [{"expr": "gate_pass_rate_percent", "legendFormat": "{{ gate }}"}],
                },
                {
                    "id": 4,
                    "title": "Total Errors by Code",
                    "type": "piechart",
                    "gridPos": {"h": 8, "w": 12, "x": 6, "y": 8},
                    "targets": [{"expr": "gate_errors_total", "legendFormat": "{{ error_code }}"}],
                },
                {
                    "id": 5,
                    "title": "Current Gate Latency (ms)",
                    "type": "gauge",
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 12},
                    "targets": [{"expr": "gate_current_latency_ms", "legendFormat": "{{ gate }}"}],
                    "fieldConfig": {"defaults": {"max": 50, "min": 0, "unit": "ms"}},
                },
                {
                    "id": 6,
                    "title": "Rejection Rate by Gate",
                    "type": "bargauge",
                    "gridPos": {"h": 4, "w": 12, "x": 12, "y": 12},
                    "targets": [
                        {
                            "expr": "rate(gate_invocations_total{outcome='reject'}[5m]) / rate(gate_invocations_total[5m])",
                            "legendFormat": "{{ gate }}",
                        }
                    ],
                },
                {
                    "id": 7,
                    "title": "Gate Latency Distribution (Heatmap)",
                    "type": "heatmap",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": "rate(gate_execution_ms_bucket[5m])",
                            "legendFormat": "{{ gate }}: {{ le }}",
                        }
                    ],
                },
            ],
        }
    }


def get_alert_rules_config() -> dict[str, Any]:
    """Get Prometheus alert rules configuration for security gates.

    Returns:
        Alert rules configuration dictionary
    """
    return {
        "groups": [
            {
                "name": "gate_security_alerts",
                "interval": "30s",
                "rules": [
                    {
                        "alert": "GateHighRejectionRate",
                        "expr": "rate(gate_invocations_total{outcome='reject'}[5m]) / rate(gate_invocations_total[5m]) > 0.05",
                        "for": "2m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High rejection rate on {{ $labels.gate }}",
                            "description": "{{ $labels.gate }} rejection rate is {{ $value | humanizePercentage }}",
                        },
                    },
                    {
                        "alert": "URLGateHighHostDenials",
                        "expr": "rate(gate_errors_total{gate='url_gate', error_code='E_HOST_DENY'}[5m]) > 0.1",
                        "for": "5m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Potential network attack detected",
                            "description": "URL gate blocking {{ $value | humanize }}/sec",
                        },
                    },
                    {
                        "alert": "ExtractionGateZipBombDetection",
                        "expr": "rate(gate_errors_total{gate='extraction_gate', error_code=~'E_BOMB_RATIO|E_ENTRY_RATIO'}[5m]) > 0.05",
                        "for": "1m",
                        "labels": {"severity": "critical"},
                        "annotations": {"summary": "Zip bomb attempts detected"},
                    },
                    {
                        "alert": "FilesystemGateTraversalDetection",
                        "expr": "rate(gate_errors_total{gate='filesystem_gate', error_code='E_TRAVERSAL'}[5m]) > 0.02",
                        "for": "1m",
                        "labels": {"severity": "critical"},
                        "annotations": {"summary": "Path traversal attacks detected"},
                    },
                    {
                        "alert": "GateLatencyHigh",
                        "expr": "histogram_quantile(0.99, rate(gate_execution_ms_bucket[5m])) > 10",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "{{ $labels.gate }} P99 latency elevated",
                            "description": "Current: {{ $value }}ms",
                        },
                    },
                    {
                        "alert": "DBBoundaryViolationAttempt",
                        "expr": "increase(gate_errors_total{gate='db_boundary_gate'}[1m]) > 0",
                        "for": "0m",
                        "labels": {"severity": "critical"},
                        "annotations": {"summary": "Database boundary violation attempt"},
                    },
                ],
            }
        ]
    }


def export_grafana_config(output_dir: str = "/etc/grafana/provisioning") -> None:
    """Export Grafana provisioning configuration files.

    Args:
        output_dir: Directory to export configuration to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export datasource config
    datasource_dir = output_path / "datasources"
    datasource_dir.mkdir(exist_ok=True)
    datasource_file = datasource_dir / "prometheus.yaml"

    with open(datasource_file, "w") as f:
        f.write("# Auto-generated Prometheus data source configuration\n")
        f.write("apiVersion: 1\n\n")
        f.write("datasources:\n")
        f.write("  - name: 'Prometheus'\n")
        f.write("    type: 'prometheus'\n")
        f.write("    access: 'proxy'\n")
        f.write("    url: 'http://localhost:9090'\n")
        f.write("    isDefault: true\n")
        f.write("    editable: true\n")
        f.write("    jsonData:\n")
        f.write("      timeInterval: '5s'\n")

    print(f"✅ Exported Prometheus data source config to {datasource_file}")

    # Export dashboard config
    dashboard_dir = output_path / "dashboards"
    dashboard_dir.mkdir(exist_ok=True)
    dashboard_file = dashboard_dir / "gates.json"

    dashboard_config = get_gates_dashboard_config()
    with open(dashboard_file, "w") as f:
        json.dump(dashboard_config, f, indent=2)

    print(f"✅ Exported Grafana dashboard config to {dashboard_file}")

    # Export alert rules
    rules_dir = output_path / "rules"
    rules_dir.mkdir(exist_ok=True)
    rules_file = rules_dir / "gate_alerts.yaml"

    alert_rules = get_alert_rules_config()
    if yaml:
        with open(rules_file, "w") as f:
            yaml.dump(alert_rules, f, default_flow_style=False)
    else:
        print(
            f"⚠️ Skipping export of alert rules config to {rules_file} because 'yaml' module is not available."
        )

    print(f"✅ Exported alert rules config to {rules_file}")


__all__ = [
    "get_prometheus_datasource_config",
    "get_gates_dashboard_config",
    "get_alert_rules_config",
    "export_grafana_config",
]
