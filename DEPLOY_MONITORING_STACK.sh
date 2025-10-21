#!/bin/bash

##############################################################################
# DEPLOY_MONITORING_STACK.sh
# 
# Automated deployment script for Prometheus, Grafana, and alert configuration
# for the OntologyDownload security gates monitoring system.
#
# Usage: ./DEPLOY_MONITORING_STACK.sh [start|stop|restart|status]
# Default: start
##############################################################################

set -euo pipefail

# Configuration
PROMETHEUS_VERSION="2.48.0"
GRAFANA_VERSION="10.2.0"
NODE_EXPORTER_VERSION="1.7.0"

INSTALL_DIR="${HOME}/monitoring"
PROMETHEUS_DIR="${INSTALL_DIR}/prometheus-${PROMETHEUS_VERSION}.linux-amd64"
GRAFANA_DIR="${INSTALL_DIR}/grafana-${GRAFANA_VERSION}"
NODE_EXPORTER_DIR="${INSTALL_DIR}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64"

PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
NODE_EXPORTER_PORT=9100

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create installation directory
setup_directories() {
    log_info "Setting up directories..."
    mkdir -p "${INSTALL_DIR}"
    mkdir -p "${INSTALL_DIR}/data/prometheus"
    mkdir -p "${INSTALL_DIR}/data/grafana"
    mkdir -p "${INSTALL_DIR}/config"
    log_success "Directories created"
}

# Download and install Prometheus
install_prometheus() {
    log_info "Installing Prometheus ${PROMETHEUS_VERSION}..."
    
    if [ -d "${PROMETHEUS_DIR}" ]; then
        log_warning "Prometheus already installed"
        return
    fi
    
    cd "${INSTALL_DIR}"
    
    wget -q "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
    tar xzf "prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
    rm "prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
    
    log_success "Prometheus installed"
}

# Download and install Grafana
install_grafana() {
    log_info "Installing Grafana ${GRAFANA_VERSION}..."
    
    if [ -d "${GRAFANA_DIR}" ]; then
        log_warning "Grafana already installed"
        return
    fi
    
    cd "${INSTALL_DIR}"
    
    wget -q "https://github.com/grafana/grafana/releases/download/v${GRAFANA_VERSION}/grafana-${GRAFANA_VERSION}.linux-x64.tar.gz"
    tar xzf "grafana-${GRAFANA_VERSION}.linux-x64.tar.gz"
    rm "grafana-${GRAFANA_VERSION}.linux-x64.tar.gz"
    
    log_success "Grafana installed"
}

# Download and install Node Exporter
install_node_exporter() {
    log_info "Installing Node Exporter ${NODE_EXPORTER_VERSION}..."
    
    if [ -d "${NODE_EXPORTER_DIR}" ]; then
        log_warning "Node Exporter already installed"
        return
    fi
    
    cd "${INSTALL_DIR}"
    
    wget -q "https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz"
    tar xzf "node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz"
    rm "node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz"
    
    log_success "Node Exporter installed"
}

# Configure Prometheus
configure_prometheus() {
    log_info "Configuring Prometheus..."
    
    cat > "${INSTALL_DIR}/config/prometheus.yml" << 'PROMETHEUS_CONFIG'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'docstokg-gates'

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files:
  - 'alert_rules.yml'

scrape_configs:
  - job_name: 'gates'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
PROMETHEUS_CONFIG

    log_success "Prometheus configured"
}

# Configure alert rules
configure_alerts() {
    log_info "Configuring alert rules..."
    
    cat > "${INSTALL_DIR}/config/alert_rules.yml" << 'ALERT_RULES'
groups:
- name: gate_security_alerts
  interval: 30s
  rules:

  - alert: GateHighRejectionRate
    expr: |
      (
        rate(gate_invocations_total{outcome="reject"}[5m]) /
        rate(gate_invocations_total[5m])
      ) > 0.05
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High rejection rate detected on {{ $labels.gate }}"
      description: "{{ $labels.gate }} rejection rate is {{ $value | humanizePercentage }}"

  - alert: URLGateHighHostDenials
    expr: |
      rate(gate_errors_total{gate="url_gate", error_code="E_HOST_DENY"}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Potential network attack: URL gate blocking {{ $value | humanize }}/sec"

  - alert: ExtractionGateZipBombDetection
    expr: |
      rate(gate_errors_total{gate="extraction_gate", error_code=~"E_BOMB_RATIO|E_ENTRY_RATIO"}[5m]) > 0.05
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Zip bomb attempts detected"

  - alert: FilesystemGateTraversalDetection
    expr: |
      rate(gate_errors_total{gate="filesystem_gate", error_code="E_TRAVERSAL"}[5m]) > 0.02
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Path traversal attacks detected"

  - alert: GateLatencyHigh
    expr: |
      histogram_quantile(0.99, rate(gate_execution_ms_bucket[5m])) > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "{{ $labels.gate }} P99 latency elevated"

  - alert: GateUnavailable
    expr: |
      increase(gate_invocations_total[5m]) == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "{{ $labels.gate }} is not being called"

  - alert: DBBoundaryViolationAttempt
    expr: |
      increase(gate_errors_total{gate="db_boundary_gate"}[1m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Database boundary violation attempt"
ALERT_RULES

    log_success "Alert rules configured"
}

# Configure Grafana
configure_grafana() {
    log_info "Configuring Grafana..."
    
    # Create custom configuration
    cat > "${GRAFANA_DIR}/conf/custom.ini" << 'GRAFANA_CONFIG'
[server]
http_port = 3000
protocol = http

[database]
type = sqlite3
path = data/grafana.db

[security]
admin_user = admin
admin_password = admin

[users]
allow_sign_up = false

[auth.anonymous]
enabled = false
GRAFANA_CONFIG

    log_success "Grafana configured"
}

# Start services
start_services() {
    log_info "Starting services..."
    
    # Start Node Exporter
    log_info "Starting Node Exporter on port ${NODE_EXPORTER_PORT}..."
    nohup "${NODE_EXPORTER_DIR}/node_exporter" > "${INSTALL_DIR}/node_exporter.log" 2>&1 &
    echo $! > "${INSTALL_DIR}/node_exporter.pid"
    sleep 2
    log_success "Node Exporter started (PID: $(cat ${INSTALL_DIR}/node_exporter.pid))"
    
    # Start Prometheus
    log_info "Starting Prometheus on port ${PROMETHEUS_PORT}..."
    cd "${PROMETHEUS_DIR}"
    nohup ./prometheus \
        --config.file="${INSTALL_DIR}/config/prometheus.yml" \
        --storage.tsdb.path="${INSTALL_DIR}/data/prometheus" \
        --web.console.templates=consoles \
        --web.console.libraries=console_libraries \
        > "${INSTALL_DIR}/prometheus.log" 2>&1 &
    echo $! > "${INSTALL_DIR}/prometheus.pid"
    sleep 2
    log_success "Prometheus started (PID: $(cat ${INSTALL_DIR}/prometheus.pid))"
    
    # Start Grafana
    log_info "Starting Grafana on port ${GRAFANA_PORT}..."
    cd "${GRAFANA_DIR}"
    nohup ./bin/grafana-server \
        --config="${GRAFANA_DIR}/conf/custom.ini" \
        > "${INSTALL_DIR}/grafana.log" 2>&1 &
    echo $! > "${INSTALL_DIR}/grafana.pid"
    sleep 2
    log_success "Grafana started (PID: $(cat ${INSTALL_DIR}/grafana.pid))"
    
    log_success "All services started successfully"
}

# Stop services
stop_services() {
    log_info "Stopping services..."
    
    for pid_file in "${INSTALL_DIR}"/{node_exporter,prometheus,grafana}.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                rm "$pid_file"
                log_success "Stopped process with PID $pid"
            fi
        fi
    done
    
    log_success "All services stopped"
}

# Check service status
check_status() {
    log_info "Checking service status..."
    
    for name in "Prometheus" "Grafana" "Node Exporter"; do
        case $name in
            "Prometheus")
                url="http://localhost:${PROMETHEUS_PORT}"
                ;;
            "Grafana")
                url="http://localhost:${GRAFANA_PORT}"
                ;;
            "Node Exporter")
                url="http://localhost:${NODE_EXPORTER_PORT}"
                ;;
        esac
        
        if curl -s "$url" >/dev/null 2>&1; then
            log_success "$name is running at $url"
        else
            log_error "$name is not responding at $url"
        fi
    done
}

# Display access information
show_access_info() {
    cat << 'ACCESS_INFO'

╔═══════════════════════════════════════════════════════════════════════════╗
║                   MONITORING STACK DEPLOYMENT COMPLETE                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

📊 ACCESS INFORMATION:

1. Prometheus
   URL: http://localhost:9090
   Status: http://localhost:9090/-/healthy
   Metrics: http://localhost:9090/api/v1/targets

2. Grafana
   URL: http://localhost:3000
   Username: admin
   Password: admin (CHANGE THIS!)

3. Node Exporter
   Metrics: http://localhost:9100/metrics

📈 NEXT STEPS:

1. Import Dashboard
   - Go to Grafana: http://localhost:3000
   - Click "+" → "Import"
   - Paste dashboard JSON from MONITORING_AND_DASHBOARDS.md

2. Configure Data Source
   - Settings → Data Sources → Add
   - URL: http://localhost:9090
   - Save & Test

3. Set Up Notifications
   - Alerts → Notification channels
   - Configure Slack/PagerDuty/Email

4. Monitor Metrics
   - Gates metrics should appear in Prometheus
   - Verify endpoint: http://localhost:8000/metrics

📋 LOGS:
   - Prometheus: ~/monitoring/prometheus.log
   - Grafana: ~/monitoring/grafana.log
   - Node Exporter: ~/monitoring/node_exporter.log

⚠️  SECURITY:
   - Change Grafana default password immediately
   - Enable HTTPS for production
   - Configure authentication/TLS
   - Restrict network access

🔧 COMMANDS:
   ./DEPLOY_MONITORING_STACK.sh start     - Start services
   ./DEPLOY_MONITORING_STACK.sh stop      - Stop services
   ./DEPLOY_MONITORING_STACK.sh restart   - Restart services
   ./DEPLOY_MONITORING_STACK.sh status    - Check status

ACCESS_INFO
}

# Main execution
main() {
    local action="${1:-start}"
    
    case "$action" in
        start)
            setup_directories
            install_prometheus
            install_grafana
            install_node_exporter
            configure_prometheus
            configure_alerts
            configure_grafana
            start_services
            sleep 3
            check_status
            show_access_info
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 1
            start_services
            sleep 3
            check_status
            ;;
        status)
            check_status
            ;;
        *)
            log_error "Unknown action: $action"
            echo "Usage: $0 [start|stop|restart|status]"
            exit 1
            ;;
    esac
}

main "$@"
