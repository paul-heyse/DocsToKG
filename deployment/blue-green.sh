#!/bin/bash
set -e

# Blue-Green Deployment Script
# Orchestrates zero-downtime deployment with traffic switching

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${1:-latest}"
ENVIRONMENT="${2:-prod}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get current active environment (blue or green)
get_active_env() {
    local config_file="$SCRIPT_DIR/lb-config.txt"
    if [ -f "$config_file" ]; then
        cat "$config_file"
    else
        echo "blue"
    fi
}

# Get standby environment
get_standby_env() {
    local active=$(get_active_env)
    if [ "$active" = "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Health check for service
health_check() {
    local service=$1
    local max_retries=10
    local retry=0

    log_info "Health checking $service..."
    while [ $retry -lt $max_retries ]; do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            log_info "$service is healthy"
            return 0
        fi
        retry=$((retry + 1))
        log_warn "Health check attempt $retry/$max_retries failed, retrying..."
        sleep 5
    done

    log_error "$service failed health check after $max_retries attempts"
    return 1
}

# Deploy to standby environment
deploy_standby() {
    local standby=$(get_standby_env)
    log_info "Deploying version $VERSION to $standby environment..."

    # Build image
    log_info "Building Docker image for version $VERSION..."
    docker build -t docstokg/ontology-download:$VERSION \
        -f "$PROJECT_ROOT/Dockerfile" \
        "$PROJECT_ROOT"

    # Start standby services
    log_info "Starting $standby services..."
    docker-compose -f "$SCRIPT_DIR/docker-compose.$ENVIRONMENT.yml" up -d \
        --no-deps --build ontology-download

    # Wait for service to be ready
    sleep 10
}

# Run smoke tests on standby
run_smoke_tests() {
    log_info "Running smoke tests on standby environment..."

    # Test basic endpoints
    local endpoints=(
        "/health"
        "/metrics"
        "/api/version"
    )

    for endpoint in "${endpoints[@]}"; do
        log_info "Testing endpoint: $endpoint"
        if ! curl -sf http://localhost:8000$endpoint >/dev/null 2>&1; then
            log_error "Smoke test failed for endpoint: $endpoint"
            return 1
        fi
    done

    log_info "All smoke tests passed"
    return 0
}

# Switch traffic to standby
switch_traffic() {
    local active=$(get_active_env)
    local standby=$(get_standby_env)

    log_info "Switching traffic from $active to $standby..."

    # Update load balancer configuration
    echo "$standby" > "$SCRIPT_DIR/lb-config.txt"

    # Reload nginx
    docker-compose -f "$SCRIPT_DIR/docker-compose.$ENVIRONMENT.yml" exec nginx nginx -s reload

    log_info "Traffic switched to $standby"
}

# Monitor deployment
monitor_deployment() {
    local duration=${1:-300}  # 5 minutes default
    local check_interval=10
    local elapsed=0

    log_info "Monitoring deployment for $duration seconds..."

    while [ $elapsed -lt $duration ]; do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            log_info "Health check passed"
        else
            log_error "Health check failed"
            return 1
        fi

        # Check error rate from metrics
        local error_rate=$(curl -s http://localhost:9090/api/v1/query?query='rate(http_requests_total{status=~"5.."}[5m])' | grep -oP '(?<="value":\[.*?,")\d+(\.\d+)?')
        if [ ! -z "$error_rate" ] && [ $(echo "$error_rate > 0.05" | bc) -eq 1 ]; then
            log_error "High error rate detected: $error_rate"
            return 1
        fi

        elapsed=$((elapsed + check_interval))
        sleep $check_interval
    done

    log_info "Deployment monitoring completed successfully"
    return 0
}

# Rollback function
rollback() {
    local previous=$(get_active_env)
    log_warn "Rolling back to $previous..."

    # Switch traffic back
    local standby=$(get_standby_env)
    echo "$previous" > "$SCRIPT_DIR/lb-config.txt"
    docker-compose -f "$SCRIPT_DIR/docker-compose.$ENVIRONMENT.yml" exec nginx nginx -s reload

    log_info "Rollback completed"
}

# Main deployment flow
main() {
    log_info "Starting blue-green deployment for version $VERSION"

    # Pre-deployment checks
    log_info "Running pre-deployment checks..."
    if ! docker-compose -f "$SCRIPT_DIR/docker-compose.$ENVIRONMENT.yml" exec ontology-download curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        log_warn "Current service is unhealthy, proceeding with caution"
    fi

    # Deploy to standby
    if ! deploy_standby; then
        log_error "Deployment to standby failed"
        return 1
    fi

    # Health check standby
    if ! health_check "standby"; then
        log_error "Standby health check failed"
        rollback
        return 1
    fi

    # Run smoke tests
    if ! run_smoke_tests; then
        log_error "Smoke tests failed"
        rollback
        return 1
    fi

    # Switch traffic
    switch_traffic

    # Monitor deployment
    if ! monitor_deployment; then
        log_error "Monitoring detected issues"
        rollback
        return 1
    fi

    log_info "Blue-green deployment completed successfully"
    return 0
}

# Error handler
trap 'log_error "Deployment failed"; rollback; exit 1' ERR

# Run main deployment
main
exit $?
