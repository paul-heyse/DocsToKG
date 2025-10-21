#!/bin/bash

# Circuit Breaker Deployment Verification Script
# Version: 1.0
# Purpose: Comprehensive pre-deployment and post-deployment checks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNED=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✅${NC} $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}❌${NC} $1"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠️${NC}  $1"
    ((CHECKS_WARNED++))
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_result() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    if [ $CHECKS_FAILED -eq 0 ]; then
        echo -e "${GREEN}✅ ALL CHECKS PASSED${NC}"
        echo "   Passed: $CHECKS_PASSED | Warned: $CHECKS_WARNED"
    else
        echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
        echo "   Passed: $CHECKS_PASSED | Failed: $CHECKS_FAILED | Warned: $CHECKS_WARNED"
        return 1
    fi
    echo "═══════════════════════════════════════════════════════════"
}

# 1. Environment Checks
check_environment() {
    print_header "1. ENVIRONMENT VERIFICATION"

    # Python version
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        check_pass "Python installed: $PYTHON_VERSION"
    else
        check_fail "Python not found in PATH"
        return 1
    fi

    # .venv existence
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        check_pass ".venv directory exists"
    else
        check_fail ".venv directory not found"
        return 1
    fi

    # .venv/bin/python
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        check_pass ".venv/bin/python executable"
    else
        check_fail ".venv/bin/python not executable"
        return 1
    fi

    # Disk space
    DISK_AVAILABLE=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$DISK_AVAILABLE" -gt 1048576 ]; then  # 1GB in KB
        check_pass "Disk space available: $(numfmt --to=iec-i --suffix=B $((DISK_AVAILABLE*1024)))"
    else
        check_warn "Low disk space: $(numfmt --to=iec-i --suffix=B $((DISK_AVAILABLE*1024)))"
    fi
}

# 2. Module Checks
check_modules() {
    print_header "2. MODULE VERIFICATION"

    local modules=(
        "breakers.py"
        "breakers_loader.py"
        "sqlite_cooldown_store.py"
        "networking_breaker_listener.py"
        "cli_breakers.py"
        "cli_breaker_advisor.py"
        "breaker_advisor.py"
        "breaker_autotune.py"
    )

    for module in "${modules[@]}"; do
        if [ -f "$PROJECT_ROOT/src/DocsToKG/ContentDownload/$module" ]; then
            check_pass "Module exists: $module"
        else
            check_fail "Module missing: $module"
        fi
    done

    # Configuration YAML
    if [ -f "$PROJECT_ROOT/src/DocsToKG/ContentDownload/config/breakers.yaml" ]; then
        check_pass "Configuration YAML exists"
    else
        check_fail "Configuration YAML missing"
    fi
}

# 3. Import Checks
check_imports() {
    print_header "3. IMPORT VERIFICATION"

    cd "$PROJECT_ROOT"

    if "$PROJECT_ROOT/.venv/bin/python" -c "from DocsToKG.ContentDownload.breakers import BreakerRegistry, BreakerConfig" 2>/dev/null; then
        check_pass "BreakerRegistry imports"
    else
        check_fail "BreakerRegistry import failed"
    fi

    if "$PROJECT_ROOT/.venv/bin/python" -c "from DocsToKG.ContentDownload.breakers_loader import load_breaker_config" 2>/dev/null; then
        check_pass "breakers_loader imports"
    else
        check_fail "breakers_loader import failed"
    fi

    if "$PROJECT_ROOT/.venv/bin/python" -c "from DocsToKG.ContentDownload.sqlite_cooldown_store import SQLiteCooldownStore" 2>/dev/null; then
        check_pass "SQLiteCooldownStore imports"
    else
        check_fail "SQLiteCooldownStore import failed"
    fi

    if "$PROJECT_ROOT/.venv/bin/python" -c "from DocsToKG.ContentDownload.breaker_advisor import BreakerAdvisor, HostMetrics" 2>/dev/null; then
        check_pass "BreakerAdvisor imports"
    else
        check_fail "BreakerAdvisor import failed"
    fi

    if "$PROJECT_ROOT/.venv/bin/python" -c "from DocsToKG.ContentDownload.breaker_autotune import BreakerAutoTuner" 2>/dev/null; then
        check_pass "BreakerAutoTuner imports"
    else
        check_fail "BreakerAutoTuner import failed"
    fi
}

# 4. Configuration Checks
check_configuration() {
    print_header "4. CONFIGURATION VERIFICATION"

    if "$PROJECT_ROOT/.venv/bin/python" -c "
import yaml
with open('src/DocsToKG/ContentDownload/config/breakers.yaml') as f:
    config = yaml.safe_load(f)
assert 'defaults' in config, 'Missing defaults section'
assert 'advanced' in config, 'Missing advanced section'
assert 'hosts' in config, 'Missing hosts section'
print(f'Hosts configured: {len(config[\"hosts\"])}')
" 2>/dev/null; then
        check_pass "Configuration YAML syntax valid"
    else
        check_fail "Configuration YAML invalid"
    fi

    # Check for host entries
    if "$PROJECT_ROOT/.venv/bin/python" -c "
import yaml
with open('src/DocsToKG/ContentDownload/config/breakers.yaml') as f:
    config = yaml.safe_load(f)
assert len(config.get('hosts', {})) > 0, 'No hosts configured'
" 2>/dev/null; then
        check_pass "At least one host configured"
    else
        check_fail "No hosts configured"
    fi
}

# 5. Linting Checks
check_linting() {
    print_header "5. LINTING VERIFICATION"

    if command -v ruff &> /dev/null; then
        cd "$PROJECT_ROOT"
        if ruff check src/DocsToKG/ContentDownload/breakers.py \
                      src/DocsToKG/ContentDownload/breakers_loader.py \
                      src/DocsToKG/ContentDownload/sqlite_cooldown_store.py \
                      2>/dev/null | grep -q "All checks passed"; then
            check_pass "Ruff linting passed"
        else
            check_warn "Ruff linting has warnings (non-blocking)"
        fi
    else
        check_warn "ruff not found (skipping)"
    fi
}

# 6. Test Files
check_test_files() {
    print_header "6. TEST FILES VERIFICATION"

    local test_files=(
        "tests/content_download/test_breakers_core.py"
        "tests/content_download/test_breakers_networking.py"
        "tests/content_download/test_cli_breakers.py"
        "tests/content_download/test_breaker_advisor.py"
    )

    for test_file in "${test_files[@]}"; do
        if [ -f "$PROJECT_ROOT/$test_file" ]; then
            TEST_COUNT=$(grep -c "def test_" "$PROJECT_ROOT/$test_file" || echo "0")
            check_pass "Test file exists: $test_file ($TEST_COUNT tests)"
        else
            check_fail "Test file missing: $test_file"
        fi
    done
}

# 7. Directory Checks
check_directories() {
    print_header "7. DIRECTORY VERIFICATION"

    # Create breaker store directory
    BREAKER_DIR="/var/run/docstokg/breakers"
    if [ -d "$BREAKER_DIR" ]; then
        check_pass "Breaker store directory exists: $BREAKER_DIR"
    else
        if mkdir -p "$BREAKER_DIR" 2>/dev/null; then
            check_pass "Breaker store directory created: $BREAKER_DIR"
        else
            check_warn "Cannot create breaker store directory (will create on first run)"
        fi
    fi
}

# 8. Permission Checks
check_permissions() {
    print_header "8. PERMISSION VERIFICATION"

    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        check_pass "Python executable has execute permission"
    else
        check_fail "Python executable missing execute permission"
    fi

    if [ -r "$PROJECT_ROOT/src/DocsToKG/ContentDownload/breakers.py" ]; then
        check_pass "Breaker modules readable"
    else
        check_fail "Breaker modules not readable"
    fi
}

# 9. Deployment Readiness
check_deployment_readiness() {
    print_header "9. DEPLOYMENT READINESS SUMMARY"

    if [ $CHECKS_FAILED -eq 0 ]; then
        check_pass "All critical checks passed"
        check_pass "Code is deployment-ready"
    else
        check_fail "Some critical checks failed - do not deploy"
    fi
}

# Main execution
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     Circuit Breaker Deployment Verification Script         ║"
    echo "║     Version: 1.0 | Status: Production-Ready               ║"
    echo "╚════════════════════════════════════════════════════════════╝"

    check_environment || {
        print_result
        exit 1
    }
    check_modules
    check_imports
    check_configuration
    check_linting
    check_test_files
    check_directories
    check_permissions
    check_deployment_readiness

    print_result
}

# Run main
main "$@"
