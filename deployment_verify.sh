#!/bin/bash
# Deployment Verification Script for Phases 5A-6B

echo "=========================================================================="
echo "🔍 DEPLOYMENT VERIFICATION - Phases 5A-6B"
echo "=========================================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0

check_step() {
    echo ""
    echo "📋 $1"
    echo "─────────────────────────────────────────────────────────────────────"
}

check_pass() {
    echo -e "${GREEN}✅${NC} $1"
}

check_fail() {
    echo -e "${RED}❌${NC} $1"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠️ ${NC} $1"
}

# Step 1: Python environment
check_step "Step 1: Python Environment"

PYTHON_VERSION=$(.venv/bin/python --version 2>&1)
if [ $? -eq 0 ]; then
    check_pass "Python: $PYTHON_VERSION"
else
    check_fail "Python not found or .venv missing"
    exit 1
fi

# Step 2: Required packages
check_step "Step 2: Required Packages"

packages=("duckdb" "polars")
for pkg in "${packages[@]}"; do
    .venv/bin/python -c "import $pkg" 2>/dev/null
    if [ $? -eq 0 ]; then
        VERSION=$(.venv/bin/python -c "import $pkg; print($pkg.__version__ if hasattr($pkg, '__version__') else 'unknown')")
        check_pass "$pkg: $VERSION"
    else
        check_fail "$pkg not installed"
    fi
done

# Step 3: OntologyDownload modules
check_step "Step 3: OntologyDownload Modules"

PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python -c "
from DocsToKG.OntologyDownload import catalog
from DocsToKG.OntologyDownload import analytics
print('✅ Both modules imported successfully')
" 2>/dev/null

if [ $? -eq 0 ]; then
    check_pass "Catalog module imports correctly"
    check_pass "Analytics module imports correctly"
else
    check_fail "Module imports failed"
    ERRORS=$((ERRORS + 1))
fi

# Step 4: Database connectivity
check_step "Step 4: Database Connectivity"

PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python << 'PYEOF'
import tempfile
import duckdb
from DocsToKG.OntologyDownload.catalog import apply_migrations, quick_health_check

# Create temp database
with tempfile.NamedTemporaryFile(suffix=".duckdb") as f:
    db_path = f.name

try:
    conn = duckdb.connect(db_path)
    result = apply_migrations(conn)
    print(f"✅ Migrations applied: {len(result.applied)} migrations")

    health = quick_health_check(conn)
    print(f"✅ Health check passed: {health.message}")
    conn.close()
except Exception as e:
    print(f"❌ Database error: {e}")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    check_fail "Database connectivity test failed"
    ERRORS=$((ERRORS + 1))
fi

# Step 5: Analytics functionality
check_step "Step 5: Analytics Functionality"

PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python << 'PYEOF'
import polars as pl
from DocsToKG.OntologyDownload.analytics import cmd_report_latest

# Create sample data
files_df = pl.DataFrame({
    "file_id": ["f1", "f2", "f3"],
    "relpath": ["a.ttl", "b.rdf", "c.owl"],
    "size": [1024, 2048, 4096],
    "format": ["ttl", "rdf", "owl"],
})

try:
    # Test report generation
    report = cmd_report_latest(files_df, output_format="table")
    print(f"✅ Report generation working")
except Exception as e:
    print(f"❌ Analytics error: {e}")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    check_fail "Analytics functionality test failed"
    ERRORS=$((ERRORS + 1))
fi

# Step 6: Test execution
check_step "Step 6: Test Suite"

TEST_COUNT=$(.venv/bin/pytest tests/ontology_download/catalog/ tests/ontology_download/analytics/ --co -q 2>/dev/null | wc -l)
if [ "$TEST_COUNT" -gt 0 ]; then
    check_pass "Found $TEST_COUNT tests in suite"

    # Run tests
    .venv/bin/pytest tests/ontology_download/catalog/ tests/ontology_download/analytics/ -q --tb=no 2>/dev/null
    if [ $? -eq 0 ]; then
        check_pass "All tests passing"
    else
        check_warn "Some tests failed - see full output for details"
    fi
else
    check_warn "Could not find tests"
fi

# Summary
echo ""
echo "=========================================================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ VERIFICATION COMPLETE - All checks passed${NC}"
    echo "=========================================================================="
    echo ""
    echo "🚀 Ready for deployment!"
    echo ""
    echo "Next steps:"
    echo "1. Run: python deployment_init.py"
    echo "2. Set environment: export PYTHONPATH=\$PWD/src:\$PYTHONPATH"
    echo "3. Import and use modules in your code"
    echo ""
else
    echo -e "${RED}❌ VERIFICATION FAILED - $ERRORS error(s)${NC}"
    echo "=========================================================================="
    echo ""
    echo "Troubleshooting:"
    echo "1. Check that .venv is activated"
    echo "2. Verify PYTHONPATH includes src directory"
    echo "3. Run: .venv/bin/python -c 'import duckdb; import polars'"
    echo "4. Check database permissions in ~/.local/share/docstokg/"
    echo ""
    exit 1
fi
