# Circuit Breaker Implementation - Deployment Guide

**Version**: 1.0
**Status**: Production-Ready
**Last Updated**: October 21, 2025

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Steps](#deployment-steps)
3. [Migration Guide](#migration-guide)
4. [Verification & Testing](#verification--testing)
5. [Monitoring & Observability](#monitoring--observability)
6. [Rollback Procedures](#rollback-procedures)
7. [Support & Troubleshooting](#support--troubleshooting)

---

## Pre-Deployment Checklist

### Environment Verification

```bash
# 1. Verify Python environment
python -c "import sys; print(f'Python {sys.version}')"

# 2. Check required packages
pip list | grep -E "pybreaker|pyyaml|sqlite3|idna"

# 3. Verify disk space for SQLite cooldown store
df -h /path/to/data

# 4. Check file permissions for .venv and src/
ls -la .venv/bin/python
ls -la src/DocsToKG/ContentDownload/
```

### Code Validation

```bash
# 1. Run all tests
pytest tests/content_download/test_breakers_core.py \
        tests/content_download/test_breakers_networking.py \
        tests/content_download/test_cli_breakers.py \
        tests/content_download/test_breaker_advisor.py \
        -v --tb=short

# 2. Verify linting
ruff check src/DocsToKG/ContentDownload/

# 3. Check configuration YAML
python -c "
import yaml
with open('src/DocsToKG/ContentDownload/config/breakers.yaml') as f:
    config = yaml.safe_load(f)
print(f'✅ Config loads: {len(config.get(\"hosts\", {}))} hosts configured')
"

# 4. Verify module imports
python -c "
from DocsToKG.ContentDownload.breakers import BreakerRegistry, BreakerConfig
from DocsToKG.ContentDownload.breakers_loader import load_breaker_config
from DocsToKG.ContentDownload.sqlite_cooldown_store import SQLiteCooldownStore
print('✅ All core modules import successfully')
"
```

### Configuration Review

- [ ] Review `src/DocsToKG/ContentDownload/config/breakers.yaml`
- [ ] Verify `fail_max` values are appropriate for each host
- [ ] Confirm `reset_timeout_s` aligns with expected service recovery times
- [ ] Check `retry_after_cap_s` is set (default: 900s)
- [ ] Validate rolling window thresholds if enabled
- [ ] Test with `--dry-run` on small workload

---

## Deployment Steps

### Step 1: Code Deployment

```bash
# 1. Backup current state (optional but recommended)
git stash
git tag pre-circuit-breaker-$(date +%Y%m%d)

# 2. Deploy code
git pull origin main  # or your deployment branch

# 3. Verify new files exist
ls -l src/DocsToKG/ContentDownload/breakers.py
ls -l src/DocsToKG/ContentDownload/config/breakers.yaml
ls -l tests/content_download/test_breakers_*.py

# 4. Run health checks
./.venv/bin/ruff check src/DocsToKG/ContentDownload/
```

### Step 2: Database Initialization

```bash
# 1. Create cooldown store directory (if not exists)
mkdir -p /var/run/docstokg/breakers

# 2. SQLite will auto-create database on first use
# (SQLiteCooldownStore handles DDL on initialization)

# 3. Verify permissions
chmod 755 /var/run/docstokg/breakers
```

### Step 3: Configuration Deployment

```bash
# 1. Validate YAML before deployment
python -c "
from DocsToKG.ContentDownload.breakers_loader import load_breaker_config
cfg = load_breaker_config(
    yaml_path='src/DocsToKG/ContentDownload/config/breakers.yaml',
    env={},
)
print(f'✅ Config loaded: {len(cfg.hosts)} hosts')
for host in list(cfg.hosts.keys())[:3]:
    print(f'   • {host}')
"

# 2. (Optional) Set environment variable for custom config location
export DOCSTOKG_BREAKERS_YAML=/path/to/custom/breakers.yaml

# 3. (Optional) Enable circuit breaker debug logging
export DOCSTOKG_BREAKER_DEBUG=1
```

### Step 4: Gradual Rollout

#### Phase 1: Dry-Run (No Changes)

```bash
python -m DocsToKG.ContentDownload.cli \
  --topic "test" \
  --year-start 2024 \
  --year-end 2024 \
  --max 10 \
  --dry-run \
  --out /tmp/test_run \
  2>&1 | head -50
```

#### Phase 2: Small Workload

```bash
python -m DocsToKG.ContentDownload.cli \
  --topic "test" \
  --year-start 2024 \
  --year-end 2024 \
  --max 100 \
  --workers 2 \
  --out runs/content_phase2 \
  --manifest /tmp/phase2_manifest.jsonl
```

#### Phase 3: Monitor & Analyze

```bash
# Check for breaker events
sqlite3 runs/content_phase2/manifest.sqlite3 << 'SQL'
SELECT host, COUNT(*) as events
FROM breaker_events
GROUP BY host
ORDER BY events DESC;
SQL

# Analyze success rates
sqlite3 runs/content_phase2/manifest.sqlite3 << 'SQL'
SELECT
  event_type,
  COUNT(*) as count
FROM breaker_events
GROUP BY event_type;
SQL
```

#### Phase 4: Full Production

```bash
python -m DocsToKG.ContentDownload.cli \
  --topic "your_topic" \
  --year-start 2024 \
  --year-end 2024 \
  --workers 4 \
  --out runs/content_production
```

---

## Migration Guide

### From Legacy Throttle Configuration

**Old (Legacy):**

```yaml
# Old per-host throttle settings (DEPRECATED)
resolver_throttles:
  api.crossref.org: 10/s
  export.arxiv.org: 1/3s
```

**New (Circuit Breaker):**

```yaml
# New circuit breaker configuration
defaults:
  fail_max: 5
  reset_timeout_s: 60
  retry_after_cap_s: 900

hosts:
  api.crossref.org:
    fail_max: 6
    reset_timeout_s: 90
  export.arxiv.org:
    fail_max: 10
    reset_timeout_s: 120

# ALSO use rate limiter for traffic shaping:
# --rate api.crossref.org=10/s,1000/h
# --rate export.arxiv.org=1/3s,100/h
```

### Environment Variables

**If using environment-based config:**

```bash
# Base configuration from YAML
export DOCSTOKG_BREAKERS_YAML=/path/to/breakers.yaml

# Host-specific overrides
export DOCSTOKG_BREAKER_HOSTS="api.example.org=fail_max:3,reset:45"

# Role-specific overrides
export DOCSTOKG_BREAKER_ROLES="api.example.org:metadata=fail_max:2,success_threshold:2"

# Cooldown store location (optional, default: in-memory)
export DOCSTOKG_BREAKER_COOLDOWN_STORE="sqlite:///var/run/docstokg/breakers/cooldowns.db"
```

### CLI Arguments

```bash
# Host-specific breaker tuning
--breaker api.example.org=fail_max:4,reset:90

# Role-specific tuning
--breaker-role api.example.org:metadata=success_threshold:2

# Resolver-specific tuning
--breaker-resolver crossref=fail_max:3

# Default breaker settings
--breaker-defaults fail_max:5,reset:60,retry_after_cap:900

# Configuration file
--breaker-config /path/to/custom/breakers.yaml
```

---

## Verification & Testing

### Test Connectivity

```bash
# Test a single resolver endpoint
python -m DocsToKG.ContentDownload.cli \
  --topic "machine learning" \
  --year-start 2024 \
  --year-end 2024 \
  --max 5 \
  --dry-run \
  --manifest /tmp/test_manifest.jsonl

echo "✅ CLI accepts breaker arguments"
```

### Verify Telemetry

```bash
# Check that breaker_events table exists
sqlite3 runs/content/manifest.sqlite3 ".schema breaker_events"

# Sample recent events
sqlite3 runs/content/manifest.sqlite3 << 'SQL'
SELECT ts, host, event_type, details
FROM breaker_events
ORDER BY ts DESC
LIMIT 10;
SQL
```

### CLI Commands

```bash
# Inspect breaker state
python -m DocsToKG.ContentDownload.cli breaker show

# Show a specific host
python -m DocsToKG.ContentDownload.cli breaker show --host api.crossref.org

# Manually open a breaker (maintenance)
python -m DocsToKG.ContentDownload.cli breaker open api.example.org --seconds 300 --reason "maintenance"

# Close breaker
python -m DocsToKG.ContentDownload.cli breaker close api.example.org

# Analyze telemetry and get tuning recommendations
python -m DocsToKG.ContentDownload.cli breaker-advise --window-s 3600
```

---

## Monitoring & Observability

### Key Metrics to Monitor

```sql
-- Opens per host in last hour
SELECT host, COUNT(*) as opens
FROM breaker_events
WHERE event_type = 'state_change'
  AND details LIKE '%"new_state":"OPEN"%'
  AND ts > (julianday('now') - 1/24)
GROUP BY host
ORDER BY opens DESC;

-- Average open duration
SELECT host, AVG(duration_s) as avg_duration_s
FROM (
  SELECT host,
         (close_ts - open_ts) as duration_s
  FROM breaker_state_transitions
  WHERE old_state = 'OPEN'
) t
GROUP BY host;

-- Half-open success rate
SELECT host,
       SUM(CASE WHEN event_type='success' THEN 1 ELSE 0 END) as successes,
       SUM(CASE WHEN event_type='failure' THEN 1 ELSE 0 END) as failures
FROM breaker_events
WHERE details LIKE '%"state":"HALF_OPEN"%'
GROUP BY host;
```

### Logging

```python
# Check logs for breaker events
grep "breaker" /var/log/docstokg/download.log | head -20

# Filter by host
grep "api.crossref.org.*breaker" /var/log/docstokg/download.log

# Check for errors
grep "BreakerOpenError\|circuit.*open" /var/log/docstokg/download.log
```

### Health Check Script

```bash
#!/bin/bash
echo "Circuit Breaker Deployment Health Check"
echo "========================================"

# 1. Check modules exist
python -c "from DocsToKG.ContentDownload.breakers import BreakerRegistry" && echo "✅ Core module" || echo "❌ Core module"

# 2. Check config loads
python -c "from DocsToKG.ContentDownload.breakers_loader import load_breaker_config; load_breaker_config('src/DocsToKG/ContentDownload/config/breakers.yaml', env={})" && echo "✅ Config loads" || echo "❌ Config loads"

# 3. Check CLI works
python -m DocsToKG.ContentDownload.cli breaker show > /dev/null 2>&1 && echo "✅ CLI works" || echo "❌ CLI works"

# 4. Check telemetry table
[ -f "runs/content/manifest.sqlite3" ] && echo "✅ Telemetry DB present" || echo "⚠️  Telemetry DB not yet created"

echo "Health check complete!"
```

---

## Rollback Procedures

### Quick Rollback (No Data Loss)

```bash
# 1. Stop current downloads
pkill -f "DocsToKG.ContentDownload.cli"

# 2. Revert code
git checkout main~1 src/DocsToKG/ContentDownload/breakers*.py
git checkout main~1 src/DocsToKG/ContentDownload/cli_breaker*.py
git checkout main~1 src/DocsToKG/ContentDownload/*breaker*.py

# 3. No database migration needed - breaker events are additive

# 4. Resume downloads without --breaker args
python -m DocsToKG.ContentDownload.cli \
  --topic "your_topic" \
  --year-start 2024 \
  --year-end 2024 \
  --out runs/content
```

### Full Rollback (With State Reset)

```bash
# 1. Stop downloads
pkill -f "DocsToKG.ContentDownload.cli"

# 2. Backup telemetry (if needed for analysis)
cp runs/content/manifest.sqlite3 runs/content/manifest.sqlite3.backup

# 3. Revert code
git checkout main~1 .

# 4. Clear breaker state (optional)
rm -rf /var/run/docstokg/breakers/cooldowns.db

# 5. Resume
python -m DocsToKG.ContentDownload.cli \
  --topic "your_topic" \
  --year-start 2024 \
  --year-end 2024 \
  --out runs/content
```

### Disable Circuit Breakers (Keep Running)

```bash
# If you need circuit breakers off but code deployed:
export DOCSTOKG_BREAKER_DISABLED=1

python -m DocsToKG.ContentDownload.cli \
  --topic "your_topic" \
  --year-start 2024 \
  --year-end 2024 \
  --out runs/content
```

---

## Support & Troubleshooting

### Common Issues

#### Issue 1: "pybreaker is not installed"

```
Error: RuntimeError: pybreaker is required for BreakerRegistry
```

**Solution:**

```bash
pip install pybreaker
# Or if using requirements.txt:
pip install -r requirements.txt
```

#### Issue 2: "Cannot create cooldown store"

```
Error: No such file or directory for /var/run/docstokg/breakers/cooldowns.db
```

**Solution:**

```bash
mkdir -p /var/run/docstokg/breakers
chmod 755 /var/run/docstokg/breakers
```

#### Issue 3: "BreakerOpenError: host is open"

```
Expected behavior when a host is overloaded.
Resume later or manually close the breaker.
```

**Solution:**

```bash
# Option 1: Wait for cooldown to expire
sleep 120  # default reset_timeout_s is 60

# Option 2: Manually close
python -m DocsToKG.ContentDownload.cli breaker close api.example.org
```

### Debug Mode

```bash
# Enable verbose logging
export DOCSTOKG_BREAKER_DEBUG=1
export PYTHONUNBUFFERED=1

# Run with debug output
python -m DocsToKG.ContentDownload.cli \
  --topic "test" \
  --year-start 2024 \
  --year-end 2024 \
  --max 10 \
  --dry-run \
  -vvv  # Verbose output
```

### Getting Help

1. **Check logs**:

   ```bash
   grep -i breaker /var/log/docstokg/*.log
   ```

2. **Review telemetry**:

   ```bash
   sqlite3 runs/content/manifest.sqlite3 "SELECT * FROM breaker_events LIMIT 10;"
   ```

3. **Analyze recent runs**:

   ```bash
   jq '.[] | select(.record_type=="attempt") | {host, status, reason}' runs/content/manifest.jsonl | head -20
   ```

---

## Success Criteria

After deployment, verify:

- [ ] Tests pass: `pytest tests/content_download/test_breakers_*.py`
- [ ] Linting clean: `ruff check src/DocsToKG/ContentDownload/`
- [ ] Small test run completes successfully
- [ ] Breaker events logged to telemetry
- [ ] CLI commands work (`breaker show`, `breaker open`, etc.)
- [ ] Configuration loads without errors
- [ ] No regressions in existing download functionality
- [ ] Telemetry database schema correct

---

## Post-Deployment (Week 1)

1. **Monitor** breaker events for first 100 downloads
2. **Verify** that Retry-After headers are being honored
3. **Check** for any false positives (breakers opening unnecessarily)
4. **Tune** fail_max/reset_timeout_s based on observed patterns
5. **Document** any custom configurations applied

---

## Support Contact

For issues or questions:

- Check AGENTS.md Circuit Breaker Operations section
- Review logs: `/var/log/docstokg/ContentDownload.log`
- Analyze telemetry: `runs/content/manifest.sqlite3`
- Run health check script: `scripts/breaker_health_check.sh`

---

**Deployment Package v1.0 - Ready for Production** ✅
