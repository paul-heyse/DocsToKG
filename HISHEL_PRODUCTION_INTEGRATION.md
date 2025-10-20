# Hishel HTTP Caching System - Production Integration Guide

**Status**: ✅ PRODUCTION-READY
**Date**: October 21, 2025
**Version**: Phase 4 Complete
**RFC Compliance**: RFC 9111 (HTTP Caching) + RFC 7232 (Conditional Requests)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Production Deployment](#production-deployment)
3. [Configuration Management](#configuration-management)
4. [Monitoring & Observability](#monitoring--observability)
5. [Troubleshooting](#troubleshooting)
6. [Rollback Procedures](#rollback-procedures)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Scenarios](#advanced-scenarios)

---

## Quick Start

### Minimal Deployment (File Storage)

```bash
# 1. Deploy the system using the automated script
./scripts/deploy_hishel_production.sh

# 2. Run ContentDownload with caching enabled
python -m DocsToKG.ContentDownload.cli \
  --topic "machine learning" \
  --year-start 2023 \
  --year-end 2024 \
  --cache-config config/cache.yaml \
  --workers 4

# 3. Monitor cache performance
python -m DocsToKG.ContentDownload.cache_cli stats
```

### Production Deployment (Redis Storage)

```bash
# 1. Set up Redis storage
export DOCSTOKG_CACHE_STORAGE_KIND=redis
export DOCSTOKG_CACHE_STORAGE_REDIS_HOST=redis.prod.example.com
export DOCSTOKG_CACHE_STORAGE_REDIS_PORT=6379

# 2. Deploy the system
./scripts/deploy_hishel_production.sh

# 3. Run ContentDownload with caching
python -m DocsToKG.ContentDownload.cli \
  --cache-config config/cache.yaml \
  --resolver-preset fast
```

---

## Production Deployment

### Step 1: Pre-Deployment Validation

```bash
# Check that all dependencies are installed
./.venv/bin/python -c "
import httpx
import hishel
import pydantic
import yaml
import idna
print('✓ All dependencies installed')
"

# Verify cache configuration
./.venv/bin/python -c "
import yaml
with open('src/DocsToKG/ContentDownload/config/cache.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'✓ Configuration valid')
print(f'  - Storage: {cfg[\"storage\"][\"kind\"]}')
print(f'  - Hosts: {len(cfg.get(\"hosts\", {}))}')
"
```

### Step 2: Automated Deployment

```bash
# Run the comprehensive deployment script
./scripts/deploy_hishel_production.sh

# This will:
# ✓ Check virtual environment
# ✓ Verify dependencies
# ✓ Validate configuration file
# ✓ Run core tests
# ✓ Create cache directory
# ✓ Validate Redis connection (if configured)
# ✓ Deploy configuration
# ✓ Enable Hishel caching
# ✓ Test cache functionality
# ✓ Establish performance baseline
# ✓ Set up monitoring
# ✓ Generate deployment report
```

### Step 3: Initialize Cache

```bash
# Warm up the cache with metadata from main resolvers
python -m DocsToKG.ContentDownload.cli \
  --topic "test" \
  --year-start 2024 \
  --year-end 2024 \
  --max 100 \
  --dry-run \
  --cache-config config/cache.yaml

# Verify cache is active
ls -lh ${DOCSTOKG_DATA_ROOT:-./Data}/cache/http/ContentDownload/
```

### Step 4: Production Validation (24-Hour Window)

```bash
# Monitor these metrics after 24 hours:

# 1. Cache hit rate (target: >50%)
python -m DocsToKG.ContentDownload.cache_cli stats

# 2. Response times (target: <50ms)
# (Measured in cache_statistics)

# 3. Error rate (target: <1%)
# Check logs for errors

# 4. Storage growth (target: predictable)
du -sh ${DOCSTOKG_DATA_ROOT:-./Data}/cache/http/

# 5. Memory usage (target: stable)
# Monitor via system tools
```

---

## Configuration Management

### Storage Backend Selection

#### Development: File Storage (Recommended)

```yaml
# config/cache.yaml
storage:
  kind: file
  path: "./Data/cache/http"
  ttl: 259200  # 3 days
  check_ttl_every: 600  # 10 minutes
```

**Pros**: No external dependencies, simple setup
**Cons**: Single-instance only, slower than Redis
**Best for**: Development, testing, single-machine deployments

#### Production: Redis Storage (Recommended)

```yaml
# config/cache.yaml
storage:
  kind: redis
  ttl: 259200

# Environment variables
export DOCSTOKG_CACHE_STORAGE_KIND=redis
export DOCSTOKG_CACHE_STORAGE_REDIS_HOST=redis.prod.example.com
export DOCSTOKG_CACHE_STORAGE_REDIS_PORT=6379
export DOCSTOKG_CACHE_STORAGE_REDIS_DB=0
```

**Pros**: Distributed caching, high performance, multi-instance
**Cons**: Requires Redis infrastructure
**Best for**: Production, multi-instance, distributed systems

#### Alternative: SQLite Storage

```bash
export DOCSTOKG_CACHE_STORAGE_KIND=sqlite
export DOCSTOKG_CACHE_STORAGE_SQLITE_PATH=./cache.db
```

**Pros**: Persistent, queryable, lightweight
**Cons**: Slower than Redis, limited multi-instance support
**Best for**: Medium-scale deployments, single-instance with persistence

### Host-Specific Policies

Edit `config/cache.yaml` to customize caching for specific hosts:

```yaml
hosts:
  api.crossref.org:
    ttl_s: 259200  # 3 days
    role:
      metadata:
        ttl_s: 259200
        swrv_s: 180  # Serve stale while revalidating
      landing:
        ttl_s: 86400
        swrv_s: 60

  api.openalex.org:
    ttl_s: 259200
    role:
      metadata:
        ttl_s: 259200
        swrv_s: 180
```

### Environment Variable Overrides

Override configuration via environment variables:

```bash
# Storage configuration
export DOCSTOKG_CACHE_STORAGE_KIND=redis
export DOCSTOKG_CACHE_STORAGE_REDIS_HOST=redis.internal
export DOCSTOKG_CACHE_STORAGE_REDIS_PORT=6379

# TTL configuration
export DOCSTOKG_CACHE_TTL=259200

# Statistics collection
export DOCSTOKG_CACHE_STATISTICS_ENABLED=true
export DOCSTOKG_CACHE_STATISTICS_EXPORT_INTERVAL=300

# Disable caching (emergency)
export DOCSTOKG_CACHE_DISABLE=false
```

### CLI Argument Overrides

Override configuration via CLI arguments:

```bash
python -m DocsToKG.ContentDownload.cli \
  --cache-config config/cache.yaml \
  --cache-storage redis \
  --cache-host api.crossref.org=259200 \
  --cache-role api.openalex.org:metadata=259200:180 \
  --cache-defaults CACHE:true:false
```

---

## Monitoring & Observability

### Real-Time Statistics

```bash
# View cache statistics
python -m DocsToKG.ContentDownload.cache_cli stats

# Output:
# ┌─────────────────────────────────┐
# │ Cache Statistics (Real-time)    │
# ├─────────────────────────────────┤
# │ Cache Hits:        1,234,567    │
# │ Cache Misses:        567,890    │
# │ Hit Rate:             68.4%     │
# │ Bandwidth Saved:     12.3 GB    │
# │ Response Time Avg:   23.4 ms    │
# │ Revalidations:       12,345     │
# └─────────────────────────────────┘
```

### Per-Host Statistics

```bash
# View statistics for specific host
python -m DocsToKG.ContentDownload.cache_cli stats --host api.crossref.org

# View statistics for specific role
python -m DocsToKG.ContentDownload.cache_cli stats --role metadata
```

### Metrics Export

```bash
# Export metrics as JSON
python -m DocsToKG.ContentDownload.cache_cli export --format json --output metrics.json

# Export metrics as CSV
python -m DocsToKG.ContentDownload.cache_cli export --format csv --output metrics.csv

# Example CSV output:
# timestamp,cache_hits,cache_misses,hit_rate,bandwidth_saved_mb,response_time_ms
# 1729519200,1234567,567890,68.4,12345.6,23.4
```

### Health Checks

```bash
# Perform health check
python -m DocsToKG.ContentDownload.cache_cli health

# Output:
# Cache System Health Check
# ✓ Storage backend: OPERATIONAL
# ✓ Cache directory: ACCESSIBLE (12.3 GB)
# ✓ Redis connection: OPERATIONAL (latency: 2.3ms)
# ✓ Statistics collection: ENABLED
# ✓ Cache hit rate: 68.4%
# Overall Status: HEALTHY
```

### Structured Logging

```bash
# Enable debug logging for cache decisions
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export DOCSTOKG_LOG_LEVEL=DEBUG

python -m DocsToKG.ContentDownload.cli \
  --cache-config config/cache.yaml \
  2>&1 | grep -E "cache-hit|cache-miss|cache-decision"

# Example output:
# [cache-hit] host=api.crossref.org role=metadata time_ms=5.2
# [cache-miss] host=api.openalex.org role=metadata time_ms=145.3
# [cache-decision] url=https://... use_cache=true ttl=259200
```

---

## Troubleshooting

### Issue: Low Cache Hit Rate (<30%)

**Symptoms**: Cache hits are rare, most requests result in misses

**Diagnosis**:

```bash
# Check if caching is enabled
python -m DocsToKG.ContentDownload.cache_cli stats

# Verify cache configuration
python -c "
import yaml
with open('config/cache.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'Default TTL: {cfg[\"storage\"][\"ttl\"]}')
print(f'Controller default: {cfg[\"controller\"][\"default\"]}')
"
```

**Solutions**:

1. Increase TTL values in `config/cache.yaml`
2. Verify that `controller.default: CACHE` is set
3. Check that hosts are properly configured
4. Ensure cache is writable: `ls -l ./Data/cache/http/`

### Issue: Redis Connection Failed

**Symptoms**: Error messages like "Cannot connect to Redis"

**Diagnosis**:

```bash
# Test Redis connectivity
python -c "
import redis
r = redis.Redis(host='localhost', port=6379)
r.ping()
print('✓ Redis OK')
"

# Check Redis configuration
echo $DOCSTOKG_CACHE_STORAGE_REDIS_HOST
echo $DOCSTOKG_CACHE_STORAGE_REDIS_PORT
```

**Solutions**:

1. Verify Redis server is running: `redis-cli ping`
2. Check network connectivity: `telnet redis.host 6379`
3. Verify credentials if Redis requires auth
4. Fallback to file storage: `export DOCSTOKG_CACHE_STORAGE_KIND=file`

### Issue: High Memory Usage

**Symptoms**: Cache process using excessive memory

**Diagnosis**:

```bash
# Check cache size
du -sh ./Data/cache/http/

# If using in-memory storage, check capacity
python -c "
import yaml
with open('config/cache.yaml') as f:
    cfg = yaml.safe_load(f)
if cfg['storage']['kind'] == 'memory':
    print(f'Capacity: {cfg[\"storage\"].get(\"capacity\", \"unlimited\")}')
"
```

**Solutions**:

1. Reduce TTL values to cache for shorter periods
2. If using memory storage, reduce capacity limit
3. Implement cache invalidation: `python -m DocsToKG.ContentDownload.cache_cli clear`
4. Switch to Redis for distributed memory

### Issue: Cache Invalidation Not Working

**Symptoms**: Old data still being served after updates

**Diagnosis**:

```bash
# Check cache invalidation policy
python -c "
from DocsToKG.ContentDownload.cache_invalidation import create_error_invalidation_policy
policy = create_error_invalidation_policy(on_404=True, on_410=True, on_5xx=False)
print(f'Policy: {policy}')
"
```

**Solutions**:

1. Manually clear cache: `python -m DocsToKG.ContentDownload.cache_cli clear`
2. Clear specific host: `python -m DocsToKG.ContentDownload.cache_cli clear --host api.example.com`
3. Use error-triggered invalidation for 4xx responses
4. Set shorter TTL for frequently-changing data

---

## Rollback Procedures

### Emergency Disable Caching

```bash
# Option 1: Environment variable
export DOCSTOKG_CACHE_DISABLE=true

# Option 2: CLI argument
python -m DocsToKG.ContentDownload.cli --cache-disable

# Option 3: Remove configuration file
rm config/cache.yaml
```

### Clear Cache

```bash
# Clear entire cache
python -m DocsToKG.ContentDownload.cache_cli clear

# Clear cache for specific host
python -m DocsToKG.ContentDownload.cache_cli clear --host api.crossref.org

# Clear file storage manually
rm -rf ./Data/cache/http/ContentDownload/
```

### Revert Configuration

```bash
# Restore previous configuration
git checkout HEAD -- config/cache.yaml

# Or use default minimal config
cp src/DocsToKG/ContentDownload/config/cache.yaml config/cache.yaml
```

---

## Performance Optimization

### Tuning for High Throughput

```yaml
# config/cache.yaml
storage:
  kind: redis
  ttl: 604800  # 7 days (longer caching)
  check_ttl_every: 3600  # Check less frequently

controller:
  cacheable_methods:
    - GET
    - HEAD
  cacheable_statuses:
    - 200
    - 301
    - 308
  allow_heuristics: true  # Cache more aggressively
  default: CACHE
  always_revalidate: false
```

### Auto-Tuning Recommendations

Phase 4B enables automatic TTL optimization:

```yaml
optimization:
  enabled: true
  auto_apply: false  # Start with recommendations only
  min_requests: 100
  efficiency_threshold: 10
  update_interval_s: 3600
```

Monitor and review recommendations:

```bash
# Check optimization analysis
python -c "
from DocsToKG.ContentDownload.cache_optimization import get_cache_optimizer
optimizer = get_cache_optimizer()
print(optimizer.print_optimization_report())
"
```

---

## Advanced Scenarios

### Multi-Instance Deployment (Redis)

```bash
# Deploy on multiple instances with shared Redis cache
export DOCSTOKG_CACHE_STORAGE_KIND=redis
export DOCSTOKG_CACHE_STORAGE_REDIS_HOST=redis-cluster.internal
export DOCSTOKG_CACHE_STORAGE_REDIS_PORT=6379

# Each instance will share the same cache
python -m DocsToKG.ContentDownload.cli --workers 8
```

### Conditional Request Revalidation

Automatic ETag/Last-Modified handling (no configuration needed):

```bash
# Hishel automatically:
# 1. Stores ETag and Last-Modified headers
# 2. Sends conditional requests when cache expires
# 3. Handles 304 Not Modified responses
# 4. Saves bandwidth (~50KB per revalidated response)

# Monitor revalidation statistics
python -m DocsToKG.ContentDownload.cache_cli stats | grep -i revalidation
```

### Custom TTL Per Role

Different TTL for different request roles:

```yaml
hosts:
  api.example.com:
    ttl_s: 259200  # Default
    role:
      metadata:
        ttl_s: 604800  # 7 days for metadata
      landing:
        ttl_s: 86400   # 1 day for landing pages
      artifact:
        ttl_s: 0       # Don't cache artifacts
```

### Disaster Recovery

```bash
# Back up Redis cache
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb backup/redis_dump_$(date +%s).rdb

# Back up file storage cache
tar czf backup/cache_$(date +%Y%m%d).tar.gz ./Data/cache/http/

# Restore from backup
redis-cli SHUTDOWN
cp backup/redis_dump_*.rdb /var/lib/redis/dump.rdb
redis-server
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] All dependencies installed
- [ ] Configuration file validated
- [ ] Cache directory writable
- [ ] Redis accessible (if configured)
- [ ] Tests passing (pytest)

### Deployment

- [ ] Run deployment script
- [ ] Configuration deployed
- [ ] Hishel integration verified
- [ ] Cache functionality tested
- [ ] Performance baseline established

### Post-Deployment (24 Hours)

- [ ] Cache hit rate > 50%
- [ ] Response times < 50ms average
- [ ] Error rate < 1%
- [ ] No memory leaks detected
- [ ] Storage growth predictable

### Monitoring

- [ ] Statistics collection enabled
- [ ] Health checks passing
- [ ] Metrics being exported
- [ ] Alerts configured (if applicable)
- [ ] Documentation updated

---

## Summary

The Hishel HTTP caching system is **production-ready** and provides:

✅ **RFC 9111 Compliance** - Fully compliant HTTP caching
✅ **Distributed Caching** - Shared cache via Redis
✅ **Automatic Revalidation** - ETag/Last-Modified handling
✅ **Performance Metrics** - Real-time statistics and monitoring
✅ **Auto-Tuning** - Adaptive TTL optimization (Phase 4B)
✅ **Fallback Reliability** - Graceful degradation to file storage
✅ **Easy Deployment** - Automated deployment script
✅ **Easy Monitoring** - CLI tools for cache management

**For support, troubleshooting, or advanced configuration, refer to:**

- `HISHEL_PLANNING_SUMMARY.md` - Executive summary
- `HISHEL_CACHING_COMPREHENSIVE_PLAN.md` - Technical architecture
- `HISHEL_IMPLEMENTATION_SPECIFICATION.md` - Implementation details
- `src/DocsToKG/ContentDownload/AGENTS.md` - AI agent guidance
