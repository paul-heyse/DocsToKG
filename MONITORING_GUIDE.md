# Monitoring & Observability Guide

**Date**: October 21, 2025
**Version**: 1.0
**Status**: PRODUCTION READY

---

## 🎯 MONITORING OVERVIEW

This guide covers comprehensive monitoring and observability for the OntologyDownload system using:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboard visualization
- **Alert Rules**: Automated alerting system
- **Health Checks**: Service availability monitoring

---

## 📊 KEY METRICS

### Application Metrics

#### Request Metrics
- `http_requests_total`: Total HTTP requests by status
- `http_request_duration_seconds`: Request latency histogram
- `http_requests_in_progress`: Active requests gauge

#### Error Metrics
- `http_requests_total{status=~"5.."}`: Server error count
- `http_requests_total{status=~"4.."}`: Client error count
- `application_errors_total`: Application-level errors

#### Business Metrics
- `artifacts_downloaded_total`: Total artifacts processed
- `artifacts_failed_total`: Failed artifacts
- `artifacts_deduplicated_total`: Deduplicated artifacts

### Database Metrics

#### Query Metrics
- `duckdb_query_duration_seconds`: Query execution time
- `duckdb_query_errors_total`: Query failures
- `duckdb_connections_active`: Active connections
- `duckdb_database_size_bytes`: Total database size

#### Connection Pool
- `duckdb_connection_pool_active`: Active connections
- `duckdb_connection_pool_max`: Pool size
- `duckdb_connection_wait_time_seconds`: Wait time

### Cache Metrics

#### Cache Performance
- `cache_hits_total`: Cache hit count
- `cache_misses_total`: Cache miss count
- `cache_hit_ratio`: Hit rate percentage

#### Cache Operations
- `cache_get_duration_seconds`: Get operation latency
- `cache_set_duration_seconds`: Set operation latency

### System Metrics

#### Resource Usage
- `container_cpu_usage_seconds_total`: CPU usage
- `container_memory_usage_bytes`: Memory usage
- `node_filesystem_avail_bytes`: Disk space available

#### Network
- `network_bytes_sent_total`: Outbound traffic
- `network_bytes_recv_total`: Inbound traffic

---

## 📈 DASHBOARDS

### Dashboard 1: Overview
**Purpose**: High-level system health and performance

**Panels**:
1. Request Rate (req/s)
2. Error Rate (% of requests)
3. Latency p95 (ms)
4. Active Requests (count)
5. Cache Hit Rate (%)
6. Database Query Time (ms)

**Refresh**: Every 30 seconds
**Time Range**: Last 1 hour (default)

### Dashboard 2: Database
**Purpose**: Database performance and health

**Panels**:
1. Query Duration (p95, p99)
2. Connection Pool Usage (%)
3. Query Errors (rate)
4. Database Size (GB)
5. Connection Wait Time (ms)
6. Transaction Duration (ms)

**Refresh**: Every 60 seconds
**Time Range**: Last 6 hours (default)

### Dashboard 3: System
**Purpose**: Infrastructure and resource utilization

**Panels**:
1. CPU Usage (%)
2. Memory Usage (GB)
3. Disk Space (%)
4. Network Throughput (MB/s)
5. Docker Container Health
6. Service Uptime (%)

**Refresh**: Every 60 seconds
**Time Range**: Last 24 hours (default)

### Dashboard 4: Deployments
**Purpose**: Deployment status and traffic distribution

**Panels**:
1. Blue/Green Status
2. Active Version
3. Traffic Distribution (%)
4. Error Rate by Version (%)
5. Deployment History (events)
6. Rollback Status

**Refresh**: Every 10 seconds
**Time Range**: Last 1 hour (default)

---

## 🚨 ALERT RULES

### Critical Alerts (Immediate Action Required)

#### Service Down
```yaml
alert: ServiceDown
condition: up{job="ontology-download"} == 0
duration: 2 minutes
action: Page on-call engineer
```
**Response**: Check service logs, verify database connectivity, execute rollback if needed

#### High Error Rate (>5%)
```yaml
alert: HighErrorRate
condition: error_rate > 0.05
duration: 5 minutes
action: Page on-call engineer
```
**Response**: Check application logs, identify error pattern, scale if necessary

#### Database Unavailable
```yaml
alert: DatabaseUnavailable
condition: up{job="duckdb-catalog"} == 0
duration: 1 minute
action: Page database team
```
**Response**: Check database logs, verify connections, restart if necessary

### Warning Alerts (Monitor Closely)

#### High Latency (p95 > 1s)
```yaml
alert: HighLatency
condition: latency_p95 > 1000ms
duration: 5 minutes
action: Send Slack notification
```
**Response**: Check database queries, identify slow operations, optimize if needed

#### High Memory Usage (>80%)
```yaml
alert: HighMemoryUsage
condition: memory_usage > 80%
duration: 5 minutes
action: Send Slack notification
```
**Response**: Monitor trend, scale if persistent, investigate memory leaks

#### High CPU Usage (>80%)
```yaml
alert: HighCPUUsage
condition: cpu_usage > 80%
duration: 5 minutes
action: Send Slack notification
```
**Response**: Scale horizontally, optimize hot paths, profile if needed

#### Low Cache Hit Rate (<60%)
```yaml
alert: LowCacheHitRate
condition: cache_hit_rate < 0.6
duration: 10 minutes
action: Send Slack notification
```
**Response**: Warm cache, increase cache size, optimize cache strategy

#### Disk Space Low (<20%)
```yaml
alert: DiskSpaceLow
condition: disk_available_percent < 20%
duration: 5 minutes
action: Send Slack notification
```
**Response**: Clean old data, increase disk, investigate growth trend

---

## 🔍 MONITORING PROCEDURES

### Daily Monitoring Checklist

- [ ] Check error rate (should be <1%)
- [ ] Verify latency trends (p95 should be <500ms)
- [ ] Review disk space usage
- [ ] Check cache hit rate (should be >80%)
- [ ] Verify database size trend
- [ ] Review alert history

### Weekly Review

- [ ] Analyze performance trends
- [ ] Review capacity planning
- [ ] Check for any patterns in errors
- [ ] Verify backup integrity
- [ ] Update runbooks if needed

### Monthly Review

- [ ] Capacity planning assessment
- [ ] Alert threshold adjustment
- [ ] Performance baseline update
- [ ] Cost analysis
- [ ] Team retrospective

---

## 📊 PERFORMANCE BASELINES

### Target SLIs (Service Level Indicators)

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Error Rate | <1% | 2-5% | >5% |
| Latency p95 | <500ms | 500-1000ms | >1000ms |
| Availability | 99.9% | 99%-99.9% | <99% |
| Cache Hit Rate | >80% | 60-80% | <60% |
| Database Uptime | 99.99% | 99.9%-99.99% | <99.9% |

### Historical Baselines

```
Error Rate:        0.2% (avg), 0.5% (peak)
Latency p95:       350ms (avg), 450ms (peak)
Latency p99:       450ms (avg), 600ms (peak)
Memory Usage:      1.5GB (avg), 2GB (peak)
CPU Usage:         20% (avg), 40% (peak)
Cache Hit Rate:    85% (avg), 90% (peak)
```

---

## 🔧 TROUBLESHOOTING WITH METRICS

### High Error Rate

1. Check error type:
   ```promql
   http_requests_total{status=~"5.."}
   ```

2. Check affected endpoints:
   ```promql
   http_requests_total{status=~"5..", path!="/health"}
   ```

3. Check latency correlation:
   ```promql
   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
   ```

### High Latency

1. Check database query time:
   ```promql
   histogram_quantile(0.95, rate(duckdb_query_duration_seconds_bucket[5m]))
   ```

2. Check connection pool:
   ```promql
   duckdb_connection_pool_active / duckdb_connection_pool_max
   ```

3. Check resource usage:
   ```promql
   container_cpu_usage_seconds_total, container_memory_usage_bytes
   ```

### Low Cache Hit Rate

1. Check cache operations:
   ```promql
   rate(cache_hits_total[5m]) / rate(cache_requests_total[5m])
   ```

2. Check cache size:
   ```promql
   cache_size_bytes, cache_max_size_bytes
   ```

---

## 📞 ESCALATION PROCEDURES

### Level 1 (First Response - 5 min)
- Check alert details in Prometheus
- Review Grafana dashboards
- Check recent deployments
- Review application logs

### Level 2 (Investigation - 15 min)
- Contact on-call database engineer if database issue
- Check recent configuration changes
- Review recent code deployments
- Check upstream service status

### Level 3 (Resolution - 30 min)
- Engage team lead
- Execute mitigation procedures
- Begin root cause analysis
- Update incident ticket

---

## 📋 MONITORING CONFIGURATION

### Prometheus Configuration
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ontology-download'
    scrape_interval: 10s
    targets: ['ontology-download:8000']
```

### Grafana Configuration
- Datasource: Prometheus (http://prometheus:9090)
- Refresh: 30s (dashboards)
- Retention: 15 days (default)

### Alert Channels
- Slack: #ontology-download-alerts
- Email: oncall@example.com
- PagerDuty: (if configured)

---

## ✅ MONITORING SIGN-OFF CHECKLIST

- [ ] Prometheus configured and running
- [ ] Grafana dashboards created and tested
- [ ] Alert rules configured and tested
- [ ] Team trained on monitoring systems
- [ ] Runbooks prepared
- [ ] Alert channels configured
- [ ] Baseline metrics established
- [ ] Monitoring verified in staging

---

**Document Version**: 1.0
**Last Updated**: October 21, 2025
**Next Review**: October 28, 2025
