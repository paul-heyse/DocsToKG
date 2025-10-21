# Production Deployment Guide

**Date**: October 21, 2025
**Version**: 1.0
**Status**: PRODUCTION READY

---

## 🎯 DEPLOYMENT OVERVIEW

This guide covers production deployment of the OntologyDownload system using:
- Docker Compose for container orchestration
- Blue-Green strategy for zero-downtime deployments
- Prometheus & Grafana for monitoring
- Nginx for load balancing
- Automated health checks and rollback

---

## 📋 PRE-DEPLOYMENT CHECKLIST

### 1. Environment Verification
- [ ] Docker and Docker Compose installed
- [ ] Required disk space (100GB minimum)
- [ ] Network connectivity to all upstream services
- [ ] Database backups completed
- [ ] SSL certificates configured

### 2. Code Verification
- [ ] All tests passing locally (pytest -v)
- [ ] No linting errors (ruff check)
- [ ] Type checks passing (mypy src)
- [ ] Security scans completed

### 3. Configuration Validation
- [ ] All environment variables configured
- [ ] Database connection strings verified
- [ ] Rate limiting configuration tested
- [ ] Monitoring endpoints accessible

---

## 🚀 BLUE-GREEN DEPLOYMENT

### Quick Start

```bash
# Deploy new version
./deployment/blue-green.sh v1.2.3 prod

# The script will:
# 1. Build Docker image
# 2. Deploy to standby environment
# 3. Run health checks
# 4. Execute smoke tests
# 5. Switch traffic to new version
# 6. Monitor for errors (5 minutes)
# 7. Automatic rollback on failure
```

### Deployment Stages

#### Stage 1: Pre-Deployment (15 min)
```bash
# Health check current service
docker-compose -f deployment/docker-compose.prod.yml exec ontology-download curl -sf http://localhost:8000/health

# Verify database connectivity
docker-compose -f deployment/docker-compose.prod.yml exec duckdb-catalog ls /var/lib/duckdb
```

#### Stage 2: Deploy to Standby (10 min)
```bash
# Build new Docker image
docker build -t docstokg/ontology-download:v1.2.3 .

# Start standby services
docker-compose -f deployment/docker-compose.prod.yml up -d --no-deps ontology-download
```

#### Stage 3: Verification (20 min)
```bash
# Wait for service startup
sleep 10

# Health check standby
curl http://localhost:8000/health

# Run smoke tests
curl http://localhost:8000/metrics
curl http://localhost:8000/api/version
```

#### Stage 4: Traffic Switch (5 min)
```bash
# Update load balancer configuration
echo "green" > deployment/lb-config.txt

# Reload nginx
docker-compose -f deployment/docker-compose.prod.yml exec nginx nginx -s reload
```

#### Stage 5: Monitoring (5+ min)
```bash
# Monitor error rate
curl 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])'

# Check application logs
docker-compose -f deployment/docker-compose.prod.yml logs -f ontology-download
```

### Rollback Procedure

Automatic rollback triggers:
- Health check failures
- Smoke test failures
- High error rate (>5%)
- Service unavailability

Manual rollback:
```bash
# Switch traffic back to previous version
echo "blue" > deployment/lb-config.txt
docker-compose -f deployment/docker-compose.prod.yml exec nginx nginx -s reload
```

---

## 🔄 CANARY DEPLOYMENT

### Overview
Gradually shift traffic to new version (5% → 25% → 50% → 100%)

### Configuration
```yaml
# deployment/canary-config.yml
canary:
  initial_traffic_percent: 5
  traffic_increase_percent: 25
  traffic_increase_interval_minutes: 5
  success_threshold_percent: 99.5
  error_rate_threshold_percent: 1
```

### Execution
```bash
# Start canary deployment
./deployment/canary.sh v1.2.3 prod

# Monitor progression
watch -n 10 'docker-compose -f deployment/docker-compose.prod.yml logs | tail -20'
```

---

## 📊 MONITORING & ALERTS

### Prometheus Metrics
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `http_requests_total{status=~"5.."}`: Server errors
- `duckdb_query_duration_seconds`: Database query time

### Grafana Dashboards
- **Overview**: Request rate, latency, error rate
- **Database**: Query performance, connection pool
- **System**: CPU, memory, disk usage
- **Deployments**: Version status, traffic split

### Alert Rules

#### Critical Alerts
- Service down (health check fails)
- High error rate (>5% 5xx errors)
- Database unavailable

#### Warning Alerts
- High latency (p95 > 1s)
- Cache hit rate low (<60%)
- Disk space low (<20%)

---

## 🔧 HEALTH CHECKS

### Startup Health Check
```bash
# Check service is responding
curl -f http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2025-10-21T10:30:00Z",
  "version": "1.2.3"
}
```

### Liveness Probe
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

---

## 📝 OPERATIONAL PROCEDURES

### Scaling Up (Add Capacity)

```bash
# Increase resource limits
docker-compose -f deployment/docker-compose.prod.yml down
# Edit docker-compose.prod.yml to increase resources
docker-compose -f deployment/docker-compose.prod.yml up -d

# Verify scaling
docker-compose -f deployment/docker-compose.prod.yml ps
```

### Scaling Down (Reduce Capacity)

```bash
# Drain traffic from instance
# Then scale down services
docker-compose -f deployment/docker-compose.prod.yml down ontology-download
```

### Update Certificates

```bash
# Update SSL certificates
cp /path/to/new/cert.pem deployment/certs/
cp /path/to/new/key.pem deployment/certs/

# Reload nginx
docker-compose -f deployment/docker-compose.prod.yml exec nginx nginx -s reload
```

---

## 🚨 TROUBLESHOOTING

### Service Not Starting
```bash
# Check logs
docker-compose -f deployment/docker-compose.prod.yml logs ontology-download

# Verify configuration
docker-compose -f deployment/docker-compose.prod.yml config

# Check resource availability
docker stats
```

### High Error Rate
```bash
# Check application logs
docker-compose -f deployment/docker-compose.prod.yml logs -f ontology-download --tail 100

# Check database connectivity
docker-compose -f deployment/docker-compose.prod.yml exec ontology-download curl -f http://duckdb-catalog:8432/

# Check rate limiting
docker-compose -f deployment/docker-compose.prod.yml exec ontology-download cat /data/rate-limits.db
```

### Memory Issues
```bash
# Check memory usage
docker stats ontology-download

# Increase memory limit in docker-compose.prod.yml
# Then recreate container
docker-compose -f deployment/docker-compose.prod.yml up -d --force-recreate ontology-download
```

---

## 📞 SUPPORT & ESCALATION

### On-Call Support
- **Level 1**: Check logs, verify health checks
- **Level 2**: Execute rollback, assess impact
- **Level 3**: Investigate root cause, implement fix

### Escalation Path
```
Issue Detected
    ↓
Health Check Failed → Automatic Rollback
    ↓
Manual Assessment Required → On-Call Engineer
    ↓
Root Cause Analysis → Team Lead
    ↓
Post-Incident Review
```

---

## ✅ SIGN-OFF CHECKLIST

- [ ] All pre-deployment checks completed
- [ ] Team notified of deployment window
- [ ] Rollback procedure verified
- [ ] Monitoring alerts configured
- [ ] Deployment executed successfully
- [ ] Post-deployment monitoring verified
- [ ] Incident log updated
- [ ] Team debriefing completed

---

**Document Version**: 1.0
**Last Updated**: October 21, 2025
**Next Review**: October 28, 2025
