Amazing — here’s a **ready-to-import Grafana dashboard** (JSON) that visualizes the KPIs we instrumented in PR #7 with OpenTelemetry metrics:

* **Cache hit ratio** (overall & by resolver)
* **Revalidation ratio** (304 rate)
* **Success rate**
* **GET/HEAD latency** p50/p90/p99 (histogram_quantile on `contentdownload_http_latency_ms_bucket`)
* **Retries by reason**
* **Bytes/sec** by resolver
* **Attempt rate by status** (cache-hit, http-200, http-304)
* **Outcomes** (success/skip/error), stacked by resolver

It assumes your metrics land in **Prometheus** (directly or via the OTel Collector’s Prometheus exporter/remote-write). You can import this as a dashboard, point it to your Prometheus datasource, and you’re off.

---

## How to use

1. In Grafana: **Dashboards → New → Import**.
2. Paste the JSON below.
3. When prompted, select your **Prometheus** datasource.
4. The dashboard includes variables:

   * **datasource** (Prometheus)
   * **resolver** (multi-select; defaults to All)
   * **verb** (GET/HEAD; defaults to All)
5. Time pickers & refresh are set; tune as desired.

> **Metric names expected** (from PR #7):
> `contentdownload_attempts_total`, `contentdownload_http_latency_ms_bucket` (+ `_sum/_count`),
> `contentdownload_retries_total`, `contentdownload_bytes_total`, `contentdownload_outcomes_total`.
> (Optional) `contentdownload_rate_sleep_ms_bucket` if you emit rate-limit sleeps as a histogram.

---

## Dashboard JSON (ContentDownload (hishel) — Overview)

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 1,
  "id": null,
  "links": [],
  "liveNow": false,
  "panels": [
    {
      "type": "stat",
      "title": "Cache Hit Ratio (overall)",
      "id": 1,
      "gridPos": { "h": 4, "w": 6, "x": 0, "y": 0 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum(rate(contentdownload_attempts_total{status=\"cache-hit\"}[$__rate_interval])) / sum(rate(contentdownload_attempts_total{status=~\"cache-hit|http-200|http-304\"}[$__rate_interval]))",
          "legendFormat": "hit ratio",
          "instant": false
        }
      ],
      "options": {
        "reduceOptions": { "calcs": ["lastNotNull"], "fields": "", "values": false },
        "orientation": "auto",
        "colorMode": "value",
        "graphMode": "none",
        "justifyMode": "auto",
        "text": {}
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "decimals": 2,
          "mappings": [],
          "thresholds": {
            "mode": "percentage",
            "steps": [
              { "color": "red", "value": null },
              { "color": "yellow", "value": 0.4 },
              { "color": "green", "value": 0.7 }
            ]
          }
        },
        "overrides": []
      }
    },
    {
      "type": "stat",
      "title": "Revalidation Ratio (overall)",
      "id": 2,
      "gridPos": { "h": 4, "w": 6, "x": 6, "y": 0 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum(rate(contentdownload_attempts_total{status=\"http-304\"}[$__rate_interval])) / sum(rate(contentdownload_attempts_total{status=~\"http-200|http-304\"}[$__rate_interval]))",
          "legendFormat": "revalidation ratio",
          "instant": false
        }
      ],
      "options": {
        "reduceOptions": { "calcs": ["lastNotNull"] },
        "orientation": "auto",
        "colorMode": "value",
        "graphMode": "none",
        "justifyMode": "auto"
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "decimals": 2,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "red", "value": null },
              { "color": "yellow", "value": 0.2 },
              { "color": "green", "value": 0.5 }
            ]
          }
        },
        "overrides": []
      }
    },
    {
      "type": "stat",
      "title": "Success Rate (overall)",
      "id": 3,
      "gridPos": { "h": 4, "w": 12, "x": 12, "y": 0 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum(rate(contentdownload_outcomes_total{outcome=\"success\"}[$__rate_interval])) / sum(rate(contentdownload_outcomes_total{outcome=~\"success|error\"}[$__rate_interval]))",
          "legendFormat": "success rate",
          "instant": false
        }
      ],
      "options": {
        "reduceOptions": { "calcs": ["lastNotNull"] },
        "orientation": "auto",
        "colorMode": "value",
        "graphMode": "none",
        "justifyMode": "auto"
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "decimals": 2,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "red", "value": null },
              { "color": "yellow", "value": 0.8 },
              { "color": "green", "value": 0.95 }
            ]
          }
        },
        "overrides": []
      }
    },
    {
      "type": "timeseries",
      "title": "Cache Hit Ratio by Resolver",
      "id": 4,
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 4 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (resolver) (rate(contentdownload_attempts_total{resolver=~\"$resolver\",status=\"cache-hit\"}[$__rate_interval])) / sum by (resolver) (rate(contentdownload_attempts_total{resolver=~\"$resolver\",status=~\"cache-hit|http-200|http-304\"}[$__rate_interval]))",
          "legendFormat": "{{resolver}}",
          "instant": false
        }
      ],
      "fieldConfig": { "defaults": { "unit": "percentunit", "decimals": 2 }, "overrides": [] },
      "options": {
        "legend": { "displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull"] },
        "tooltip": { "mode": "multi", "sort": "none" }
      }
    },
    {
      "type": "timeseries",
      "title": "Revalidation Ratio by Resolver",
      "id": 5,
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 4 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (resolver) (rate(contentdownload_attempts_total{resolver=~\"$resolver\",status=\"http-304\"}[$__rate_interval])) / sum by (resolver) (rate(contentdownload_attempts_total{resolver=~\"$resolver\",status=~\"http-200|http-304\"}[$__rate_interval]))",
          "legendFormat": "{{resolver}}",
          "instant": false
        }
      ],
      "fieldConfig": { "defaults": { "unit": "percentunit", "decimals": 2 }, "overrides": [] },
      "options": {
        "legend": { "displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull"] },
        "tooltip": { "mode": "multi", "sort": "none" }
      }
    },
    {
      "type": "timeseries",
      "title": "GET Latency p50/p90/p99 by Resolver",
      "id": 6,
      "gridPos": { "h": 8, "w": 24, "x": 0, "y": 12 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "histogram_quantile(0.50, sum by (le, resolver) (rate(contentdownload_http_latency_ms_bucket{resolver=~\"$resolver\", verb=~\"$verb\"}[$__rate_interval])))",
          "legendFormat": "p50 {{resolver}}",
          "instant": false
        },
        {
          "refId": "B",
          "expr": "histogram_quantile(0.90, sum by (le, resolver) (rate(contentdownload_http_latency_ms_bucket{resolver=~\"$resolver\", verb=~\"$verb\"}[$__rate_interval])))",
          "legendFormat": "p90 {{resolver}}",
          "instant": false
        },
        {
          "refId": "C",
          "expr": "histogram_quantile(0.99, sum by (le, resolver) (rate(contentdownload_http_latency_ms_bucket{resolver=~\"$resolver\", verb=~\"$verb\"}[$__rate_interval])))",
          "legendFormat": "p99 {{resolver}}",
          "instant": false
        }
      ],
      "fieldConfig": { "defaults": { "unit": "ms", "decimals": 0 }, "overrides": [] },
      "options": {
        "legend": { "displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull"] },
        "tooltip": { "mode": "multi" }
      }
    },
    {
      "type": "timeseries",
      "title": "Bytes/sec by Resolver",
      "id": 7,
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 20 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (resolver) (rate(contentdownload_bytes_total{resolver=~\"$resolver\"}[$__rate_interval]))",
          "legendFormat": "{{resolver}}",
          "instant": false
        }
      ],
      "fieldConfig": { "defaults": { "unit": "Bps", "decimals": 2 }, "overrides": [] },
      "options": { "legend": { "displayMode": "table", "placement": "bottom" }, "tooltip": { "mode": "multi" } }
    },
    {
      "type": "timeseries",
      "title": "Attempt Rate by Status (cache-hit / http-200 / http-304)",
      "id": 8,
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 20 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (status) (rate(contentdownload_attempts_total{resolver=~\"$resolver\",status=~\"cache-hit|http-200|http-304\"}[$__rate_interval]))",
          "legendFormat": "{{status}}",
          "instant": false
        }
      ],
      "options": {
        "legend": { "displayMode": "table", "placement": "bottom" },
        "tooltip": { "mode": "multi" },
        "stacking": { "mode": "normal", "group": "A" }
      },
      "fieldConfig": { "defaults": { "unit": "req/s", "decimals": 2 }, "overrides": [] }
    },
    {
      "type": "timeseries",
      "title": "Retries by Reason",
      "id": 9,
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 28 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (reason) (rate(contentdownload_retries_total{resolver=~\"$resolver\"}[$__rate_interval]))",
          "legendFormat": "{{reason}}",
          "instant": false
        }
      ],
      "options": {
        "legend": { "displayMode": "table", "placement": "bottom" },
        "tooltip": { "mode": "multi" },
        "stacking": { "mode": "normal", "group": "A" }
      },
      "fieldConfig": { "defaults": { "unit": "req/s", "decimals": 2 }, "overrides": [] }
    },
    {
      "type": "timeseries",
      "title": "Outcomes by Resolver (success/skip/error)",
      "id": 10,
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 28 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (resolver,outcome) (rate(contentdownload_outcomes_total{resolver=~\"$resolver\"}[$__rate_interval]))",
          "legendFormat": "{{resolver}} • {{outcome}}",
          "instant": false
        }
      ],
      "options": {
        "legend": { "displayMode": "table", "placement": "bottom" },
        "tooltip": { "mode": "multi" },
        "stacking": { "mode": "normal", "group": "A" }
      },
      "fieldConfig": { "defaults": { "unit": "1/s", "decimals": 2 }, "overrides": [] }
    }
    ,
    {
      "type": "timeseries",
      "title": "Rate-limit Sleep (ms) — if exposed",
      "id": 11,
      "gridPos": { "h": 8, "w": 24, "x": 0, "y": 36 },
      "datasource": { "type": "prometheus", "uid": "$datasource" },
      "targets": [
        {
          "refId": "A",
          "expr": "histogram_quantile(0.90, sum by (le, resolver) (rate(contentdownload_rate_sleep_ms_bucket{resolver=~\"$resolver\"}[$__rate_interval])))",
          "legendFormat": "p90 {{resolver}}",
          "instant": false
        }
      ],
      "fieldConfig": { "defaults": { "unit": "ms", "decimals": 0 }, "overrides": [] },
      "options": {
        "legend": { "displayMode": "table", "placement": "bottom" },
        "tooltip": { "mode": "multi" }
      }
    }
  ],
  "refresh": "10s",
  "schemaVersion": 39,
  "style": "dark",
  "tags": ["contentdownload", "hishel", "otel"],
  "templating": {
    "list": [
      {
        "type": "datasource",
        "name": "datasource",
        "label": "Prometheus",
        "query": "prometheus",
        "refresh": 1,
        "current": {}
      },
      {
        "type": "query",
        "name": "resolver",
        "label": "Resolver",
        "datasource": { "type": "prometheus", "uid": "$datasource" },
        "query": "label_values(contentdownload_attempts_total, resolver)",
        "includeAll": true,
        "multi": true,
        "refresh": 2,
        "current": {}
      },
      {
        "type": "custom",
        "name": "verb",
        "label": "Verb",
        "query": "GET,HEAD",
        "includeAll": true,
        "multi": true,
        "current": { "selected": true, "text": "All", "value": ["GET","HEAD"] }
      }
    ]
  },
  "time": { "from": "now-6h", "to": "now" },
  "timepicker": { "refresh_intervals": ["5s","10s","30s","1m","2m","5m"] },
  "timezone": "",
  "title": "ContentDownload (hishel) — Overview",
  "uid": "cd-hishel-overview",
  "version": 1
}
```

---

### Notes & tips

* **Cache Hit Ratio** denominator uses `cache-hit | http-200 | http-304` (the three steady-state outcomes for GET). If you sometimes stream from cache with a 200 (replay), keep `cache-hit` as the indicator.
* **Revalidation Ratio** intentionally excludes `cache-hit`; it’s a ratio of **304** among **network** validations (`200+304`).
* **Latency panel** uses `$verb` — you can set to only **GET** for data fetch or include **HEAD** for preflight probe timing.
* If you implement **rate-limit sleep** as a histogram (suggested name `contentdownload_rate_sleep_ms_bucket`), the last panel will light up; otherwise you can delete it.

If you want a second **“Resolver Drilldown”** dashboard (single resolver focus with more granular legends and per-reason breakdowns), I can generate that too.
