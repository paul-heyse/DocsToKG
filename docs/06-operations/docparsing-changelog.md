# DocParsing Observability Changelog

## Added

- Stage runner telemetry now records `queue_p95_ms`, `exec_p99_ms`, and `cpu_time_total_ms`
  alongside existing latency metrics in stage summaries and logs. These values are surfaced
  through the DocTags, Chunking, and Embedding summaries to aid SLO tracking and alerting.
