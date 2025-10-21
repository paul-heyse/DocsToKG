PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=4000;

-- 2.1 HTTP calls (after cache & limiter; one row per request)
CREATE TABLE IF NOT EXISTS http_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,                   -- wall clock (time.time())
  host TEXT NOT NULL,
  role TEXT NOT NULL,                 -- metadata|landing|artifact
  method TEXT NOT NULL,               -- GET|HEAD
  status INTEGER,                     -- NULL if exception
  url_hash TEXT NOT NULL,             -- sha256 of canonical URL (to avoid PII)
  from_cache INTEGER,                 -- 0/1/NULL
  revalidated INTEGER,                -- 0/1/NULL
  stale INTEGER,                      -- 0/1/NULL (Hishel SWrV)
  retry_count INTEGER,                -- Tenacity attempts-1
  retry_after_s INTEGER,              -- if honored
  rate_delay_ms INTEGER,              -- limiter wait
  breaker_state TEXT,                 -- closed|half_open|open
  breaker_recorded TEXT,              -- success|failure|none
  elapsed_ms INTEGER,                 -- end-to-end
  error TEXT                          -- short exception class if any
);
CREATE INDEX IF NOT EXISTS idx_http_run ON http_events(run_id);
CREATE INDEX IF NOT EXISTS idx_http_host ON http_events(host);
CREATE INDEX IF NOT EXISTS idx_http_role ON http_events(role);

-- 2.2 Rate limiter events
CREATE TABLE IF NOT EXISTS rate_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,
  host TEXT NOT NULL,
  role TEXT NOT NULL,
  action TEXT NOT NULL,               -- acquire|block|head_skip
  delay_ms INTEGER,
  max_delay_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_rate_run ON rate_events(run_id);
CREATE INDEX IF NOT EXISTS idx_rate_host_role ON rate_events(host, role);

-- 2.3 Breaker transitions (listener)
CREATE TABLE IF NOT EXISTS breaker_transitions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,
  host TEXT NOT NULL,
  scope TEXT NOT NULL,                -- host|resolver
  old_state TEXT NOT NULL,
  new_state TEXT NOT NULL,
  reset_timeout_s INTEGER
);
CREATE INDEX IF NOT EXISTS idx_brk_run ON breaker_transitions(run_id);
CREATE INDEX IF NOT EXISTS idx_brk_host ON breaker_transitions(host);

-- 2.4 Fallback attempts (one row per adapter try)
CREATE TABLE IF NOT EXISTS fallback_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,
  work_id TEXT,
  artifact_id TEXT,
  tier TEXT NOT NULL,
  source TEXT NOT NULL,               -- e.g., unpaywall_pdf
  host TEXT,
  outcome TEXT NOT NULL,              -- success|retryable|nonretryable|timeout|skipped|error|no_pdf
  reason TEXT,                        -- short code
  status INTEGER,
  elapsed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_fb_run ON fallback_attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_fb_source ON fallback_attempts(source);

-- 2.5 Downloads table already has new columns (from migration #7). Ensure indices exist:
CREATE INDEX IF NOT EXISTS idx_dl_sha ON downloads(sha256);
CREATE INDEX IF NOT EXISTS idx_dl_final ON downloads(final_path);
CREATE INDEX IF NOT EXISTS idx_dl_dedupe ON downloads(dedupe_action);

-- 2.6 Run summary (single row per run after aggregation)
CREATE TABLE IF NOT EXISTS run_summary (
  run_id TEXT PRIMARY KEY,
  started_at REAL,
  finished_at REAL,
  artifacts_total INTEGER,
  artifacts_success INTEGER,
  yield_pct REAL,
  ttfp_p50_ms INTEGER,                -- time-to-first-pdf
  ttfp_p95_ms INTEGER,
  cache_hit_pct REAL,                 -- metadata cache hit rate
  rate_delay_p95_ms INTEGER,
  http_429_ratio REAL,
  breaker_open_events INTEGER,
  dedupe_saved_mb REAL,               -- bytes saved / (1024*1024)
  corruption_count INTEGER
);
