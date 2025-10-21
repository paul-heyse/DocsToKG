-- Artifact Catalog Schema (SQLite/Postgres-compatible)
-- Enables deduplication, verification, retention, and content-addressed storage
-- Version: 1.0 (PR #9)

PRAGMA foreign_keys = ON;

-- Core documents table: one row per unique (artifact_id, source_url, resolver) tuple
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id TEXT NOT NULL,
  source_url TEXT NOT NULL,
  resolver TEXT NOT NULL,
  content_type TEXT,
  bytes INTEGER NOT NULL,
  sha256 TEXT,
  storage_uri TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  run_id TEXT
);

-- Uniqueness constraint: prevents duplicate (artifact_id, source_url, resolver) tuples
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_unique
  ON documents(artifact_id, source_url, resolver);

-- Lookup indexes for common queries
CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_documents_ct ON documents(content_type);
CREATE INDEX IF NOT EXISTS idx_documents_run ON documents(run_id);
CREATE INDEX IF NOT EXISTS idx_documents_artifact ON documents(artifact_id);
CREATE INDEX IF NOT EXISTS idx_documents_resolver ON documents(resolver);

-- Optional variants table: for PDF/HTML/supplement variants of the same artifact
CREATE TABLE IF NOT EXISTS variants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  variant TEXT NOT NULL,
  storage_uri TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  content_type TEXT,
  sha256 TEXT,
  created_at TEXT NOT NULL
);

-- Uniqueness: one variant type per document
CREATE UNIQUE INDEX IF NOT EXISTS idx_variants_unique
  ON variants(document_id, variant);

-- Lookup index
CREATE INDEX IF NOT EXISTS idx_variants_sha ON variants(sha256);
