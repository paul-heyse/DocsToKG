# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.job_effects",
#   "purpose": "Exactly-once operation logging and replay",
#   "sections": [
#     {
#       "id": "extract-result-json",
#       "name": "_extract_result_json",
#       "anchor": "function-extract-result-json",
#       "kind": "function"
#     },
#     {
#       "id": "run-effect",
#       "name": "run_effect",
#       "anchor": "function-run-effect",
#       "kind": "function"
#     },
#     {
#       "id": "get-effect-result",
#       "name": "get_effect_result",
#       "anchor": "function-get-effect-result",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Exactly-once operation logging and replay.

This module provides mechanisms to ensure side-effects (HEAD requests, streaming,
finalization, indexing, deduplication) are performed exactly once, even across
worker crashes and retries. It works by recording operation attempts and their
results in a ledger; if the same operation is attempted again, the stored result
is returned without repeating the effect.

Key Features:
  - Idempotent operation tracking via operation keys
  - Automatic result caching and replay
  - Efficient result storage (small JSON payloads)
  - Multi-worker safe with SQLite foreign keys

Example:
  ```python
  import sqlite3
  from DocsToKG.ContentDownload.job_effects import run_effect

  conn = sqlite3.connect("manifest.sqlite3")

  def fetch_head(url):
      response = requests.head(url, timeout=10)
      return {
          "code": "OK",
          "status": response.status_code,
          "content_length": int(response.headers.get("Content-Length", 0))
      }

  # First call: performs the effect and stores result
  result = run_effect(
      conn,
      job_id="job-123",
      kind="HEAD",
      opkey="op-key-abc123",
      effect_fn=lambda: fetch_head("https://example.org/paper.pdf")
  )

  # Second call with same opkey: returns stored result immediately
  result2 = run_effect(
      conn,
      job_id="job-123",
      kind="HEAD",
      opkey="op-key-abc123",
      effect_fn=lambda: fetch_head("https://example.org/paper.pdf")
  )
  # result2 == result (no actual HTTP call made)
  ```
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Callable, Optional


def _extract_result_json(row: Any) -> Optional[str]:
    """Return the ``result_json`` column from a SQLite row.

    The helper accepts rows returned both as ``sqlite3.Row`` mappings and as
    tuple/list sequences (the default when ``row_factory`` is unset).
    """

    if row is None:
        return None

    # ``sqlite3.Row`` implements ``keys`` while tuples/lists do not.
    keys = getattr(row, "keys", None)
    if callable(keys) and "result_json" in keys():
        return row["result_json"]

    if isinstance(row, (tuple, list)):
        return row[0] if row else None

    return None


def run_effect(
    cx: sqlite3.Connection,
    *,
    job_id: str,
    kind: str,
    opkey: str,
    effect_fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Execute a side-effect exactly once, with result caching.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection (must have artifact_ops table)
    job_id : str
        Job ID (for referential integrity)
    kind : str
        Operation type: HEAD, STREAM, FINALIZE, INDEX, DEDUPE, etc.
    opkey : str
        Operation idempotency key (result of op_key())
    effect_fn : callable
        Function that performs the side-effect and returns a dict.
        Called only if the operation hasn't been recorded yet.

    Returns
    -------
    dict
        Result returned by effect_fn (or cached from previous run)

    Notes
    -----
    If the operation is being attempted for the first time:
      1. INSERT the operation record with status started_at
      2. Call effect_fn() to perform the side-effect
      3. UPDATE the operation record with finished_at, result_code, result_json
      4. Return the result

    If the operation was already recorded:
      1. Query the existing record
      2. Return the cached result_json (deserialized)

    The result_json is capped at 20KB to avoid bloating the DB.
    If effect_fn() returns a larger result, it should store the bulk
    elsewhere and put a small reference in the returned dict.
    """
    now = time.time()

    try:
        cx.execute(
            """INSERT INTO artifact_ops(op_key, job_id, op_type, started_at)
               VALUES (?, ?, ?, ?)""",
            (opkey, job_id, kind, now),
        )
    except sqlite3.IntegrityError:
        # Operation already recorded; fetch and return cached result
        row = cx.execute(
            "SELECT result_json FROM artifact_ops WHERE op_key=?",
            (opkey,),
        ).fetchone()
        result_json = _extract_result_json(row)
        if result_json:
            return json.loads(result_json)
        return {}

    # First attempt: perform the effect
    result = effect_fn()

    # Store the result
    cx.execute(
        """UPDATE artifact_ops
           SET finished_at=?, result_code=?, result_json=?
           WHERE op_key=?""",
        (
            time.time(),
            result.get("code", "OK"),
            json.dumps(result)[:20000],
            opkey,
        ),
    )

    return result


def get_effect_result(
    cx: sqlite3.Connection,
    *,
    opkey: str,
) -> Optional[dict[str, Any]]:
    """Retrieve a previously recorded operation result.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection
    opkey : str
        Operation idempotency key

    Returns
    -------
    dict or None
        Stored result_json (deserialized), or None if not found

    Notes
    -----
    This is a read-only query useful for debugging or auditing.
    For normal operation, use run_effect() which handles both
    lookup and execution in one call.
    """
    row = cx.execute(
        "SELECT result_json FROM artifact_ops WHERE op_key=?",
        (opkey,),
    ).fetchone()
    result_json = _extract_result_json(row)
    if result_json:
        return json.loads(result_json)
    return None
