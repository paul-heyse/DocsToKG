# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.redis_cooldown_store",
#   "purpose": "Redis-backed cooldown store for distributed circuit breaker coordination.",
#   "sections": [
#     {
#       "id": "rediscooldownstore",
#       "name": "RedisCooldownStore",
#       "anchor": "class-rediscooldownstore",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Redis-backed cooldown store for distributed circuit breaker coordination.

Implements the CooldownStore protocol from breakers.py using Redis for
cross-host/multi-machine sharing. Handles wall-clock to monotonic time
conversion to ensure immunity to clock drift.

Example:
    from DocsToKG.ContentDownload.redis_cooldown_store import RedisCooldownStore

    store = RedisCooldownStore("redis://localhost:6379/3")
    store.set_until(host="api.example.org", until_monotonic=150.5, reason="429-timeout")
    deadline = store.get_until(host="api.example.org")  # returns monotonic time
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse

try:
    import redis  # pip install redis
except ImportError as e:  # pragma: no cover
    raise RuntimeError("redis-py is required for RedisCooldownStore") from e


@dataclass
class RedisCooldownStore:
    """
    Cross-host cooldown store in Redis.
    Stores wall-clock deadlines and converts to monotonic when read.

    Parameters
    ----------
    dsn : str
        Redis connection string. Examples:
        - redis://localhost:6379/3
        - rediss://user:pass@host:6380/1 (TLS)
    key_prefix : str
        Prefix for Redis keys (default "breaker:cooldown:")
    now_wall : Callable
        Function returning wall-clock time (default time.time)
    now_mono : Callable
        Function returning monotonic time (default time.monotonic)

    Notes
    -----
    Keys stored as JSON: {"until_wall": <float>, "reason": "<str>"}
    TTL automatically set to ceil(until_wall - now)
    """

    dsn: str
    key_prefix: str = "breaker:cooldown:"
    now_wall: Callable[[], float] = time.time
    now_mono: Callable[[], float] = time.monotonic
    _client: Optional[redis.Redis] = None

    def __post_init__(self) -> None:
        url = urlparse(self.dsn)
        db = int((url.path or "/0").lstrip("/"))
        # Don't enable decode_responses; we store JSON bytes.
        self._client = redis.Redis(
            host=url.hostname or "localhost",
            port=url.port or 6379,
            db=db,
            username=url.username,
            password=url.password,
            ssl=(url.scheme == "rediss"),
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            decode_responses=False,
        )
        # Verify connection
        self._client.ping()

    def _key(self, host: str) -> str:
        """Generate Redis key for host."""
        return f"{self.key_prefix}{host}"

    # ---- CooldownStore API (protocol from breakers.py) ----

    def get_until(self, host: str) -> Optional[float]:
        """
        Return monotonic deadline for host if a cooldown exists and is in the future.

        Parameters
        ----------
        host : str
            Hostname (punycode normalized)

        Returns
        -------
        Optional[float]
            Monotonic deadline if active, None if expired or not set
        """
        raw = self._client.get(self._key(host))
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            until_wall = float(obj.get("until_wall", 0.0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

        now_w = self.now_wall()
        now_m = self.now_mono()

        if until_wall <= now_w:
            # Expired → clean it up (best-effort)
            self.clear(host)
            return None

        # Convert wall-clock → monotonic so callers compare apples-to-apples
        return now_m + max(0.0, until_wall - now_w)

    def set_until(self, host: str, until_monotonic: float, reason: str) -> None:
        """
        Write/update cooldown for host with a monotonic deadline.

        Parameters
        ----------
        host : str
            Hostname (punycode normalized)
        until_monotonic : float
            Deadline in monotonic time
        reason : str
            Reason for cooldown (e.g., "429-retry-after", "rolling-window-5xx")
        """
        now_w = self.now_wall()
        now_m = self.now_mono()

        # Convert monotonic → wall-clock for cross-process sharing
        until_wall = now_w + max(0.0, until_monotonic - now_m)
        ttl = max(1, int(round(until_wall - now_w)))

        obj = json.dumps(
            {
                "until_wall": until_wall,
                "reason": str(reason)[:128],
            }
        ).encode("utf-8")

        self._client.set(self._key(host), obj, ex=ttl)

    def clear(self, host: str) -> None:
        """Clear cooldown for host."""
        self._client.delete(self._key(host))

    # ---- Optional maintenance helpers ----

    def prune_all(self) -> int:
        """Delete all expired cooldowns. Returns count removed."""
        pattern = f"{self.key_prefix}*"
        keys = self._client.keys(pattern)
        if not keys:
            return 0
        count = 0
        now_w = self.now_wall()
        for key in keys:
            try:
                raw = self._client.get(key)
                if raw:
                    obj = json.loads(raw)
                    until_wall = float(obj.get("until_wall", 0.0))
                    if until_wall < now_w:
                        if self._client.delete(key):
                            count += 1
            except Exception:
                pass
        return count

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
