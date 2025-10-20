"""Cache policy resolution and request routing.

Responsibilities
----------------
- Provide CacheDecision dataclass to represent caching decisions
- Implement CacheRouter to resolve (host, role) → CacheDecision
- Generate human-readable policy table for operations/debugging
- Handle host key normalization transparently

Design Notes
------------
- Conservative by default: unknown hosts not cached
- Role-based isolation: metadata/landing cached, artifacts raw
- Hierarchical TTL fallback: role-specific → host → default
- Fast O(1) lookups after initialization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from DocsToKG.ContentDownload.cache_loader import (
    CacheConfig,
    CacheDefault,
    _normalize_host_key,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheDecision:
    """Decision whether and how to cache a specific request."""

    use_cache: bool
    ttl_s: Optional[int] = None
    swrv_s: Optional[int] = None
    body_key: bool = False


class CacheRouter:
    """Stateful policy resolver that routes requests to cached vs raw clients.

    Responsibilities:
        - Hold CacheConfig loaded from cache_loader
        - Resolve (host, role) → CacheDecision
        - Print effective policy table at startup for ops
        - Handle host key normalization
    """

    def __init__(self, config: CacheConfig) -> None:
        """Initialize router with cache configuration.

        Args:
            config: CacheConfig from cache_loader.load_cache_config()

        Side Effects:
            Logs effective policy table at INFO level
        """
        self.config = config
        LOGGER.info("Cache routing policy loaded with %d hosts", len(config.hosts))

    def resolve_policy(
        self,
        host: str,
        role: str = "metadata",
    ) -> CacheDecision:
        """Resolve caching policy for a request.

        Logic:
            1. Normalize host key (lowercase + punycode)
            2. If host not in config.hosts → CacheDecision(use_cache=False)
            3. If role == "artifact" → CacheDecision(use_cache=False)
            4. If role in host_policy.role → use role-specific TTL/SWrV
            5. Else if host_policy.ttl_s is set → use host TTL
            6. Else → use controller.default

        Args:
            host: Hostname (will be normalized)
            role: "metadata" | "landing" | "artifact" (default: "metadata")

        Returns:
            CacheDecision with use_cache flag and optional TTL/SWrV

        Examples:
            >>> router.resolve_policy("api.crossref.org", "metadata")
            CacheDecision(use_cache=True, ttl_s=259200, swrv_s=180)

            >>> router.resolve_policy("example.com", "metadata")
            CacheDecision(use_cache=False)

            >>> router.resolve_policy("api.crossref.org", "artifact")
            CacheDecision(use_cache=False)
        """
        # 1. Normalize host key
        normalized_host = _normalize_host_key(host)

        # 2. Unknown host → not cached
        if normalized_host not in self.config.hosts:
            return CacheDecision(use_cache=False)

        # 3. Artifacts never cached
        if role == "artifact":
            return CacheDecision(use_cache=False)

        # 4. Get host policy
        host_policy = self.config.hosts[normalized_host]

        # 5. Try role-specific policy
        if role in host_policy.role:
            role_policy = host_policy.role[role]
            if role_policy.ttl_s is not None:
                swrv = (
                    role_policy.swrv_s
                    if role == "metadata" and role_policy.swrv_s is not None
                    else None
                )
                return CacheDecision(
                    use_cache=True,
                    ttl_s=role_policy.ttl_s,
                    swrv_s=swrv,
                    body_key=role_policy.body_key,
                )

        # 6. Fall back to host-level TTL
        if host_policy.ttl_s is not None:
            return CacheDecision(use_cache=True, ttl_s=host_policy.ttl_s)

        # 7. Fall back to controller default
        use_cache = self.config.controller.default == CacheDefault.CACHE
        return CacheDecision(use_cache=use_cache)

    def print_effective_policy(self) -> str:
        """Generate human-readable policy table for operations.

        Returns:
            Multi-line string with table of hosts and their TTL/SWrV values

        Example output:
            Host                    Role        TTL (days)  SWrV (min)
            ──────────────────────  ──────────  ──────────  ──────────
            api.crossref.org        metadata    3           3
            api.crossref.org        landing     1           -
            api.openalex.org        metadata    3           3
            ...
        """
        if not self.config.hosts:
            return "Cache routing: no hosts configured (default=DO_NOT_CACHE)"

        lines = [
            "Effective Cache Routing Policy",
            "=" * 80,
            "",
            f"{'Host':<28}{'Role':<12}{'TTL (days)':<12}{'SWrV (min)':<12}",
            "─" * 80,
        ]

        for host_key in sorted(self.config.hosts.keys()):
            host_policy = self.config.hosts[host_key]
            roles = list(host_policy.role.keys()) if host_policy.role else ["(default)"]

            for i, role in enumerate(roles):
                if i > 0:
                    host_display = ""
                else:
                    host_display = host_key

                if role == "(default)":
                    ttl_days = f"{host_policy.ttl_s / 86400:.1f}" if host_policy.ttl_s else "-"
                    swrv = "-"
                else:
                    role_policy = host_policy.role.get(role)
                    if role_policy:
                        ttl_days = f"{role_policy.ttl_s / 86400:.1f}" if role_policy.ttl_s else "-"
                        swrv = f"{role_policy.swrv_s / 60:.0f}" if role_policy.swrv_s else "-"
                    else:
                        ttl_days = "-"
                        swrv = "-"

                lines.append(f"{host_display:<28}{role:<12}{ttl_days:<12}{swrv:<12}")

        lines.extend(["", "─" * 80])
        lines.append(f"Default for unknown hosts: {self.config.controller.default.value}")

        return "\n".join(lines)
