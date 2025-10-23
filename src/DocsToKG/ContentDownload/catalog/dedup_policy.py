# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.dedup_policy",
#   "purpose": "Smart deduplication policy and recommendations.",
#   "sections": [
#     {
#       "id": "dedupcandidate",
#       "name": "DedupCandidate",
#       "anchor": "class-dedupcandidate",
#       "kind": "class"
#     },
#     {
#       "id": "deduppolicy",
#       "name": "DedupPolicy",
#       "anchor": "class-deduppolicy",
#       "kind": "class"
#     },
#     {
#       "id": "deduppolicyanalyzer",
#       "name": "DedupPolicyAnalyzer",
#       "anchor": "class-deduppolicyanalyzer",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Smart deduplication policy and recommendations.

Provides cost-aware dedup analysis with:
  - File size thresholds
  - Age constraints
  - Cost/benefit analysis
  - Per-resolver policies
  - Actionable recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from DocsToKG.ContentDownload.catalog.models import DocumentRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DedupCandidate:
    """A file eligible for deduplication."""

    sha256: str
    count: int
    total_size_bytes: int
    storage_savings_bytes: int
    avg_age_days: int
    resolvers: list[str]
    priority: float  # 0.0-1.0, higher = higher priority


@dataclass
class DedupPolicy:
    """Policy for deduplication decisions.

    Attributes:
        min_file_size: Minimum file size to dedup (bytes)
        max_age_days: Don't dedup files older than N days (0 = no limit)
        cost_threshold: Only dedup if savings > cost * this factor
        resolver_allowlist: Only dedup for these resolvers (None = all)
    """

    min_file_size: int = 1_000_000  # 1MB default
    max_age_days: int = 90
    cost_threshold: float = 0.10  # 10% cost savings threshold
    resolver_allowlist: Optional[list[str]] = None

    def __post_init__(self):
        """Validate policy."""
        if self.min_file_size < 0:
            raise ValueError("min_file_size must be >= 0")
        if self.max_age_days < 0:
            raise ValueError("max_age_days must be >= 0")
        if not (0.0 <= self.cost_threshold <= 1.0):
            raise ValueError("cost_threshold must be 0.0-1.0")


class DedupPolicyAnalyzer:
    """Analyze dedup opportunities using smart policies."""

    def __init__(self, policy: Optional[DedupPolicy] = None):
        """Initialize analyzer.

        Args:
            policy: Dedup policy (creates default if not provided)
        """
        self.policy = policy or DedupPolicy()

    def should_dedup(self, candidate: DedupCandidate) -> bool:
        """Determine if a duplicate group should be deduplicated.

        Args:
            candidate: DedupCandidate to evaluate

        Returns:
            True if should dedup
        """
        # Check file size
        avg_size = candidate.total_size_bytes // candidate.count
        if avg_size < self.policy.min_file_size:
            return False

        # Check age
        if self.policy.max_age_days > 0 and candidate.avg_age_days > self.policy.max_age_days:
            return False

        # Check resolver allowlist
        if self.policy.resolver_allowlist:
            if not any(r in self.policy.resolver_allowlist for r in candidate.resolvers):
                return False

        # Check cost-benefit
        savings = candidate.storage_savings_bytes
        cost = avg_size * 0.05  # Assume 5% cost to verify/dedup
        if savings < cost * self.policy.cost_threshold:
            return False

        return True

    def analyze_duplicates(
        self,
        all_records: list[DocumentRecord],
    ) -> list[DedupCandidate]:
        """Analyze all duplicates against policy.

        Args:
            all_records: All catalog records

        Returns:
            List of DedupCandidate sorted by priority
        """
        if not all_records:
            return []

        now = datetime.utcnow()

        # Group by SHA-256
        groups: dict[str, list] = {}
        for record in all_records:
            if record.sha256 and record.sha256 not in groups:
                groups[record.sha256] = []
            if record.sha256:
                groups[record.sha256].append(record)

        # Find duplicates and analyze
        candidates = []
        for sha256, records in groups.items():
            if len(records) < 2:
                continue

            total_size = sum(r.bytes for r in records)
            avg_size = total_size // len(records)

            # Calculate avg age
            ages = [(now - r.created_at).days for r in records]
            avg_age = sum(ages) // len(ages) if ages else 0

            resolvers = list(set(r.resolver for r in records))

            # Calculate savings
            savings = (len(records) - 1) * avg_size

            # Calculate priority
            priority = self._calculate_priority(
                count=len(records),
                total_size=total_size,
                avg_age=avg_age,
            )

            candidate = DedupCandidate(
                sha256=sha256,
                count=len(records),
                total_size_bytes=total_size,
                storage_savings_bytes=savings,
                avg_age_days=avg_age,
                resolvers=resolvers,
                priority=priority,
            )

            candidates.append(candidate)

        # Sort by priority (highest first)
        candidates.sort(key=lambda c: c.priority, reverse=True)

        return candidates

    def _calculate_priority(
        self,
        count: int,
        total_size: int,
        avg_age: int,
    ) -> float:
        """Calculate priority score for a duplicate group.

        Args:
            count: Number of duplicates
            total_size: Total size of all copies
            avg_age: Average age in days

        Returns:
            Priority 0.0-1.0
        """
        # Factors:
        # - More duplicates = higher priority
        # - Larger files = higher priority
        # - Newer files = higher priority (hot data)

        dup_factor = min(1.0, (count - 1) / 10)  # Up to 10 duplicates
        size_factor = min(1.0, total_size / (1024 * 1024 * 1024))  # Up to 1GB
        age_factor = max(0.0, 1.0 - (avg_age / 365.0))  # Decay over 1 year

        # Weighted average (size matters most)
        priority = 0.2 * dup_factor + 0.6 * size_factor + 0.2 * age_factor

        return priority

    def recommendations(
        self,
        all_records: list[DocumentRecord],
    ) -> list[str]:
        """Generate dedup recommendations.

        Args:
            all_records: All catalog records

        Returns:
            List of recommendation strings
        """
        recommendations = []

        candidates = self.analyze_duplicates(all_records)
        good_candidates = [c for c in candidates if self.should_dedup(c)]

        if not good_candidates:
            recommendations.append("No dedup opportunities at current policy settings")
            return recommendations

        # Total potential savings
        total_savings = sum(c.storage_savings_bytes for c in good_candidates)
        recommendations.append(
            f"Total savings potential: {total_savings / 1024 / 1024 / 1024:.1f}GB "
            f"across {len(good_candidates)} duplicate groups"
        )

        # Top opportunity
        if good_candidates:
            top = good_candidates[0]
            recommendations.append(
                f"Top opportunity: {top.count} copies, "
                f"saves {top.storage_savings_bytes / 1024 / 1024:.0f}MB"
            )

        # Per-resolver analysis
        by_resolver: dict[str, int] = {}
        for candidate in good_candidates:
            for resolver in candidate.resolvers:
                by_resolver[resolver] = (
                    by_resolver.get(resolver, 0) + candidate.storage_savings_bytes
                )

        if by_resolver:
            top_resolver = max(by_resolver.items(), key=lambda x: x[1])
            recommendations.append(
                f"Top resolver for dedup: {top_resolver[0]} "
                f"({top_resolver[1] / 1024 / 1024 / 1024:.1f}GB)"
            )

        # Policy suggestions
        if len(good_candidates) > 100:
            recommendations.append(
                "High dedup opportunity detected. Consider lowering min_file_size "
                "or max_age_days for more savings."
            )

        return recommendations

    def get_top_candidates(
        self,
        all_records: list[DocumentRecord],
        n: int = 10,
    ) -> list[DedupCandidate]:
        """Get top N dedup candidates by priority.

        Args:
            all_records: All catalog records
            n: Number to return

        Returns:
            Top N candidates
        """
        candidates = self.analyze_duplicates(all_records)
        good = [c for c in candidates if self.should_dedup(c)]
        return good[:n]
