# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.retention",
#   "purpose": "Multi-dimensional retention policy engine.",
#   "sections": [
#     {
#       "id": "retentiondecision",
#       "name": "RetentionDecision",
#       "anchor": "class-retentiondecision",
#       "kind": "class"
#     },
#     {
#       "id": "retentionpolicy",
#       "name": "RetentionPolicy",
#       "anchor": "class-retentionpolicy",
#       "kind": "class"
#     },
#     {
#       "id": "retentionevaluator",
#       "name": "RetentionEvaluator",
#       "anchor": "class-retentionevaluator",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Multi-dimensional retention policy engine.

Provides fine-grained lifecycle management with:
  - Age-based retention
  - Content-type policies
  - Resolver-specific rules
  - Size-tier policies
  - Replica constraints
  - Score-based decisions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from DocsToKG.ContentDownload.catalog.models import DocumentRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionDecision:
    """Decision on whether to retain a record."""

    record_id: int
    artifact_id: str
    should_retain: bool
    reasons: list[str]
    score: float  # 0.0-1.0, higher = more important to keep


@dataclass
class RetentionPolicy:
    """Multi-dimensional retention policy.

    Attributes:
        age_days: Base retention age in days (0 = disabled)
        content_type_rules: Override age per content-type (days)
        resolver_rules: Override age per resolver (days)
        size_tier_rules: Override age per size tier (bytes -> days)
        min_replicas: Keep if at least N replicas exist globally
    """

    age_days: int = 180
    content_type_rules: dict[str, int] = field(default_factory=dict)
    resolver_rules: dict[str, int] = field(default_factory=dict)
    size_tier_rules: dict[str, int] = field(default_factory=dict)
    min_replicas: int = 1

    def __post_init__(self):
        """Validate policy."""
        if self.age_days < 0:
            raise ValueError("age_days must be >= 0")
        if self.min_replicas < 1:
            raise ValueError("min_replicas must be >= 1")

    def get_retention_age(
        self,
        record: DocumentRecord,
    ) -> int:
        """Get retention age for a specific record.

        Applies overrides in order:
          1. content_type_rules
          2. resolver_rules
          3. size_tier_rules
          4. base age_days

        Args:
            record: Document record

        Returns:
            Retention age in days
        """
        age = self.age_days

        # Content-type override
        if record.content_type and record.content_type in self.content_type_rules:
            age = self.content_type_rules[record.content_type]

        # Resolver override (takes priority)
        if record.resolver in self.resolver_rules:
            age = self.resolver_rules[record.resolver]

        # Size-tier override (takes priority)
        tier = self._get_size_tier(record.bytes)
        if tier in self.size_tier_rules:
            age = self.size_tier_rules[tier]

        return age

    def _get_size_tier(self, bytes_size: int) -> str:
        """Classify file into size tier.

        Args:
            bytes_size: File size in bytes

        Returns:
            Size tier name
        """
        mb = bytes_size / 1024 / 1024

        if mb < 1:
            return "tiny"
        elif mb < 10:
            return "small"
        elif mb < 100:
            return "medium"
        elif mb < 1000:
            return "large"
        else:
            return "huge"


class RetentionEvaluator:
    """Evaluate retention decisions using policies."""

    def __init__(
        self,
        catalog_catalog,
        policy: Optional[RetentionPolicy] = None,
    ):
        """Initialize evaluator.

        Args:
            catalog_catalog: Catalog store for replica counting
            policy: Retention policy (creates default if not provided)
        """
        self.catalog = catalog_catalog
        self.policy = policy or RetentionPolicy()

    def evaluate_record(
        self,
        record: DocumentRecord,
        now: Optional[datetime] = None,
        replica_count: Optional[int] = None,
    ) -> RetentionDecision:
        """Evaluate whether to retain a record.

        Args:
            record: Document record
            now: Current time (uses now() if not provided)
            replica_count: Number of replicas (counts if not provided)

        Returns:
            RetentionDecision with reasoning
        """
        if now is None:
            now = datetime.utcnow()

        reasons = []
        should_retain = True
        score = 0.5  # Neutral default

        # Check age
        retention_age = self.policy.get_retention_age(record)
        if retention_age > 0:
            created = record.created_at
            age_days = (now - created).days

            if age_days > retention_age:
                reasons.append(f"Age exceeded: {age_days} > {retention_age} days")
                should_retain = False
                score -= 0.3
            else:
                reasons.append(f"Age acceptable: {age_days}/{retention_age} days")
                score += 0.2

        # Check replica constraint
        if replica_count is None:
            try:
                replicas = self._count_replicas(record.sha256) if record.sha256 else 1
            except Exception:
                replicas = 1
        else:
            replicas = replica_count

        if replicas >= self.policy.min_replicas:
            reasons.append(f"Replicas ok: {replicas} >= {self.policy.min_replicas}")
            score += 0.1
        else:
            reasons.append(f"Keep for replication: {replicas} < {self.policy.min_replicas}")
            should_retain = True  # Always keep if below min_replicas
            score += 0.4

        # Clamp score
        score = max(0.0, min(1.0, score))

        return RetentionDecision(
            record_id=record.id,
            artifact_id=record.artifact_id,
            should_retain=should_retain,
            reasons=reasons,
            score=score,
        )

    def _count_replicas(self, sha256: Optional[str]) -> int:
        """Count replicas of a file (records with same sha256).

        Args:
            sha256: SHA-256 hash to count

        Returns:
            Number of replicas
        """
        if not sha256:
            return 1

        try:
            records = self.catalog.get_by_sha256(sha256)
            return len(records)
        except NotImplementedError:
            return 1

    def evaluate_batch(
        self,
        records: list[DocumentRecord],
        now: Optional[datetime] = None,
    ) -> list[RetentionDecision]:
        """Evaluate batch of records.

        Args:
            records: Document records
            now: Current time

        Returns:
            List of RetentionDecision
        """
        decisions = []
        for record in records:
            decision = self.evaluate_record(record, now=now)
            decisions.append(decision)

        return decisions

    def candidates_for_deletion(
        self,
        records: list[DocumentRecord],
        now: Optional[datetime] = None,
    ) -> list[RetentionDecision]:
        """Find records eligible for deletion.

        Args:
            records: Document records
            now: Current time

        Returns:
            List of RetentionDecision with should_retain=False
        """
        decisions = self.evaluate_batch(records, now=now)
        return [d for d in decisions if not d.should_retain]

    def retention_stats(
        self,
        records: list[DocumentRecord],
        now: Optional[datetime] = None,
    ) -> dict:
        """Generate retention statistics.

        Args:
            records: Document records
            now: Current time

        Returns:
            Statistics dict
        """
        if not records:
            return {
                "total_records": 0,
                "retained": 0,
                "deletable": 0,
                "retention_rate": 0.0,
            }

        decisions = self.evaluate_batch(records, now=now)

        retained = sum(1 for d in decisions if d.should_retain)
        deletable = sum(1 for d in decisions if not d.should_retain)

        return {
            "total_records": len(records),
            "retained": retained,
            "deletable": deletable,
            "retention_rate": retained / len(records) if records else 0.0,
            "avg_score": sum(d.score for d in decisions) / len(decisions),
        }
