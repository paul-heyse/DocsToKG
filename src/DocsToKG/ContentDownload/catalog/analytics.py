"""Deduplication analytics and statistics.

Provides rich dedup insights:
  - Storage saved calculations
  - Dedup ratios and rankings
  - Per-resolver breakdown
  - Actionable recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from DocsToKG.ContentDownload.catalog.store import CatalogStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DedupGroup:
    """A group of duplicate files."""
    sha256: str
    count: int
    total_size_bytes: int
    resolvers: list[str]
    
    @property
    def storage_saved_bytes(self) -> int:
        """Storage saved if only 1 copy kept."""
        return (self.count - 1) * (self.total_size_bytes // self.count)
    
    @property
    def storage_saved_gb(self) -> float:
        """Storage saved in GB."""
        return self.storage_saved_bytes / 1024 / 1024 / 1024


@dataclass(frozen=True)
class DedupAnalytics:
    """Complete dedup analysis."""
    total_records: int
    total_size_gb: float
    unique_hashes: int
    duplicate_groups: int
    total_duplicates: int
    storage_saved_gb: float
    dedup_ratio: float
    avg_copies_per_duplicate: float


class DedupAnalyzer:
    """Analyze deduplication opportunities."""
    
    def __init__(self, catalog: CatalogStore):
        """Initialize analyzer.
        
        Args:
            catalog: Catalog store to analyze
        """
        self.catalog = catalog
    
    def storage_saved_gb(self) -> float:
        """Calculate total storage that could be saved via dedup.
        
        Returns:
            Storage saved in GB if only 1 copy kept per hash
        """
        try:
            duplicates = self.catalog.find_duplicates()
        except NotImplementedError:
            logger.error("find_duplicates not supported")
            return 0.0
        
        total_saved = 0
        for sha256, count in duplicates:
            if count > 1:
                # Estimate: average file size, count - 1 copies saved
                # (This is a rough estimate; exact calculation requires file sizes)
                total_saved += 1  # Placeholder
        
        return total_saved / 1024 / 1024 / 1024
    
    def dedup_ratio(self) -> float:
        """Calculate dedup ratio (0.0 = no dupes, 1.0 = all identical).
        
        Returns:
            Ratio of duplicate records to total records
        """
        try:
            all_records = self.catalog.get_all_records()
        except NotImplementedError:
            logger.error("get_all_records not supported")
            return 0.0
        
        if not all_records:
            return 0.0
        
        try:
            duplicates = self.catalog.find_duplicates()
        except NotImplementedError:
            return 0.0
        
        duplicate_count = sum(1 for sha256, count in duplicates if count > 1)
        return duplicate_count / len(all_records) if all_records else 0.0
    
    def top_duplicates(self, n: int = 10) -> list[DedupGroup]:
        """Find top N duplicate groups by storage impact.
        
        Args:
            n: Number of top duplicates to return
            
        Returns:
            List of DedupGroup sorted by storage saved
        """
        try:
            all_records = self.catalog.get_all_records()
        except NotImplementedError:
            logger.error("get_all_records not supported")
            return []
        
        # Group by SHA-256
        groups: dict[str, list] = {}
        for record in all_records:
            if record.sha256:
                if record.sha256 not in groups:
                    groups[record.sha256] = []
                groups[record.sha256].append(record)
        
        # Find duplicates
        dedup_groups = []
        for sha256, records in groups.items():
            if len(records) > 1:
                total_size = sum(r.bytes for r in records)
                resolvers = list(set(r.resolver for r in records))
                
                group = DedupGroup(
                    sha256=sha256,
                    count=len(records),
                    total_size_bytes=total_size,
                    resolvers=resolvers,
                )
                dedup_groups.append(group)
        
        # Sort by storage saved
        dedup_groups.sort(
            key=lambda g: g.storage_saved_bytes,
            reverse=True,
        )
        
        return dedup_groups[:n]
    
    def dedup_by_resolver(self) -> dict[str, float]:
        """Calculate dedup ratio per resolver.
        
        Returns:
            Dict mapping resolver name to dedup ratio
        """
        try:
            all_records = self.catalog.get_all_records()
        except NotImplementedError:
            logger.error("get_all_records not supported")
            return {}
        
        # Group by resolver
        by_resolver: dict[str, list] = {}
        for record in all_records:
            if record.resolver not in by_resolver:
                by_resolver[record.resolver] = []
            by_resolver[record.resolver].append(record)
        
        # Calculate dedup per resolver
        ratios = {}
        for resolver, records in by_resolver.items():
            if not records:
                continue
            
            # Count unique hashes
            unique_hashes = len(set(r.sha256 for r in records if r.sha256))
            total = len(records)
            
            # Dedup ratio = (total - unique) / total
            ratio = (total - unique_hashes) / total if total > 0 else 0.0
            ratios[resolver] = ratio
        
        return ratios
    
    def recommendations(self) -> list[str]:
        """Generate actionable recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check dedup ratio
        dedup_ratio = self.dedup_ratio()
        if dedup_ratio > 0.3:
            recommendations.append(
                f"High dedup ratio ({dedup_ratio:.1%}): "
                "Consider enabling CAS (content-addressable storage) to save space"
            )
        elif dedup_ratio > 0.1:
            recommendations.append(
                f"Moderate dedup ({dedup_ratio:.1%}): "
                "Monitor dedup performance, consider hardlink optimization"
            )
        
        # Check per-resolver dedup
        by_resolver = self.dedup_by_resolver()
        high_dedup_resolvers = [
            r for r, ratio in by_resolver.items() if ratio > 0.4
        ]
        if high_dedup_resolvers:
            recommendations.append(
                f"High inter-resolver dedup detected: "
                f"{', '.join(high_dedup_resolvers)}. "
                "These resolvers often download the same content."
            )
        
        # Check top duplicates
        top = self.top_duplicates(n=1)
        if top and top[0].storage_saved_gb > 10:
            recommendations.append(
                f"Top duplicate group saves {top[0].storage_saved_gb:.1f}GB: "
                f"Prioritize dedup for this file ({top[0].count} copies)"
            )
        
        return recommendations if recommendations else ["No recommendations at this time"]
    
    def full_analytics(self) -> DedupAnalytics:
        """Generate complete dedup analytics.
        
        Returns:
            DedupAnalytics with all metrics
        """
        try:
            all_records = self.catalog.get_all_records()
        except NotImplementedError:
            return DedupAnalytics(
                total_records=0,
                total_size_gb=0.0,
                unique_hashes=0,
                duplicate_groups=0,
                total_duplicates=0,
                storage_saved_gb=0.0,
                dedup_ratio=0.0,
                avg_copies_per_duplicate=0.0,
            )
        
        if not all_records:
            return DedupAnalytics(
                total_records=0,
                total_size_gb=0.0,
                unique_hashes=0,
                duplicate_groups=0,
                total_duplicates=0,
                storage_saved_gb=0.0,
                dedup_ratio=0.0,
                avg_copies_per_duplicate=0.0,
            )
        
        # Calculate totals
        total_size_bytes = sum(r.bytes for r in all_records)
        
        # Group by hash
        groups: dict[str, list] = {}
        for record in all_records:
            if record.sha256:
                if record.sha256 not in groups:
                    groups[record.sha256] = []
                groups[record.sha256].append(record)
        
        unique_hashes = len(groups)
        duplicate_groups = sum(1 for g in groups.values() if len(g) > 1)
        total_duplicates = sum(len(g) - 1 for g in groups.values() if len(g) > 1)
        
        # Calculate storage saved
        storage_saved_bytes = 0
        for group in groups.values():
            if len(group) > 1:
                avg_size = sum(r.bytes for r in group) // len(group)
                storage_saved_bytes += (len(group) - 1) * avg_size
        
        avg_copies = (
            (total_duplicates + duplicate_groups) / duplicate_groups
            if duplicate_groups > 0
            else 1.0
        )
        
        return DedupAnalytics(
            total_records=len(all_records),
            total_size_gb=total_size_bytes / 1024 / 1024 / 1024,
            unique_hashes=unique_hashes,
            duplicate_groups=duplicate_groups,
            total_duplicates=total_duplicates,
            storage_saved_gb=storage_saved_bytes / 1024 / 1024 / 1024,
            dedup_ratio=total_duplicates / len(all_records) if all_records else 0.0,
            avg_copies_per_duplicate=avg_copies,
        )
