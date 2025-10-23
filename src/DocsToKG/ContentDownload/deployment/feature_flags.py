# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.deployment.feature_flags",
#   "purpose": "Feature flags system for safe feature deployment and experimentation.",
#   "sections": [
#     {
#       "id": "rolloutstrategy",
#       "name": "RolloutStrategy",
#       "anchor": "class-rolloutstrategy",
#       "kind": "class"
#     },
#     {
#       "id": "featureflag",
#       "name": "FeatureFlag",
#       "anchor": "class-featureflag",
#       "kind": "class"
#     },
#     {
#       "id": "featureflagmanager",
#       "name": "FeatureFlagManager",
#       "anchor": "class-featureflagmanager",
#       "kind": "class"
#     },
#     {
#       "id": "get-feature-flag-manager",
#       "name": "get_feature_flag_manager",
#       "anchor": "function-get-feature-flag-manager",
#       "kind": "function"
#     },
#     {
#       "id": "is-feature-enabled",
#       "name": "is_feature_enabled",
#       "anchor": "function-is-feature-enabled",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Feature flags system for safe feature deployment and experimentation.

Enables safe rollout of new features without requiring code redeployment.
Supports per-request feature toggling, gradual rollouts, and A/B testing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class RolloutStrategy(str, Enum):
    """Rollout strategies for gradual feature enabling."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    CANARY = "canary"  # Percentage-based gradual rollout
    AB_TEST = "ab_test"  # A/B testing with control/treatment groups


@dataclass(frozen=True)
class FeatureFlag:
    """Feature flag configuration."""

    name: str
    enabled: bool = False
    strategy: RolloutStrategy = RolloutStrategy.DISABLED
    rollout_percentage: int = 0  # 0-100 for canary rollouts
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def should_enable(self, user_id: str | None = None) -> bool:
        """Determine if feature should be enabled for user."""
        if not self.enabled:
            return False
        if self.strategy == RolloutStrategy.DISABLED:
            return False
        if self.strategy == RolloutStrategy.ENABLED:
            return True
        if self.strategy == RolloutStrategy.CANARY:
            if user_id is None:
                return False
            # Hash-based canary: deterministic per user
            hash_val = hash(f"{self.name}:{user_id}") % 100
            return hash_val < self.rollout_percentage
        if self.strategy == RolloutStrategy.AB_TEST:
            if user_id is None:
                return False
            hash_val = hash(f"{self.name}:{user_id}") % 2
            return hash_val == 0  # 50/50 split
        return False


class FeatureFlagManager:
    """Manages feature flags for application features."""

    def __init__(self, flags: dict[str, FeatureFlag] | None = None) -> None:
        """Initialize feature flag manager.

        Args:
            flags: Dictionary of feature flags keyed by name
        """
        self._flags: dict[str, FeatureFlag] = flags or {}
        self._lock = __import__("threading").Lock()

    def register_flag(self, flag: FeatureFlag) -> None:
        """Register a feature flag.

        Args:
            flag: Feature flag to register
        """
        with self._lock:
            self._flags[flag.name] = flag
            LOGGER.info(f"Registered feature flag: {flag.name}")

    def is_enabled(self, flag_name: str, user_id: str | None = None) -> bool:
        """Check if feature is enabled for user.

        Args:
            flag_name: Name of feature flag
            user_id: Optional user ID for canary/A/B testing

        Returns:
            True if feature is enabled for user
        """
        flag = self._flags.get(flag_name)
        if flag is None:
            LOGGER.warning(f"Feature flag not found: {flag_name}")
            return False
        return flag.should_enable(user_id)

    def get_flag(self, flag_name: str) -> FeatureFlag | None:
        """Get feature flag by name.

        Args:
            flag_name: Name of feature flag

        Returns:
            Feature flag or None if not found
        """
        return self._flags.get(flag_name)

    def update_flag(self, flag: FeatureFlag) -> None:
        """Update feature flag configuration.

        Args:
            flag: Updated feature flag
        """
        with self._lock:
            self._flags[flag.name] = flag
            LOGGER.info(f"Updated feature flag: {flag.name}")

    def set_rollout_percentage(self, flag_name: str, percentage: int) -> None:
        """Update canary rollout percentage.

        Args:
            flag_name: Name of feature flag
            percentage: Rollout percentage (0-100)
        """
        flag = self._flags.get(flag_name)
        if flag is None:
            LOGGER.error(f"Feature flag not found: {flag_name}")
            return

        if not (0 <= percentage <= 100):
            LOGGER.error(f"Invalid percentage: {percentage}")
            return

        updated = FeatureFlag(
            name=flag.name,
            enabled=flag.enabled,
            strategy=RolloutStrategy.CANARY if percentage > 0 else flag.strategy,
            rollout_percentage=percentage,
            config=flag.config,
            metadata=flag.metadata,
        )
        self.update_flag(updated)

    def list_flags(self) -> dict[str, dict[str, Any]]:
        """List all feature flags with status.

        Returns:
            Dictionary of flag statuses
        """
        result = {}
        for name, flag in self._flags.items():
            result[name] = {
                "enabled": flag.enabled,
                "strategy": flag.strategy.value,
                "rollout_percentage": flag.rollout_percentage,
                "config": flag.config,
            }
        return result

    def export_flags_json(self) -> str:
        """Export feature flags as JSON.

        Returns:
            JSON string of flag configurations
        """
        flags_dict = {}
        for name, flag in self._flags.items():
            flags_dict[name] = {
                "enabled": flag.enabled,
                "strategy": flag.strategy.value,
                "rollout_percentage": flag.rollout_percentage,
                "config": flag.config,
                "metadata": flag.metadata,
            }
        return json.dumps(flags_dict, indent=2)


# Global feature flag manager instance
_GLOBAL_FLAG_MANAGER: FeatureFlagManager | None = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get global feature flag manager instance.

    Returns:
        Global FeatureFlagManager instance
    """
    global _GLOBAL_FLAG_MANAGER
    if _GLOBAL_FLAG_MANAGER is None:
        _GLOBAL_FLAG_MANAGER = FeatureFlagManager()
    return _GLOBAL_FLAG_MANAGER


def is_feature_enabled(flag_name: str, user_id: str | None = None) -> bool:
    """Check if feature is enabled (convenience function).

    Args:
        flag_name: Name of feature flag
        user_id: Optional user ID for canary/A/B testing

    Returns:
        True if feature is enabled
    """
    manager = get_feature_flag_manager()
    return manager.is_enabled(flag_name, user_id)
