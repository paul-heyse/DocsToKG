# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.config.feature_flags",
#   "purpose": "Feature flags for Pydantic v2 optimization features.",
#   "sections": [
#     {
#       "id": "featureflag",
#       "name": "FeatureFlag",
#       "anchor": "class-featureflag",
#       "kind": "class"
#     },
#     {
#       "id": "featureflags",
#       "name": "FeatureFlags",
#       "anchor": "class-featureflags",
#       "kind": "class"
#     },
#     {
#       "id": "get-feature-flags",
#       "name": "get_feature_flags",
#       "anchor": "function-get-feature-flags",
#       "kind": "function"
#     },
#     {
#       "id": "reset-feature-flags",
#       "name": "reset_feature_flags",
#       "anchor": "function-reset-feature-flags",
#       "kind": "function"
#     },
#     {
#       "id": "set-feature-flags",
#       "name": "set_feature_flags",
#       "anchor": "function-set-feature-flags",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Feature flags for Pydantic v2 optimization features.

Allows easy enabling/disabling of new features without impacting existing functionality.
All features are disabled by default for safety.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class FeatureFlag(Enum):
    """Feature flag names."""

    UNIFIED_BOOTSTRAP = "DTKG_FEATURE_UNIFIED_BOOTSTRAP"
    CLI_CONFIG_COMMANDS = "DTKG_FEATURE_CLI_CONFIG_COMMANDS"
    CONFIG_AUDIT_TRAIL = "DTKG_FEATURE_CONFIG_AUDIT_TRAIL"
    POLICY_MODULES = "DTKG_FEATURE_POLICY_MODULES"


@dataclass
class FeatureFlags:
    """Feature flag configuration."""

    unified_bootstrap: bool = False
    cli_config_commands: bool = False
    config_audit_trail: bool = False
    policy_modules: bool = False

    @classmethod
    def from_env(cls) -> FeatureFlags:
        """Load feature flags from environment variables.

        Environment variables:
          - DTKG_FEATURE_UNIFIED_BOOTSTRAP=1
          - DTKG_FEATURE_CLI_CONFIG_COMMANDS=1
          - DTKG_FEATURE_CONFIG_AUDIT_TRAIL=1
          - DTKG_FEATURE_POLICY_MODULES=1

        Returns:
            FeatureFlags instance with values from environment
        """
        return cls(
            unified_bootstrap=os.environ.get(FeatureFlag.UNIFIED_BOOTSTRAP.value, "0") == "1",
            cli_config_commands=os.environ.get(FeatureFlag.CLI_CONFIG_COMMANDS.value, "0") == "1",
            config_audit_trail=os.environ.get(FeatureFlag.CONFIG_AUDIT_TRAIL.value, "0") == "1",
            policy_modules=os.environ.get(FeatureFlag.POLICY_MODULES.value, "0") == "1",
        )

    def enable_all(self) -> None:
        """Enable all features."""
        self.unified_bootstrap = True
        self.cli_config_commands = True
        self.config_audit_trail = True
        self.policy_modules = True

    def disable_all(self) -> None:
        """Disable all features."""
        self.unified_bootstrap = False
        self.cli_config_commands = False
        self.config_audit_trail = False
        self.policy_modules = False

    def enable(self, flag: FeatureFlag) -> None:
        """Enable a specific feature flag."""
        if flag == FeatureFlag.UNIFIED_BOOTSTRAP:
            self.unified_bootstrap = True
        elif flag == FeatureFlag.CLI_CONFIG_COMMANDS:
            self.cli_config_commands = True
        elif flag == FeatureFlag.CONFIG_AUDIT_TRAIL:
            self.config_audit_trail = True
        elif flag == FeatureFlag.POLICY_MODULES:
            self.policy_modules = True

    def disable(self, flag: FeatureFlag) -> None:
        """Disable a specific feature flag."""
        if flag == FeatureFlag.UNIFIED_BOOTSTRAP:
            self.unified_bootstrap = False
        elif flag == FeatureFlag.CLI_CONFIG_COMMANDS:
            self.cli_config_commands = False
        elif flag == FeatureFlag.CONFIG_AUDIT_TRAIL:
            self.config_audit_trail = False
        elif flag == FeatureFlag.POLICY_MODULES:
            self.policy_modules = False

    def is_enabled(self, flag: FeatureFlag) -> bool:
        """Check if a feature flag is enabled."""
        if flag == FeatureFlag.UNIFIED_BOOTSTRAP:
            return self.unified_bootstrap
        elif flag == FeatureFlag.CLI_CONFIG_COMMANDS:
            return self.cli_config_commands
        elif flag == FeatureFlag.CONFIG_AUDIT_TRAIL:
            return self.config_audit_trail
        elif flag == FeatureFlag.POLICY_MODULES:
            return self.policy_modules
        return False


# Global feature flags instance
_feature_flags: FeatureFlags | None = None


def get_feature_flags() -> FeatureFlags:
    """Get global feature flags instance (singleton).

    Returns:
        Global FeatureFlags instance
    """
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags.from_env()
    return _feature_flags


def reset_feature_flags() -> None:
    """Reset global feature flags (useful for testing)."""
    global _feature_flags
    _feature_flags = None


def set_feature_flags(flags: FeatureFlags) -> None:
    """Set global feature flags (useful for testing)."""
    global _feature_flags
    _feature_flags = flags


__all__ = [
    "FeatureFlag",
    "FeatureFlags",
    "get_feature_flags",
    "reset_feature_flags",
    "set_feature_flags",
]
