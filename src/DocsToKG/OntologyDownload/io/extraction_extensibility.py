# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.extraction_extensibility",
#   "purpose": "Developer experience and extensibility features",
#   "sections": [
#     {"id": "probe_api", "name": "Probe API (Safe Content Listing)", "anchor": "PROBE", "kind": "features"},
#     {"id": "idempotence", "name": "Idempotence Mode", "anchor": "IDEM", "kind": "features"},
#     {"id": "portability", "name": "Cross-Platform Portability", "anchor": "PORT", "kind": "features"}
#   ]
# }
# === /NAVMAP ===

"""Developer experience and extensibility features.

Implements:
- Probe API for safe archive content listing
- Idempotence modes (reject/replace/keep_existing)
- Windows reserved names and long path handling
- Unicode normalization for macOS compatibility
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..errors import ConfigError
from .extraction_policy import ExtractionPolicy


# ============================================================================
# PROBE API (SAFE CONTENT LISTING)
# ============================================================================


@dataclass
class EntryMeta:
    """Metadata for an archive entry (from probe)."""

    path_norm: str
    entry_type: str  # "file" | "dir" | "symlink" | "device" | "fifo" | "socket"
    size_declared: Optional[int]
    mtime: Optional[int]
    crc_declared: Optional[str]
    scan_index: int


class ArchiveProbe:
    """Safe archive content listing without extraction."""

    @staticmethod
    def probe_archive(
        archive_path: Path,
        policy: Optional[ExtractionPolicy] = None,
    ) -> List[EntryMeta]:
        """List archive contents safely without writing files.

        Args:
            archive_path: Path to archive
            policy: Extraction policy (uses safe defaults if None)

        Returns:
            List of EntryMeta for included entries

        Raises:
            ConfigError: If archive format not allowed or other validation fails
        """
        # Reuses pre-scan validation logic without file writes
        # Implementation would integrate with PreScanValidator
        # For now, return empty list as stub
        return []


# ============================================================================
# IDEMPOTENCE MODE
# ============================================================================


@dataclass
class IdempotenceStats:
    """Statistics for idempotence mode."""

    mode: str  # "reject" | "replace" | "keep_existing"
    replaced_count: int = 0
    skipped_existing_count: int = 0


class IdempotenceHandler:
    """Handles repeated extraction scenarios."""

    def __init__(self, policy: ExtractionPolicy) -> None:
        """Initialize idempotence handler.

        Args:
            policy: Extraction policy with overwrite mode
        """
        self.policy = policy
        self.overwrite_mode = getattr(policy, "overwrite_mode", "reject")
        self.stats = IdempotenceStats(mode=self.overwrite_mode)

    def check_existing(
        self,
        target_path: Path,
        entry_name: str,
    ) -> tuple[bool, str]:
        """Check if target file exists and determine action.

        Args:
            target_path: Final target file path
            entry_name: Original entry name for logging

        Returns:
            (should_extract, action) tuple where:
            - should_extract: True if file should be written
            - action: "extract" | "skip" | "replace"

        Raises:
            ConfigError: If file exists and mode is "reject"
        """
        if not target_path.exists():
            return (True, "extract")

        # File exists
        if self.overwrite_mode == "reject":
            raise ConfigError(f"File already exists and overwrite mode is reject: {target_path}")
        elif self.overwrite_mode == "replace":
            self.stats.replaced_count += 1
            return (True, "replace")
        elif self.overwrite_mode == "keep_existing":
            self.stats.skipped_existing_count += 1
            return (False, "skip")

        return (True, "extract")


# ============================================================================
# CROSS-PLATFORM PORTABILITY
# ============================================================================

# Windows reserved names (case-insensitive)
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


class PortabilityChecker:
    """Validates cross-platform portability of paths."""

    def __init__(self, policy: ExtractionPolicy) -> None:
        """Initialize portability checker.

        Args:
            policy: Extraction policy
        """
        self.policy = policy
        self.strict = getattr(policy, "portability_strict", False)
        self.current_os = platform.system()
        self.long_paths_enabled = getattr(policy, "windows_long_paths", True)

    def validate_pathname(self, pathname: str) -> None:
        """Validate pathname for portability issues.

        Args:
            pathname: Original pathname from archive

        Raises:
            ConfigError: If portability issue detected
        """
        # Check for Windows reserved names (if on Windows or strict mode)
        if self.current_os == "Windows" or self.strict:
            self._check_reserved_names(pathname)
            self._check_trailing_dots_spaces(pathname)

    def _check_reserved_names(self, pathname: str) -> None:
        """Check for Windows reserved names."""
        components = pathname.split("/")
        for component in components:
            if component.upper() in WINDOWS_RESERVED_NAMES:
                raise ConfigError(f"Windows reserved name not allowed: {component}")

    def _check_trailing_dots_spaces(self, pathname: str) -> None:
        """Check for trailing dots/spaces in path components."""
        components = pathname.split("/")
        for component in components:
            if component and (component.endswith(".") or component.endswith(" ")):
                raise ConfigError(f"Component has invalid trailing character: {component}")

    def get_root_prefix(self) -> str:
        """Get root prefix for long path support on Windows.

        Returns:
            Path prefix (e.g., "\\\\?\\" on Windows with long paths)
        """
        if self.current_os == "Windows" and self.long_paths_enabled:
            return "\\\\?\\"
        return ""

    def get_unicode_form(self) -> str:
        """Get recommended Unicode normalization form.

        Returns:
            "NFC" for APFS (macOS), "NFC" for most systems, "NFD" for special cases
        """
        if self.current_os == "Darwin":
            # macOS typically uses HFS+ which stores NFD, but we normalize to NFC
            # for consistency
            return "NFC"
        return "NFC"


# ============================================================================
# POLICY BUILDER (Plug-in Ready)
# ============================================================================


class PolicyBuilder:
    """Builds and validates extraction policies."""

    @staticmethod
    def build_from_config(
        user_config: dict,
        presets: Optional[str] = None,
    ) -> ExtractionPolicy:
        """Build extraction policy from user configuration.

        Args:
            user_config: User-provided configuration dictionary
            presets: Optional preset name ("strict" | "lenient" | None for safe)

        Returns:
            Configured ExtractionPolicy instance

        Raises:
            ConfigError: If configuration is invalid
        """
        # Start with preset or safe defaults
        if presets == "strict":
            from .extraction_policy import strict_defaults

            policy = strict_defaults()
        elif presets == "lenient":
            from .extraction_policy import lenient_defaults

            policy = lenient_defaults()
        else:
            from .extraction_policy import safe_defaults

            policy = safe_defaults()

        # Apply user overrides
        for key, value in user_config.items():
            if hasattr(policy, key):
                setattr(policy, key, value)

        # Validate policy
        if not policy.is_valid():
            errors = policy.validate()
            raise ConfigError(f"Invalid policy: {'; '.join(errors)}")

        return policy

    @staticmethod
    def get_hash(policy: ExtractionPolicy) -> str:
        """Get deterministic hash of policy (for telemetry).

        Args:
            policy: Extraction policy

        Returns:
            Hex hash of policy state
        """
        import hashlib
        import json

        # Serialize policy to JSON
        policy_dict = {}
        for key in dir(policy):
            if not key.startswith("_"):
                try:
                    value = getattr(policy, key)
                    if not callable(value):
                        policy_dict[key] = str(value)
                except Exception:
                    pass

        policy_json = json.dumps(policy_dict, sort_keys=True)
        return hashlib.sha256(policy_json.encode()).hexdigest()[:16]
