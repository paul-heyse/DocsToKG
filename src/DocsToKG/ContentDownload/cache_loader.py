"""Cache configuration loader with YAML/env/CLI overlay support.

Responsibilities
----------------
- Load and parse YAML cache configuration files
- Apply environment variable and CLI argument overlays with clear precedence
- Normalize host keys using IDNA 2008 + UTS #46 for RFC-compliant hostname handling
- Validate all configuration values (TTLs, role names, etc.)
- Return immutable CacheConfig dataclass for use throughout the system

Design Notes
------------
- Configuration precedence: YAML → env → CLI (later wins)
- All host keys normalized to lowercase ASCII using IDNA before storage
- Graceful fallback when IDNA encoding fails (logs debug, uses lowercase)
- Frozen dataclasses ensure immutability
- Deep merging for complex configuration overlays
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    import idna
except ImportError as e:
    raise RuntimeError("idna is required for hostname normalization") from e

try:
    import yaml
except ImportError as e:
    raise RuntimeError("PyYAML is required for configuration loading") from e  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)


class StorageKind(str, Enum):
    """Storage backend types for Hishel cache."""

    FILE = "file"
    MEMORY = "memory"
    REDIS = "redis"
    SQLITE = "sqlite"
    S3 = "s3"


class CacheDefault(str, Enum):
    """Global caching policy for unknown hosts."""

    DO_NOT_CACHE = "DO_NOT_CACHE"
    CACHE = "CACHE"


@dataclass(frozen=True)
class CacheStorage:
    """Storage backend configuration."""

    kind: StorageKind
    path: str
    check_ttl_every_s: int = 600

    def __post_init__(self) -> None:
        """Validate storage settings."""
        if self.check_ttl_every_s < 60:
            raise ValueError("check_ttl_every_s must be >= 60")


@dataclass(frozen=True)
class CacheRolePolicy:
    """Per-role cache policy (e.g., metadata vs landing)."""

    ttl_s: Optional[int] = None
    swrv_s: Optional[int] = None
    body_key: bool = False

    def __post_init__(self) -> None:
        """Validate role policy settings."""
        if self.ttl_s is not None and self.ttl_s < 0:
            raise ValueError("ttl_s must be >= 0")
        if self.swrv_s is not None and self.swrv_s < 0:
            raise ValueError("swrv_s must be >= 0")


@dataclass(frozen=True)
class CacheHostPolicy:
    """Per-host cache policy with role-specific overrides."""

    ttl_s: Optional[int] = None
    role: Dict[str, CacheRolePolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate host policy settings."""
        if self.ttl_s is not None and self.ttl_s < 0:
            raise ValueError("ttl_s must be >= 0")
        for role_name in self.role.keys():
            if role_name not in ("metadata", "landing", "artifact"):
                raise ValueError(f"Invalid role: {role_name}")


@dataclass(frozen=True)
class CacheControllerDefaults:
    """Global cache controller defaults."""

    cacheable_methods: List[str] = field(default_factory=lambda: ["GET", "HEAD"])
    cacheable_statuses: List[int] = field(default_factory=lambda: [200, 301, 308])
    allow_heuristics: bool = False
    default: CacheDefault = CacheDefault.DO_NOT_CACHE


@dataclass(frozen=True)
class CacheConfig:
    """Complete cache configuration."""

    storage: CacheStorage
    controller: CacheControllerDefaults
    hosts: Dict[str, CacheHostPolicy]

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.controller.default == CacheDefault.DO_NOT_CACHE and not self.hosts:
            raise ValueError("If default=DO_NOT_CACHE, must specify at least one host")


def _normalize_host_key(host: str) -> str:
    """Normalize host to lowercase ASCII using IDNA 2008 + UTS #46.

    Examples:
        "API.Crossref.Org" → "api.crossref.org"
        "münchen.example" → "xn--mnich-kva.example"

    Args:
        host: Hostname to normalize

    Returns:
        Normalized ASCII hostname suitable as dictionary key

    Notes:
        - Strips whitespace and trailing dots before encoding
        - Uses IDNA 2008 + UTS #46 for RFC-compliant handling
        - Gracefully falls back to lowercase on IDNA errors
        - Logs debug message if fallback used
    """
    h = host.strip().rstrip(".")
    if not h:
        return h

    try:
        h_ascii = idna.encode(h, uts46=True).decode("ascii")
        return h_ascii
    except idna.IDNAError as e:
        LOGGER.debug(f"IDNA encoding failed for '{h}': {e}; falling back to lowercase")
        return h.lower()
    except Exception:
        return h.lower()


def _load_yaml(yaml_path: Optional[str | Path]) -> Dict[str, Any]:
    """Load and parse YAML file.

    Args:
        yaml_path: Path to YAML file (or None to skip)

    Returns:
        Parsed YAML as dictionary

    Raises:
        FileNotFoundError: If path provided but file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if yaml_path is None:
        return {}

    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Cache YAML not found: {p}")

    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse cache YAML: {e}") from e


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary to merge into base

    Returns:
        Merged dictionary
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _parse_policy_value(value_str: str) -> Dict[str, Any]:
    """Parse a policy value string (e.g., 'ttl_s:259200,swrv_s:180').

    Args:
        value_str: Policy string with key:value pairs separated by commas

    Returns:
        Dictionary of parsed values
    """
    result: Dict[str, Any] = {}
    for part in value_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, val = part.split(":", 1)
            try:
                result[key.strip()] = int(val.strip())
            except ValueError:
                result[key.strip()] = val.strip()
        else:
            result[part] = True
    return result


def _apply_env_overlays(doc: Dict[str, Any], env: Mapping[str, str]) -> Dict[str, Any]:
    """Apply environment variable overlays to configuration.

    Env var patterns:
    - DOCSTOKG_CACHE_HOST__api_crossref_org=ttl_s:259200
    - DOCSTOKG_CACHE_ROLE__api_openalex_org__metadata=ttl_s:259200,swrv_s:180
    - DOCSTOKG_CACHE_DEFAULTS=cacheable_methods:GET

    Args:
        doc: Base configuration dictionary
        env: Environment variables

    Returns:
        Configuration with env overlays applied
    """
    result = dict(doc)

    for key, value in env.items():
        if not key.startswith("DOCSTOKG_CACHE_"):
            continue

        # Remove prefix
        suffix = key[len("DOCSTOKG_CACHE_") :]

        # Parse key parts (replace __ with actual separators)
        if suffix.startswith("HOST__"):
            # DOCSTOKG_CACHE_HOST__api_crossref_org
            host_key = suffix[len("HOST__") :].replace("_", ".")
            normalized = _normalize_host_key(host_key)
            parsed = _parse_policy_value(value)
            if "hosts" not in result:
                result["hosts"] = {}
            result["hosts"][normalized] = parsed
        elif suffix.startswith("ROLE__"):
            # DOCSTOKG_CACHE_ROLE__api_openalex_org__metadata
            rest = suffix[len("ROLE__") :]
            parts = rest.rsplit("__", 1)
            if len(parts) == 2:
                host_key, role = parts
                host_key = host_key.replace("_", ".")
                normalized = _normalize_host_key(host_key)
                parsed = _parse_policy_value(value)
                if "hosts" not in result:
                    result["hosts"] = {}
                if normalized not in result["hosts"]:
                    result["hosts"][normalized] = {}
                if "role" not in result["hosts"][normalized]:
                    result["hosts"][normalized]["role"] = {}
                result["hosts"][normalized]["role"][role] = parsed
        elif suffix == "DEFAULTS":
            parsed = _parse_policy_value(value)
            if "controller" not in result:
                result["controller"] = {}
            result["controller"].update(parsed)

    return result


def _apply_cli_overrides(
    doc: Dict[str, Any],
    cli_host_overrides: Sequence[str] | None = None,
    cli_role_overrides: Sequence[str] | None = None,
    cli_defaults_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply CLI argument overlays to configuration.

    Args:
        doc: Base configuration dictionary
        cli_host_overrides: List of "host=ttl_s:259200" strings
        cli_role_overrides: List of "host:role=ttl_s:259200" strings
        cli_defaults_override: "cacheable_methods:GET,cacheable_statuses:200"

    Returns:
        Configuration with CLI overlays applied
    """
    result = dict(doc)

    if cli_host_overrides:
        if "hosts" not in result:
            result["hosts"] = {}
        for override in cli_host_overrides:
            if "=" not in override:
                continue
            host_key, policy_str = override.split("=", 1)
            normalized = _normalize_host_key(host_key.strip())
            parsed = _parse_policy_value(policy_str)
            result["hosts"][normalized] = parsed

    if cli_role_overrides:
        if "hosts" not in result:
            result["hosts"] = {}
        for override in cli_role_overrides:
            if "=" not in override:
                continue
            host_role, policy_str = override.split("=", 1)
            if ":" not in host_role:
                continue
            host_key, role = host_role.split(":", 1)
            normalized = _normalize_host_key(host_key.strip())
            parsed = _parse_policy_value(policy_str)
            if normalized not in result["hosts"]:
                result["hosts"][normalized] = {}
            if "role" not in result["hosts"][normalized]:
                result["hosts"][normalized]["role"] = {}
            result["hosts"][normalized]["role"][role] = parsed

    if cli_defaults_override:
        parsed = _parse_policy_value(cli_defaults_override)
        if "controller" not in result:
            result["controller"] = {}
        result["controller"].update(parsed)

    return result


def load_cache_config(
    yaml_path: Optional[str | Path] = None,
    *,
    env: Mapping[str, str],
    cli_host_overrides: Sequence[str] | None = None,
    cli_role_overrides: Sequence[str] | None = None,
    cli_defaults_override: Optional[str] = None,
) -> CacheConfig:
    """Load cache configuration with YAML → env → CLI precedence.

    Args:
        yaml_path: Path to cache.yaml configuration file
        env: Environment variables (typically os.environ)
        cli_host_overrides: List of "host=ttl_s:259200" strings
        cli_role_overrides: List of "host:role=ttl_s:259200,swrv_s:180" strings
        cli_defaults_override: "cacheable_methods:GET,cacheable_statuses:200"

    Returns:
        CacheConfig with all values resolved and validated

    Raises:
        FileNotFoundError: If yaml_path provided but not found
        ValueError: If any config value fails validation
        yaml.YAMLError: If YAML parsing fails
    """
    # Load YAML
    doc = _load_yaml(yaml_path)

    # Apply env overlays
    doc = _apply_env_overlays(doc, env)

    # Apply CLI overlays
    doc = _apply_cli_overrides(
        doc,
        cli_host_overrides=cli_host_overrides,
        cli_role_overrides=cli_role_overrides,
        cli_defaults_override=cli_defaults_override,
    )

    # Parse storage
    storage_doc = doc.get("storage", {})
    storage = CacheStorage(
        kind=StorageKind(storage_doc.get("kind", "file")),
        path=storage_doc.get("path", "${DOCSTOKG_DATA_ROOT}/cache/http"),
        check_ttl_every_s=int(storage_doc.get("check_ttl_every_s", 600)),
    )

    # Parse controller
    controller_doc = doc.get("controller", {})
    cacheable_methods = controller_doc.get("cacheable_methods", ["GET", "HEAD"])
    if isinstance(cacheable_methods, str):
        cacheable_methods = cacheable_methods.split(",")
    cacheable_statuses = controller_doc.get("cacheable_statuses", [200, 301, 308])
    if isinstance(cacheable_statuses, str):
        cacheable_statuses = [int(s.strip()) for s in cacheable_statuses.split(",")]
    controller = CacheControllerDefaults(
        cacheable_methods=cacheable_methods,
        cacheable_statuses=cacheable_statuses,
        allow_heuristics=bool(controller_doc.get("allow_heuristics", False)),
        default=CacheDefault(controller_doc.get("default", "DO_NOT_CACHE")),
    )

    # Parse hosts
    hosts: Dict[str, CacheHostPolicy] = {}
    for host_key, host_policy_doc in doc.get("hosts", {}).items():
        normalized = _normalize_host_key(host_key)
        host_ttl = host_policy_doc.get("ttl_s")
        role_policies: Dict[str, CacheRolePolicy] = {}
        for role_name, role_doc in host_policy_doc.get("role", {}).items():
            role_policies[role_name] = CacheRolePolicy(
                ttl_s=role_doc.get("ttl_s"),
                swrv_s=role_doc.get("swrv_s"),
                body_key=bool(role_doc.get("body_key", False)),
            )
        hosts[normalized] = CacheHostPolicy(ttl_s=host_ttl, role=role_policies)

    # Create and return config
    return CacheConfig(storage=storage, controller=controller, hosts=hosts)
