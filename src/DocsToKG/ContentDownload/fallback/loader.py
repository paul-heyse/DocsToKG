"""Configuration loader for fallback strategy.

This module provides functions to load fallback configuration from multiple
sources (YAML, environment variables, CLI arguments) and merge them with
proper precedence to produce a FallbackPlan object.

Configuration Precedence (highest to lowest):
  1. CLI arguments (--fallback-*)
  2. Environment variables (DOCSTOKG_FALLBACK_*)
  3. YAML configuration (config/fallback.yaml)
  4. Built-in defaults

Example:
    ```python
    from DocsToKG.ContentDownload.fallback.loader import load_fallback_plan

    # Load plan with CLI overrides
    plan = load_fallback_plan(
        yaml_path="config/fallback.yaml",
        env_overrides={"total_timeout_ms": "60000"},
        cli_overrides={"fast_mode": True},
    )
    ```
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore[import-untyped]

from DocsToKG.ContentDownload.fallback.types import (
    AttemptPolicy,
    FallbackPlan,
    TierPlan,
)

logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Raised when configuration is invalid."""

    pass


def load_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If YAML file not found
        yaml.YAMLError: If YAML is invalid
        ConfigurationError: If configuration is incomplete
    """
    if not yaml_path.exists():
        msg = f"Fallback config YAML not found: {yaml_path}"
        raise FileNotFoundError(msg)

    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Invalid YAML in {yaml_path}: {e}"
        raise ConfigurationError(msg) from e

    if config is None:
        msg = f"YAML file is empty: {yaml_path}"
        raise ConfigurationError(msg)

    logger.debug(f"Loaded fallback config from {yaml_path}")
    return config


def load_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables.

    Supports variables matching pattern: DOCSTOKG_FALLBACK_*

    Environment variable to config mapping:
      DOCSTOKG_FALLBACK_TOTAL_TIMEOUT_MS → budgets.total_timeout_ms
      DOCSTOKG_FALLBACK_TOTAL_ATTEMPTS → budgets.total_attempts
      DOCSTOKG_FALLBACK_MAX_CONCURRENT → budgets.max_concurrent
      DOCSTOKG_FALLBACK_OFFLINE_BEHAVIOR → gates.offline_behavior

    Returns:
        Dictionary with environment overrides
    """
    config: Dict[str, Any] = {}

    # Check for overall enable flag
    if os.getenv("DOCSTOKG_ENABLE_FALLBACK") == "0":
        logger.debug("Fallback disabled via DOCSTOKG_ENABLE_FALLBACK=0")
        return config

    # Budget overrides
    budgets = {}
    for key in ["total_timeout_ms", "total_attempts", "max_concurrent", "per_source_timeout_ms"]:
        env_key = f"DOCSTOKG_FALLBACK_{key.upper()}"
        if env_key in os.environ:
            try:
                budgets[key] = int(os.environ[env_key])
                logger.debug(f"Loaded {key} from {env_key}")
            except ValueError:
                logger.warning(f"Invalid value for {env_key}: {os.environ[env_key]}")

    if budgets:
        config["budgets"] = budgets

    # Gate overrides
    gates: Dict[str, Any] = {}
    if "DOCSTOKG_FALLBACK_OFFLINE_BEHAVIOR" in os.environ:
        gates["offline_behavior"] = os.environ["DOCSTOKG_FALLBACK_OFFLINE_BEHAVIOR"]
        logger.debug("Loaded offline_behavior from env")

    if "DOCSTOKG_FALLBACK_SKIP_IF_BREAKER_OPEN" in os.environ:
        val = os.environ["DOCSTOKG_FALLBACK_SKIP_IF_BREAKER_OPEN"].lower()
        gates["skip_if_breaker_open"] = val in ("true", "1", "yes")
        logger.debug("Loaded skip_if_breaker_open from env")

    if gates:
        config["gates"] = gates

    logger.debug(f"Loaded fallback config from environment: {len(config)} sections")
    return config


def load_from_cli(cli_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Load configuration from CLI arguments dictionary.

    Expected keys:
      - fallback_timeout_ms: int (total timeout)
      - fallback_attempts: int (total attempts)
      - fallback_concurrency: int (max concurrent)
      - fallback_offline_mode: str (metadata_only, block_all, cache_only)
      - fallback_fast_mode: bool (use FAST tuning profile)
      - fallback_reliable_mode: bool (use HIGH RELIABILITY tuning profile)

    Args:
        cli_dict: Dictionary of CLI overrides (or None)

    Returns:
        Dictionary with CLI overrides
    """
    if not cli_dict:
        return {}

    config: Dict[str, Any] = {}

    # Handle preset modes
    if cli_dict.get("fallback_fast_mode"):
        logger.info("Applying FAST tuning profile from CLI")
        config.update(_get_fast_profile())

    elif cli_dict.get("fallback_reliable_mode"):
        logger.info("Applying HIGH RELIABILITY tuning profile from CLI")
        config.update(_get_reliable_profile())

    # Individual budget overrides
    budgets = {}
    if "fallback_timeout_ms" in cli_dict:
        budgets["total_timeout_ms"] = cli_dict["fallback_timeout_ms"]

    if "fallback_attempts" in cli_dict:
        budgets["total_attempts"] = cli_dict["fallback_attempts"]

    if "fallback_concurrency" in cli_dict:
        budgets["max_concurrent"] = cli_dict["fallback_concurrency"]

    if budgets:
        config["budgets"] = budgets

    # Gate overrides
    gates = {}
    if "fallback_offline_mode" in cli_dict:
        gates["offline_behavior"] = cli_dict["fallback_offline_mode"]

    if gates:
        config["gates"] = gates

    logger.debug(f"Loaded fallback config from CLI: {len(config)} sections")
    return config


def _get_fast_profile() -> Dict[str, Any]:
    """Return FAST tuning profile."""
    return {
        "budgets": {
            "total_timeout_ms": 60_000,
            "total_attempts": 12,
            "max_concurrent": 2,
            "per_source_timeout_ms": 5_000,
        },
    }


def _get_reliable_profile() -> Dict[str, Any]:
    """Return HIGH RELIABILITY tuning profile."""
    return {
        "budgets": {
            "total_timeout_ms": 180_000,
            "total_attempts": 30,
            "max_concurrent": 2,
            "per_source_timeout_ms": 15_000,
        },
    }


def merge_configs(
    yaml_config: Dict[str, Any],
    env_config: Dict[str, Any],
    cli_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge configurations with proper precedence.

    Precedence (highest to lowest):
      1. CLI config
      2. Environment config
      3. YAML config

    For each section, merges keys with later sources overriding earlier ones.

    Args:
        yaml_config: Configuration from YAML
        env_config: Configuration from environment
        cli_config: Configuration from CLI

    Returns:
        Merged configuration dictionary
    """
    merged: Dict[str, Any] = {}

    # Merge each section (budgets, tiers, policies, gates, thresholds)
    for section in ["budgets", "tiers", "policies", "gates", "thresholds", "metadata"]:
        # Start with YAML
        if section in yaml_config:
            if section in ("tiers", "metadata"):
                # These are list/object, not merged
                merged[section] = yaml_config[section]
            else:
                merged[section] = dict(yaml_config[section])

        # Overlay env
        if section in env_config:
            if section in ("tiers", "metadata"):
                # Replace entirely
                merged[section] = env_config[section]
            else:
                merged[section] = merged.get(section, {})
                merged[section].update(env_config[section])

        # Overlay CLI
        if section in cli_config:
            if section in ("tiers", "metadata"):
                # Replace entirely
                merged[section] = cli_config[section]
            else:
                merged[section] = merged.get(section, {})
                merged[section].update(cli_config[section])

    logger.debug(f"Merged configuration: {len(merged)} sections")
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure and content.

    Checks:
      - Required sections present
      - Tier sources have policies
      - Timeouts are positive
      - Budgets are sensible

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ConfigurationError: If invalid
    """
    # Check required sections
    required = ["budgets", "tiers", "policies"]
    missing = [s for s in required if s not in config]
    if missing:
        msg = f"Configuration missing required sections: {missing}"
        raise ConfigurationError(msg)

    # Validate budgets
    budgets = config["budgets"]
    if budgets.get("total_timeout_ms", 0) <= 0:
        msg = "budgets.total_timeout_ms must be positive"
        raise ConfigurationError(msg)

    if budgets.get("total_attempts", 0) <= 0:
        msg = "budgets.total_attempts must be positive"
        raise ConfigurationError(msg)

    # Validate tiers reference sources with policies
    all_sources = set()
    for tier in config.get("tiers", []):
        all_sources.update(tier.get("sources", []))

    policies = config.get("policies", {})
    missing_policies = all_sources - set(policies.keys())
    if missing_policies:
        msg = f"Sources missing policies: {missing_policies}"
        raise ConfigurationError(msg)

    logger.info("Configuration validation passed")
    return True


def build_fallback_plan(config: Dict[str, Any]) -> FallbackPlan:
    """Build FallbackPlan from configuration dictionary.

    Constructs TierPlan, AttemptPolicy, and FallbackPlan objects
    from the merged configuration.

    Args:
        config: Merged configuration dictionary

    Returns:
        FallbackPlan ready for use by orchestrator

    Raises:
        ConfigurationError: If configuration is invalid
    """
    validate_config(config)

    # Build tier plans
    tiers: list[TierPlan] = []
    for tier_config in config["tiers"]:
        tier = TierPlan(
            name=tier_config["name"],
            parallel=tier_config["parallel"],
            sources=tier_config["sources"],  # Changed from tuple() to list
        )
        tiers.append(tier)

    # Build policies
    policies: Dict[str, AttemptPolicy] = {}
    for source_name, policy_config in config["policies"].items():
        policy = AttemptPolicy(
            name=source_name,
            timeout_ms=policy_config["timeout_ms"],
            retries_max=policy_config["retries_max"],
            robots_respect=policy_config.get("robots_respect", False),
        )
        policies[source_name] = policy

    # Build plan
    plan = FallbackPlan(
        budgets=config["budgets"],
        tiers=tiers,  # Changed from tuple(tiers) to tiers (already a list)
        policies=policies,
        gates=config.get("gates", {}),
    )

    logger.info(f"Built FallbackPlan: {len(tiers)} tiers, {len(policies)} sources")
    return plan


def load_fallback_plan(
    yaml_path: Optional[Path] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> FallbackPlan:
    """Load and merge fallback configuration into a FallbackPlan.

    Loads configuration from:
      1. YAML file (default: config/fallback.yaml)
      2. Environment variables (DOCSTOKG_FALLBACK_*)
      3. CLI overrides dictionary

    Merges with proper precedence: CLI > ENV > YAML > defaults

    Args:
        yaml_path: Path to YAML config (None uses default)
        env_overrides: Manual env overrides (uses os.environ if None)
        cli_overrides: Manual CLI overrides (uses None if not provided)

    Returns:
        FallbackPlan ready for use

    Raises:
        FileNotFoundError: If YAML not found
        ConfigurationError: If configuration invalid
    """
    # Default YAML path
    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "config" / "fallback.yaml"

    # Load from all sources
    yaml_config = load_from_yaml(yaml_path)
    env_config = load_from_env() if env_overrides is None else env_overrides
    cli_config = load_from_cli(cli_overrides)

    # Merge with precedence
    merged = merge_configs(yaml_config, env_config, cli_config)

    # Build plan
    plan = build_fallback_plan(merged)

    logger.info(
        f"Loaded FallbackPlan from {yaml_path} (+env={bool(env_config)} +cli={bool(cli_config)})"
    )
    return plan


__all__ = [
    "ConfigurationError",
    "load_from_yaml",
    "load_from_env",
    "load_from_cli",
    "merge_configs",
    "validate_config",
    "build_fallback_plan",
    "load_fallback_plan",
]
