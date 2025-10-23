# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.ratelimits_loader",
#   "purpose": "Load rate limit configuration from YAML, environment variables, and CLI arguments.",
#   "sections": [
#     {
#       "id": "normalize-host-key",
#       "name": "_normalize_host_key",
#       "anchor": "function-normalize-host-key",
#       "kind": "function"
#     },
#     {
#       "id": "load-yaml-file",
#       "name": "_load_yaml_file",
#       "anchor": "function-load-yaml-file",
#       "kind": "function"
#     },
#     {
#       "id": "merge-dicts",
#       "name": "_merge_dicts",
#       "anchor": "function-merge-dicts",
#       "kind": "function"
#     },
#     {
#       "id": "parse-rate-list",
#       "name": "_parse_rate_list",
#       "anchor": "function-parse-rate-list",
#       "kind": "function"
#     },
#     {
#       "id": "apply-env-overlays",
#       "name": "_apply_env_overlays",
#       "anchor": "function-apply-env-overlays",
#       "kind": "function"
#     },
#     {
#       "id": "apply-cli-overrides",
#       "name": "_apply_cli_overrides",
#       "anchor": "function-apply-cli-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "build-role-rates",
#       "name": "_build_role_rates",
#       "anchor": "function-build-role-rates",
#       "kind": "function"
#     },
#     {
#       "id": "build-host-policy",
#       "name": "_build_host_policy",
#       "anchor": "function-build-host-policy",
#       "kind": "function"
#     },
#     {
#       "id": "build-aimd-config",
#       "name": "_build_aimd_config",
#       "anchor": "function-build-aimd-config",
#       "kind": "function"
#     },
#     {
#       "id": "get-default-config",
#       "name": "_get_default_config",
#       "anchor": "function-get-default-config",
#       "kind": "function"
#     },
#     {
#       "id": "load-rate-config",
#       "name": "load_rate_config",
#       "anchor": "function-load-rate-config",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Load rate limit configuration from YAML, environment variables, and CLI arguments.

Supports hierarchical configuration with precedence:
  CLI > Environment > YAML > Defaults

Features:
- Multi-window rate parsing (e.g., "10/SECOND,5000/HOUR")
- Per-host and per-role configuration
- AIMD tuning parameters
- Backend selection (memory, multiprocess, sqlite, redis)
- Deep configuration merging
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise RuntimeError("PyYAML is required for loading rate limit configurations") from e

from DocsToKG.ContentDownload.ratelimit import (
    AIMDConfig,
    BackendConfig,
    HostPolicy,
    RateConfig,
    RoleRates,
)

LOGGER = logging.getLogger(__name__)


def _normalize_host_key(host: str) -> str:
    """Normalize hostname to lowercase for consistent keying."""
    return host.strip().rstrip(".").lower()


def _load_yaml_file(yaml_path: str | Path | None) -> dict[str, Any]:
    """Load and parse a YAML configuration file."""
    if not yaml_path:
        return {}

    path = Path(yaml_path)
    if not path.exists():
        LOGGER.warning(f"Rate limit config file not found: {yaml_path}")
        return {}

    try:
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        LOGGER.info(f"Loaded rate limit config from {yaml_path}")
        return doc
    except Exception as e:
        LOGGER.error(f"Failed to parse rate limit YAML {yaml_path}: {e}")
        raise


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base dictionary."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _parse_rate_list(rate_str: str) -> list[str]:
    """Parse comma-separated rate strings like '10/SECOND,5000/HOUR'."""
    if not rate_str:
        return []
    return [r.strip() for r in rate_str.split(",") if r.strip()]


def _apply_env_overlays(cfg: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Apply environment variable overlays to configuration."""
    # DOCSTOKG_RLIMITS_YAML: path to additional YAML file
    # DOCSTOKG_RLIMIT_BACKEND: backend kind (memory, sqlite, redis)
    # DOCSTOKG_RLIMIT_GLOBAL_INFLIGHT: global in-flight ceiling
    # DOCSTOKG_RLIMIT_AIMD_ENABLED: enable AIMD
    # DOCSTOKG_RLIMIT__host__role: role-specific overrides

    overlay = dict(cfg)

    # Global settings
    if "DOCSTOKG_RLIMIT_BACKEND" in env:
        overlay.setdefault("backend", {})["kind"] = env["DOCSTOKG_RLIMIT_BACKEND"]

    if "DOCSTOKG_RLIMIT_GLOBAL_INFLIGHT" in env:
        try:
            overlay["global_max_inflight"] = int(env["DOCSTOKG_RLIMIT_GLOBAL_INFLIGHT"])
        except ValueError:
            LOGGER.warning(
                f"Invalid DOCSTOKG_RLIMIT_GLOBAL_INFLIGHT: {env['DOCSTOKG_RLIMIT_GLOBAL_INFLIGHT']}"
            )

    if "DOCSTOKG_RLIMIT_AIMD_ENABLED" in env:
        overlay.setdefault("aimd", {})["enabled"] = env["DOCSTOKG_RLIMIT_AIMD_ENABLED"].lower() in (
            "true",
            "1",
            "yes",
        )

    # Per-host/role overrides (DOCSTOKG_RLIMIT__host__role format)
    hosts_overlay: dict[str, Any] = {}
    for key, value in env.items():
        if key.startswith("DOCSTOKG_RLIMIT__"):
            parts = key[len("DOCSTOKG_RLIMIT__") :].split("__")
            if len(parts) >= 2:
                host = _normalize_host_key(parts[0])
                role = parts[1].lower()
                if host not in hosts_overlay:
                    hosts_overlay[host] = {}
                if role not in hosts_overlay[host]:
                    hosts_overlay[host][role] = {}

                # Parse key-value pairs from value (e.g., "rates:10/SECOND+5000/HOUR,max_delay_ms:200")
                for pair in value.split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        k = k.strip().lower()
                        v = v.strip()
                        if k == "rates":
                            v = v.replace("+", ",")
                        hosts_overlay[host][role][k] = v

    if hosts_overlay:
        overlay.setdefault("hosts", {}).update(hosts_overlay)

    return overlay


def _apply_cli_overrides(
    cfg: dict[str, Any],
    cli_host_role_overrides: Sequence[str] | None = None,
    cli_backend: str | None = None,
    cli_global_max_inflight: int | None = None,
    cli_aimd_enabled: bool | None = None,
) -> dict[str, Any]:
    """Apply CLI argument overrides to configuration."""
    overlay = dict(cfg)

    if cli_backend:
        overlay.setdefault("backend", {})["kind"] = cli_backend

    if cli_global_max_inflight is not None:
        overlay["global_max_inflight"] = cli_global_max_inflight

    if cli_aimd_enabled is not None:
        overlay.setdefault("aimd", {})["enabled"] = cli_aimd_enabled

    if cli_host_role_overrides:
        hosts_overlay: dict[str, Any] = {}
        for override in cli_host_role_overrides:
            # Format: "host:role=rates:10/SECOND+5000/HOUR,max_delay_ms:200"
            if ":" not in override or "=" not in override:
                LOGGER.warning(f"Invalid CLI override format: {override}")
                continue

            host_role_part, config_part = override.split("=", 1)
            if ":" not in host_role_part:
                LOGGER.warning(f"Invalid host:role format in override: {host_role_part}")
                continue

            host, role = host_role_part.split(":", 1)
            host = _normalize_host_key(host)
            role = role.lower()

            if host not in hosts_overlay:
                hosts_overlay[host] = {}
            if role not in hosts_overlay[host]:
                hosts_overlay[host][role] = {}

            # Parse config part
            for pair in config_part.split(","):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    if k == "rates":
                        v = v.replace("+", ",")
                    hosts_overlay[host][role][k] = v

        if hosts_overlay:
            overlay.setdefault("hosts", {}).update(hosts_overlay)

    return overlay


def _build_role_rates(role_cfg: dict[str, Any]) -> RoleRates:
    """Build a RoleRates object from configuration dictionary."""
    rates_str = role_cfg.get("rates", "")
    if isinstance(rates_str, list):
        rates = rates_str
    else:
        rates = _parse_rate_list(rates_str)

    return RoleRates(
        rates=rates,
        max_delay_ms=int(role_cfg.get("max_delay_ms", 200)),
        count_head=bool(role_cfg.get("count_head", False)),
        max_concurrent=int(role_cfg["max_concurrent"]) if role_cfg.get("max_concurrent") else None,
    )


def _build_host_policy(host_cfg: dict[str, Any]) -> HostPolicy:
    """Build a HostPolicy object from configuration dictionary."""
    return HostPolicy(
        metadata=_build_role_rates(host_cfg["metadata"]) if "metadata" in host_cfg else None,
        landing=_build_role_rates(host_cfg["landing"]) if "landing" in host_cfg else None,
        artifact=_build_role_rates(host_cfg["artifact"]) if "artifact" in host_cfg else None,
    )


def _build_aimd_config(aimd_cfg: dict[str, Any]) -> AIMDConfig:
    """Build an AIMDConfig object from configuration dictionary."""
    return AIMDConfig(
        enabled=bool(aimd_cfg.get("enabled", False)),
        window_s=int(aimd_cfg.get("window_s", 60)),
        high_429_ratio=float(aimd_cfg.get("high_429_ratio", 0.05)),
        increase_step_pct=int(aimd_cfg.get("increase_step_pct", 5)),
        decrease_step_pct=int(aimd_cfg.get("decrease_step_pct", 20)),
        min_multiplier=float(aimd_cfg.get("min_multiplier", 0.3)),
        max_multiplier=float(aimd_cfg.get("max_multiplier", 1.0)),
    )


def _get_default_config() -> dict[str, Any]:
    """Return default rate limiting configuration."""
    return {
        "defaults": {
            "metadata": {
                "rates": ["10/SECOND", "5000/HOUR"],
                "max_delay_ms": 200,
                "count_head": False,
            },
            "landing": {
                "rates": ["5/SECOND", "2000/HOUR"],
                "max_delay_ms": 250,
                "count_head": False,
            },
            "artifact": {
                "rates": ["2/SECOND", "500/HOUR"],
                "max_delay_ms": 2000,
                "count_head": False,
            },
        },
        "hosts": {},
        "backend": {"kind": "memory", "dsn": ""},
        "aimd": {"enabled": False},
        "global_max_inflight": 500,
    }


def load_rate_config(
    yaml_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cli_host_role_overrides: Sequence[str] | None = None,
    cli_backend: str | None = None,
    cli_global_max_inflight: int | None = None,
    cli_aimd_enabled: bool | None = None,
) -> RateConfig:
    """Load rate limit configuration with hierarchical precedence.

    Precedence (highest to lowest):
      1. CLI arguments
      2. Environment variables
      3. YAML file
      4. Defaults

    Args:
        yaml_path: Path to YAML configuration file
        env: Environment variables (defaults to os.environ)
        cli_host_role_overrides: List of CLI host:role overrides
        cli_backend: CLI backend selection
        cli_global_max_inflight: CLI global in-flight ceiling
        cli_aimd_enabled: CLI AIMD enable flag

    Returns:
        Fully resolved RateConfig object
    """
    if env is None:
        env = os.environ

    # Start with defaults
    cfg = _get_default_config()

    # Apply YAML file
    yaml_cfg = _load_yaml_file(yaml_path)
    if yaml_cfg:
        cfg = _merge_dicts(cfg, yaml_cfg)

    # Apply environment overlays
    cfg = _apply_env_overlays(cfg, env)

    # Apply CLI overrides
    cfg = _apply_cli_overrides(
        cfg,
        cli_host_role_overrides=cli_host_role_overrides,
        cli_backend=cli_backend,
        cli_global_max_inflight=cli_global_max_inflight,
        cli_aimd_enabled=cli_aimd_enabled,
    )

    # Build RateConfig from merged configuration
    defaults_cfg = cfg.get("defaults", {})
    defaults: dict[str, RoleRates] = {}
    for role in ["metadata", "landing", "artifact"]:
        if role in defaults_cfg:
            defaults[role] = _build_role_rates(defaults_cfg[role])  # type: ignore[arg-type]

    hosts_cfg = cfg.get("hosts", {})
    hosts: dict[str, HostPolicy] = {}
    for host_str, host_cfg in hosts_cfg.items():
        host_key = _normalize_host_key(host_str)
        if isinstance(host_cfg, dict):
            hosts[host_key] = _build_host_policy(host_cfg)

    backend_cfg = cfg.get("backend", {})
    backend = BackendConfig(
        kind=backend_cfg.get("kind", "memory"),
        dsn=backend_cfg.get("dsn", ""),
    )

    aimd_cfg = cfg.get("aimd", {})
    aimd = _build_aimd_config(aimd_cfg)

    global_max_inflight = cfg.get("global_max_inflight", 500)

    return RateConfig(
        defaults=defaults,
        hosts=hosts,
        backend=backend,
        aimd=aimd,
        global_max_inflight=global_max_inflight,
    )
