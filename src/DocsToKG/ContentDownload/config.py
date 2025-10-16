"""
Resolver configuration helpers shared by the DocsToKG downloader CLI.

This module centralises logic for reading resolver configuration files,
applying overrides, and constructing :class:`ResolverConfig` instances so the
command-line entrypoint can remain thin and testable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from DocsToKG.ContentDownload import resolvers

ResolverConfig = resolvers.ResolverConfig

__all__ = (
    "ResolverConfig",
    "apply_config_overrides",
    "load_resolver_config",
    "read_resolver_config",
)


def read_resolver_config(path: Path) -> Dict[str, Any]:
    """Read resolver configuration from JSON or YAML files."""

    text = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install PyYAML to load YAML resolver configs, or provide JSON.") from exc
        return yaml.safe_load(text) or {}

    if ext in {".json", ".jsn"}:
        return json.loads(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Unable to parse resolver config. Install PyYAML or provide JSON."
            ) from exc
        return yaml.safe_load(text) or {}


def _seed_resolver_toggle_defaults(config: ResolverConfig, resolver_names: Sequence[str]) -> None:
    """Ensure resolver toggles include defaults for every known resolver."""

    for name in resolver_names:
        default_enabled = resolvers.DEFAULT_RESOLVER_TOGGLES.get(name, True)
        config.resolver_toggles.setdefault(name, default_enabled)


def apply_config_overrides(
    config: ResolverConfig,
    data: Dict[str, Any],
    resolver_names: Sequence[str],
) -> None:
    """Apply overrides from configuration data onto a ResolverConfig."""

    for field_name in (
        "resolver_order",
        "resolver_toggles",
        "max_attempts_per_work",
        "timeout",
        "sleep_jitter",
        "polite_headers",
        "unpaywall_email",
        "core_api_key",
        "semantic_scholar_api_key",
        "doaj_api_key",
        "resolver_timeouts",
        "resolver_min_interval_s",
        "mailto",
        "resolver_head_precheck",
        "head_precheck_host_overrides",
        "host_accept_overrides",
        "domain_token_buckets",
        "resolver_circuit_breakers",
    ):
        if field_name in data and data[field_name] is not None:
            setattr(config, field_name, data[field_name])

    if "resolver_rate_limits" in data:
        raise ValueError(
            "resolver_rate_limits is no longer supported. Rename entries to resolver_min_interval_s."
        )

    _seed_resolver_toggle_defaults(config, resolver_names)


def load_resolver_config(
    args: argparse.Namespace,
    resolver_names: Sequence[str],
    resolver_order_override: Optional[List[str]] = None,
) -> ResolverConfig:
    """Construct resolver configuration combining CLI, config files, and env vars."""

    config = ResolverConfig()
    if args.resolver_config:
        config_data = read_resolver_config(Path(args.resolver_config))
        apply_config_overrides(config, config_data, resolver_names)

    config.unpaywall_email = (
        args.unpaywall_email
        or config.unpaywall_email
        or os.getenv("UNPAYWALL_EMAIL")
        or args.mailto
    )
    config.core_api_key = args.core_api_key or config.core_api_key or os.getenv("CORE_API_KEY")
    config.semantic_scholar_api_key = (
        args.semantic_scholar_api_key or config.semantic_scholar_api_key or os.getenv("S2_API_KEY")
    )
    config.doaj_api_key = args.doaj_api_key or config.doaj_api_key or os.getenv("DOAJ_API_KEY")
    config.mailto = args.mailto or config.mailto

    if getattr(args, "max_resolver_attempts", None):
        config.max_attempts_per_work = args.max_resolver_attempts
    if getattr(args, "resolver_timeout", None):
        config.timeout = args.resolver_timeout
    if hasattr(args, "concurrent_resolvers") and args.concurrent_resolvers is not None:
        config.max_concurrent_resolvers = args.concurrent_resolvers

    if resolver_order_override:
        ordered: List[str] = []
        for name in resolver_order_override:
            if name not in ordered:
                ordered.append(name)
        for name in resolver_names:
            if name not in ordered:
                ordered.append(name)
        config.resolver_order = ordered

    for disabled in getattr(args, "disable_resolver", []) or []:
        config.resolver_toggles[disabled] = False

    for enabled in getattr(args, "enable_resolver", []) or []:
        config.resolver_toggles[enabled] = True

    _seed_resolver_toggle_defaults(config, resolver_names)

    if hasattr(args, "global_url_dedup") and args.global_url_dedup is not None:
        config.enable_global_url_dedup = args.global_url_dedup

    if getattr(args, "domain_min_interval", None):
        domain_limits = dict(config.domain_min_interval_s)
        for domain, interval in args.domain_min_interval:
            domain_limits[domain] = interval
        config.domain_min_interval_s = domain_limits

    headers = dict(config.polite_headers)
    existing_mailto = headers.get("mailto")
    mailto_value = config.mailto or existing_mailto
    base_agent = headers.get("User-Agent") or "DocsToKGDownloader/1.0"
    if mailto_value:
        config.mailto = config.mailto or mailto_value
        headers["mailto"] = mailto_value
        user_agent = f"DocsToKGDownloader/1.0 (+{mailto_value}; mailto:{mailto_value})"
    else:
        headers.pop("mailto", None)
        user_agent = base_agent
    headers["User-Agent"] = user_agent
    accept_override = getattr(args, "accept", None)
    if accept_override:
        headers["Accept"] = accept_override
    elif not headers.get("Accept"):
        headers["Accept"] = "application/pdf, text/html;q=0.9, */*;q=0.8"
    config.polite_headers = headers

    if hasattr(args, "head_precheck") and args.head_precheck is not None:
        config.enable_head_precheck = args.head_precheck

    config.resolver_min_interval_s.setdefault("unpaywall", 1.0)

    return config
