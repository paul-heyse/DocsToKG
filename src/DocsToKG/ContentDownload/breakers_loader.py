# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.breakers_loader",
#   "purpose": "Load breaker configuration from YAML with env/CLI overlays",
#   "sections": [
#     {
#       "id": "parse-kv-overrides",
#       "name": "_parse_kv_overrides",
#       "anchor": "function-parse-kv-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-host-key",
#       "name": "_normalize_host_key",
#       "anchor": "function-normalize-host-key",
#       "kind": "function"
#     },
#     {
#       "id": "load-yaml",
#       "name": "_load_yaml",
#       "anchor": "function-load-yaml",
#       "kind": "function"
#     },
#     {
#       "id": "config-from-yaml",
#       "name": "_config_from_yaml",
#       "anchor": "function-config-from-yaml",
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
#       "id": "validate",
#       "name": "_validate",
#       "anchor": "function-validate",
#       "kind": "function"
#     },
#     {
#       "id": "load-breaker-config",
#       "name": "load_breaker_config",
#       "anchor": "function-load-breaker-config",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Breaker config loader: YAML -> BreakerConfig with env/CLI overlays.

This module loads breaker configuration from YAML files and applies environment
variable and CLI command-line overlays with proper precedence handling.

Usage:
    from pathlib import Path
    from DocsToKG.ContentDownload.breakers_loader import load_breaker_config

    cfg = load_breaker_config(
        yaml_path=os.getenv("DOCSTOKG_BREAKERS_YAML"),
        env=os.environ,
        cli_host_overrides=[
            # --breaker api.crossref.org=fail_max:5,reset:60,retry_after_cap:900
            "api.crossref.org=fail_max:5,reset:60,retry_after_cap:900"
        ],
        cli_role_overrides=[
            # --breaker-role web.archive.org:artifact=fail_max:2,reset:120,trial_calls:2
            "web.archive.org:artifact=fail_max:2,reset:120,trial_calls:2"
        ],
        cli_resolver_overrides=[
            # --breaker-resolver landing_page=fail_max:4,reset:45
            "landing_page=fail_max:4,reset:45"
        ],
    )
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    import idna  # IDNA 2008 with UTS #46 support
except Exception as e:  # pragma: no cover
    raise RuntimeError("idna is required for proper hostname normalization") from e

try:
    import yaml  # PyYAML  # type: ignore[import-untyped]
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load breaker YAML configs") from e

import logging

from DocsToKG.ContentDownload.breakers import (
    BreakerClassification,
    BreakerConfig,
    BreakerPolicy,
    BreakerRolePolicy,
    HalfOpenPolicy,
    RequestRole,
    RollingWindowPolicy,
)

LOGGER = logging.getLogger(__name__)

# ------------------------------
# Parsing helpers
# ------------------------------

_ROLE_ALIASES = {
    "meta": RequestRole.METADATA,
    "metadata": RequestRole.METADATA,
    "landing": RequestRole.LANDING,
    "artifact": RequestRole.ARTIFACT,
}


def _parse_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(v: str) -> int:
    return int(v.strip().replace("_", ""))


def _normalize_host_key(host: str) -> str:
    """
    Normalize a host to the canonical breaker key using IDNA 2008 + UTS #46.

    This function ensures stable, RFC-compliant hostname keys across the system:
    - Strips whitespace & trailing dots
    - Converts to lowercase
    - Applies IDNA 2008 with UTS #46 mapping for internationalized domain names (IDNs)
    - Falls back gracefully to lowercase-only for edge cases

    Args:
        host: The hostname to normalize (Unicode or ASCII)

    Returns:
        Lowercase ASCII-compatible encoding (punycode) of the host, suitable as a dict key

    Examples:
        >>> _normalize_host_key("Example.COM")
        'example.com'
        >>> _normalize_host_key("münchen.example")
        'xn--mnchen-3ya.example'
        >>> _normalize_host_key("  api.crossref.org.")
        'api.crossref.org'

    Notes:
        - Uses IDNA 2008 with UTS #46 compatibility mapping for user input normalization
        - IDNA handles internationalized domain names (IDNs) like café.example → xn--caf-dma.example
        - UTS #46 pre-processes the input to handle case folding and dot-like character normalization
        - Falls back to simple lowercase if IDNA encoding fails (preserves robustness)
    """
    h = host.strip().rstrip(".")
    if not h:
        return h

    try:
        # Use IDNA 2008 with UTS #46 mapping for maximum compatibility
        # uts46=True enables compatibility mapping: case-fold, width-normalize, dot-like chars, etc.
        h_ascii = idna.encode(h, uts46=True).decode("ascii")
        return h_ascii
    except idna.IDNAError as e:
        # Log the IDNA error but fall back to lowercase for robustness
        # This preserves backward compatibility while attempting IDNA normalization
        LOGGER.debug(f"IDNA encoding failed for '{h}': {e}; falling back to lowercase")
        return h.lower()
    except Exception:
        # Final fallback for any other exception (encoding issues, etc.)
        return h.lower()


def _parse_kv_overrides(s: str) -> Dict[str, str]:
    """
    Parse "fail_max:5,reset:60,retry_after_cap:900,trial_calls:2"
    into a dict of raw strings.
    """
    out: Dict[str, str] = {}
    if not s:
        return out
    for part in s.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            # allow "enabled=true" style too (with =)
            if "=" in part:
                k, v = part.split("=", 1)
            else:
                # bare tokens treated as flags with true
                k, v = part, "true"
        else:
            k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _merge_policy(base: BreakerPolicy, raw: Mapping[str, str]) -> BreakerPolicy:
    """
    Merge simple host-level overrides into a BreakerPolicy.
    Accepts keys: fail_max, reset, reset_timeout_s, retry_after_cap, retry_after_cap_s.
    """
    fail_max = int(raw["fail_max"]) if "fail_max" in raw else base.fail_max
    reset = raw.get("reset") or raw.get("reset_timeout_s")
    reset_timeout_s = int(reset) if reset is not None else base.reset_timeout_s
    rac = raw.get("retry_after_cap") or raw.get("retry_after_cap_s")
    retry_after_cap_s = int(rac) if rac is not None else base.retry_after_cap_s
    return replace(
        base,
        fail_max=fail_max,
        reset_timeout_s=reset_timeout_s,
        retry_after_cap_s=retry_after_cap_s,
    )


def _merge_role_policy(base: BreakerRolePolicy, raw: Mapping[str, str]) -> BreakerRolePolicy:
    """
    Merge role-level overrides into BreakerRolePolicy.
    Keys: fail_max, reset, reset_timeout_s, success_threshold, trial_calls
    """
    fm = int(raw["fail_max"]) if "fail_max" in raw else base.fail_max
    r = raw.get("reset") or raw.get("reset_timeout_s")
    rs = int(r) if r is not None else base.reset_timeout_s
    st = int(raw["success_threshold"]) if "success_threshold" in raw else base.success_threshold
    tc = int(raw["trial_calls"]) if "trial_calls" in raw else base.trial_calls
    return replace(base, fail_max=fm, reset_timeout_s=rs, success_threshold=st, trial_calls=tc)


def _apply_classify_override(cur: BreakerClassification, s: str) -> BreakerClassification:
    """
    e.g. "failure=429,500,502,503,504,408 neutral=401,403,404,410,451"
    """
    if not s:
        return cur
    # Support both spaces and commas; accept "failure:" or "failure="
    m_fail = re.search(r"(failure\s*[:=]\s*)([0-9,\s]+)", s, re.I)
    m_neut = re.search(r"(neutral\s*[:=]\s*)([0-9,\s]+)", s, re.I)
    fset = cur.failure_statuses
    nset = cur.neutral_statuses
    if m_fail:
        nums = [int(x) for x in re.split(r"[,\s]+", m_fail.group(2).strip()) if x]
        fset = frozenset(nums)
    if m_neut:
        nums = [int(x) for x in re.split(r"[,\s]+", m_neut.group(2).strip()) if x]
        nset = frozenset(nums)
    return BreakerClassification(
        failure_statuses=fset, neutral_statuses=nset, failure_exceptions=cur.failure_exceptions
    )


def _apply_rolling_override(cur: RollingWindowPolicy, s: str) -> RollingWindowPolicy:
    """
    e.g. "enabled:true,window:30,thresh:6,cooldown:60"
    """
    if not s:
        return cur
    kv = _parse_kv_overrides(s)
    enabled = _parse_bool(kv["enabled"]) if "enabled" in kv else cur.enabled
    window_s = _parse_int(kv["window"]) if "window" in kv else cur.window_s
    threshold = _parse_int(kv["thresh"]) if "thresh" in kv else cur.threshold_failures
    cooldown_s = _parse_int(kv["cooldown"]) if "cooldown" in kv else cur.cooldown_s
    return RollingWindowPolicy(
        enabled=enabled, window_s=window_s, threshold_failures=threshold, cooldown_s=cooldown_s
    )


def _role_from_str(s: str) -> RequestRole:
    key = s.strip().lower()
    if key not in _ROLE_ALIASES:
        raise ValueError(f"Unknown role: {s}")
    return _ROLE_ALIASES[key]


def _merge_docs(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge breaker policy dictionaries preserving nested role maps."""

    if not override:
        return dict(base)

    result: Dict[str, Any] = dict(base)

    for key, value in override.items():
        if isinstance(value, Mapping):
            if key == "defaults":
                existing = dict(result.get("defaults", {}))
                roles_override = (
                    value.get("roles") if isinstance(value.get("roles"), Mapping) else None
                )
                if roles_override:
                    existing_roles = dict(existing.get("roles", {}))
                    for role_key, role_value in roles_override.items():
                        existing_roles[role_key] = role_value
                    if existing_roles:
                        existing["roles"] = existing_roles
                    value = {k: v for k, v in value.items() if k != "roles"}
                existing.update(value)
                result["defaults"] = existing
                continue
            if key in {"hosts", "resolvers"}:
                existing_map = dict(result.get(key, {}))
                for subkey, subval in value.items():
                    if key == "hosts" and isinstance(subval, Mapping):
                        host_entry = dict(existing_map.get(subkey, {}))
                        roles_override = (
                            subval.get("roles")
                            if isinstance(subval.get("roles"), Mapping)
                            else None
                        )
                        if roles_override:
                            host_roles = dict(host_entry.get("roles", {}))
                            for role_key, role_value in roles_override.items():
                                host_roles[role_key] = role_value
                            if host_roles:
                                host_entry["roles"] = host_roles
                            subval = {k: v for k, v in subval.items() if k != "roles"}
                        host_entry.update(subval)
                        existing_map[subkey] = host_entry
                    else:
                        existing_map[subkey] = subval
                result[key] = existing_map
                continue
            if key == "advanced":
                existing_adv = dict(result.get("advanced", {}))
                existing_adv.update(value)
                result["advanced"] = existing_adv
                continue
        result[key] = value

    return result


def merge_breaker_docs(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Public helper to merge breaker policy documents."""

    return _merge_docs(base, override)


# ------------------------------
# YAML loader & overlays
# ------------------------------


def _load_yaml(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Breaker YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Breaker YAML root must be a mapping")
    return data


def _config_from_yaml(doc: Dict) -> BreakerConfig:
    # Defaults
    d = doc.get("defaults", {}) or {}
    defaults = BreakerPolicy(
        fail_max=int(d.get("fail_max", 5)),
        reset_timeout_s=int(d.get("reset_timeout_s", d.get("reset", 60))),
        retry_after_cap_s=int(d.get("retry_after_cap_s", d.get("retry_after_cap", 900))),
        roles={},  # filled below if provided
    )
    # Role default overrides (optional)
    roles_cfg = (d.get("roles") or {}) if isinstance(d.get("roles"), dict) else {}
    roles_map = {}
    for role_key, rvals in roles_cfg.items():
        role = _role_from_str(role_key)
        roles_map[role] = BreakerRolePolicy(
            fail_max=_maybe_int(rvals.get("fail_max")),
            reset_timeout_s=_maybe_int(rvals.get("reset_timeout_s", rvals.get("reset"))),
            success_threshold=_maybe_int(rvals.get("success_threshold")),
            trial_calls=int(rvals.get("trial_calls", 1)),
        )
    defaults = replace(defaults, roles=roles_map)

    # Classification
    classify_doc = doc.get("defaults", {}).get("classify", {}) or {}
    failure_statuses = (
        frozenset(int(x) for x in classify_doc.get("failure_statuses", []) or []) or None
    )
    neutral_statuses = (
        frozenset(int(x) for x in classify_doc.get("neutral_statuses", []) or []) or None
    )
    classify = BreakerClassification(
        failure_statuses=failure_statuses or BreakerClassification().failure_statuses,
        neutral_statuses=neutral_statuses or BreakerClassification().neutral_statuses,
        failure_exceptions=BreakerClassification().failure_exceptions,
    )

    # Half-open
    half_open_doc = doc.get("defaults", {}).get("half_open", {}) or {}
    half_open = HalfOpenPolicy(jitter_ms=int(half_open_doc.get("jitter_ms", 150)))

    # Rolling policy
    rolling_doc = (doc.get("advanced") or {}).get("rolling", {}) or {}
    rolling = RollingWindowPolicy(
        enabled=bool(rolling_doc.get("enabled", False)),
        window_s=int(rolling_doc.get("window_s", rolling_doc.get("window", 30))),
        threshold_failures=int(rolling_doc.get("threshold_failures", rolling_doc.get("thresh", 6))),
        cooldown_s=int(rolling_doc.get("cooldown_s", rolling_doc.get("cooldown", 60))),
    )

    # Hosts
    hosts_map: Dict[str, BreakerPolicy] = {}
    for host, hvals in (doc.get("hosts") or {}).items():
        key = _normalize_host_key(host)
        pol = BreakerPolicy(
            fail_max=int(hvals.get("fail_max", defaults.fail_max)),
            reset_timeout_s=int(
                hvals.get("reset_timeout_s", hvals.get("reset", defaults.reset_timeout_s))
            ),
            retry_after_cap_s=int(
                hvals.get(
                    "retry_after_cap_s", hvals.get("retry_after_cap", defaults.retry_after_cap_s)
                )
            ),
            roles={},  # fill per-role below if present
        )
        rmap = {}
        for role_key, rvals in (hvals.get("roles") or {}).items():
            role = _role_from_str(role_key)
            rmap[role] = BreakerRolePolicy(
                fail_max=_maybe_int(rvals.get("fail_max")),
                reset_timeout_s=_maybe_int(rvals.get("reset_timeout_s", rvals.get("reset"))),
                success_threshold=_maybe_int(rvals.get("success_threshold")),
                trial_calls=int(rvals.get("trial_calls", 1)),
            )
        pol = replace(pol, roles=rmap)
        hosts_map[key] = pol

    # Resolvers (optional)
    resolvers_map: Dict[str, BreakerPolicy] = {}
    for res, rvals in (doc.get("resolvers") or {}).items():
        resolvers_map[res] = BreakerPolicy(
            fail_max=int(rvals.get("fail_max", defaults.fail_max)),
            reset_timeout_s=int(
                rvals.get("reset_timeout_s", rvals.get("reset", defaults.reset_timeout_s))
            ),
            retry_after_cap_s=int(
                rvals.get(
                    "retry_after_cap_s", rvals.get("retry_after_cap", defaults.retry_after_cap_s)
                )
            ),
            roles={},  # role overrides for resolvers generally unnecessary
        )

    return BreakerConfig(
        defaults=defaults,
        classify=classify,
        half_open=half_open,
        rolling=rolling,
        hosts=hosts_map,
        resolvers=resolvers_map,
    )


def _maybe_int(v) -> Optional[int]:
    if v is None:
        return None
    return int(v)


# ------------------------------
# Env + CLI overlays
# ------------------------------


def _apply_env_overlays(cfg: BreakerConfig, env: Mapping[str, str]) -> BreakerConfig:
    """
    Supported envs:
      DOCSTOKG_BREAKER__<HOST>=fail_max:5,reset:60,retry_after_cap:900
      DOCSTOKG_BREAKER_ROLE__<HOST>__<ROLE>=fail_max:4,reset:45,trial_calls:2
      DOCSTOKG_BREAKER_RESOLVER__<RESOLVER>=fail_max:4,reset:45
      DOCSTOKG_BREAKER_CLASSIFY="failure=..., neutral=..."
      DOCSTOKG_BREAKER_ROLLING="enabled:true,window:30,thresh:6,cooldown:60"
      DOCSTOKG_BREAKER_DEFAULTS="fail_max:5,reset:60,retry_after_cap:900"
    """
    new_cfg = cfg

    # Defaults
    if s := env.get("DOCSTOKG_BREAKER_DEFAULTS"):
        new_cfg = replace(new_cfg, defaults=_merge_policy(new_cfg.defaults, _parse_kv_overrides(s)))

    # Classification
    if s := env.get("DOCSTOKG_BREAKER_CLASSIFY"):
        new_cfg = replace(new_cfg, classify=_apply_classify_override(new_cfg.classify, s))

    # Rolling
    if s := env.get("DOCSTOKG_BREAKER_ROLLING"):
        new_cfg = replace(new_cfg, rolling=_apply_rolling_override(new_cfg.rolling, s))

    # Host overrides
    prefix = "DOCSTOKG_BREAKER__"
    for k, v in env.items():
        if not k.startswith(prefix):
            continue
        host = _normalize_host_key(k[len(prefix) :])
        pol = new_cfg.hosts.get(host, new_cfg.defaults)
        hosts_map = dict(new_cfg.hosts)
        hosts_map[host] = _merge_policy(pol, _parse_kv_overrides(v))
        new_cfg = replace(new_cfg, hosts=hosts_map)

    # Role overrides per host
    prefix = "DOCSTOKG_BREAKER_ROLE__"
    for k, v in env.items():
        if not k.startswith(prefix):
            continue
        tail = k[len(prefix) :]  # "<HOST>__<ROLE>"
        try:
            host_raw, role_raw = tail.split("__", 1)
        except ValueError:
            raise ValueError(f"Invalid env var key (expected HOST__ROLE): {k}")
        host = _normalize_host_key(host_raw)
        role = _role_from_str(role_raw)
        base_pol = new_cfg.hosts.get(host, new_cfg.defaults)
        cur_role_pol = base_pol.roles.get(role, BreakerRolePolicy())
        merged_role = _merge_role_policy(cur_role_pol, _parse_kv_overrides(v))
        roles_updated = dict(base_pol.roles)
        roles_updated[role] = merged_role
        hosts_map = dict(new_cfg.hosts)
        hosts_map[host] = replace(base_pol, roles=roles_updated)
        new_cfg = replace(new_cfg, hosts=hosts_map)

    # Resolver overrides
    prefix = "DOCSTOKG_BREAKER_RESOLVER__"
    for k, v in env.items():
        if not k.startswith(prefix):
            continue
        res = k[len(prefix) :]
        base_pol = new_cfg.resolvers.get(res, new_cfg.defaults)
        resolvers_map = dict(new_cfg.resolvers)
        resolvers_map[res] = _merge_policy(base_pol, _parse_kv_overrides(v))
        new_cfg = replace(new_cfg, resolvers=resolvers_map)

    return new_cfg


def _apply_cli_overrides(
    cfg: BreakerConfig,
    *,
    cli_host_overrides: Sequence[str] | None,
    cli_role_overrides: Sequence[str] | None,
    cli_resolver_overrides: Sequence[str] | None,
    cli_defaults_override: Optional[str],
    cli_classify_override: Optional[str],
    cli_rolling_override: Optional[str],
) -> BreakerConfig:
    """
    CLI formats:
      --breaker HOST=fail_max:5,reset:60,retry_after_cap:900
      --breaker-role HOST:ROLE=fail_max:4,reset:45,trial_calls:2
      --breaker-resolver NAME=fail_max:4,reset:45
      --breaker-defaults "fail_max:5,reset:60"
      --breaker-classify "failure=429,500,... neutral=401,403,..."
      --breaker-rolling "enabled:true,window:30,thresh:6,cooldown:60"
    """
    new_cfg = cfg

    if cli_defaults_override:
        new_cfg = replace(
            new_cfg,
            defaults=_merge_policy(new_cfg.defaults, _parse_kv_overrides(cli_defaults_override)),
        )

    if cli_classify_override:
        new_cfg = replace(
            new_cfg, classify=_apply_classify_override(new_cfg.classify, cli_classify_override)
        )

    if cli_rolling_override:
        new_cfg = replace(
            new_cfg, rolling=_apply_rolling_override(new_cfg.rolling, cli_rolling_override)
        )

    # Hosts
    for item in cli_host_overrides or ():
        if "=" not in item:
            raise ValueError(f"Invalid --breaker item (expected HOST=...): {item}")
        host_raw, settings = item.split("=", 1)
        host = _normalize_host_key(host_raw)
        base_pol = new_cfg.hosts.get(host, new_cfg.defaults)
        hosts_map = dict(new_cfg.hosts)
        hosts_map[host] = _merge_policy(base_pol, _parse_kv_overrides(settings))
        new_cfg = replace(new_cfg, hosts=hosts_map)

    # Host role
    for item in cli_role_overrides or ():
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(f"Invalid --breaker-role (expected HOST:ROLE=...): {item}")
        left, settings = item.split("=", 1)
        host_raw, role_raw = left.split(":", 1)
        host = _normalize_host_key(host_raw)
        role = _role_from_str(role_raw)
        base_pol = new_cfg.hosts.get(host, new_cfg.defaults)
        cur_role_pol = base_pol.roles.get(role, BreakerRolePolicy())
        merged_role = _merge_role_policy(cur_role_pol, _parse_kv_overrides(settings))
        roles_updated = dict(base_pol.roles)
        roles_updated[role] = merged_role
        hosts_map = dict(new_cfg.hosts)
        hosts_map[host] = replace(base_pol, roles=roles_updated)
        new_cfg = replace(new_cfg, hosts=hosts_map)

    # Resolver
    for item in cli_resolver_overrides or ():
        if "=" not in item:
            raise ValueError(f"Invalid --breaker-resolver (expected NAME=...): {item}")
        name, settings = item.split("=", 1)
        base_pol = new_cfg.resolvers.get(name, new_cfg.defaults)
        resolvers_map = dict(new_cfg.resolvers)
        resolvers_map[name] = _merge_policy(base_pol, _parse_kv_overrides(settings))
        new_cfg = replace(new_cfg, resolvers=resolvers_map)

    return new_cfg


# ------------------------------
# Validation
# ------------------------------


def _validate(cfg: BreakerConfig) -> None:
    def _chk_pol(pol: BreakerPolicy, ctx: str) -> None:
        if pol.fail_max < 1:
            raise ValueError(f"{ctx}: fail_max must be >=1")
        if pol.reset_timeout_s <= 0:
            raise ValueError(f"{ctx}: reset_timeout_s must be >0")
        if pol.retry_after_cap_s <= 0:
            raise ValueError(f"{ctx}: retry_after_cap_s must be >0")
        for role, rpol in (pol.roles or {}).items():
            if rpol.fail_max is not None and rpol.fail_max < 1:
                raise ValueError(f"{ctx}:{role.value}: role.fail_max must be >=1 if set")
            if rpol.reset_timeout_s is not None and rpol.reset_timeout_s <= 0:
                raise ValueError(f"{ctx}:{role.value}: role.reset_timeout_s must be >0 if set")

    _chk_pol(cfg.defaults, "defaults")
    for host, pol in cfg.hosts.items():
        _chk_pol(pol, f"host[{host}]")
    for name, pol in cfg.resolvers.items():
        _chk_pol(pol, f"resolver[{name}]")


# ------------------------------
# Public entrypoint
# ------------------------------


def load_breaker_config(
    yaml_path: Optional[str | Path],
    *,
    env: Mapping[str, str],
    cli_host_overrides: Sequence[str] | None = None,
    cli_role_overrides: Sequence[str] | None = None,
    cli_resolver_overrides: Sequence[str] | None = None,
    cli_defaults_override: Optional[str] = None,
    cli_classify_override: Optional[str] = None,
    cli_rolling_override: Optional[str] = None,
    base_doc: Optional[Mapping[str, Any]] = None,
    extra_yaml_paths: Sequence[str | Path] | None = None,
) -> BreakerConfig:
    """
    Load breaker configuration with precedence:
      YAML (if provided) -> env overlays -> CLI overlays.

    - Host keys are normalized to lowercased punycode.
    - Role strings are case-insensitive (meta/metadata, landing, artifact).
    - Validates basic invariants (fail_max >=1, reset>0, caps>0).
    """
    merged_doc: Dict[str, Any] = {}
    if base_doc:
        merged_doc = merge_breaker_docs(merged_doc, base_doc)

    merged_doc = merge_breaker_docs(merged_doc, _load_yaml(yaml_path))

    for extra in extra_yaml_paths or ():
        merged_doc = merge_breaker_docs(merged_doc, _load_yaml(extra))

    cfg = _config_from_yaml(merged_doc)
    cfg = _apply_env_overlays(cfg, env)
    cfg = _apply_cli_overrides(
        cfg,
        cli_host_overrides=cli_host_overrides,
        cli_role_overrides=cli_role_overrides,
        cli_resolver_overrides=cli_resolver_overrides,
        cli_defaults_override=cli_defaults_override,
        cli_classify_override=cli_classify_override,
        cli_rolling_override=cli_rolling_override,
    )

    # Ensure all host keys are normalized (handles cases where YAML/env were mixed)
    if cfg.hosts:
        cfg = replace(
            cfg,
            hosts={_normalize_host_key(h): pol for h, pol in cfg.hosts.items()},
        )

    _validate(cfg)
    return cfg
