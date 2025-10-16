# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.pipeline",
#   "purpose": "Planning helpers for the ontology downloader pipeline",
#   "sections": []
# }
# === /NAVMAP ===

"""Fetch planning helpers for the ontology downloader pipeline."""

from __future__ import annotations

from typing import Mapping, Optional

from .config import ConfigError, DefaultsConfig, _coerce_sequence


__all__ = ["merge_defaults"]


def _make_fetch_spec(
    raw_spec: Mapping[str, object],
    defaults: DefaultsConfig,
    *,
    allow_missing_resolvers: bool = False,
):
    """Create a fetch specification from raw configuration and defaults."""

    from .ontology_download import FetchSpec  # local import to avoid circular dependency
    from .resolvers import ResolverRegistry

    if "id" not in raw_spec:
        raise ConfigError("Ontology specification missing required field 'id'")

    ontology_id = str(raw_spec["id"]).strip()
    if not ontology_id:
        raise ConfigError("Ontology ID cannot be empty")

    fallback = _coerce_sequence(raw_spec.get("fallback"))
    prefer_source = _coerce_sequence(raw_spec.get("prefer_source")) or defaults.prefer_source
    normalize_to = _coerce_sequence(raw_spec.get("normalize_to")) or defaults.normalize_to
    accept_licenses = _coerce_sequence(raw_spec.get("accept_licenses")) or defaults.accept_licenses
    plan_validation = raw_spec.get("plan_validation", True)
    extras = raw_spec.get("extras")
    if extras is not None and not isinstance(extras, Mapping):
        raise ConfigError(f"Ontology '{ontology_id}' extras must be a mapping if provided")

    if not allow_missing_resolvers:
        missing = [resolver for resolver in prefer_source if resolver not in ResolverRegistry]
        if missing:
            raise ConfigError(
                "Unknown resolver(s) specified: " + ", ".join(sorted(set(missing)))
            )

    return FetchSpec(
        id=ontology_id,
        fallback=fallback,
        prefer_source=prefer_source,
        normalize_to=normalize_to,
        accept_licenses=accept_licenses,
        plan_validation=bool(plan_validation),
        extras=dict(extras or {}),
    )


def merge_defaults(
    raw_spec: Mapping[str, object], defaults: DefaultsConfig, *, index: Optional[int] = None
):
    """Merge user-provided specification with defaults to create a fetch spec."""

    allow_missing_resolvers = bool(raw_spec.get("allow_missing_resolvers", False))

    try:
        return _make_fetch_spec(
            raw_spec,
            defaults,
            allow_missing_resolvers=allow_missing_resolvers,
        )
    except ConfigError as exc:
        location = f"ontologies[{index}]" if index is not None else "ontologies[]"
        raise ConfigError(f"{location}: {exc}") from exc

