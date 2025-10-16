# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload",
#   "purpose": "Package initialization for DocsToKG.OntologyDownload",
#   "sections": []
# }
# === /NAVMAP ===

"""Public API for the DocsToKG ontology downloader and resolver pipeline.

This facade exposes the primary fetch utilities used by external callers to
plan resolver fallback chains, download ontologies with hardened validation,
perform stream normalization, and emit schema-compliant manifests with
deterministic fingerprints.
"""

from __future__ import annotations

from .api import (
    CACHE_DIR,
    CONFIG_DIR,
    LOCAL_ONTOLOGY_DIR,
    LOG_DIR,
    MANIFEST_SCHEMA_VERSION,
    RDF_MIME_ALIASES,
    STORAGE,
    ConfigError,
    ConfigurationError,
    DefaultsConfig,
    DownloadConfiguration,
    DownloadFailure,
    DownloadResult,
    FetchResult,
    FetchSpec,
    LoggingConfiguration,
    OntologyDownloadError,
    PlannedFetch,
    ResolvedConfig,
    ResolverCandidate,
    ValidationConfig,
    ValidationRequest,
    ValidationResult,
    ValidationTimeout,
    ValidatorSubprocessError,
    __version__,
    about,
    cli_main,
    build_resolved_config,
    download_stream,
    ensure_python_version,
    extract_archive_safe,
    fetch_all,
    fetch_one,
    generate_correlation_id,
    get_env_overrides,
    get_manifest_schema,
    get_owlready2,
    get_pronto,
    get_pystow,
    get_rdflib,
    infer_version_timestamp,
    list_plugins,
    load_config,
    load_raw_yaml,
    main,
    mask_sensitive_data,
    merge_defaults,
    normalize_streaming,
    parse_http_datetime,
    parse_iso_datetime,
    parse_rate_limit_to_rps,
    parse_version_timestamp,
    plan_all,
    plan_one,
    retry_with_backoff,
    run_validators,
    sanitize_filename,
    setup_logging,
    validate_arelle,
    validate_config,
    validate_manifest_dict,
    validate_owlready2,
    validate_pronto,
    validate_rdflib,
    validate_robot,
    validate_url_security,
)

# --- Globals ---

# --- Globals ---

__all__ = [
    "FetchSpec",
    "FetchResult",
    "PlannedFetch",
    "DownloadResult",
    "DownloadFailure",
    "OntologyDownloadError",
    "ValidationRequest",
    "ValidationResult",
    "ValidationTimeout",
    "ValidatorSubprocessError",
    "ResolverCandidate",
    "ResolvedConfig",
    "ConfigError",
    "ConfigurationError",
    "DefaultsConfig",
    "DownloadConfiguration",
    "ValidationConfig",
    "LoggingConfiguration",
    "build_resolved_config",
    "get_env_overrides",
    "load_raw_yaml",
    "merge_defaults",
    "validate_config",
    "CACHE_DIR",
    "CONFIG_DIR",
    "LOG_DIR",
    "LOCAL_ONTOLOGY_DIR",
    "STORAGE",
    "RDF_MIME_ALIASES",
    "MANIFEST_SCHEMA_VERSION",
    "fetch_one",
    "fetch_all",
    "plan_one",
    "plan_all",
    "download_stream",
    "extract_archive_safe",
    "validate_manifest_dict",
    "validate_url_security",
    "run_validators",
    "normalize_streaming",
    "validate_rdflib",
    "validate_pronto",
    "validate_owlready2",
    "validate_robot",
    "validate_arelle",
    "main",
    "parse_iso_datetime",
    "parse_http_datetime",
    "parse_version_timestamp",
    "infer_version_timestamp",
    "setup_logging",
    "load_config",
    "ensure_python_version",
    "retry_with_backoff",
    "sanitize_filename",
    "generate_correlation_id",
    "parse_rate_limit_to_rps",
    "get_pystow",
    "get_rdflib",
    "get_pronto",
    "get_owlready2",
    "get_manifest_schema",
    "mask_sensitive_data",
    "list_plugins",
    "about",
    "cli_main",
    "__version__",
    "PUBLIC_API_MANIFEST",
]

PUBLIC_API_MANIFEST = {
    "FetchSpec": "class",
    "FetchResult": "class",
    "PlannedFetch": "class",
    "DownloadResult": "class",
    "DownloadFailure": "class",
    "OntologyDownloadError": "class",
    "ValidationRequest": "class",
    "ValidationResult": "class",
    "ResolverCandidate": "class",
    "ResolvedConfig": "class",
    "ConfigError": "class",
    "ConfigurationError": "class",
    "DefaultsConfig": "class",
    "DownloadConfiguration": "class",
    "LoggingConfiguration": "class",
    "ValidationConfig": "class",
    "build_resolved_config": "function",
    "merge_defaults": "function",
    "validate_config": "function",
    "load_config": "function",
    "load_raw_yaml": "function",
    "get_env_overrides": "function",
    "fetch_one": "function",
    "fetch_all": "function",
    "plan_one": "function",
    "plan_all": "function",
    "download_stream": "function",
    "extract_archive_safe": "function",
    "run_validators": "function",
    "normalize_streaming": "function",
    "validate_rdflib": "function",
    "validate_pronto": "function",
    "validate_owlready2": "function",
    "validate_robot": "function",
    "validate_arelle": "function",
    "setup_logging": "function",
    "validate_manifest_dict": "function",
    "validate_url_security": "function",
    "mask_sensitive_data": "function",
    "list_plugins": "function",
    "about": "function",
    "cli_main": "function",
    "format_table": "function",
    "format_plan_rows": "function",
    "format_results_table": "function",
    "format_validation_summary": "function",
    "CACHE_DIR": "const",
    "CONFIG_DIR": "const",
    "LOG_DIR": "const",
    "LOCAL_ONTOLOGY_DIR": "const",
    "STORAGE": "const",
    "RDF_MIME_ALIASES": "const",
    "MANIFEST_SCHEMA_VERSION": "const",
    "PUBLIC_API_MANIFEST": "const",
}


def _install_legacy_aliases() -> None:
    """Populate ``sys.modules`` with legacy module aliases for backwards compatibility."""

    import sys
    import types

    from . import api as _api
    from . import io as _io
    from . import planning as _planning
    from . import settings as _settings
    from . import validation as _validation

    alias_map = {
        "DocsToKG.OntologyDownload.ontology_download": _api,
        "DocsToKG.OntologyDownload.config": _settings,
        "DocsToKG.OntologyDownload.storage": _settings,
        "DocsToKG.OntologyDownload.optdeps": _settings,
        "DocsToKG.OntologyDownload.errors": _settings,
        "DocsToKG.OntologyDownload.io_safe": _io,
        "DocsToKG.OntologyDownload.net": _io,
        "DocsToKG.OntologyDownload.ratelimit": _io,
        "DocsToKG.OntologyDownload.utils": _io,
        "DocsToKG.OntologyDownload.pipeline": _planning,
        "DocsToKG.OntologyDownload.resolvers": _planning,
        "DocsToKG.OntologyDownload.plugins": _validation,
        "DocsToKG.OntologyDownload.validation_core": _validation,
    }

    for name, module in alias_map.items():
        sys.modules.setdefault(name, module)

    class _LegacyModule(types.ModuleType):
        def __getattr__(self, name: str):  # pragma: no cover - thin wrapper
            if name == "main":
                return _api.cli_main
            return getattr(_api, name)

        def __setattr__(self, name: str, value) -> None:  # pragma: no cover - thin wrapper
            setattr(_api, name, value)
            super().__setattr__(name, value)

    cli_module = _LegacyModule("DocsToKG.OntologyDownload.cli")
    original_main = _api.main
    cli_module.main = _api.cli_main
    cli_module.cli_main = _api.cli_main
    cli_module.format_table = _api.format_table
    cli_module.format_plan_rows = _api.format_plan_rows
    cli_module.format_results_table = _api.format_results_table
    cli_module.format_validation_summary = _api.format_validation_summary
    cli_module.CACHE_DIR = CACHE_DIR
    cli_module.CONFIG_DIR = CONFIG_DIR
    cli_module.LOCAL_ONTOLOGY_DIR = LOCAL_ONTOLOGY_DIR
    cli_module.LOG_DIR = LOG_DIR
    cli_module.ONTOLOGY_DIR = getattr(_api, "ONTOLOGY_DIR", LOCAL_ONTOLOGY_DIR)
    if hasattr(_api, "DEFAULT_PLAN_BASELINE"):
        cli_module.DEFAULT_PLAN_BASELINE = getattr(_api, "DEFAULT_PLAN_BASELINE")
    if hasattr(_api, "PLAN_DIFF_FIELDS"):
        cli_module.PLAN_DIFF_FIELDS = getattr(_api, "PLAN_DIFF_FIELDS")
    _api.main = original_main
    sys.modules.setdefault("DocsToKG.OntologyDownload.cli", cli_module)


_install_legacy_aliases()
