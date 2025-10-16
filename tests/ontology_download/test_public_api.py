# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_public_api",
#   "purpose": "Guards the public surface of DocsToKG.OntologyDownload",
#   "sections": []
# }
# === /NAVMAP ===

"""Tests ensuring the ontology downloader package facade stays stable."""

from __future__ import annotations

EXPECTED_PUBLIC_API = (
    "FetchSpec",
    "FetchResult",
    "PlannedFetch",
    "DownloadResult",
    "DownloadFailure",
    "OntologyDownloadError",
    "ValidationRequest",
    "ValidationResult",
    "ResolverCandidate",
    "ResolvedConfig",
    "ConfigError",
    "ConfigurationError",
    "DefaultsConfig",
    "DownloadConfiguration",
    "LoggingConfiguration",
    "ValidationConfig",
    "build_resolved_config",
    "merge_defaults",
    "validate_config",
    "load_config",
    "load_raw_yaml",
    "get_env_overrides",
    "fetch_one",
    "fetch_all",
    "plan_one",
    "plan_all",
    "download_stream",
    "extract_archive_safe",
    "run_validators",
    "setup_logging",
    "validate_manifest_dict",
    "validate_url_security",
    "mask_sensitive_data",
    "list_plugins",
    "about",
    "CACHE_DIR",
    "CONFIG_DIR",
    "LOG_DIR",
    "LOCAL_ONTOLOGY_DIR",
    "STORAGE",
    "RDF_MIME_ALIASES",
    "MANIFEST_SCHEMA_VERSION",
    "PUBLIC_API_MANIFEST",
)

EXPECTED_PUBLIC_API_MANIFEST = {
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
    "setup_logging": "function",
    "validate_manifest_dict": "function",
    "validate_url_security": "function",
    "mask_sensitive_data": "function",
    "list_plugins": "function",
    "about": "function",
    "CACHE_DIR": "const",
    "CONFIG_DIR": "const",
    "LOG_DIR": "const",
    "LOCAL_ONTOLOGY_DIR": "const",
    "STORAGE": "const",
    "RDF_MIME_ALIASES": "const",
    "MANIFEST_SCHEMA_VERSION": "const",
    "PUBLIC_API_MANIFEST": "const",
}


def test_public_api_surface_is_stable() -> None:
    """__all__ should remain unchanged unless intentionally versioned."""

    import DocsToKG.OntologyDownload as pkg

    assert tuple(pkg.__all__) == EXPECTED_PUBLIC_API
    for name in EXPECTED_PUBLIC_API:
        attr = getattr(pkg, name)
        assert attr is not None
    assert pkg.PUBLIC_API_MANIFEST == EXPECTED_PUBLIC_API_MANIFEST
    assert set(pkg.PUBLIC_API_MANIFEST) == set(pkg.__all__)


def test_list_plugins_returns_mapping() -> None:
    import DocsToKG.OntologyDownload as pkg

    result = pkg.list_plugins("resolver")
    assert isinstance(result, dict)


def test_about_reports_expected_keys() -> None:
    import DocsToKG.OntologyDownload as pkg

    info = pkg.about()
    assert info["package_version"] == pkg.__version__
    assert "manifest_schema_version" in info
    assert "plugins" in info and isinstance(info["plugins"], dict)
