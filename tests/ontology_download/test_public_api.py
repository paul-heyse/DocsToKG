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
    "ValidationTimeout",
    "ValidatorSubprocessError",
    "ResolverCandidate",
    "ResolvedConfig",
    "ExpectedChecksum",
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
)


def test_public_api_surface_is_stable() -> None:
    """__all__ should remain unchanged unless intentionally versioned."""

    import DocsToKG.OntologyDownload as pkg

    assert tuple(pkg.__all__) == EXPECTED_PUBLIC_API
    for name in EXPECTED_PUBLIC_API:
        attr = getattr(pkg, name)
        assert attr is not None
    manifest = pkg.PUBLIC_API_MANIFEST
    assert isinstance(manifest, dict)
    assert set(manifest).issubset(set(pkg.__all__))


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
