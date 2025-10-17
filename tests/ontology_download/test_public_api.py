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
    "PlannedFetch",
    "FetchResult",
    "DownloadResult",
    "DownloadFailure",
    "OntologyDownloadError",
    "ValidationRequest",
    "ValidationResult",
    "ValidationTimeout",
    "ValidatorSubprocessError",
    "plan_all",
    "plan_one",
    "fetch_all",
    "fetch_one",
    "run_validators",
    "validate_manifest_dict",
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


def test_removed_symbols_are_not_exported() -> None:
    import DocsToKG.OntologyDownload as pkg

    for symbol in {"list_plugins", "about", "cli_main"}:
        assert not hasattr(pkg, symbol)

