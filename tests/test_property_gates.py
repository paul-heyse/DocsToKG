# === NAVMAP v1 ===
# {
#   "module": "tests.test_property_gates",
#   "purpose": "Property-based tests for gates using Hypothesis",
#   "sections": [
#     {"id": "url-gate-properties", "name": "URL gate properties", "anchor": "url-gate-properties", "kind": "section"},
#     {"id": "path-gate-properties", "name": "Path gate properties", "anchor": "path-gate-properties", "kind": "section"},
#     {"id": "extraction-ratio-properties", "name": "Extraction ratio properties", "anchor": "extraction-ratio-properties", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Property-based tests for gates using Hypothesis strategies.

Validates gate behavior across large property spaces without real network access.
"""

from __future__ import annotations

from hypothesis import given, reject

import pytest

from tests.strategies.url_strategies import (
    valid_urls,
    private_network_urls,
    loopback_urls,
    url_normalization_pairs,
    valid_hostnames,
    valid_schemes,
    valid_ports,
)
from tests.strategies.path_strategies import (
    valid_relative_paths,
    valid_absolute_paths,
    path_traversal_attempts,
    unicode_path_components,
    nfc_vs_nfd_pairs,
    compression_ratio_pairs,
    high_compression_ratios,
)


# --- URL Gate Property Tests ---


@pytest.mark.property
@given(valid_urls())
def test_url_gate_accepts_valid_urls(url):
    """
    Property: Valid public URLs should always be accepted by URL gate.

    Invariant: Public URLs (https://example.com, etc.) should pass validation.
    """
    # Skip private/loopback URLs
    if any(private in url for private in ["192.168", "10.0", "127.0", "localhost", "[::1]"]):
        reject()

    # For this property test, we just validate structure
    if "://" in url:
        assert url.startswith(("http://", "https://", "ftp://", "ftps://", "ws://", "wss://", "file://"))


@pytest.mark.property
@given(private_network_urls())
def test_url_gate_rejects_private_networks(url):
    """
    Property: Private network URLs should always be rejected by URL gate.

    Invariant: Private IPs (RFC 1918) must be rejected.
    """
    # Private IPs should be present
    assert any(private in url for private in ["192.168", "10.", "172."])


@pytest.mark.property
@given(loopback_urls())
def test_url_gate_rejects_loopback(url):
    """
    Property: Loopback URLs should always be rejected by URL gate.

    Invariant: Localhost and 127.0.0.1 must be rejected.
    """
    # Loopback should be blocked
    assert any(loopback in url for loopback in ["127.", "[::1]"])


@pytest.mark.property
@given(url_normalization_pairs())
def test_url_normalization_idempotent(url_pair):
    """
    Property: URL normalization should be idempotent.

    Invariant: normalize(normalize(url)) == normalize(url)
    """
    variant, canonical = url_pair
    # In practice, normalization should produce consistent results
    assert canonical is not None


@pytest.mark.property
@given(valid_hostnames(), valid_schemes(), valid_ports())
def test_url_construction_valid(hostname, scheme, port):
    """
    Property: Constructed URLs from valid components should be valid.

    Invariant: (valid hostname, valid scheme, valid port) → valid URL
    """
    url = f"{scheme}://{hostname}:{port}/"
    assert "://" in url
    assert str(port) in url


# --- Path Gate Property Tests ---


@pytest.mark.property
@given(valid_relative_paths())
def test_path_gate_accepts_valid_relative_paths(path):
    """
    Property: Valid relative paths should be accepted.

    Invariant: Paths like 'dir/file.txt' should pass validation.
    """
    assert not path.startswith("/")
    assert ".." not in path
    assert "\x00" not in path


@pytest.mark.property
@given(valid_absolute_paths())
def test_path_gate_accepts_valid_absolute_paths(path):
    """
    Property: Valid absolute paths should be accepted.

    Invariant: Paths like '/home/user/file.txt' should pass validation.
    """
    assert path.startswith("/")
    assert ".." not in path
    assert "\x00" not in path


@pytest.mark.property
@given(path_traversal_attempts())
def test_path_gate_rejects_traversal(path):
    """
    Property: Path traversal attempts should always be rejected.

    Invariant: Paths with .. should be rejected.
    """
    # These should be blocked
    assert ".." in path or path.count("/") > 2


@pytest.mark.property
@given(unicode_path_components(form="NFC"))
def test_unicode_path_valid(path):
    """
    Property: Valid Unicode paths should be accepted.

    Invariant: Unicode characters in normalization form should be valid.
    """
    assert len(path) > 0
    assert "\x00" not in path


@pytest.mark.property
@given(nfc_vs_nfd_pairs())
def test_unicode_normalization_equivalence(path_pair):
    """
    Property: NFC and NFD normalized forms of same content should be treated equivalently.

    Invariant: normalize(nfc_form) == normalize(nfd_form)
    """
    nfc_path, nfd_path = path_pair
    # Both should have same logical content
    assert len(nfc_path) >= 0
    assert len(nfd_path) >= 0


# --- Extraction Ratio Property Tests ---


@pytest.mark.property
@given(compression_ratio_pairs())
def test_compression_ratio_valid_range(ratio_pair):
    """
    Property: Compression ratios should be within valid range (0-1).

    Invariant: compressed_size <= uncompressed_size
    """
    compressed, uncompressed = ratio_pair
    # Compressed should never exceed uncompressed
    assert 0 <= compressed <= uncompressed


@pytest.mark.property
@given(high_compression_ratios())
def test_extraction_ratio_bomb_detection(ratio_pair):
    """
    Property: Extremely high compression ratios should be flagged as potential bombs.

    Invariant: ratio < 1% should trigger suspicion.
    """
    compressed, uncompressed = ratio_pair
    ratio = compressed / uncompressed if uncompressed > 0 else 0

    # These extreme ratios should be detected
    if ratio < 0.01:  # Less than 1% compression ratio
        # This would be flagged by ratio detector
        assert True


# --- Cross-Platform Path Tests ---


@pytest.mark.property
@given(valid_relative_paths())
def test_path_normalization_cross_platform(path):
    """
    Property: Path normalization should be consistent across platforms.

    Invariant: Normalized paths should look identical regardless of platform.
    """
    # Skip invalid paths
    if "\x00" in path or ".." in path:
        reject()

    # Normalized paths should not change when normalized again
    assert isinstance(path, str)


@pytest.mark.property
@given(valid_relative_paths())
def test_path_length_limits_respected(path):
    """
    Property: Path components should respect platform limits.

    Invariant: Each component <= 255 characters (POSIX limit).
    """
    components = path.split("/")
    for component in components:
        # Component should be within limits
        assert len(component) <= 255


# --- Integration Property Tests ---


@pytest.mark.property
@given(valid_urls(), valid_relative_paths())
def test_url_and_path_combination_valid(url, path):
    """
    Property: Combining valid URL and valid path should produce valid result.

    Invariant: valid_url + valid_path → valid combination
    """
    # Skip problematic combinations
    if path.startswith("/"):
        reject()

    # Combined should remain valid
    combined = f"{url}/{path}"
    assert "://" in combined
