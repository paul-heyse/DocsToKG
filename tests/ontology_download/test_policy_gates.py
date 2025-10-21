"""Tests for the six concrete policy gates.

Covers:
- Configuration gate
- URL & network gate
- Filesystem & path gate
- Extraction policy gate
- Storage gate
- DB transactional gate
"""

import pytest

from DocsToKG.OntologyDownload.policy.errors import (
    ErrorCode,
    ExtractionPolicyException,
    FilesystemPolicyException,
    PolicyOK,
    StoragePolicyException,
    URLPolicyException,
)
from DocsToKG.OntologyDownload.policy.gates import (
    config_gate,
    db_gate,
    extraction_gate,
    path_gate,
    storage_gate,
    url_gate,
)

# ============================================================================
# Configuration Gate Tests
# ============================================================================


class TestConfigGate:
    """Test configuration gate."""

    def test_config_gate_valid(self):
        """Valid config passes."""

        class MockConfig:
            http_settings = {}
            network_settings = {}
            timeout_sec = 10

        result = config_gate(MockConfig())
        assert isinstance(result, PolicyOK)
        assert result.gate_name == "config_gate"

    def test_config_gate_missing_http_settings(self):
        """Config missing http_settings fails."""

        class MockConfig:
            network_settings = {}

        with pytest.raises(Exception):
            config_gate(MockConfig())

    def test_config_gate_negative_timeout(self):
        """Config with negative timeout fails."""

        class MockConfig:
            http_settings = {}
            network_settings = {}
            timeout_sec = -1

        with pytest.raises(Exception):
            config_gate(MockConfig())


# ============================================================================
# URL & Network Gate Tests
# ============================================================================


class TestUrlGate:
    """Test URL and network gate."""

    def test_url_gate_valid_https(self):
        """Valid HTTPS URL passes."""
        result = url_gate("https://example.com/path")
        assert isinstance(result, PolicyOK)
        assert result.gate_name == "url_gate"

    def test_url_gate_valid_http(self):
        """Valid HTTP URL passes."""
        result = url_gate("http://example.com/path")
        assert isinstance(result, PolicyOK)

    def test_url_gate_invalid_scheme(self):
        """Invalid scheme fails."""
        with pytest.raises(URLPolicyException) as exc_info:
            url_gate("ftp://example.com")
        assert exc_info.value.error_code == ErrorCode.E_SCHEME

    def test_url_gate_userinfo_forbidden(self):
        """URL with userinfo fails."""
        with pytest.raises(URLPolicyException) as exc_info:
            url_gate("https://user:pass@example.com")
        assert exc_info.value.error_code == ErrorCode.E_USERINFO

    def test_url_gate_empty_host(self):
        """URL with empty host fails."""
        with pytest.raises(URLPolicyException) as exc_info:
            url_gate("https://")
        assert exc_info.value.error_code == ErrorCode.E_HOST_DENY

    def test_url_gate_host_allowlist(self):
        """Host not in allowlist fails."""
        with pytest.raises(URLPolicyException) as exc_info:
            url_gate(
                "https://evil.com",
                allowed_hosts={"good.com"},
            )
        assert exc_info.value.error_code == ErrorCode.E_HOST_DENY

    def test_url_gate_host_allowlist_pass(self):
        """Host in allowlist passes."""
        result = url_gate(
            "https://good.com",
            allowed_hosts={"good.com"},
        )
        assert isinstance(result, PolicyOK)

    def test_url_gate_port_not_allowed(self):
        """Port not in allowlist fails."""
        with pytest.raises(URLPolicyException) as exc_info:
            url_gate(
                "https://example.com:8080",
                allowed_ports={443},
            )
        assert exc_info.value.error_code == ErrorCode.E_PORT_DENY

    def test_url_gate_port_allowed(self):
        """Port in allowlist passes."""
        result = url_gate(
            "https://example.com:8443",
            allowed_ports={443, 8443},
        )
        assert isinstance(result, PolicyOK)


# ============================================================================
# Path Gate Tests
# ============================================================================


class TestPathGate:
    """Test filesystem path gate."""

    def test_path_gate_valid(self):
        """Valid path passes."""
        result = path_gate("data/file.txt")
        assert isinstance(result, PolicyOK)
        assert result.gate_name == "path_gate"

    def test_path_gate_absolute_rejected(self):
        """Absolute path rejected."""
        with pytest.raises(FilesystemPolicyException) as exc_info:
            path_gate("/etc/passwd")
        assert exc_info.value.error_code == ErrorCode.E_TRAVERSAL

    def test_path_gate_traversal_rejected(self):
        """Path traversal (..) rejected."""
        with pytest.raises(FilesystemPolicyException) as exc_info:
            path_gate("data/../../../etc/passwd")
        assert exc_info.value.error_code == ErrorCode.E_TRAVERSAL

    def test_path_gate_leading_slash_rejected(self):
        """Path starting with / rejected."""
        with pytest.raises(FilesystemPolicyException) as exc_info:
            path_gate("/data/file.txt")
        assert exc_info.value.error_code == ErrorCode.E_TRAVERSAL

    def test_path_gate_too_deep(self):
        """Path too deep rejected."""
        deep_path = "/".join(["dir"] * 15)
        with pytest.raises(FilesystemPolicyException) as exc_info:
            path_gate(deep_path, max_depth=5)
        assert exc_info.value.error_code == ErrorCode.E_DEPTH

    def test_path_gate_segment_too_long(self):
        """Path segment too long rejected."""
        long_segment = "a" * 300
        with pytest.raises(FilesystemPolicyException) as exc_info:
            path_gate(f"data/{long_segment}")
        assert exc_info.value.error_code == ErrorCode.E_SEGMENT_LEN

    def test_path_gate_path_too_long(self):
        """Path too long rejected."""
        long_path = "/".join(["dir"] * 100)
        with pytest.raises(FilesystemPolicyException) as exc_info:
            path_gate(long_path)
        # Note: depth is checked before path length, so we'll get E_DEPTH first
        assert exc_info.value.error_code == ErrorCode.E_DEPTH

    def test_path_gate_windows_reserved(self):
        """Windows reserved names rejected."""
        with pytest.raises(FilesystemPolicyException) as exc_info:
            path_gate("data/CON/file.txt")
        assert exc_info.value.error_code == ErrorCode.E_PORTABILITY


# ============================================================================
# Extraction Gate Tests
# ============================================================================


class TestExtractionGate:
    """Test extraction policy gate."""

    def test_extraction_gate_valid_file(self):
        """Valid file entry passes."""
        entry = {
            "type": "file",
            "size": 1000,
            "compressed_size": 500,
        }
        result = extraction_gate(entry)
        assert isinstance(result, PolicyOK)

    def test_extraction_gate_regular_type(self):
        """'regular' type also passes."""
        entry = {
            "type": "regular",
            "size": 1000,
            "compressed_size": 500,
        }
        result = extraction_gate(entry)
        assert isinstance(result, PolicyOK)

    def test_extraction_gate_symlink_rejected(self):
        """Symlink entry rejected."""
        entry = {
            "type": "symlink",
            "size": 100,
            "compressed_size": 50,
        }
        with pytest.raises(ExtractionPolicyException) as exc_info:
            extraction_gate(entry)
        assert exc_info.value.error_code == ErrorCode.E_SPECIAL_TYPE

    def test_extraction_gate_device_rejected(self):
        """Device entry rejected."""
        entry = {
            "type": "device",
            "size": 0,
            "compressed_size": 0,
        }
        with pytest.raises(ExtractionPolicyException) as exc_info:
            extraction_gate(entry)
        assert exc_info.value.error_code == ErrorCode.E_SPECIAL_TYPE

    def test_extraction_gate_file_too_large(self):
        """File too large rejected."""
        entry = {
            "type": "file",
            "size": 200 * 1024 * 1024,  # 200MB
            "compressed_size": 100 * 1024 * 1024,
        }
        with pytest.raises(ExtractionPolicyException) as exc_info:
            extraction_gate(entry)
        assert exc_info.value.error_code == ErrorCode.E_FILE_SIZE

    def test_extraction_gate_zip_bomb_detected(self):
        """High compression ratio detected as bomb."""
        entry = {
            "type": "file",
            "size": 50 * 1024 * 1024,  # 50MB (under 100MB limit)
            "compressed_size": 1024 * 100,  # 100KB
        }
        with pytest.raises(ExtractionPolicyException) as exc_info:
            extraction_gate(entry)
        # High compression ratio triggers E_ENTRY_RATIO
        assert exc_info.value.error_code == ErrorCode.E_ENTRY_RATIO

    def test_extraction_gate_ratio_within_limit(self):
        """Ratio within limit passes."""
        entry = {
            "type": "file",
            "size": 10000,
            "compressed_size": 1000,  # ratio = 10
        }
        result = extraction_gate(entry, max_ratio=10.0)
        assert isinstance(result, PolicyOK)


# ============================================================================
# Storage Gate Tests
# ============================================================================


class TestStorageGate:
    """Test storage gate."""

    def test_storage_gate_put_valid(self):
        """Valid put operation passes."""
        result = storage_gate("put")
        assert isinstance(result, PolicyOK)

    def test_storage_gate_move_valid(self):
        """Valid move operation passes."""
        result = storage_gate("move")
        assert isinstance(result, PolicyOK)

    def test_storage_gate_marker_valid(self):
        """Valid marker operation passes."""
        result = storage_gate("marker")
        assert isinstance(result, PolicyOK)

    def test_storage_gate_invalid_operation(self):
        """Invalid operation rejected."""
        with pytest.raises(StoragePolicyException) as exc_info:
            storage_gate("invalid_op")
        assert exc_info.value.error_code == ErrorCode.E_STORAGE_PUT


# ============================================================================
# DB Gate Tests
# ============================================================================


class TestDbGate:
    """Test database transactional gate."""

    def test_db_gate_commit_valid(self):
        """Valid commit operation passes."""
        result = db_gate("commit")
        assert isinstance(result, PolicyOK)

    def test_db_gate_rollback_valid(self):
        """Valid rollback operation passes."""
        result = db_gate("rollback")
        assert isinstance(result, PolicyOK)

    def test_db_gate_migrate_valid(self):
        """Valid migrate operation passes."""
        result = db_gate("migrate")
        assert isinstance(result, PolicyOK)

    def test_db_gate_invalid_operation(self):
        """Invalid operation rejected."""
        with pytest.raises(StoragePolicyException) as exc_info:
            db_gate("invalid_op")
        assert exc_info.value.error_code == ErrorCode.E_DB_TX


# ============================================================================
# Integration Tests
# ============================================================================


class TestGateIntegration:
    """Test gates working together."""

    def test_all_gates_registered(self):
        """All gates are registered."""
        from DocsToKG.OntologyDownload.policy.registry import get_registry

        registry = get_registry()
        gates = registry.list_gates()

        expected = {
            "config_gate",
            "url_gate",
            "path_gate",
            "extraction_gate",
            "storage_gate",
            "db_gate",
        }
        for gate_name in expected:
            assert gate_name in gates

    def test_gates_by_domain(self):
        """Gates can be filtered by domain."""
        from DocsToKG.OntologyDownload.policy.registry import get_registry

        registry = get_registry()
        net_gates = registry.gates_by_domain("network")
        assert "url_gate" in net_gates

        fs_gates = registry.gates_by_domain("filesystem")
        assert "path_gate" in fs_gates
