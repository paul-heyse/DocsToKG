"""Tests for policy error codes and contracts.

Covers:
- Error code enums
- Result types (PolicyOK, PolicyReject)
- Policy exceptions
- Error scrubbing
"""

import pytest

from DocsToKG.OntologyDownload.policy.errors import (
    ConfigurationPolicyException,
    ErrorCode,
    ExtractionPolicyException,
    FilesystemPolicyException,
    PolicyException,
    PolicyOK,
    PolicyReject,
    StoragePolicyException,
    URLPolicyException,
    raise_policy_error,
)

# ============================================================================
# Error Code Tests
# ============================================================================


class TestErrorCode:
    """Test ErrorCode enum."""

    def test_error_codes_exist(self):
        """All expected error codes are defined."""
        codes = {
            "E_NET_CONNECT",
            "E_NET_READ",
            "E_SCHEME",
            "E_HOST_DENY",
            "E_TRAVERSAL",
            "E_BOMB_RATIO",
            "E_STORAGE_PUT",
            "E_DB_TX",
        }
        for code_name in codes:
            assert hasattr(ErrorCode, code_name)

    def test_error_code_values(self):
        """Error codes have correct string values."""
        assert ErrorCode.E_HOST_DENY.value == "E_HOST_DENY"
        assert ErrorCode.E_TRAVERSAL.value == "E_TRAVERSAL"
        assert ErrorCode.E_DB_TX.value == "E_DB_TX"

    def test_error_code_is_string_enum(self):
        """ErrorCode is a string enum."""
        assert isinstance(ErrorCode.E_HOST_DENY, str)
        assert ErrorCode.E_HOST_DENY == "E_HOST_DENY"


# ============================================================================
# Result Type Tests
# ============================================================================


class TestPolicyOK:
    """Test PolicyOK result type."""

    def test_policy_ok_creation(self):
        """PolicyOK can be created."""
        result = PolicyOK(gate_name="url_gate", elapsed_ms=12.5)
        assert result.gate_name == "url_gate"
        assert result.elapsed_ms == 12.5
        assert result.details is None

    def test_policy_ok_with_details(self):
        """PolicyOK accepts optional details."""
        details = {"host": "example.com", "port": 443}
        result = PolicyOK(gate_name="url_gate", elapsed_ms=12.5, details=details)
        assert result.details == details

    def test_policy_ok_frozen(self):
        """PolicyOK is immutable."""
        result = PolicyOK(gate_name="url_gate", elapsed_ms=12.5)
        with pytest.raises(Exception):  # FrozenInstanceError
            result.gate_name = "other_gate"


class TestPolicyReject:
    """Test PolicyReject result type."""

    def test_policy_reject_creation(self):
        """PolicyReject can be created."""
        result = PolicyReject(
            gate_name="url_gate",
            error_code=ErrorCode.E_HOST_DENY,
            elapsed_ms=5.2,
            details={"host": "evil.com"},
        )
        assert result.gate_name == "url_gate"
        assert result.error_code == ErrorCode.E_HOST_DENY
        assert result.elapsed_ms == 5.2
        assert result.details == {"host": "evil.com"}

    def test_policy_reject_frozen(self):
        """PolicyReject is immutable."""
        result = PolicyReject(
            gate_name="url_gate",
            error_code=ErrorCode.E_HOST_DENY,
            elapsed_ms=5.2,
            details={"host": "evil.com"},
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.gate_name = "other_gate"


# ============================================================================
# Exception Tests
# ============================================================================


class TestPolicyExceptions:
    """Test exception types."""

    def test_policy_exception_creation(self):
        """PolicyException can be created."""
        exc = PolicyException(
            ErrorCode.E_HOST_DENY,
            "Host not allowed",
            {"host": "evil.com"},
        )
        assert exc.error_code == ErrorCode.E_HOST_DENY
        assert exc.message == "Host not allowed"
        assert exc.details == {"host": "evil.com"}

    def test_policy_exception_message(self):
        """PolicyException formats message correctly."""
        exc = PolicyException(ErrorCode.E_HOST_DENY, "Host not allowed")
        assert "E_HOST_DENY" in str(exc)
        assert "Host not allowed" in str(exc)

    def test_url_policy_exception(self):
        """URLPolicyException is a subclass."""
        exc = URLPolicyException(ErrorCode.E_SCHEME, "Invalid scheme")
        assert isinstance(exc, PolicyException)
        assert exc.error_code == ErrorCode.E_SCHEME

    def test_filesystem_policy_exception(self):
        """FilesystemPolicyException is a subclass."""
        exc = FilesystemPolicyException(ErrorCode.E_TRAVERSAL, "Path traversal")
        assert isinstance(exc, PolicyException)
        assert exc.error_code == ErrorCode.E_TRAVERSAL

    def test_extraction_policy_exception(self):
        """ExtractionPolicyException is a subclass."""
        exc = ExtractionPolicyException(ErrorCode.E_BOMB_RATIO, "Zip bomb detected")
        assert isinstance(exc, PolicyException)

    def test_storage_policy_exception(self):
        """StoragePolicyException is a subclass."""
        exc = StoragePolicyException(ErrorCode.E_STORAGE_PUT, "Write failed")
        assert isinstance(exc, PolicyException)

    def test_configuration_policy_exception(self):
        """ConfigurationPolicyException is a subclass."""
        exc = ConfigurationPolicyException(ErrorCode.E_CONFIG_INVALID, "Invalid config")
        assert isinstance(exc, PolicyException)


# ============================================================================
# Error Scrubbing Tests
# ============================================================================


class TestErrorScrubbing:
    """Test sensitive data scrubbing in error details."""

    def test_scrub_removes_password(self):
        """Scrubbing removes password field."""
        with pytest.raises(URLPolicyException) as exc_info:
            raise_policy_error(
                ErrorCode.E_HOST_DENY,
                "Host denied",
                {"host": "evil.com", "password": "secret123"},
                URLPolicyException,
            )
        assert "password" not in exc_info.value.details

    def test_scrub_removes_token(self):
        """Scrubbing removes token field."""
        with pytest.raises(URLPolicyException) as exc_info:
            raise_policy_error(
                ErrorCode.E_HOST_DENY,
                "Host denied",
                {"host": "evil.com", "api_token": "abc123"},
                URLPolicyException,
            )
        assert "api_token" not in exc_info.value.details

    def test_scrub_removes_secret(self):
        """Scrubbing removes secret field."""
        with pytest.raises(URLPolicyException) as exc_info:
            raise_policy_error(
                ErrorCode.E_HOST_DENY,
                "Host denied",
                {"host": "evil.com", "secret_key": "xyz"},
                URLPolicyException,
            )
        assert "secret_key" not in exc_info.value.details

    def test_scrub_redacts_url(self):
        """Scrubbing redacts URLs."""
        with pytest.raises(URLPolicyException) as exc_info:
            raise_policy_error(
                ErrorCode.E_HOST_DENY,
                "Host denied",
                {"url": "https://evil.com/path?api_key=secret"},
                URLPolicyException,
            )
        assert exc_info.value.details["url"] == "[REDACTED_URL]"

    def test_scrub_redacts_path(self):
        """Scrubbing redacts full paths."""
        with pytest.raises(FilesystemPolicyException) as exc_info:
            raise_policy_error(
                ErrorCode.E_TRAVERSAL,
                "Path traversal",
                {"path": "/home/user/secret/file.txt"},
                FilesystemPolicyException,
            )
        # Should only show basename
        assert "secret" not in exc_info.value.details["path"]
        assert "file.txt" in exc_info.value.details["path"]

    def test_scrub_keeps_safe_fields(self):
        """Scrubbing keeps non-sensitive fields."""
        with pytest.raises(URLPolicyException) as exc_info:
            raise_policy_error(
                ErrorCode.E_HOST_DENY,
                "Host denied",
                {"host": "evil.com", "reason": "private network"},
                URLPolicyException,
            )
        assert exc_info.value.details["host"] == "evil.com"
        assert exc_info.value.details["reason"] == "private network"


# ============================================================================
# Error Raising Tests
# ============================================================================


class TestRaisePolicyError:
    """Test raise_policy_error helper."""

    def test_raise_policy_error_with_default_exception(self):
        """raise_policy_error uses PolicyException by default."""
        with pytest.raises(PolicyException) as exc_info:
            raise_policy_error(
                ErrorCode.E_HOST_DENY,
                "Host not allowed",
                {"host": "evil.com"},
            )
        exc = exc_info.value
        assert exc.error_code == ErrorCode.E_HOST_DENY
        assert exc.message == "Host not allowed"

    def test_raise_policy_error_with_custom_exception(self):
        """raise_policy_error uses custom exception class."""
        with pytest.raises(URLPolicyException) as exc_info:
            raise_policy_error(
                ErrorCode.E_HOST_DENY,
                "Host not allowed",
                {"host": "evil.com"},
                URLPolicyException,
            )
        assert isinstance(exc_info.value, URLPolicyException)

    def test_raise_policy_error_without_details(self):
        """raise_policy_error works without details."""
        with pytest.raises(PolicyException) as exc_info:
            raise_policy_error(ErrorCode.E_HOST_DENY, "Host not allowed")
        exc = exc_info.value
        assert exc.details == {}

    def test_raise_policy_error_preserves_message(self):
        """raise_policy_error preserves message."""
        msg = "This is a detailed error message with context"
        with pytest.raises(PolicyException) as exc_info:
            raise_policy_error(ErrorCode.E_TRAVERSAL, msg)
        assert exc_info.value.message == msg


# ============================================================================
# Integration Tests
# ============================================================================


class TestErrorCodeComplete:
    """Test completeness of error catalog."""

    def test_all_error_codes_documented(self):
        """All error codes have comments."""
        # This is a manual check - verify in the source
        assert len(ErrorCode) >= 30  # Sanity check: at least 30 codes

    def test_error_codes_no_duplicates(self):
        """No duplicate error codes."""
        values = [e.value for e in ErrorCode]
        assert len(values) == len(set(values))


class TestExceptionHierarchy:
    """Test exception inheritance."""

    def test_all_exceptions_inherit_from_policy_exception(self):
        """All specific exceptions inherit from PolicyException."""
        exceptions = [
            URLPolicyException,
            FilesystemPolicyException,
            ExtractionPolicyException,
            StoragePolicyException,
            ConfigurationPolicyException,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, PolicyException)

    def test_policy_exception_inherits_from_exception(self):
        """PolicyException inherits from Exception."""
        assert issubclass(PolicyException, Exception)
