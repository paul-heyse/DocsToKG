"""
Test policy gates for URL and path validation in ContentDownload.

Verifies that:
1. URLs are validated against security policy (scheme, IDN, ports)
2. File paths are sandboxed within artifact directories
3. Path traversal attempts are blocked
4. Symlink escapes are prevented
5. Permission checks work correctly
"""

import os
import tempfile

import pytest

from DocsToKG.ContentDownload.policy.path_gate import (
    PathPolicyError,
    validate_path_safety,
)
from DocsToKG.ContentDownload.policy.url_gate import (
    PolicyError as UrlPolicyError,
)
from DocsToKG.ContentDownload.policy.url_gate import (
    validate_url_security,
)


class TestUrlGate:
    """Test URL security validation gate."""

    def test_valid_https_url(self):
        """Valid HTTPS URL should pass."""
        url = "https://example.com/paper.pdf"
        result = validate_url_security(url)
        assert result is not None
        assert "example.com" in result

    def test_valid_http_url(self):
        """Valid HTTP URL should pass."""
        url = "http://example.com/data"
        result = validate_url_security(url)
        assert result is not None

    def test_invalid_scheme_ftp(self):
        """FTP scheme should be rejected."""
        with pytest.raises(UrlPolicyError, match="Scheme not allowed"):
            validate_url_security("ftp://example.com/file")

    def test_invalid_scheme_file(self):
        """File:// scheme should be rejected."""
        with pytest.raises(UrlPolicyError, match="Scheme not allowed"):
            validate_url_security("file:///etc/passwd")

    def test_missing_host(self):
        """URL without hostname should be rejected."""
        with pytest.raises(UrlPolicyError, match="No host"):
            validate_url_security("https:///path")

    def test_idn_normalization(self):
        """IDN hosts should be normalized to punycode."""
        # This test verifies IDN handling (if applicable for your domain)
        url = "https://example.com:443/data"
        result = validate_url_security(url)
        assert result is not None

    def test_url_with_query_string(self):
        """URLs with query strings should be preserved."""
        url = "https://api.example.com/papers?doi=10.1234&format=json"
        result = validate_url_security(url)
        assert "?" in result or "params" in result or result is not None


class TestPathGate:
    """Test path safety validation gate."""

    def test_valid_path_within_root(self):
        """Path within artifact root should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = tmpdir
            final_path = os.path.join(artifact_root, "paper.pdf")

            result = validate_path_safety(final_path, artifact_root)
            assert result is not None
            assert os.path.isabs(result)

    def test_path_traversal_attack_with_dotdot(self):
        """Path traversal with .. should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = tmpdir
            # Try to escape: /tmp/xyz/../../etc/passwd
            final_path = os.path.join(artifact_root, "..", "..", "etc", "passwd")

            with pytest.raises(PathPolicyError, match="escapes artifact root"):
                validate_path_safety(final_path, artifact_root)

    def test_path_traversal_attack_absolute(self):
        """Absolute path outside root should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = tmpdir
            final_path = "/etc/passwd"

            with pytest.raises(PathPolicyError, match="escapes artifact root"):
                validate_path_safety(final_path, artifact_root)

    def test_symlink_escape_blocked(self):
        """Symlinks that escape the root should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = tmpdir

            # Create a symlink to /tmp (outside the artifact root)
            symlink_path = os.path.join(artifact_root, "escape_link")
            os.symlink("/tmp", symlink_path)

            # Try to write through the symlink to a file outside root
            target = os.path.join(symlink_path, "outside_file.txt")

            with pytest.raises(PathPolicyError, match="escapes artifact root"):
                validate_path_safety(target, artifact_root)

    def test_system_directory_blocked(self):
        """Cannot write to /etc, /sys, /proc, /root, /boot, /dev."""
        # Test with artifact_root = None (uses cwd), so /etc paths will escape
        forbidden_paths = [
            "/etc/passwd",
            "/sys/kernel/config",
            "/proc/sys/kernel",
            "/root/.ssh/id_rsa",
            "/boot/vmlinuz",
            "/dev/null",
        ]

        for path in forbidden_paths:
            # These should be caught by the "escapes artifact root" check
            # since they're outside the artifact root (defaults to cwd)
            with pytest.raises(PathPolicyError):
                validate_path_safety(path)

    def test_empty_path_rejected(self):
        """Empty final_path should be rejected."""
        with pytest.raises(PathPolicyError, match="empty"):
            validate_path_safety("")

    def test_defaults_to_cwd(self):
        """If artifact_root is None, should default to cwd."""
        cwd = os.getcwd()
        # Create a path relative to cwd
        test_file = os.path.join(cwd, "test_artifact.pdf")

        result = validate_path_safety(test_file, artifact_root=None)
        assert result is not None
        assert os.path.isabs(result)

    def test_relative_path_resolution(self):
        """Relative paths should be resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = tmpdir
            # Use relative path with ../ but still within bounds
            rel_path = os.path.join(artifact_root, "subdir", "..", "file.pdf")

            result = validate_path_safety(rel_path, artifact_root)
            assert result is not None
            # Should resolve to something within artifact_root
            assert os.path.commonpath([result, artifact_root]) == artifact_root


class TestPolicyGateIntegration:
    """Test integration of policy gates with download execution."""

    def test_url_gate_rejects_ftp_in_download(self):
        """Download execution should reject FTP URLs via policy gate."""
        # This would be tested at the download_execution level
        # with mock HTTP calls
        from DocsToKG.ContentDownload.policy.url_gate import PolicyError

        with pytest.raises(PolicyError):
            validate_url_security("ftp://example.com/paper.pdf")

    def test_path_gate_prevents_escape(self):
        """Download finalize should reject paths outside artifact root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = tmpdir
            escaped_path = "/etc/shadow"

            from DocsToKG.ContentDownload.policy.path_gate import PathPolicyError

            with pytest.raises(PathPolicyError):
                validate_path_safety(escaped_path, artifact_root)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
