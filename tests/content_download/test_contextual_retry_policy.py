"""Unit tests for contextual retry policies.

Tests cover:
- OperationType enum values
- 429 predicate logic (all operation types)
- Timeout predicate logic (all operation types)
- Policy creation and configuration
- Integration with Tenacity
"""

from __future__ import annotations

import httpx
import pytest
from tenacity import RetryError

from DocsToKG.ContentDownload.errors.tenacity_policies import (
    OperationType,
    create_contextual_retry_policy,
)


class Test429Predicate:
    """Test 429 handling per operation type."""

    def test_download_retries_on_429(self):
        """DOWNLOAD operation should retry on 429."""
        policy = create_contextual_retry_policy(
            operation=OperationType.DOWNLOAD,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        last_error = None
        try:
            for attempt in policy:
                with attempt:
                    attempt_count += 1
                    if attempt_count < 3:
                        # Create a mock response with 429 status
                        response = httpx.Response(429, request=httpx.Request("GET", "http://example.com"))
                        # Raise HTTPStatusError which carries the response
                        raise httpx.HTTPStatusError("429", request=response.request, response=response)
                    return "success"
        except httpx.HTTPStatusError as e:
            last_error = e
        
        # Should have retried multiple times (last_error means we got to max attempts)
        assert attempt_count >= 2, f"Expected at least 2 attempts, got {attempt_count}"

    def test_validate_defers_on_429(self):
        """VALIDATE operation should NOT retry on 429 (signal deferral)."""
        policy = create_contextual_retry_policy(
            operation=OperationType.VALIDATE,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        with pytest.raises(httpx.HTTPStatusError):
            for attempt in policy:
                with attempt:
                    attempt_count += 1
                    response = httpx.Response(429, request=httpx.Request("GET", "http://example.com"))
                    raise httpx.HTTPStatusError("429", request=response.request, response=response)
        
        # Should fail on first attempt (no retry on 429)
        assert attempt_count == 1

    def test_resolve_failovers_on_429(self):
        """RESOLVE operation should NOT retry on 429 (signal failover)."""
        policy = create_contextual_retry_policy(
            operation=OperationType.RESOLVE,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        with pytest.raises(httpx.HTTPStatusError):
            for attempt in policy:
                with attempt:
                    attempt_count += 1
                    response = httpx.Response(429, request=httpx.Request("GET", "http://example.com"))
                    raise httpx.HTTPStatusError("429", request=response.request, response=response)
        
        # Should fail on first attempt (no retry on 429)
        assert attempt_count == 1

    def test_manifest_fetch_defers_on_429(self):
        """MANIFEST_FETCH operation should NOT retry on 429 (signal deferral)."""
        policy = create_contextual_retry_policy(
            operation=OperationType.MANIFEST_FETCH,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        with pytest.raises(httpx.HTTPStatusError):
            for attempt in policy:
                with attempt:
                    attempt_count += 1
                    response = httpx.Response(429, request=httpx.Request("GET", "http://example.com"))
                    raise httpx.HTTPStatusError("429", request=response.request, response=response)
        
        # Should fail on first attempt (no retry on 429)
        assert attempt_count == 1


class TestTimeoutPredicate:
    """Test timeout handling per operation type."""

    def test_download_retries_on_timeout(self):
        """DOWNLOAD operation should retry on timeout."""
        policy = create_contextual_retry_policy(
            operation=OperationType.DOWNLOAD,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        last_error = None
        try:
            for attempt in policy:
                with attempt:
                    attempt_count += 1
                    if attempt_count < 3:
                        raise httpx.ConnectTimeout("timeout")
                    return "success"
        except httpx.ConnectTimeout as e:
            last_error = e
        
        # Should have retried (last_error means we exhausted retries)
        assert attempt_count >= 2

    def test_validate_defers_on_timeout(self):
        """VALIDATE operation should NOT retry on timeout (signal deferral)."""
        policy = create_contextual_retry_policy(
            operation=OperationType.VALIDATE,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        with pytest.raises(httpx.ConnectTimeout):
            for attempt in policy:
                with attempt:
                    attempt_count += 1
                    raise httpx.ConnectTimeout("timeout")
        
        # Should fail on first attempt (no retry on timeout)
        assert attempt_count == 1, f"Expected 1 attempt, got {attempt_count}"

    def test_resolve_retries_on_timeout(self):
        """RESOLVE operation should retry on timeout (then failover handled by caller)."""
        policy = create_contextual_retry_policy(
            operation=OperationType.RESOLVE,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        last_error = None
        try:
            for attempt in policy:
                with attempt:
                    attempt_count += 1
                    if attempt_count < 3:
                        raise httpx.ConnectTimeout("timeout")
                    return "success"
        except httpx.ConnectTimeout as e:
            last_error = e
        
        assert attempt_count >= 2

    def test_extract_retries_on_timeout(self):
        """EXTRACT operation should retry on timeout."""
        policy = create_contextual_retry_policy(
            operation=OperationType.EXTRACT,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        last_error = None
        try:
            for attempt in policy:
                with attempt:
                    attempt_count += 1
                    if attempt_count < 3:
                        raise httpx.ReadTimeout("timeout")
                    return "success"
        except httpx.ReadTimeout as e:
            last_error = e
        
        assert attempt_count >= 2


class TestPolicyCreation:
    """Test policy creation and configuration."""

    def test_policy_defaults_to_download(self):
        """Policy should default to DOWNLOAD operation."""
        policy = create_contextual_retry_policy()
        assert policy is not None

    def test_policy_accepts_all_operation_types(self):
        """Policy should accept all OperationType values."""
        for op_type in OperationType:
            policy = create_contextual_retry_policy(operation=op_type)
            assert policy is not None

    def test_policy_respects_max_attempts(self):
        """Policy should respect max_attempts limit."""
        policy = create_contextual_retry_policy(
            operation=OperationType.DOWNLOAD,
            max_attempts=2,
            max_delay_seconds=10,
        )
        
        attempt_count = 0
        last_error = None
        try:
            for attempt in policy:
                with attempt:
                    attempt_count += 1
                    raise httpx.ConnectError("error")
        except httpx.ConnectError as e:
            last_error = e
        
        # Should exhaust max_attempts
        assert attempt_count == 2

    def test_policy_retries_on_5xx_for_all_operations(self):
        """Policy should retry on 5xx for all operation types."""
        for op_type in OperationType:
            policy = create_contextual_retry_policy(
                operation=op_type,
                max_attempts=3,
                max_delay_seconds=1,
            )
            
            attempt_count = 0
            last_error = None
            try:
                for attempt in policy:
                    with attempt:
                        attempt_count += 1
                        if attempt_count < 3:
                            response = httpx.Response(500, request=httpx.Request("GET", "http://example.com"))
                            raise httpx.HTTPStatusError("500", request=response.request, response=response)
                        return "success"
            except httpx.HTTPStatusError as e:
                last_error = e
            
            assert attempt_count >= 2, f"Failed for operation {op_type.name}: only {attempt_count} attempts"


class TestRetryableExceptions:
    """Test retry behavior on various exceptions."""

    def test_connect_error_retried(self):
        """ConnectError should be retried."""
        policy = create_contextual_retry_policy(
            operation=OperationType.DOWNLOAD,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        for attempt in policy:
            with attempt:
                attempt_count += 1
                if attempt_count < 3:
                    raise httpx.ConnectError("connection failed")
                return "success"
        
        assert attempt_count == 3

    def test_read_timeout_retried_for_download(self):
        """ReadTimeout should be retried for DOWNLOAD."""
        policy = create_contextual_retry_policy(
            operation=OperationType.DOWNLOAD,
            max_attempts=3,
            max_delay_seconds=1,
        )
        
        attempt_count = 0
        for attempt in policy:
            with attempt:
                attempt_count += 1
                if attempt_count < 3:
                    raise httpx.ReadTimeout("read timeout")
                return "success"
        
        assert attempt_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
