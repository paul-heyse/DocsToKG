"""Tests for policy registry and gate decorator.

Covers:
- Gate registration and discovery
- Gate invocation and statistics
- Decorator API
- Thread safety
- Singleton pattern
"""

import threading
import time

import pytest

from DocsToKG.OntologyDownload.policy.errors import (
    ErrorCode,
    PolicyOK,
    PolicyReject,
)
from DocsToKG.OntologyDownload.policy.registry import (
    GateMetadata,
    PolicyRegistry,
    get_registry,
    policy_gate,
)

# ============================================================================
# Gate Metadata Tests
# ============================================================================


class TestGateMetadata:
    """Test GateMetadata dataclass."""

    def test_gate_metadata_creation(self):
        """GateMetadata can be created."""

        def dummy_gate():
            pass

        metadata = GateMetadata(
            name="test_gate",
            description="Test gate",
            domain="network",
            callable=dummy_gate,
        )
        assert metadata.name == "test_gate"
        assert metadata.description == "Test gate"
        assert metadata.domain == "network"
        assert metadata.callable == dummy_gate

    def test_gate_metadata_frozen(self):
        """GateMetadata is immutable."""

        def dummy_gate():
            pass

        metadata = GateMetadata(
            name="test_gate",
            description="Test gate",
            domain="network",
            callable=dummy_gate,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            metadata.name = "other_gate"


# ============================================================================
# Registry Tests
# ============================================================================


class TestPolicyRegistry:
    """Test PolicyRegistry class."""

    def test_registry_singleton(self):
        """PolicyRegistry is a singleton."""
        registry1 = PolicyRegistry()
        registry2 = PolicyRegistry()
        assert registry1 is registry2

    def test_register_gate_with_decorator(self):
        """Gates can be registered using decorator."""
        registry = PolicyRegistry()

        @registry.register("test_gate", "Test gate", "network")
        def test_gate():
            return PolicyOK(gate_name="test_gate", elapsed_ms=1.0)

        gates = registry.list_gates()
        assert "test_gate" in gates
        assert gates["test_gate"].description == "Test gate"

    def test_cannot_register_duplicate_gate(self):
        """Cannot register gate with same name twice."""
        registry = PolicyRegistry()

        @registry.register("dup_gate", "First", "network")
        def gate1():
            pass

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("dup_gate", "Second", "network")
            def gate2():
                pass

    def test_list_gates(self):
        """list_gates returns all registered gates."""
        registry = PolicyRegistry()

        @registry.register("gate1", "Gate 1", "network")
        def gate1():
            pass

        @registry.register("gate2", "Gate 2", "filesystem")
        def gate2():
            pass

        gates = registry.list_gates()
        assert len(gates) >= 2
        assert "gate1" in gates
        assert "gate2" in gates

    def test_gates_by_domain(self):
        """gates_by_domain filters by domain."""
        registry = PolicyRegistry()

        @registry.register("net_gate", "Network gate", "network")
        def net_gate():
            pass

        @registry.register("fs_gate", "Filesystem gate", "filesystem")
        def fs_gate():
            pass

        net_gates = registry.gates_by_domain("network")
        assert "net_gate" in net_gates
        assert "fs_gate" not in net_gates

    def test_get_gate(self):
        """get_gate retrieves gate callable."""
        registry = PolicyRegistry()

        def my_gate():
            return "result"

        registry.register("my_gate", "My gate", "network")(my_gate)

        gate = registry.get_gate("my_gate")
        assert gate() == "result"

    def test_get_nonexistent_gate_raises(self):
        """get_gate raises KeyError for nonexistent gate."""
        registry = PolicyRegistry()
        with pytest.raises(KeyError):
            registry.get_gate("nonexistent")


# ============================================================================
# Gate Invocation Tests
# ============================================================================


class TestGateInvocation:
    """Test gate invocation and statistics."""

    def test_invoke_gate(self):
        """invoke() calls gate and returns result."""
        registry = PolicyRegistry()

        @registry.register("ok_gate", "OK gate", "network")
        def ok_gate():
            return PolicyOK(gate_name="ok_gate", elapsed_ms=1.0)

        result = registry.invoke("ok_gate")
        assert isinstance(result, PolicyOK)
        assert result.gate_name == "ok_gate"

    def test_invoke_with_arguments(self):
        """invoke() passes arguments to gate."""
        registry = PolicyRegistry()

        @registry.register("arg_gate", "Arg gate", "network")
        def arg_gate(value):
            return PolicyOK(
                gate_name="arg_gate",
                elapsed_ms=1.0,
                details={"value": value},
            )

        result = registry.invoke("arg_gate", 42)
        assert result.details["value"] == 42

    def test_invoke_tracks_statistics(self):
        """invoke() tracks pass/reject statistics."""
        registry = PolicyRegistry()

        @registry.register("stat_gate", "Stat gate", "network")
        def stat_gate(should_pass):
            if should_pass:
                return PolicyOK(gate_name="stat_gate", elapsed_ms=1.0)
            else:
                return PolicyReject(
                    gate_name="stat_gate",
                    error_code=ErrorCode.E_HOST_DENY,
                    elapsed_ms=1.0,
                    details={},
                )

        registry.invoke("stat_gate", True)  # Pass
        registry.invoke("stat_gate", False)  # Reject
        registry.invoke("stat_gate", True)  # Pass

        stats = registry.get_stats("stat_gate")
        assert stats["invocations"] == 3
        assert stats["passes"] == 2
        assert stats["rejects"] == 1
        assert stats["pass_rate"] == pytest.approx(2 / 3)

    def test_invoke_tracks_timing(self):
        """invoke() tracks execution time."""
        registry = PolicyRegistry()

        @registry.register("timing_gate", "Timing gate", "network")
        def timing_gate():
            time.sleep(0.01)  # 10ms
            return PolicyOK(gate_name="timing_gate", elapsed_ms=1.0)

        registry.invoke("timing_gate")
        registry.invoke("timing_gate")  # Invoke twice to have different timing

        stats = registry.get_stats("timing_gate")
        assert stats["invocations"] == 2
        assert stats["total_ms"] > 18  # At least 18ms for two 10ms sleeps
        assert stats["min_ms"] > 0
        assert stats["max_ms"] >= stats["min_ms"]  # max >= min
        assert stats["avg_ms"] > 8  # Average should be around 10ms

    def test_invoke_nonexistent_gate_raises(self):
        """invoke() raises KeyError for nonexistent gate."""
        registry = PolicyRegistry()
        with pytest.raises(KeyError):
            registry.invoke("nonexistent")

    def test_invoke_gate_exception_is_propagated(self):
        """invoke() propagates exceptions from gate."""
        registry = PolicyRegistry()

        @registry.register("error_gate", "Error gate", "network")
        def error_gate():
            raise ValueError("Gate error")

        with pytest.raises(ValueError, match="Gate error"):
            registry.invoke("error_gate")

    def test_invoke_exception_still_tracks_stats(self):
        """invoke() tracks stats even when gate raises."""
        registry = PolicyRegistry()

        @registry.register("exc_gate", "Exception gate", "network")
        def exc_gate():
            raise RuntimeError("Test error")

        try:
            registry.invoke("exc_gate")
        except RuntimeError:
            pass

        stats = registry.get_stats("exc_gate")
        assert stats["invocations"] == 1


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Test statistics tracking."""

    def test_get_stats_basic(self):
        """get_stats returns stat dict."""
        registry = PolicyRegistry()

        @registry.register("basic_gate", "Basic gate", "network")
        def basic_gate():
            return PolicyOK(gate_name="basic_gate", elapsed_ms=1.0)

        registry.invoke("basic_gate")

        stats = registry.get_stats("basic_gate")
        assert "invocations" in stats
        assert "passes" in stats
        assert "rejects" in stats
        assert "total_ms" in stats
        assert "avg_ms" in stats
        assert "pass_rate" in stats

    def test_reset_stats_single_gate(self):
        """reset_stats clears stats for a single gate."""
        registry = PolicyRegistry()

        @registry.register("reset_gate", "Reset gate", "network")
        def reset_gate():
            return PolicyOK(gate_name="reset_gate", elapsed_ms=1.0)

        registry.invoke("reset_gate")
        registry.invoke("reset_gate")
        registry.reset_stats("reset_gate")

        stats = registry.get_stats("reset_gate")
        assert stats["invocations"] == 0
        assert stats["passes"] == 0

    def test_reset_stats_all_gates(self):
        """reset_stats clears stats for all gates."""
        registry = PolicyRegistry()

        @registry.register("reset1", "Reset 1", "network")
        def reset1():
            return PolicyOK(gate_name="reset1", elapsed_ms=1.0)

        @registry.register("reset2", "Reset 2", "filesystem")
        def reset2():
            return PolicyOK(gate_name="reset2", elapsed_ms=1.0)

        registry.invoke("reset1")
        registry.invoke("reset2")
        registry.reset_stats()

        assert registry.get_stats("reset1")["invocations"] == 0
        assert registry.get_stats("reset2")["invocations"] == 0


# ============================================================================
# Decorator API Tests
# ============================================================================


class TestDecoratorAPI:
    """Test policy_gate decorator API."""

    def test_policy_gate_decorator(self):
        """@policy_gate decorator registers gates."""

        @policy_gate("decor_gate", "Decorator gate", "network")
        def decor_gate():
            return PolicyOK(gate_name="decor_gate", elapsed_ms=1.0)

        registry = get_registry()
        gates = registry.list_gates()
        assert "decor_gate" in gates

    def test_get_registry_singleton(self):
        """get_registry returns singleton registry."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_registration(self):
        """Multiple threads can register gates safely."""
        registry = PolicyRegistry()
        errors = []

        def register_gate(name):
            try:

                @registry.register(name, f"Gate {name}", "network")
                def gate():
                    return PolicyOK(gate_name=name, elapsed_ms=1.0)

            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_gate, args=(f"gate_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_invocation(self):
        """Multiple threads can invoke gates safely."""
        registry = PolicyRegistry()
        results = []

        @registry.register("concurrent_gate", "Concurrent gate", "network")
        def concurrent_gate():
            return PolicyOK(gate_name="concurrent_gate", elapsed_ms=1.0)

        def invoke_gate():
            result = registry.invoke("concurrent_gate")
            results.append(result)

        threads = [threading.Thread(target=invoke_gate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        stats = registry.get_stats("concurrent_gate")
        assert stats["invocations"] == 10
