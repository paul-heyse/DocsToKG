"""Central registry of policy gates and decorator for gate registration.

All gates register with this registry:
- Enables central discovery
- Tracks gate metadata (name, description, domain)
- Provides decorator for easy registration
- Singleton pattern for thread-safe access
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from DocsToKG.OntologyDownload.policy.errors import PolicyResult

# ============================================================================
# Gate Registry
# ============================================================================


@dataclass(frozen=True)
class GateMetadata:
    """Metadata about a registered gate."""

    name: str  # e.g., "url_gate", "path_gate"
    description: str  # Human-readable description
    domain: str  # Domain: "network", "filesystem", "extraction", "storage", "db", "config"
    callable: Callable  # The actual gate function


class PolicyRegistry:
    """Central registry for all policy gates.

    Thread-safe singleton that manages:
    - Gate registration
    - Gate discovery
    - Gate invocation with timing
    - Gate statistics
    """

    _instance: Optional["PolicyRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PolicyRegistry":
        """Singleton constructor (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize registry (only once)."""
        if not hasattr(self, "_gates"):
            self._gates: Dict[str, GateMetadata] = {}
            self._stats: Dict[str, Dict[str, Any]] = {}
            self._registry_lock = threading.Lock()

    def register(
        self,
        name: str,
        description: str,
        domain: str,
    ) -> Callable:
        """Decorator to register a gate function.

        Args:
            name: Gate name (e.g., "url_gate")
            description: Human-readable description
            domain: Gate domain (network, filesystem, extraction, storage, db, config)

        Returns:
            Decorator function

        Example:
            @policy_registry.register(
                name="url_gate",
                description="Validate URLs against security policy",
                domain="network"
            )
            def url_gate(url: str) -> PolicyResult:
                ...
        """

        def decorator(gate_func: Callable) -> Callable:
            with self._registry_lock:
                if name in self._gates:
                    raise ValueError(f"Gate '{name}' already registered")

                metadata = GateMetadata(
                    name=name,
                    description=description,
                    domain=domain,
                    callable=gate_func,
                )
                self._gates[name] = metadata
                self._stats[name] = {
                    "invocations": 0,
                    "passes": 0,
                    "rejects": 0,
                    "total_ms": 0.0,
                    "min_ms": float("inf"),
                    "max_ms": 0.0,
                }

            return gate_func

        return decorator

    def get_gate(self, name: str) -> Callable:
        """Get a registered gate by name.

        Args:
            name: Gate name

        Returns:
            The gate callable

        Raises:
            KeyError: If gate not found
        """
        if name not in self._gates:
            available = ", ".join(self._gates.keys())
            raise KeyError(f"Gate '{name}' not found. Available: {available}")
        return self._gates[name].callable

    def list_gates(self) -> Dict[str, GateMetadata]:
        """Get all registered gates.

        Returns:
            Dict mapping gate names to metadata
        """
        with self._registry_lock:
            return dict(self._gates)

    def gates_by_domain(self, domain: str) -> Dict[str, GateMetadata]:
        """Get all gates in a specific domain.

        Args:
            domain: Gate domain (network, filesystem, extraction, storage, db, config)

        Returns:
            Dict of gates in that domain
        """
        with self._registry_lock:
            return {
                name: metadata
                for name, metadata in self._gates.items()
                if metadata.domain == domain
            }

    def invoke(self, gate_name: str, *args, **kwargs) -> PolicyResult:
        """Invoke a gate and track statistics.

        Args:
            gate_name: Name of gate to invoke
            *args: Positional args for gate
            **kwargs: Keyword args for gate

        Returns:
            PolicyResult (OK or Reject)

        Raises:
            KeyError: If gate not found
            Exception: Any exception from gate callable
        """
        gate_callable = self.get_gate(gate_name)
        start_ms = time.perf_counter() * 1000  # Convert to ms

        try:
            result = gate_callable(*args, **kwargs)
            elapsed_ms = time.perf_counter() * 1000 - start_ms

            with self._registry_lock:
                stats = self._stats[gate_name]
                stats["invocations"] += 1
                stats["total_ms"] += elapsed_ms
                stats["min_ms"] = min(stats["min_ms"], elapsed_ms)
                stats["max_ms"] = max(stats["max_ms"], elapsed_ms)

                # Track pass/reject
                from DocsToKG.OntologyDownload.policy.errors import PolicyOK

                if isinstance(result, PolicyOK):
                    stats["passes"] += 1
                else:
                    stats["rejects"] += 1

            return result

        except Exception:
            elapsed_ms = time.perf_counter() * 1000 - start_ms
            with self._registry_lock:
                stats = self._stats[gate_name]
                stats["invocations"] += 1
                stats["total_ms"] += elapsed_ms
            raise

    def get_stats(self, gate_name: str) -> Dict[str, Any]:
        """Get statistics for a gate.

        Args:
            gate_name: Name of gate

        Returns:
            Stats dict with invocations, passes, rejects, timing

        Raises:
            KeyError: If gate not found
        """
        if gate_name not in self._gates:
            raise KeyError(f"Gate '{gate_name}' not found")

        with self._registry_lock:
            stats = dict(self._stats[gate_name])

        # Calculate derived stats
        if stats["invocations"] > 0:
            stats["pass_rate"] = stats["passes"] / stats["invocations"]
            stats["reject_rate"] = stats["rejects"] / stats["invocations"]
            stats["avg_ms"] = stats["total_ms"] / stats["invocations"]
        else:
            stats["pass_rate"] = 0.0
            stats["reject_rate"] = 0.0
            stats["avg_ms"] = 0.0

        return stats

    def reset_stats(self, gate_name: Optional[str] = None) -> None:
        """Reset statistics for a gate or all gates.

        Args:
            gate_name: Gate to reset, or None for all
        """
        with self._registry_lock:
            if gate_name:
                if gate_name not in self._gates:
                    raise KeyError(f"Gate '{gate_name}' not found")
                self._stats[gate_name] = {
                    "invocations": 0,
                    "passes": 0,
                    "rejects": 0,
                    "total_ms": 0.0,
                    "min_ms": float("inf"),
                    "max_ms": 0.0,
                }
            else:
                for gate_name in self._gates:
                    self._stats[gate_name] = {
                        "invocations": 0,
                        "passes": 0,
                        "rejects": 0,
                        "total_ms": 0.0,
                        "min_ms": float("inf"),
                        "max_ms": 0.0,
                    }


# ============================================================================
# Singleton API
# ============================================================================


# Global registry instance
_registry: Optional[PolicyRegistry] = None


def get_registry() -> PolicyRegistry:
    """Get or create the global policy registry (singleton).

    Returns:
        PolicyRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PolicyRegistry()
    return _registry


def policy_gate(
    name: str,
    description: str,
    domain: str,
) -> Callable:
    """Decorator to register a policy gate with the global registry.

    Args:
        name: Gate name (e.g., "url_gate")
        description: Human-readable description
        domain: Gate domain (network, filesystem, extraction, storage, db, config)

    Returns:
        Decorator function

    Example:
        @policy_gate(
            name="url_gate",
            description="Validate URLs against security policy",
            domain="network"
        )
        def url_gate(url: str) -> PolicyResult:
            ...
    """
    registry = get_registry()
    return registry.register(name, description, domain)


__all__ = [
    "GateMetadata",
    "PolicyRegistry",
    "get_registry",
    "policy_gate",
]
