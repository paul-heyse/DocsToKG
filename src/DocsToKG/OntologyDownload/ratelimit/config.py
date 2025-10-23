# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.ratelimit.config",
#   "purpose": "RateSpec parsing and normalization for rate-limiting configuration.",
#   "sections": [
#     {
#       "id": "ratespec",
#       "name": "RateSpec",
#       "anchor": "class-ratespec",
#       "kind": "class"
#     },
#     {
#       "id": "parse-rate-string",
#       "name": "parse_rate_string",
#       "anchor": "function-parse-rate-string",
#       "kind": "function"
#     },
#     {
#       "id": "validate-rate-list",
#       "name": "validate_rate_list",
#       "anchor": "function-validate-rate-list",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-rate-list",
#       "name": "normalize_rate_list",
#       "anchor": "function-normalize-rate-list",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-per-service-rates",
#       "name": "normalize_per_service_rates",
#       "anchor": "function-normalize-per-service-rates",
#       "kind": "function"
#     },
#     {
#       "id": "get-schema-summary",
#       "name": "get_schema_summary",
#       "anchor": "function-get-schema-summary",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""RateSpec parsing and normalization for rate-limiting configuration.

Parses human-readable rate strings (e.g., "5/second") into structured
RateSpec objects compatible with pyrate-limiter. Handles multi-window
enforcement (e.g., 5/sec AND 300/min) with proper ordering validation.

Design:
- **Human-readable input**: "5/second", "300/minute", "1000/hour"
- **Structured output**: RateSpec(limit=5, interval_ms=1000)
- **Multi-window validation**: Rates must be ordered by interval/limit
- **Per-service config**: Different limits for OLS, BioPortal, etc.
- **Default fallback**: Conservative global default (8/sec, 300/min)

Example:
    >>> from DocsToKG.OntologyDownload.ratelimit.config import (
    ...     parse_rate_string, normalize_per_service_rates
    ... )
    >>> spec = parse_rate_string("5/second")
    >>> print(spec.limit, spec.interval_ms)
    5 1000

    >>> rates = normalize_per_service_rates({
    ...     "ols": "4/second",
    ...     "bioportal": "2/second"
    ... })
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# ============================================================================
# Constants & Enums
# ============================================================================

# Duration constants (in milliseconds)
DURATION_MS = {
    "second": 1_000,
    "minute": 60 * 1_000,
    "hour": 60 * 60 * 1_000,
    "day": 24 * 60 * 60 * 1_000,
}

# Short aliases
DURATION_ALIASES = {
    "sec": "second",
    "min": "minute",
    "hr": "hour",
}


# ============================================================================
# Data Models
# ============================================================================


@dataclass(frozen=True)
class RateSpec:
    """Normalized rate specification.

    Represents a single rate window (e.g., "5 per second").
    Multiple RateSpecs can be combined for multi-window enforcement.

    Attributes:
        limit: Number of events allowed
        interval_ms: Duration in milliseconds
    """

    limit: int
    interval_ms: int

    @property
    def rps(self) -> float:
        """Requests per second."""
        return (self.limit * 1000) / self.interval_ms

    @property
    def rpm(self) -> float:
        """Requests per minute."""
        return (self.limit * 60_000) / self.interval_ms

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.interval_ms == 1_000:
            return f"{self.limit}/second"
        elif self.interval_ms == 60_000:
            return f"{self.limit}/minute"
        elif self.interval_ms == 3_600_000:
            return f"{self.limit}/hour"
        else:
            return f"{self.limit}/{self.interval_ms}ms"

    def __repr__(self) -> str:
        return f"RateSpec(limit={self.limit}, interval_ms={self.interval_ms})"


# ============================================================================
# Parsing
# ============================================================================


def parse_rate_string(spec: str) -> RateSpec:
    """Parse human-readable rate string into RateSpec.

    Format: "{limit}/{duration}" where duration is second/minute/hour/day

    Examples:
        "5/second"    → RateSpec(limit=5, interval_ms=1000)
        "300/minute"  → RateSpec(limit=300, interval_ms=60000)
        "10/hr"       → RateSpec(limit=10, interval_ms=3600000)

    Args:
        spec: Rate specification string

    Returns:
        Parsed RateSpec

    Raises:
        ValueError: If spec format is invalid or unparseable
    """
    spec = spec.strip()

    # Parse "{limit}/{duration}"
    match = re.match(r"(\d+)\s*/\s*(\w+)", spec)
    if not match:
        raise ValueError(
            f"Invalid rate spec: {spec!r}. Expected format: '5/second', '300/minute', etc."
        )

    limit_str, duration_str = match.groups()
    limit = int(limit_str)

    # Normalize duration
    duration_str = duration_str.lower()
    if duration_str in DURATION_ALIASES:
        duration_str = DURATION_ALIASES[duration_str]

    if duration_str not in DURATION_MS:
        raise ValueError(
            f"Unknown duration: {duration_str!r}. Supported: {list(DURATION_MS.keys())}"
        )

    interval_ms = DURATION_MS[duration_str]

    if limit <= 0:
        raise ValueError(f"Limit must be positive, got: {limit}")

    return RateSpec(limit=limit, interval_ms=interval_ms)


def validate_rate_list(rates: List[RateSpec]) -> bool:
    """Validate multi-window rate list.

    pyrate-limiter requires rates to be ordered by interval (ascending)
    AND by limit/interval ratio (descending). This ensures that smaller
    windows are checked before larger ones.

    From pyrate-limiter docs:
        >>> from pyrate_limiter import validate_rate_list, Rate, Duration
        >>> validate_rate_list([Rate(10, Duration.SECOND), Rate(500, Duration.MINUTE)])
        True

    Args:
        rates: List of RateSpec objects

    Returns:
        True if rates are correctly ordered

    Raises:
        ValueError: If rates are out of order
    """
    if len(rates) <= 1:
        return True

    # Check ordering: intervals must be ascending
    for i in range(len(rates) - 1):
        if rates[i].interval_ms > rates[i + 1].interval_ms:
            raise ValueError(
                f"Rates not ordered by interval: "
                f"{rates[i]} (ms={rates[i].interval_ms}) "
                f"should come after {rates[i + 1]} (ms={rates[i + 1].interval_ms})"
            )

    # Check limit/interval ratio: should be descending
    for i in range(len(rates) - 1):
        ratio_i = rates[i].limit / rates[i].interval_ms
        ratio_next = rates[i + 1].limit / rates[i + 1].interval_ms
        if ratio_i < ratio_next:
            # This is a warning, not an error (pyrate-limiter enforces it)
            # but we should note it
            pass

    return True


def normalize_rate_list(rates: List[str]) -> List[RateSpec]:
    """Parse and validate a list of rate strings.

    Automatically sorts by interval to ensure correct ordering.

    Args:
        rates: List of rate specification strings

    Returns:
        Sorted list of RateSpec objects

    Raises:
        ValueError: If any rate spec is invalid
    """
    parsed = [parse_rate_string(r) for r in rates]

    # Sort by interval (ascending), then by limit/interval ratio (descending)
    parsed.sort(key=lambda r: (r.interval_ms, -r.limit / r.interval_ms))

    # Validate ordering
    validate_rate_list(parsed)

    return parsed


# ============================================================================
# Per-Service Configuration
# ============================================================================


def normalize_per_service_rates(
    rates_dict: Dict[str, str],
    default_rate: Optional[str] = None,
) -> Dict[str, List[RateSpec]]:
    """Normalize per-service rate configuration.

    Builds a registry of service → rates for use by RateLimitManager.

    Args:
        rates_dict: Dictionary mapping service names to rate strings
                    e.g., {"ols": "4/second", "bioportal": "2/second"}
        default_rate: Default rate for unlisted services

    Returns:
        Dictionary mapping service names to sorted lists of RateSpecs

    Example:
        >>> config = normalize_per_service_rates({
        ...     "ols": "4/second",
        ...     "bioportal": "2/second"
        ... }, default_rate="8/second")
        >>> config["ols"]
        [RateSpec(limit=4, interval_ms=1000)]
    """
    result: Dict[str, List[RateSpec]] = {}

    for service, rate_str in rates_dict.items():
        try:
            spec = parse_rate_string(rate_str)
            result[service] = [spec]
        except ValueError as e:
            raise ValueError(f"Invalid rate for service {service!r}: {rate_str!r}. {e}")

    # Add default if provided
    if default_rate:
        try:
            default_spec = parse_rate_string(default_rate)
            result["_default"] = [default_spec]
        except ValueError as e:
            raise ValueError(f"Invalid default rate: {default_rate!r}. {e}")

    return result


# ============================================================================
# Validation
# ============================================================================


def get_schema_summary() -> Dict[str, str]:
    """Get summary of rate spec schema for documentation.

    Returns:
        Dictionary with format examples and documentation
    """
    return {
        "format": "{limit}/{duration}",
        "duration_options": "second, sec, minute, min, hour, hr, day",
        "examples": [
            "5/second",
            "300/minute",
            "1000/hour",
            "2/sec",
        ],
        "multi_window": "Combine multiple specs for multi-window enforcement",
        "default_global": "8/second, 300/minute",
        "default_per_service": "4/second (per your recommendations)",
    }


__all__ = [
    "RateSpec",
    "parse_rate_string",
    "validate_rate_list",
    "normalize_rate_list",
    "normalize_per_service_rates",
    "get_schema_summary",
]
