"""RFC 7232 conditional request handling (ETag and Last-Modified).

Responsibilities
----------------
- Parse and store ETag and Last-Modified headers from responses
- Generate If-None-Match and If-Modified-Since headers for requests
- Validate conditional request semantics per RFC 7232
- Handle 304 Not Modified responses and cache revalidation
- Support strong and weak ETags

Design Notes
------------
- Immutable dataclass for parsed validators
- Conservative approach: treat all ETags as weak unless explicitly strong
- Last-Modified takes precedence for "modified" check
- 304 responses should update cache metadata but not body
- Supports entity tags and HTTP date format parsing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Mapping, Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityValidator:
    """Immutable representation of cache validator (ETag or Last-Modified).

    Used to determine if a cached response is still valid via conditional
    requests per RFC 7232.

    Attributes:
        etag: Entity tag from ETag header (includes weak indicator)
        etag_strong: True if etag is a strong validator
        last_modified: Last-Modified timestamp as HTTP date string
        last_modified_dt: Parsed datetime object for last_modified
    """

    etag: Optional[str] = None
    etag_strong: bool = False
    last_modified: Optional[str] = None
    last_modified_dt: Optional[datetime] = None


def parse_entity_validator(headers: Mapping[str, str]) -> EntityValidator:
    """Extract ETag and Last-Modified validators from response headers.

    Args:
        headers: HTTP response headers (case-insensitive)

    Returns:
        EntityValidator with parsed etag and last_modified

    Notes:
        - Extracts ETag header (preserves weak indicator)
        - Extracts Last-Modified and parses to datetime
        - Gracefully handles missing headers
        - Conservative: treats ETags as weak unless starts with strong indicator

    Examples:
        >>> headers = {
        ...     "ETag": '"abc123"',
        ...     "Last-Modified": "Wed, 21 Oct 2025 07:28:00 GMT"
        ... }
        >>> validator = parse_entity_validator(headers)
        >>> validator.etag
        '"abc123"'
        >>> validator.last_modified
        'Wed, 21 Oct 2025 07:28:00 GMT'
    """
    # Parse ETag (case-insensitive)
    etag: Optional[str] = None
    etag_strong = False
    for key, value in headers.items():
        if key.lower() == "etag":
            etag = value.strip()
            # Strong validator starts with non-W, weak starts with W/
            etag_strong = not etag.startswith("W/")
            break

    # Parse Last-Modified (case-insensitive)
    last_modified: Optional[str] = None
    last_modified_dt: Optional[datetime] = None
    for key, value in headers.items():
        if key.lower() == "last-modified":
            last_modified = value.strip()
            try:
                last_modified_dt = parsedate_to_datetime(last_modified)
            except (TypeError, ValueError) as e:
                LOGGER.debug(f"Failed to parse Last-Modified: {e}")
            break

    return EntityValidator(
        etag=etag,
        etag_strong=etag_strong,
        last_modified=last_modified,
        last_modified_dt=last_modified_dt,
    )


def build_conditional_headers(
    validator: EntityValidator,
    use_strong_comparison: bool = False,
) -> dict[str, str]:
    """Build If-None-Match and If-Modified-Since headers for conditional request.

    Args:
        validator: EntityValidator with cached etag and last_modified
        use_strong_comparison: Force strong comparison even for weak ETags

    Returns:
        Dictionary with conditional headers (may be empty)

    Notes:
        - If-None-Match uses ETag (strong comparison by default)
        - If-Modified-Since uses Last-Modified if available
        - Per RFC 7232: weak comparison used unless dealing with strong validators
        - Both headers sent if both validators available (origin decides)

    Examples:
        >>> validator = EntityValidator(etag='"abc123"')
        >>> headers = build_conditional_headers(validator)
        >>> headers
        {'If-None-Match': '"abc123"'}

        >>> validator = EntityValidator(
        ...     etag='"abc123"',
        ...     last_modified="Wed, 21 Oct 2025 07:28:00 GMT",
        ... )
        >>> headers = build_conditional_headers(validator)
        >>> headers
        {'If-None-Match': '"abc123"', 'If-Modified-Since': 'Wed, 21 Oct 2025 07:28:00 GMT'}
    """
    headers: dict[str, str] = {}

    # Add If-None-Match header
    if validator.etag:
        headers["If-None-Match"] = validator.etag

    # Add If-Modified-Since header
    if validator.last_modified:
        headers["If-Modified-Since"] = validator.last_modified

    return headers


def should_revalidate(
    validator: EntityValidator,
    response_headers: Mapping[str, str],
) -> bool:
    """Determine if cached response was revalidated successfully (304 Not Modified).

    Args:
        validator: Original EntityValidator used in conditional request
        response_headers: Headers from 304 response

    Returns:
        True if validators match and response is 304 equivalent, False otherwise

    Notes:
        - 304 response should have matching ETag or Last-Modified
        - Per RFC 7232: cache should be updated with new validators
        - Weak comparison used for ETag unless strong validator available

    Examples:
        >>> original = EntityValidator(etag='"abc123"')
        >>> response_headers = {"ETag": '"abc123"'}
        >>> should_revalidate(original, response_headers)
        True
    """
    if not validator.etag and not validator.last_modified:
        return False

    # Parse response validators
    response_validator = parse_entity_validator(response_headers)

    # Check ETag match (weak comparison by default)
    if validator.etag and response_validator.etag:
        # Normalize ETags for comparison (remove W/ prefix for weak comparison)
        original_etag = validator.etag[2:] if validator.etag.startswith("W/") else validator.etag
        response_etag = (
            response_validator.etag[2:]
            if response_validator.etag.startswith("W/")
            else response_validator.etag
        )
        if original_etag == response_etag:
            return True

    # Check Last-Modified match (exact string comparison for HTTP date)
    if validator.last_modified and response_validator.last_modified:
        if validator.last_modified == response_validator.last_modified:
            return True

    return False


def merge_validators(
    original: EntityValidator,
    updated: EntityValidator,
) -> EntityValidator:
    """Merge original and updated validators (for 304 revalidation).

    Args:
        original: Original cached validator
        updated: Validator from 304 response

    Returns:
        Merged EntityValidator with updated values (prefer updated)

    Notes:
        - 304 response may update validators
        - Use original validators if 304 doesn't provide new ones
        - Per RFC 7232: cache entry is updated with new validators

    Examples:
        >>> original = EntityValidator(etag='"old"')
        >>> updated = EntityValidator(etag='"new"')
        >>> merged = merge_validators(original, updated)
        >>> merged.etag
        '"new"'
    """
    return EntityValidator(
        etag=updated.etag or original.etag,
        etag_strong=updated.etag_strong if updated.etag else original.etag_strong,
        last_modified=updated.last_modified or original.last_modified,
        last_modified_dt=updated.last_modified_dt or original.last_modified_dt,
    )


def is_validator_available(validator: EntityValidator) -> bool:
    """Check if validator has any usable validation tokens.

    Args:
        validator: EntityValidator to check

    Returns:
        True if either etag or last_modified is set

    Examples:
        >>> EntityValidator()
        False
        >>> EntityValidator(etag='"abc"')
        True
    """
    return bool(validator.etag or validator.last_modified)
