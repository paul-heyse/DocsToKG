"""Helpers for converting between vector UUIDs and FAISS-compatible integers."""

from __future__ import annotations

import uuid

__all__ = ["vector_uuid_to_faiss_int"]

_MASK_63_BITS = (1 << 63) - 1


def vector_uuid_to_faiss_int(vector_id: str) -> int:
    """Return the FAISS-compatible int identifier for a vector UUID.

    Args:
        vector_id: String UUID associated with the vector.

    Returns:
        63-bit integer safe for FAISS index identifiers.
    """
    return uuid.UUID(vector_id).int & _MASK_63_BITS
