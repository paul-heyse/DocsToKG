# Module: ids

Helpers for converting between vector UUIDs and FAISS-compatible integers.

## Functions

### `vector_uuid_to_faiss_int(vector_id)`

Return the FAISS-compatible int identifier for a vector UUID.

Args:
vector_id: String UUID associated with the vector.

Returns:
63-bit integer safe for FAISS index identifiers.
