# === NAVMAP v1 ===
# {
#   "module": "tests.strategies.path_strategies",
#   "purpose": "Hypothesis strategies for filesystem path generation and testing",
#   "sections": [
#     {"id": "basic-path-strategies", "name": "Basic path strategies", "anchor": "basic-path-strategies", "kind": "section"},
#     {"id": "unicode-strategies", "name": "Unicode strategies", "anchor": "unicode-strategies", "kind": "section"},
#     {"id": "adversarial-strategies", "name": "Adversarial strategies", "anchor": "adversarial-strategies", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Hypothesis strategies for filesystem path generation and testing.

Generates valid, edge-case, and adversarial filesystem paths for testing
path validation, normalization, and traversal protection.
"""

from __future__ import annotations

import unicodedata

from hypothesis import strategies as st


# --- Basic Path Strategies ---


@st.composite
def valid_path_components(draw, max_length: int = 255) -> str:
    """
    Generate valid filesystem path components.

    Args:
        max_length: Maximum component length (255 is POSIX limit)

    Examples:
        - file.txt
        - my-document_v1
        - README.md
    """
    # Valid filename characters (exclude path separators and null)
    valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return draw(
        st.text(
            alphabet=valid_chars,
            min_size=1,
            max_size=max_length,
        )
    )


@st.composite
def valid_relative_paths(draw) -> str:
    """
    Generate valid relative filesystem paths.

    Examples:
        - file.txt
        - dir/file.txt
        - a/b/c/file.txt
    """
    components = draw(
        st.lists(
            valid_path_components(),
            min_size=1,
            max_size=5,
        )
    )
    return "/".join(components)


@st.composite
def valid_absolute_paths(draw) -> str:
    """
    Generate valid absolute filesystem paths.

    Examples:
        - /home/user/file.txt
        - /var/log/app.log
        - /usr/local/bin/tool
    """
    relative = draw(valid_relative_paths())
    return "/" + relative


@st.composite
def path_traversal_attempts(draw) -> str:
    """
    Generate path traversal attack patterns.

    Examples:
        - ../../../etc/passwd
        - ../../sensitive/file.txt
        - dir/../../../../etc/passwd
    """
    depth = draw(st.integers(1, 10))
    traversals = "/".join([".."] * depth)

    append_file = draw(st.booleans())
    if append_file:
        target = draw(st.sampled_from(["etc/passwd", "admin/users.db", "secrets.yaml"]))
        return f"{traversals}/{target}"
    return traversals


@st.composite
def path_with_null_bytes(draw) -> str:
    """
    Generate paths with null bytes (null injection attack).

    Examples:
        - file.txt\x00.jpg
        - /path/to/file\x00/secret
    """
    path = draw(valid_relative_paths())
    position = draw(st.integers(0, len(path)))
    return path[:position] + "\x00" + path[position:]


@st.composite
def path_with_double_encoding(draw) -> str:
    """
    Generate double-encoded paths.

    Examples:
        - file%252Etxt (encoded .txt)
        - %252e%252e%252fadmin (encoded ../)
    """
    path = draw(valid_relative_paths())
    # Simple percent encoding
    encoded = "".join(f"%{ord(c):02x}" if c in "./" else c for c in path)
    # Double encode
    return "".join(f"%{ord(c):02x}" if c == "%" else c for c in encoded)


# --- Unicode Path Strategies ---


@st.composite
def unicode_path_components(draw, form: str = "NFC") -> str:
    """
    Generate Unicode path components in specified normalization form.

    Args:
        form: Normalization form (NFC, NFD, NFKC, NFKD)

    Examples (NFC vs NFD):
        - café (NFC: 1 character)
        - cafe (NFD: 5 characters with combining accent)
    """
    # Unicode letters from various scripts
    unicode_chars = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"  # Latin extended
        "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"  # Greek
        "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя"  # Cyrillic
    )
    text = draw(
        st.text(
            alphabet=unicode_chars,
            min_size=1,
            max_size=20,
        )
    )
    # Normalize to specified form
    return unicodedata.normalize(form, text)


@st.composite
def nfc_vs_nfd_pairs(draw) -> tuple[str, str]:
    """
    Generate pairs of NFC and NFD normalized paths that should be equivalent.

    Returns:
        (nfc_path, nfd_path): Same content in different normalization forms
    """
    base = draw(unicode_path_components(form="NFC"))
    nfc = base
    nfd = unicodedata.normalize("NFD", base)
    return (nfc, nfd)


@st.composite
def bidirectional_text_paths(draw) -> str:
    """
    Generate paths with bidirectional text (Hebrew, Arabic, etc.).

    Challenges:
        - Display order vs logical order
        - Path parsing with RTL characters
        - Unicode normalization with bidi marks
    """
    # Simple RTL text
    rtl_scripts = {
        "hebrew": "אבגדהוזחטיכלמנסעפצקרשת",
        "arabic": "ابجدهوزحطيكلمنسعفصقرشتثخذضظغ",
    }
    script = draw(st.sampled_from(list(rtl_scripts.keys())))
    chars = rtl_scripts[script]
    text = "".join(draw(st.sampled_from(chars)) for _ in range(draw(st.integers(1, 5))))
    return f"dir_{text}/file.txt"


@st.composite
def zero_width_characters_paths(draw) -> str:
    """
    Generate paths containing zero-width characters.

    Examples:
        - file​.txt (zero-width space)
        - my‌doc.pdf (zero-width non-joiner)
    """
    path = draw(valid_relative_paths())
    # Insert zero-width space at random position
    zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff"]
    char = draw(st.sampled_from(zero_width_chars))
    position = draw(st.integers(0, len(path)))
    return path[:position] + char + path[position:]


# --- Cross-Platform Path Strategies ---


@st.composite
def windows_reserved_names(draw) -> str:
    """
    Generate paths with Windows reserved names.

    Examples:
        - CON, PRN, AUX, NUL
        - LPT1, LPT9, COM1, COM9
        - CON.txt, PRN.tar.gz
    """
    reserved = (
        ["CON", "PRN", "AUX", "NUL"]
        + [f"LPT{i}" for i in range(1, 10)]
        + [f"COM{i}" for i in range(1, 10)]
    )
    name = draw(st.sampled_from(reserved))
    # Some have extensions
    if draw(st.booleans()):
        ext = draw(st.sampled_from(["txt", "gz", "tar.gz"]))
        return f"{name}.{ext}"
    return name


@st.composite
def windows_trailing_dots_spaces(draw) -> str:
    """
    Generate paths with trailing dots/spaces (Windows restriction).

    Examples:
        - file.txt.
        - document .txt
        - my file  .txt
    """
    base = draw(valid_path_components())
    suffix = draw(st.sampled_from([".", " ", "  ", " . "]))
    return f"{base}{suffix}"


@st.composite
def long_path_components(draw, max_length: int = 255) -> str:
    """
    Generate path components near or exceeding platform limits.

    Args:
        max_length: Component length limit
    """
    # Generate near-limit and over-limit components
    length = draw(
        st.one_of(
            st.just(max_length - 1),  # Just under limit
            st.just(max_length),  # At limit
            st.just(max_length + 10),  # Over limit
        )
    )
    return "a" * length


@st.composite
def deeply_nested_paths(draw, max_depth: int = 100) -> str:
    """
    Generate deeply nested directory structures.

    Args:
        max_depth: Maximum nesting depth
    """
    depth = draw(st.integers(1, max_depth))
    components = [f"dir{i}" for i in range(depth)] + ["file.txt"]
    return "/".join(components)


# --- Extraction Ratio Strategies ---


@st.composite
def compression_ratio_pairs(draw) -> tuple[int, int]:
    """
    Generate compressed vs uncompressed size pairs.

    Returns:
        (compressed_size, uncompressed_size): Pair of sizes
    """
    # Realistic compression ratios (10-90%)
    uncompressed = draw(st.integers(1000, 1000000))
    ratio = draw(st.floats(min_value=0.1, max_value=0.9))
    compressed = int(uncompressed * ratio)
    return (compressed, uncompressed)


@st.composite
def high_compression_ratios(draw) -> tuple[int, int]:
    """
    Generate highly compressible sizes (potential zip bomb).

    Returns:
        (compressed_size, uncompressed_size): Extreme compression pair
    """
    # Highly compressible (1%, 0.1%, etc.)
    uncompressed = draw(st.integers(1000000, 100000000))  # 1MB - 100MB
    ratio = draw(st.floats(min_value=0.0001, max_value=0.01))
    compressed = int(uncompressed * ratio)
    return (compressed, uncompressed)


@st.composite
def extraction_size_limits(draw, limit_gb: int = 5) -> tuple[int, int, int]:
    """
    Generate archive sizes relative to extraction limit.

    Args:
        limit_gb: Extraction limit in GB

    Returns:
        (archive_size, entry_size, limit): Sizes in bytes
    """
    limit_bytes = limit_gb * 1024 * 1024 * 1024
    statuses = [
        ("under", int(limit_bytes * 0.8)),  # Under limit
        ("at", limit_bytes),  # At limit
        ("over", int(limit_bytes * 1.2)),  # Over limit
    ]
    status, archive_size = draw(st.sampled_from(statuses))
    entry_size = draw(st.integers(0, archive_size))
    return (archive_size, entry_size, limit_bytes)
