"""Tests for :func:`DocsToKG.DocParsing.io.compute_content_hash`."""

from __future__ import annotations

import hashlib
import unicodedata

from pathlib import Path

from DocsToKG.DocParsing.io import compute_content_hash


def _legacy_compute_content_hash(path: Path, algorithm: str = "sha1") -> str:
    """Replicate the historical content hashing strategy for regression tests."""

    hasher = hashlib.new(algorithm)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    normalised = unicodedata.normalize("NFKC", text)
    hasher.update(normalised.encode("utf-8"))
    return hasher.hexdigest()


def test_compute_content_hash_text_preserves_legacy_digest(tmp_path: Path) -> None:
    """UTF-8 inputs should match the historical digest, including combining marks."""

    content = (
        "Heading — naïve façade"  # simple ASCII + Latin-1 supplement
        "\n"
        + ("X" * 65536)
        + "f\u0301"  # combining acute accent crosses the chunk boundary
        + "\nemoji: 😀"
        + "\n"
        + "A\u030A"  # NFKC composes this into Å
    )
    target = tmp_path / "text.txt"
    target.write_text(content, encoding="utf-8")

    legacy = _legacy_compute_content_hash(target)
    updated = compute_content_hash(target)

    assert updated == legacy


def test_compute_content_hash_binary_preserves_legacy_digest(tmp_path: Path) -> None:
    """Binary inputs should fall back to byte-wise hashing unchanged."""

    payload = (b"\xff\x00\xfe\x00" * 16384) + b"\x00\xff\xfe\xfd"
    target = tmp_path / "binary.bin"
    target.write_bytes(payload)

    legacy = _legacy_compute_content_hash(target)
    updated = compute_content_hash(target)

    assert updated == legacy
