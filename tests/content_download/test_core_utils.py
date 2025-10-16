import math
from pathlib import Path

import pytest

from DocsToKG.ContentDownload.core import (
    DEFAULT_TAIL_CHECK_BYTES,
    Classification,
    atomic_write_bytes,
    atomic_write_text,
    classify_payload,
    dedupe,
    has_pdf_eof,
    normalize_url,
    parse_size,
    tail_contains_html,
    update_tail_buffer,
)


def test_parse_size_parses_units():
    assert parse_size("1KB") == 1024
    assert parse_size("1.5MB") == int(1.5 * 1024**2)
    assert parse_size("2gb") == 2 * 1024**3
    with pytest.raises(ValueError):
        parse_size("bad")


def test_normalize_url_strips_utm_and_lowercases():
    url = "HTTPS://Example.COM/Path?utm_source=abc&b=1&UTM_campaign=s"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/Path?b=1"


def test_dedupe_preserves_first_occurrence():
    assert dedupe(["a", "b", "a", "", "c", "b"]) == ["a", "b", "c"]


def test_classify_payload_pdf_and_html():
    assert classify_payload(b"%PDF-1.4", "application/pdf", "https://example.com") is Classification.PDF
    assert classify_payload(b"<html><body>", "text/html", "https://example.com") is Classification.HTML
    assert (
        classify_payload(b"", "application/octet-stream", "https://example.com/unknown")
        is Classification.UNKNOWN
    )


def test_tail_detection(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"header data\n%%EOF\n")
    assert has_pdf_eof(path, window_bytes=64)
    assert tail_contains_html(b"</html>")

    buffer = bytearray()
    update_tail_buffer(buffer, b"abc", limit=5)
    update_tail_buffer(buffer, b"defghi", limit=5)
    assert buffer == b"efghi"


def test_atomic_write_helpers(tmp_path: Path):
    target = tmp_path / "atomic.bin"
    total = atomic_write_bytes(target, [b"hello", b"world"])
    assert total == 10
    assert target.read_bytes() == b"helloworld"

    text_target = tmp_path / "atomic.txt"
    atomic_write_text(text_target, "sample text")
    assert text_target.read_text(encoding="utf-8") == "sample text"
