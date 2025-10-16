import string
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

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


def _case_variations(base: str) -> st.SearchStrategy[bytes]:
    return st.lists(
        st.booleans(),
        min_size=len(base),
        max_size=len(base),
    ).map(
        lambda mask, base=base: "".join(
            (char.upper() if flag else char.lower()) if char.isalpha() else char
            for char, flag in zip(base, mask)
        ).encode("ascii")
    )


@st.composite
def _url_components(draw):
    letters_digits = string.ascii_letters + string.digits
    scheme = draw(st.sampled_from(["HTTP", "https", "HtTp"]))
    host_labels = draw(
        st.lists(
            st.text(alphabet=letters_digits, min_size=1, max_size=5),
            min_size=1,
            max_size=3,
        )
    )
    host = ".".join(host_labels)
    path_segments = draw(
        st.lists(
            st.text(alphabet=letters_digits + "-_", min_size=1, max_size=6),
            min_size=0,
            max_size=4,
        )
    )
    path = "/" + "/".join(path_segments) if path_segments else ""
    query_pairs = draw(
        st.lists(
            st.tuples(
                st.one_of(
                    st.text(alphabet=letters_digits, min_size=1, max_size=6).filter(
                        lambda key: not key.lower().startswith("utm_")
                    ),
                    st.text(alphabet=letters_digits, min_size=1, max_size=6).map(lambda key: "utm_" + key),
                ),
                st.text(alphabet=letters_digits + "-_", min_size=0, max_size=6),
            ),
            max_size=5,
        )
    )
    fragment = draw(st.text(alphabet=letters_digits, min_size=0, max_size=6))
    return scheme, host, path, query_pairs, fragment


@given(_url_components())
def test_normalize_url_property_idempotent_and_strips_tracking(components):
    scheme, host, path, query_pairs, fragment = components
    query = urlencode(query_pairs, doseq=True)
    url = f"{scheme}://{host}{path}"
    if query:
        url = f"{url}?{query}"
    if fragment:
        url = f"{url}#{fragment}"

    normalized = normalize_url(url)
    assert normalize_url(normalized) == normalized

    parts = urlsplit(normalized)
    assert parts.scheme == parts.scheme.lower()
    assert parts.netloc == parts.netloc.lower()
    assert parts.fragment == ""
    assert all(
        not key.lower().startswith("utm_")
        for key, _ in parse_qsl(parts.query, keep_blank_values=True)
    )


@given(
    st.lists(
        st.one_of(
            st.text(alphabet=string.ascii_letters + string.digits + "-_", min_size=0, max_size=6),
            st.just(""),
        ),
        max_size=20,
    )
)
def test_dedupe_property_preserves_first_occurrences(values):
    result = dedupe(values)
    expected = []
    seen = set()
    for item in values:
        if item and item not in seen:
            expected.append(item)
            seen.add(item)
    assert result == expected


WHITESPACE_BYTES = [b"", b" ", b"\t", b"\n", b"\r"]
HTML_START_BASES = ["<html", "<body", "<head", "<!doctype html"]


html_start_strategy = st.one_of(*[_case_variations(base) for base in HTML_START_BASES])


@given(
    st.lists(st.sampled_from(WHITESPACE_BYTES), min_size=0, max_size=4).map(b"".join),
    html_start_strategy,
    st.binary(min_size=0, max_size=32),
)
def test_classify_payload_detects_html_signatures(leading, marker, remainder):
    payload = leading + marker + remainder
    result = classify_payload(payload, "application/octet-stream", "https://example.com/item")
    assert result is Classification.HTML


@given(
    st.lists(st.sampled_from(WHITESPACE_BYTES), min_size=0, max_size=6).map(b"".join),
    st.binary(min_size=0, max_size=128),
)
def test_classify_payload_detects_pdf_header(leading, trailer):
    payload = leading + b"%PDF-1.5" + trailer
    result = classify_payload(payload, None, "https://example.com/resource")
    assert result is Classification.PDF


@given(
    st.text(alphabet=string.ascii_letters + string.digits + " \n\r\t", min_size=0, max_size=512).map(
        lambda text: text.encode("ascii")
    ),
    st.binary(min_size=0, max_size=512),
)
def test_classify_payload_detects_pdf_marker_within_window(prefix, suffix):
    payload = prefix + b"%PDF" + suffix
    result = classify_payload(payload, "application/octet-stream", "https://example.com/content")
    assert result is Classification.PDF


@given(
    st.text(alphabet=string.ascii_letters + string.digits + " \n\r\t", min_size=0, max_size=128).map(
        lambda text: text.encode("ascii")
    ),
    st.text(alphabet=string.ascii_letters + string.digits + " \n\r\t", min_size=0, max_size=128).map(
        lambda text: text.encode("ascii")
    ),
)
def test_has_pdf_eof_detects_marker(prefix: bytes, suffix: bytes):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "payload.pdf"
        window = max(DEFAULT_TAIL_CHECK_BYTES, len(prefix) + len(suffix) + 10)
        path.write_bytes(prefix + b"%%EOF" + suffix)
        assert has_pdf_eof(path, window_bytes=window)
        path.write_bytes(prefix + suffix)
        assert not has_pdf_eof(path, window_bytes=window)


HTML_TAIL_BASES = ["</html", "</body", "</script", "<html"]
html_tail_strategy = st.one_of(*[_case_variations(base) for base in HTML_TAIL_BASES])


@given(
    st.binary(min_size=0, max_size=32),
    html_tail_strategy,
    st.binary(min_size=0, max_size=32),
)
def test_tail_contains_html_detects_closing_markers(prefix, marker, suffix):
    tail = prefix + marker + suffix
    assert tail_contains_html(tail)


@given(
    st.text(alphabet=string.ascii_letters + string.digits, min_size=0, max_size=64).map(
        lambda text: text.encode("ascii")
    )
)
def test_tail_contains_html_false_without_markers(tail: bytes):
    assert not tail_contains_html(tail)
