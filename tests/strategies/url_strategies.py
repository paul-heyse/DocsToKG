# === NAVMAP v1 ===
# {
#   "module": "tests.strategies.url_strategies",
#   "purpose": "Hypothesis strategies for URL generation and property-based testing",
#   "sections": [
#     {"id": "basic-url-strategies", "name": "Basic URL strategies", "anchor": "basic-url-strategies", "kind": "section"},
#     {"id": "host-strategies", "name": "Host strategies", "anchor": "host-strategies", "kind": "section"},
#     {"id": "url-gate-strategies", "name": "URL gate strategies", "anchor": "url-gate-strategies", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Hypothesis strategies for URL generation and property-based testing.

Generates valid and invalid URLs, hosts, ports, and schemes for testing URL validation,
normalization, and security gates.
"""

from __future__ import annotations

from hypothesis import strategies as st


# --- Basic URL Components ---


@st.composite
def valid_schemes(draw) -> str:
    """Generate valid URL schemes (http, https, ftp, etc.)."""
    schemes = ["http", "https", "ftp", "ftps", "ws", "wss", "file"]
    return draw(st.sampled_from(schemes))


@st.composite
def valid_ports(draw) -> int:
    """Generate valid port numbers (1-65535)."""
    return draw(st.integers(min_value=1, max_value=65535))


@st.composite
def privileged_ports(draw) -> int:
    """Generate privileged port numbers (1-1024)."""
    return draw(st.integers(min_value=1, max_value=1024))


@st.composite
def high_ports(draw) -> int:
    """Generate high port numbers (49152-65535)."""
    return draw(st.integers(min_value=49152, max_value=65535))


@st.composite
def invalid_ports(draw) -> int:
    """Generate invalid port numbers."""
    return draw(st.one_of(
        st.integers(max_value=0),  # Below range
        st.integers(min_value=65536),  # Above range
    ))


# --- Host Strategies ---


@st.composite
def valid_hostnames(draw, allow_ip: bool = True) -> str:
    """
    Generate valid hostnames.

    Args:
        allow_ip: Include IPv4 addresses as valid hosts

    Examples:
        - example.com
        - sub.example.org
        - localhost
        - 192.168.1.1
    """
    if allow_ip and draw(st.booleans()):
        # IPv4 address
        return draw(st.just(".".join(
            str(draw(st.integers(0, 255))) for _ in range(4)
        )))

    # Hostname
    labels = draw(st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
            min_size=1,
            max_size=63,
        ),
        min_size=1,
        max_size=4,
    ))
    return ".".join(labels) or "localhost"


@st.composite
def private_ips(draw) -> str:
    """Generate private IP addresses (RFC 1918)."""
    private_ranges = [
        (10, 0, 0, 0),  # 10.0.0.0/8
        (172, 16, 0, 0),  # 172.16.0.0/12
        (192, 168, 0, 0),  # 192.168.0.0/16
    ]
    base = draw(st.sampled_from(private_ranges))

    if base[0] == 10:
        octets = (
            base[0],
            draw(st.integers(0, 255)),
            draw(st.integers(0, 255)),
            draw(st.integers(0, 255)),
        )
    elif base[0] == 172:
        octets = (
            base[0],
            draw(st.integers(16, 31)),
            draw(st.integers(0, 255)),
            draw(st.integers(0, 255)),
        )
    else:  # 192.168
        octets = (
            base[0],
            base[1],
            draw(st.integers(0, 255)),
            draw(st.integers(0, 255)),
        )

    return ".".join(str(o) for o in octets)


@st.composite
def loopback_ips(draw) -> str:
    """Generate loopback IP addresses (127.0.0.0/8 or ::1)."""
    is_ipv4 = draw(st.booleans())
    if is_ipv4:
        return f"127.0.0.{draw(st.integers(0, 255))}"
    return "::1"


@st.composite
def idn_hostnames(draw) -> str:
    """Generate internationalized domain names."""
    # Simplified IDN (ASCII-compatible encoding not shown)
    scripts = {
        "cyrillic": "абвгдежзийклмнопрстуфхцчшщъыьэюя",
        "greek": "αβγδεζηθικλμνξοπρστυφχψω",
        "japanese": "あいうえおかきくけこ",
        "arabic": "ابجدهوزحطيكلمنسعفصقرشتثخذضظغ",
    }
    script = draw(st.sampled_from(list(scripts.keys())))
    chars = scripts[script]
    label = "".join(draw(st.sampled_from(chars)) for _ in range(draw(st.integers(1, 5))))
    return f"{label}.example.com"


# --- Complete URL Strategies ---


@st.composite
def valid_urls(draw, scheme: str | None = None, host: str | None = None) -> str:
    """
    Generate valid URLs.

    Args:
        scheme: Override scheme (default: random valid)
        host: Override host (default: random valid)

    Examples:
        - https://example.com/path?query=value
        - http://192.168.1.1:8080/api/v1/resource
        - ftp://ftp.example.org/pub/file.txt
    """
    s = scheme or draw(valid_schemes())
    h = host or draw(valid_hostnames(allow_ip=True))
    port = draw(st.just(None) | st.just(f":{draw(valid_ports())}"))
    path = draw(st.just("") | st.just(f"/{draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789-._~', max_size=50))}"))
    query = draw(st.just("") | st.just(f"?{draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789=&', max_size=50))}"))

    return f"{s}://{h}{port}{path}{query}"


@st.composite
def private_network_urls(draw) -> str:
    """Generate URLs on private networks."""
    host = draw(private_ips())
    scheme = draw(st.sampled_from(["http", "https"]))
    port = draw(st.just("") | st.just(f":{draw(high_ports())}"))
    return f"{scheme}://{host}{port}/api"


@st.composite
def loopback_urls(draw) -> str:
    """Generate URLs on loopback addresses."""
    host = draw(loopback_ips())
    scheme = draw(st.sampled_from(["http", "https"]))
    return f"{scheme}://{host}:8000/local"


@st.composite
def suspicious_urls(draw) -> str:
    """
    Generate suspicious/potentially dangerous URLs.

    Examples:
        - URLs with many redirects
        - URLs with unusual ports
        - URLs with encoded paths
        - URLs with mixed case/encoding
    """
    suspicious_patterns = [
        lambda: f"https://example.com/?url={''.join(draw(st.sampled_from(['a', 'b', '/', '%'])) for _ in range(50))}",
        lambda: f"https://example.com/{'../' * draw(st.integers(1, 10))}etc/passwd",
        lambda: f"https://example.com/api%2Fadmin",
        lambda: f"https://ExAmPlE.CoM/PATH",
        lambda: f"https://[::1]/admin",
    ]
    pattern = draw(st.sampled_from(suspicious_patterns))
    return pattern()


@st.composite
def url_normalization_pairs(draw) -> tuple[str, str]:
    """
    Generate pairs of URLs that should normalize to same canonical form.

    Returns:
        (variant, canonical): URL variant and its canonical form
    """
    base = "https://example.com/path"
    variants = [
        ("HTTPS://EXAMPLE.COM/PATH", base),
        ("https://example.com:443/path", base),
        ("https://example.com/path?", base),
        ("https://example.com/path/", "https://example.com/path/"),
    ]
    return draw(st.sampled_from(variants))


# --- URL Gate Test Strategies ---


@st.composite
def url_host_port_combinations(draw) -> tuple[str, int | None]:
    """Generate host and port combinations for gate testing."""
    host = draw(valid_hostnames(allow_ip=True))
    port = draw(st.just(None) | st.just(draw(valid_ports())))
    return (host, port)


@st.composite
def restricted_url_patterns(draw) -> tuple[str, bool]:
    """
    Generate URLs with known restriction status.

    Returns:
        (url, should_allow): URL and whether gate should allow it
    """
    patterns = [
        ("https://example.com/api", True),
        ("https://internal.corp/admin", False),
        ("http://192.168.1.1:8080", False),
        ("https://127.0.0.1:3000", False),
        ("https://[::1]/local", False),
    ]
    return draw(st.sampled_from(patterns))
