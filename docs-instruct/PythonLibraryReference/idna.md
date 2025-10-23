Below is a concise-but-complete **field guide to the `idna` library** for Python—written for an AI programming agent refactoring custom domain‑name handling code. It focuses on what the library does, how to use it safely, what policy switches matter (IDNA2008 vs UTS #46), what errors to expect, and common footguns.

> **What library is this?**
> **`idna`** provides **IDNA 2008** (RFC 5890–5895) conversion between Unicode domain labels (**U‑labels**) and their ASCII form (**A‑labels**, “Punycode” with the `xn--` prefix), plus optional **UTS #46** compatibility mapping. It’s BSD‑3‑Clause licensed, supports Python 3.8+, and the latest release is **3.11 (Oct 12, 2025)**. ([PyPI][1])

---

## Core mental model

* IDNs are processed **label‑by‑label** (the parts between dots).

  * **U‑label**: Unicode form users type (e.g., `ドメイン`).
  * **A‑label**: ASCII form used on the wire, created by **Punycode** then prefixed with `xn--`. ([GitHub][2])
* **IDNA 2008** defines what characters are allowed, including Bidi and contextual rules. **UTS #46** (optional) **maps** end‑user input (case, width, dot‑like characters, some legacy compat) before IDNA. ([IETF Datatracker][3])

---

## Installation

```bash
pip install idna
```

`idna` has no runtime dependencies and is widely used transitively by HTTP clients (e.g., **HTTPX** requires `idna`). ([PyPI Stats][4])

---

## Quick start (copy/paste)

```python
import idna

# Unicode host -> ASCII (A-labels as bytes)
a = idna.encode('ドメイン.テスト')   # b'xn--eckwd4c7c.xn--zckzah'
# ASCII -> Unicode (str)
u = idna.decode('xn--eckwd4c7c.xn--zckzah')  # 'ドメイン.テスト'
```

Per‑label helpers if you need to process labels yourself:

```python
idna.alabel('测试')   # b'xn--0zwm56d'
idna.ulabel('xn--0zwm56d')  # '测试'
```

All of the above are straight from the package’s README. ([GitHub][2])

---

## When to turn on UTS #46 mapping

Set `uts46=True` to map the user’s input to a canonical form (case‑fold, width‑fold, normalize “dot‑like” characters such as `。\uff0e\uff61` to `.`) **before** IDNA conversion. That’s the behavior most browsers follow for user input.

```python
# Without mapping: uppercase 'K' is disallowed by IDNA
idna.encode('Königsgäßchen')               # raises InvalidCodepoint
# With UTS #46 mapping: case‑folding etc. then IDNA 2008
idna.encode('Königsgäßchen', uts46=True)   # b'xn--knigsgchen-b4a3dun'
```

UTS #46 explicitly treats **label separators** as **U+002E `.`**, **U+FF0E**, **U+3002**, **U+FF61**; mapping normalizes them to `.`. This is why `uts46=True` is the right “be liberal in what you accept” choice for end‑user input. ([GitHub][2])

> **Note on “transitional” mode.** Unicode 16.0 removed transitional processing; the `transitional` argument no longer has any effect and is being deprecated in `idna`. ([GitHub][2])

---

## API surface you’ll actually use

* **`idna.encode(domain: str, *, uts46: bool = False, ...) -> bytes`**
  Full domain in → **A‑labels** (ASCII bytes) out, dot‑separated.
* **`idna.decode(domain: str | bytes) -> str`**
  A‑labels in → **U‑labels** (Unicode str) out.
* **Label helpers**: **`alabel(label: str) -> bytes`**, **`ulabel(label: str | bytes) -> str`**.
* **Exceptions** (all subclass **`idna.IDNAError`**):

  * `InvalidCodepoint` (disallowed character),
  * `InvalidCodepointContext` (CONTEXTJ/CONTEXTO not satisfied),
  * `IDNABidiError` (violates Bidi rules). ([PyPI][1])

> Internally the library enforces the IDNA 2008 code‑point categories (PVALID/CONTEXTJ/CONTEXTO/DISALLOWED/UNASSIGNED) and the **Bidi** rule from **RFC 5893**. ([IETF Datatracker][5])

---

## Strict vs. compatibility behavior (IDNA 2008 vs UTS #46)

* **IDNA 2008 (strict)**: what the protocol requires for registration/lookup.

  * Example: certain uppercase or width‑variant forms are **not allowed** if you don’t pre‑map.
* **UTS #46 (compatibility mapping)**: a **pre‑processing** step that makes user input friendlier by mapping case and dot‑likes, and smoothing historic IDNA 2003 vs 2008 differences. Use it for UI input; store/compare the resulting **A‑labels**. ([Unicode][6])

---

## “Do/Don’t” for an AI agent

**Do…**

* **Convert only hostnames** (not entire URLs or free text). Parse the URL, take the **host** part, and IDNA‑encode that. IDNA is **only** for domain names. ([IETF Datatracker][7])
* For **email addresses**, convert **domain** only; the local‑part uses separate rules (EAI), and many systems accept UTF‑8 there without IDNA.
* **Canonicalize for comparisons** by storing **A‑labels** (lowercase ASCII) rather than U‑labels, because registries, resolvers, and HTTP clients match on A‑labels. (HTTPX/Requests rely on `idna` under the hood.) ([PyPI Stats][4])
* **Validate lengths** *after* encoding: each label ≤ **63 octets**, whole name ≤ **255 octets** on the wire; long Unicode can expand in Punycode and overflow. ([IETF Datatracker][8])

**Don’t…**

* Don’t apply IDNA to **IP addresses** (IPv4/IPv6) or to the **email local‑part**.
* Don’t assume every Unicode string is valid under IDNA 2008 (e.g., **emoji** are disallowed; you’ll get an exception). ([GitHub][2])

---

## Robust patterns (with code)

### 1) Safe URL building from user input

```python
from urllib.parse import urlsplit, urlunsplit
import idna

def normalize_url(url: str) -> str:
    parts = urlsplit(url)  # scheme://netloc/path?query#fragment
    host, sep, port = parts.netloc.partition(":")
    # Leave IPv6 literals alone (they come as "[...]" in netloc)
    if host.startswith("[") and host.endswith("]"):
        new_netloc = parts.netloc  # no IDNA for IP literals
    elif host:
        a_host = idna.encode(host, uts46=True).decode("ascii")
        new_netloc = f"{a_host}{sep}{port}" if sep else a_host
    else:
        new_netloc = parts.netloc
    return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))
```

This normalizes separators/width/case via UTS #46, then encodes to A‑labels for network use. ([Unicode][6])

### 2) Displaying a hostname to users (pretty‑print)

```python
def display_host(ascii_host: str) -> str:
    # Defensive: decode only if it's an A-label; otherwise return as-is
    try:
        return idna.decode(ascii_host)
    except idna.IDNAError:
        return ascii_host
```

### 3) Catching and classifying failures

```python
try:
    ascii_host = idna.encode(user_supplied_host, uts46=True).decode("ascii")
except idna.IDNABidiError as e:
    # Right-to-left or mixed-directionality rule violated (RFC 5893)
    ...
except idna.InvalidCodepointContext as e:
    # CONTEXTJ/CONTEXTO rule failed (e.g., ZWJ/ZWNJ or script-specific dots)
    ...
except idna.InvalidCodepoint as e:
    # Disallowed characters (e.g., emoji)
    ...
```

`idna` raises these errors according to RFCs 5891/5892/5893—handle them to produce actionable messages. ([PyPI][1])

---

## Under the hood: what rules are enforced?

* **Code‑point eligibility** per **RFC 5892** (PVALID/CONTEXTJ/CONTEXTO/DISALLOWED/UNASSIGNED), with IANA‑published derived property tables. ([IETF Datatracker][5])
* **Bidi rule** for RTL scripts per **RFC 5893**. ([IETF Datatracker][9])
* **Punycode** (RFC 3492) for the ASCII encoding itself. ([IETF Datatracker][10])
* Optional **UTS #46** mapping for case/width/dot normalization and legacy compat (often used by browsers). ([Unicode][6])

---

## Interop notes

* The Python stdlib codec **`'idna'`** implements the **older IDNA 2003** rules. This library is a **drop‑in replacement** for modern IDNA 2008 semantics and can also expose a codec (e.g., `'idna2008'`) via `idna.codec` if you want to use `str.encode(...)`. Prefer the `idna.encode/decode` functions unless you know you want the codec. ([PyPI][1])
* Popular HTTP clients (e.g., **HTTPX**) require `idna`; if you normalize hosts yourself you can still hand the client a canonical **ASCII** hostname. ([PyPI Stats][4])

---

## Footguns (and how to avoid them)

1. **“Works in Unicode, fails on the wire.”** A human‑readable domain might exceed DNS limits after Punycode expansion. Always check length **after** encoding (labels ≤ 63, whole name ≤ 255 octets). ([IETF Datatracker][8])
2. **Dot variants** (`。\u3002`, `．\uff0e`, `｡\uff61`). Without UTS #46, these won’t be treated as separators—turn on `uts46=True` for end‑user input. ([Unicode][6])
3. **Emoji & symbols**: Not permitted by IDNA 2008—expect `InvalidCodepoint`. ([GitHub][2])
4. **Whole URL / email local‑part**: IDNA is for **domain labels only**. Parse first; apply IDNA to the host (or email **domain**). ([IETF Datatracker][7])
5. **Assuming stdlib `'idna'` is “the same.”** It isn’t (IDNA 2003). Use this library’s APIs or its registered **`idna2008`** codec if you need codec semantics. ([PyPI][1])

---

## “Custom code” → `idna` migration map

| If your code currently…                                   | Replace with…                                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Lowercases and strips “weird dots” by hand                | `idna.encode(host, uts46=True)` (maps case/width/separators per UTS #46) ([Unicode][6])                |
| Directly punycodes labels with a generic “punycode” codec | `idna.alabel(...)` (ensures full IDNA 2008 validity, not just raw RFC 3492) ([IETF Datatracker][10])   |
| Accepts full URLs and mutates strings                     | Parse → IDNA only the **host** → reassemble URL                                                        |
| Stores user-typed Unicode hostnames                       | Store **A‑labels** (ASCII), decode to U‑labels only for display                                        |
| Swallows invalid input                                    | Catch `IDNAError` subclasses and render helpful messages (Bidi/context/invalid code point) ([PyPI][1]) |

---

## Reference snippets

**A‑label ⇄ U‑label round‑trip**

```python
host = "mañana.example"
a = idna.encode(host, uts46=True)      # b'xn--maana-pta.example'
u = idna.decode(a)                     # 'mañana.example'
```

**Label‑by‑label processing**

```python
labels = ["παράδειγμα", "δοκιμή"]
encoded = b".".join(idna.alabel(l) for l in labels)  # b'xn--hxajbheg2az3al.xn--jxalpdlp'
```

---

## Why “strict” matters (for registries) vs “friendly” mapping (for users)

* **Registries/DNS** must follow **IDNA 2008** strictly.
* **User input** often needs **UTS #46 mapping** so `K\u00D6NIGSG\u00C4\u00DFCHEN` or `example。\u3002com` normalize to a valid canonical form before conversion. The spec explicitly defines both the mapping and the set of equivalent label separators. ([Unicode][6])

---

## Sources worth bookmarking

* **Project README & PyPI**: usage, `uts46=True` example, exceptions, Python support, release history. ([GitHub][2])
* **RFCs**: IDNA protocol & rules (**5891**, **5892**, **5893**); Punycode (**3492**). ([IETF Datatracker][3])
* **UTS #46**: compatibility mapping; dot‑like separators. ([Unicode][6])

---

If you share a snippet of your current hostname/URL handling, I’ll rewrite it to (1) parse robustly, (2) apply `idna.encode(..., uts46=True)` to the host only, (3) enforce length limits post‑encoding, and (4) surface actionable errors for Bidi/context violations.

[1]: https://pypi.org/project/idna/ "idna · PyPI"
[2]: https://github.com/kjd/idna "GitHub - kjd/idna: Internationalized Domain Names for Python (IDNA 2008 and UTS #46)"
[3]: https://datatracker.ietf.org/doc/html/rfc5891?utm_source=chatgpt.com "RFC 5891 - Internationalized Domain Names in ..."
[4]: https://pypistats.org/packages/httpx?utm_source=chatgpt.com "httpx"
[5]: https://datatracker.ietf.org/doc/html/rfc5892?utm_source=chatgpt.com "RFC 5892 - The Unicode Code Points and ..."
[6]: https://unicode.org/reports/tr46/?utm_source=chatgpt.com "UTS #46: Unicode IDNA Compatibility Processing"
[7]: https://datatracker.ietf.org/doc/html/rfc3490?utm_source=chatgpt.com "RFC 3490 - Internationalizing Domain Names in ..."
[8]: https://datatracker.ietf.org/doc/html/rfc1035?utm_source=chatgpt.com "RFC 1035 - Domain names - implementation and ..."
[9]: https://datatracker.ietf.org/doc/html/rfc5893?utm_source=chatgpt.com "RFC 5893 - Right-to-Left Scripts for Internationalized ..."
[10]: https://datatracker.ietf.org/doc/rfc3492/?utm_source=chatgpt.com "RFC 3492 - Punycode: A Bootstring encoding of Unicode ..."
