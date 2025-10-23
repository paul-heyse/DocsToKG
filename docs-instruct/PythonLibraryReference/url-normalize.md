Below is a practical, “drop‑in” engineering reference to **url‑normalize** (Python) aimed at an AI agent refactoring custom URL cleanup code. It covers what the library normalizes, what it does *not* do, the public API and CLI, version differences, and patterns to replace common home‑grown logic.

---

## 1) What url‑normalize is for

**url‑normalize** is a tiny library that converts a URL (or URL‑like string) into a standardized string in line with RFC 3986/3987 best practices: case normalization (scheme/host), percent‑encoding normalization, removal of dot‑segments, IDN (internationalized domain name) handling, and sensible defaults for missing parts. It has one primary API function: `url_normalize(...)` returning a normalized `str`. The project README lists the exact normalizations and shows basic usage. ([PyPI][1])

Under the hood, its behavior mirrors the normalization techniques described in **RFC 3986 §6.2** (case normalization, percent‑encoding normalization, removal of dot‑segments), with IRI/IDN handling per **RFC 3987**. ([IETF Datatracker][2])

---

## 2) Current version, footprint, and dependencies

* **Latest**: 2.2.1 (released Apr 26, 2025). Python 3.8+ only. License MIT. ([PyPI][1])
* **Dependency**: `idna` (for IDNA2008 with UTS46 processing). ([PyPI Stats][3])
* The library includes `py.typed` as of 2.2.1, so type checkers can consume its typing metadata. ([GitHub][4])

---

## 3) Public API (Python) and CLI

### Python API (single entry point)

```python
from url_normalize import url_normalize

url_normalize("www.foo.com:80/foo")                    # basic
url_normalize("www.foo.com/foo", default_scheme="http")
url_normalize("www.google.com/search?q=x&utm_source=y",
              filter_params=True)                      # drop common tracking params
url_normalize("example.com?page=1&id=123&ref=z",
              filter_params=True,
              param_allowlist=["page", "id"])          # keep only a list of params
url_normalize("/images/logo.png", default_domain="example.com")  # absolute path + domain
```

These examples and supported keyword arguments are shown in the README (not exhaustive, but representative):

* `default_scheme="https"` (default)
* `filter_params` (bool)
* `param_allowlist` (list of names **or** dict mapping domain→list)
* `default_domain` (string) ([PyPI][1])

> **Note.** Since 2.0.0 the library **removed** any option to sort query parameters, explicitly noting that parameter order can be semantically meaningful. If your custom code sorts/deduplicates query params, keep that logic outside url‑normalize. ([GitHub][5])

### CLI

A `url-normalize` command is provided. Key options added in 2.1.0:

* `--charset/-c`, `--default-scheme/-s`, `--filter-params/-f`, `--param-allowlist/-p` (CSV), plus `--version/-v`. ([GitHub][6])
  The README shows basic CLI examples matching the Python API behavior. ([PyPI][1])

---

## 4) Exactly what gets normalized (and why)

The README enumerates the normalizations that url‑normalize performs; these map closely to RFC 3986/3987 guidance. Here they are with context and impact for refactoring:

### A. Case and character normalization

* **Lowercase the scheme and host.** (e.g., `HTTP://Example.COM` → `http://example.com`). RFC 3986 marks scheme/host as case‑insensitive; lowering eliminates equivalence ambiguity. ([PyPI][1])
* **Percent‑encoding normalization**:

  * Use **uppercase hex** for percent‑escapes (e.g., `%2a` → `%2A`).
  * **Only** percent‑encode where essential (decode percent‑encoded *unreserved* characters). These follow RFC 3986 rules. ([PyPI][1])
* **Unicode normalization**: ensure **UTF‑8 NFC** across URL portions; for IDNs, perform **IDNA2008 with UTS 46 processing via `idna`** (punycode ASCII host). This is what enables inputs like `https://пример.рф/` to become a valid ASCII hostname. ([PyPI][1])

### B. Path normalization

* **Remove dot‑segments** (`.` and `..`) in non‑relative paths (RFC 3986 algorithm).
* For schemes that define an empty path to be equivalent to `/`, **normalize to `/`** (e.g., `http://example.com` → `http://example.com/`). ([PyPI][1])

### C. Authority / port / defaults

* For schemes that define a default authority, use the empty authority when the default is desired (effectively: avoid superfluous authority text).
* **Drop the port if it equals the scheme’s default** (e.g., `http://example.com:80/` → `http://example.com/`; `https://example.com:443/` → `https://example.com/`). The docs summarize this behavior; it’s part of “use empty port if default is desired.” ([PyPI][1])

> **Gotcha (docs example conflict).** The README shows:
>
> * Python API example: `url_normalize("www.foo.com:80/foo")` → `https://www.foo.com/foo` (drops `:80`),
> * while a later `uvx` example shows it **kept**: `https://www.foo.com:80/foo`.
>   With **default_scheme="https"**, port 80 is *not* the default for HTTPS, so keeping `:80` is the conservative outcome; the “kept” example matches that interpretation. When refactoring logic around default ports, assume **only the scheme’s default gets dropped** and confirm with tests in your environment. ([GitHub][7])

### D. Query string handling

* **No reordering or sorting of parameters** (since 2.0.0 the sort option was removed as incorrect). If your custom code sorted parameters for dedupe/canonicalization, you must keep that logic outside url‑normalize. ([GitHub][5])
* **Optional parameter filtering**: `filter_params=True` removes common “noise”/tracking parameters; you can supply a **global allowlist** (`["page","id"]`) or a **domain→allowlist** map (`{"example.com": ["page","id"]}`) to constrain what remains. Example in README shows dropping `utm_*` by enabling filtering. Use this if your custom code stripped analytics tags. ([PyPI][1])

### E. Special forms it understands

* **Empty string** (returns sensible default), **protocol‑relative** (`//domain.tld`) inputs, and **shebang (`#!`) URLs** are handled—these behaviors were called out historically and carried forward. ([PyPI][8])

---

## 5) What url‑normalize intentionally does **not** do

* **No validation/sanitization** beyond what’s needed for normalization; garbage in → normalized garbage out. (This is a common stance in normalization tools; treat normalization as canonicalization, not security.) Aligns with RFC’s focus on syntax‑level normalization. ([IETF Datatracker][2])
* **No query param sorting/deduplication** (removed; order can be meaningful). If your system relies on sorted query strings for cache keys or de‑dupe, keep that step. ([GitHub][5])
* **No content‑aware canonicalization** like stripping `index.html`, collapsing site‑specific aliases, or removing path elements based on application semantics. For those you need site‑specific logic (or a crawler‑oriented toolkit).

---

## 6) Version changes that matter for refactors

* **2.0.0 (Mar 30, 2025)**

  * **BREAKING**: default scheme **changed from `http` → `https`**.
  * **BREAKING**: switched to **IDNA 2008 + UTS 46** via `idna`.
  * **BREAKING**: **removed “sort query params”** option.
  * **Added**: query parameter filtering + allowlists.
  * **Python**: now **3.8+**.
  * Host normalization refined to handle labels separately. ([GitHub][5])
* **2.1.0**: Added the **CLI** and flags for charset, scheme, filtering, allowlists. ([GitHub][6])

> If you’re upgrading from **1.4.x**: your tests may change where (a) URLs without scheme now default to **https**; (b) any reliance on sorted query parameters must be removed; (c) IDN behavior uses IDNA2008/UTS46, which may slightly change punycode results for edge‑case inputs. ([PyPI][8])

---

## 7) “Replace my custom code” — a refactor checklist

Match your custom behavior to url‑normalize in a one‑to‑one fashion:

| If your code currently does…                        | Replace with url‑normalize by…                                                                             |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Lowercase scheme and host                           | `url_normalize(url)` (built‑in) ([PyPI][1])                                                                |
| Uppercase percent‑escapes & decode unreserved chars | `url_normalize(url)` (built‑in per RFC rules) ([PyPI][1])                                                  |
| Remove `.` and `..` from paths                      | `url_normalize(url)` (built‑in) ([PyPI][1])                                                                |
| Force `/` when path is empty (http/https)           | `url_normalize(url)` (built‑in) ([PyPI][1])                                                                |
| Drop default ports (80 on http, 443 on https)       | `url_normalize(url)` (built‑in; only drop when port == scheme default) ([PyPI][1])                         |
| Normalize/accept IDNs (e.g., café.example)          | `url_normalize(url)` (IDNA2008/UTS46 via `idna`) ([GitHub][5])                                             |
| Strip trackers like `utm_*`                         | `url_normalize(url, filter_params=True)`; optionally add `param_allowlist=` for strict control ([PyPI][1]) |
| Default a missing scheme                            | Set `default_scheme=` (`"https"` by default since 2.0) ([GitHub][5])                                       |
| Turn “/path” into a full URL                        | `url_normalize("/path", default_domain="example.com")` (and optional `default_scheme`) ([PyPI][1])         |
| Sort or dedupe query params                         | **Do this outside url‑normalize** (feature removed in 2.0; don’t rely on the lib here) ([GitHub][5])       |

---

## 8) Subtleties & pitfalls to keep in mind

1. **Default scheme vs explicit port**
   If input is `www.example.com:80/foo` and you accept the new default scheme `https`, port 80 is **not** default for HTTPS. Expect the port to be **kept** unless you explicitly set `default_scheme="http"`. (The project README examples show both outcomes; test in your environment and pin behavior in tests.) ([GitHub][7])

2. **Parameter order matters**
   Don’t assume the library will re‑order query params; this is deliberate per 2.0.0 release notes. If you used sorting as part of cache‑key canonicalization, keep that logic separate. ([GitHub][5])

3. **Normalization ≠ validation**
   url‑normalize normalizes strings; it doesn’t guarantee the resource is reachable, secure, or even syntactically valid in all contexts. Treat it like a canonicalizer, not a sanitizer. (This is consistent with RFC 3986 scope.) ([IETF Datatracker][2])

---

## 9) Minimal “equivalent behavior” recipe (if you must keep some custom steps)

If you’re consolidating custom code, a common pattern is:

1. Parse with `urllib.parse.urlsplit`, detect missing scheme/host; apply `default_scheme` / `default_domain` policies.
2. For host: if Unicode, apply **IDNA2008/UTS46** via `idna` (to ASCII punycode).
3. Lowercase scheme, host.
4. Normalize path via RFC dot‑segment removal; percent‑encode only when required and uppercase escapes; ensure `/` when applicable.
5. Query: **keep order**, optionally filter names using an allowlist; do **not** sort (unless your application explicitly requires it).
6. Drop port **only if** it’s the default for the chosen scheme.
7. Recombine.

url‑normalize does these steps for you and keeps up with spec nuances (notably IDNA and percent‑encoding rules). Use the library instead of replicating this logic unless you have application‑specific canonicalization needs. ([PyPI][1])

---

## 10) Example I/O you can use as tests

These showcase behavior your refactor can lean on. (They reflect library docs & RFC‑aligned expectations.)

```python
cases = [
    # Case & percent-encoding normalization
    ("HTTP://User@ExAmple.COM/%7efoo/%2e%2E/bar%2a", "http://User@example.com/bar%2A"),
    # Dot-segment removal + default '/' path
    ("http://example.com", "http://example.com/"),
    ("http://example.com/a/b/.././c", "http://example.com/a/c"),
    # IDN normalization (punycode host)
    ("http://münich.example/straße", "http://xn--mnich-kva.example/stra%C3%9Fe"),
    # Default port removal
    ("http://example.com:80/", "http://example.com/"),
    ("https://example.com:443/", "https://example.com/"),
    # Query filtering
    ("https://www.google.com/search?q=test&utm_source=foo",
     "https://www.google.com/search?q=test"),
    # Absolute path + default_domain
    ("/img/logo.png", "https://example.com/img/logo.png"),  # when default_domain="example.com"
]
```

* Lowercasing scheme/host, uppercase escapes, dot‑segments, and default `/` are covered by the README’s “what it does” list and RFC 3986. ([PyPI][1])
* IDN behavior depends on IDNA2008/UTS46 via `idna`. Exact punycode will be stable, but test with your inputs. ([GitHub][5])
* Query filtering examples mirror the README behavior (dropping `utm_*` when `filter_params=True`). ([PyPI][1])

> **Tip.** Pin `url-normalize` (e.g., `==2.2.*`) in environments where canonicalization stability is critical, and record expected outputs in golden tests.

---

## 11) When not to use url‑normalize alone

* **Canonicalization for deduping across a specific site** (e.g., remove `index.html`, strip trailing slashes only on known directories, merge vanity aliases) — that is application/site‑specific logic beyond spec normalization. See RFC 3986 notes that some normalizations may *change* semantics; prefer explicit rules. ([Wikipedia][9])
* **Security filtering/sanitization** (e.g., blocking traversal or dangerous schemes). Do validation/sanitization separately and *before* using normalized URLs for routing or storage.

---

## 12) Usage snippets for common refactors

**A. Replace “strip UTM & normalize”**

```python
def canonicalize(url: str) -> str:
    return url_normalize(url, filter_params=True)   # preserves param order; trims trackers
```

(You may further *allowlist* params if you know what must remain.) ([PyPI][1])

**B. Replace “make absolute from path”**

```python
def to_absolute(path_or_url: str, domain: str, scheme: str = "https") -> str:
    return url_normalize(path_or_url, default_domain=domain, default_scheme=scheme)
```

([PyPI][1])

**C. Preserve non‑default ports and drop only defaults**

No additional code needed—this is the library default. Just be mindful of which **scheme** you chose for missing schemes. ([PyPI][1])

---

## 13) CLI quick reference (for batch jobs)

```bash
# Normalize a list in a file, dropping tracking params and keeping only “page,id”
xargs -I{} url-normalize -f -p page,id "{}" < urls.txt

# Force http as the default scheme (instead of the library default https)
url-normalize -s http "www.example.com"
```

Flags & behavior per v2.1.0 notes. ([GitHub][6])

---

## 14) Alternatives / complements

* **`yarl`**: a rich URL object model for building/modifying URLs, not primarily a normalizer. You can combine it with url‑normalize if you need structured edits then a final canonical string. ([PyPI][10])
* **RFC‑3986‑focused libs** (e.g., `rfc3986`): give fine‑grained control/validation but lack the 2.x parameter filtering and convenience defaults this library offers. Align your choice with your goals (validation vs normalization vs canonicalization).

---

## 15) Source of truth & where to check behaviors

* **Project README (features, examples, usage)** — the best first stop. ([PyPI][1])
* **Release notes** — use them to understand 2.x behavior changes (default `https`, no query sorting, IDNA2008/UTS46). ([GitHub][5])
* **RFC 3986/3987** — reference sections for normalization rationales (case, percent‑encoding, dot‑segments) and internationalization. ([IETF Datatracker][2])

---

### Bottom line

If your custom code already does: (1) lowercasing scheme/host, (2) percent‑encoding normalization, (3) dot‑segment cleanup, (4) default port dropping, (5) IDN handling, and (6) optional tracker stripping—**you can replace most/all of it** with a single call to `url_normalize(...)`, plus a small wrapper for any site‑specific rules and (if needed) query sorting/deduping. The 2.x series is explicit about **not** reordering query params, and about **defaulting to HTTPS**, so update your tests accordingly. ([PyPI][1])

If you want, I can turn your existing normalization rules into a few `url_normalize(...)` wrappers and produce a compact test suite mirroring your current behavior.

[1]: https://pypi.org/project/url-normalize/ "url-normalize · PyPI"
[2]: https://datatracker.ietf.org/doc/html/rfc3986?utm_source=chatgpt.com "RFC 3986 - Uniform Resource Identifier (URI): Generic ..."
[3]: https://pypistats.org/packages/url-normalize "
        PyPI Download Stats
    "
[4]: https://github.com/niksite/url-normalize/releases/tag/v2.2.1 "Release v2.2.1 · niksite/url-normalize · GitHub"
[5]: https://github.com/niksite/url-normalize/releases/tag/2.0.0 "Release 2.0.0 · niksite/url-normalize · GitHub"
[6]: https://github.com/niksite/url-normalize/releases/tag/v2.1.0 "Release v2.1.0 · niksite/url-normalize · GitHub"
[7]: https://github.com/niksite/url-normalize "GitHub - niksite/url-normalize: URL normalization for Python"
[8]: https://pypi.org/project/url-normalize/1.4.0/ "url-normalize · PyPI"
[9]: https://en.wikipedia.org/wiki/URI_normalization?utm_source=chatgpt.com "URI normalization"
[10]: https://pypi.org/project/yarl/?utm_source=chatgpt.com "yarl"
