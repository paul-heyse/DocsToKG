Absolutely—here’s a **commit-ready checklist** focused solely on **rewriting `validate_url_security` internals** (no call-site changes). It’s structured so an AI programming agent can execute it end-to-end without guessing. No code—just precise actions and acceptance checks.

---

# PR-3 Checklist — Rewrite `validate_url_security` Internals (No Call-Site Changes)

## 0) Scope & Signature Freeze

* [ ] Confirm the function lives where your tree expects it for this PR (e.g., `src/DocsToKG/OntologyDownload/ontology_download.py`; if you’ve already consolidated into `io/network.py`, keep the same public import path).
* [ ] Do **not** change the signature:
  `validate_url_security(url: str, http_config: Optional[DownloadConfiguration]) -> str`
* [ ] Preserve exception taxonomy used by callers: raise **PolicyError** for policy violations (scheme/host/port/redirect), **ConfigError** for environmental/ DNS issues when `strict_dns=True`.
* [ ] Keep **all call-sites unchanged** (e.g., `_populate_plan_metadata` already passes the full `http_config`).

## 1) Inputs/Config Contract (read-only)

* [ ] Read and honor these `DownloadConfiguration` fields (names as in your codebase):

  * [ ] `allowed_hosts` (supports exact domains, wildcard suffixes, per-host ports, and IP literals).
  * [ ] `allowed_port_set()` (global allowed ports).
  * [ ] `allow_private_networks_for_host_allowlist` (bool).
  * [ ] `allow_plain_http_for_host_allowlist` (bool).
  * [ ] `strict_dns` (bool).

## 2) Library & API Readiness (no behavior change)

* [ ] Ensure the following libraries are available for internal use (already pinned in prior work): **rfc3986** (strict parse), **idna** (IDN), **tldextract/publicsuffix2** (registrable domain), **ipaddress** (IP classifying).
* [ ] No new public dependencies or flags introduced by this PR.

## 3) Parse & Canonicalize (strict RFC 3986)

* [ ] Parse as an **absolute** URI; reject missing **scheme** or **netloc**.
* [ ] **Allow only `http` and `https`**; reject all others (e.g., `ftp:`, `file:`, `data:`).
* [ ] **Forbid userinfo** (`user:pass@`) in netloc.
* [ ] Normalize host:

  * [ ] Lowercase scheme and host; trim any trailing dot from host.
  * [ ] Convert Unicode host → **punycode** (IDNA 2008); retain ASCII form for checks.
  * [ ] Reject invisible/mixed-script host labels (basic homograph safety).
* [ ] Normalize path:

  * [ ] Remove dot-segments (`.`/`..`), keep leading slash semantics.
  * [ ] Preserve percent-encoding as given (no lossy decoding/re-encoding).
* [ ] Disallow backslashes in authority/path; normalize to forward slashes if necessary before checks.
* [ ] Preserve query/fragment verbatim (no reordering, no decoding).

## 4) Allowlist Decision (domain/IP + ports)

* [ ] Compute allowlist structures once (exact domains, wildcard suffixes, per-host ports map, IP-literal set).
* [ ] Determine host kind:

  * [ ] **IP literal** (IPv4/IPv6 bracketed) → compare against IP allowlist set.
  * [ ] **Domain name** → check **exact** match or **suffix** match (e.g., `*.example.org`).
* [ ] If `allowed_hosts` is present and **no match** → **PolicyError**.
* [ ] Determine **effective allowed ports**:

  * [ ] Start from `allowed_port_set()` (global).
  * [ ] If a per-host port set exists (from allowlist), use it in addition.
  * [ ] Resolve the URL port (`explicit` or `default(80/443)`) and **enforce** membership; otherwise **PolicyError**.
* [ ] Canonicalize port in output: **omit default port** (80 for http, 443 for https).

## 5) TLS Policy (upgrade vs allow plain HTTP)

* [ ] If scheme is `http`:

  * [ ] If host is allowlisted **and** `allow_plain_http_for_host_allowlist=True` → **permit** HTTP (log a warning).
  * [ ] Else if host is allowlisted but flag is **False** → **upgrade to `https`** in place (log an info/warn).
  * [ ] Else (not allowlisted) → **reject** (PolicyError).
* [ ] After upgrade step, if scheme is still not `https` and plain HTTP is not allowed → **PolicyError**.

## 6) DNS Resolution & Network Class (respect `strict_dns`)

* [ ] If host is an **IP literal**, skip DNS; classify directly via **ipaddress**.
* [ ] Else **resolve** with a cached `getaddrinfo` helper (short TTL within process):

  * [ ] If resolution fails and `strict_dns=True` → **ConfigError**.
  * [ ] If resolution fails and `strict_dns=False` → **warn** and proceed.
* [ ] Classify each resolved address:

  * [ ] Deny **loopback, link-local, private, multicast, reserved** when **not** allowed.
  * [ ] Compute `allow_private_networks`:

    * [ ] **True** if (a) the allowlist matched an **IP literal**, or (b) `allow_private_networks_for_host_allowlist=True`.
    * [ ] **False** otherwise.
  * [ ] If any address violates the above policy → **ConfigError** (environment) or **PolicyError** (if you treat network class as policy; pick the same one your code uses today and keep it consistent).

## 7) Recompose & Return

* [ ] Rebuild a **normalized** URL string:

  * [ ] Lowercased scheme/host (ASCII punycode).
  * [ ] Bracketed IPv6 when present.
  * [ ] Default port omitted; explicit non-default port preserved.
  * [ ] Cleaned path with dot-segments removed.
  * [ ] Original query/fragment preserved.
* [ ] Return this normalized URL string to callers.

## 8) Logging & Telemetry (stable keys)

* [ ] Emit structured logs (no key renames) for these events:

  * [ ] `validator.scheme_upgrade` (from http→https, host, reason).
  * [ ] `validator.allow_plain_http` (host, reason).
  * [ ] `validator.allowlist_match` (`exact|suffix|ip|none`).
  * [ ] `validator.port_policy` (port, allowed set, per-host override boolean).
  * [ ] `validator.dns_resolution` (strict flag, success/failure).
  * [ ] `validator.address_classification` (public/private/loopback/etc).
* [ ] Keep log volume bounded (debug for per-attempt details; info/warn for policy decisions).

## 9) Tests (augment; no public API changes)

Create/adjust **unit tests** targeting only this function’s behavior:

**Scheme & Netloc**

* [ ] Reject non-HTTP(S) schemes (`ftp:`, `file:`, `data:`).
* [ ] Reject missing netloc or userinfo in netloc.

**IDN & Host Canonicalization**

* [ ] Unicode host → punycode ASCII; allowlist compares on ASCII form.
* [ ] Reject mixed/invisible host labels (basic homograph filters).

**Allowlist Matches**

* [ ] Exact domain accepted; wildcard suffix accepted; non-match rejected (when allowlist provided).
* [ ] IP literal accepted only when present in allowlist IP set.

**Port Policy**

* [ ] Accept default ports; omit from output.
* [ ] Accept per-host custom ports when configured.
* [ ] Reject disallowed ports.

**TLS Policy**

* [ ] Allowlisted `http://` upgraded to `https://` when plain HTTP flag is false.
* [ ] Allowlisted `http://` permitted as HTTP when plain HTTP flag is true.
* [ ] Non-allowlisted `http://` rejected.

**DNS & Network Class**

* [ ] Public DNS result accepted.
* [ ] Private/loopback result rejected when `allow_private_networks_for_host_allowlist=False`.
* [ ] Same accepted when `allow_private_networks_for_host_allowlist=True` and host is allowlisted.
* [ ] Strict DNS failure → ConfigError; non-strict → warning and continue.
* [ ] IPv6 literal bracket handling (with and without explicit port).

**Path & Query**

* [ ] Dot-segment removal validated.
* [ ] Query/fragment preserved; no reordering.

**Performance/Determinism**

* [ ] Cached resolver used on repeated calls (validate memoization via timing/call count or instrumentation).

## 10) Helper Consolidation (single source of truth)

* [ ] Ensure **one authoritative implementation** of:
  `_cached_getaddrinfo`, `_rebuild_netloc`, `_enforce_idn_safety`, and the allowlist normalizer used by the validator.
* [ ] Delete (or re-export) any **duplicates** in adjacent modules (e.g., `io_safe.py`) so tests and prod code patch the **same** module.

## 11) Documentation & Comments

* [ ] Update the function docstring to enumerate: inputs, outputs, raised exceptions, and policy semantics (HTTP upgrade rules, private network toggles, per-host ports).
* [ ] Add a short developer note (in comments) explaining why **registrable-domain** checks (PSL) are used and how wildcard matching is performed.

## 12) Final Acceptance

* [ ] `grep` shows **single** authoritative `validate_url_security` (others removed/re-exported).
* [ ] All tests that previously called this function **still pass without call-site edits**.
* [ ] New unit tests for edge cases pass (see §9).
* [ ] Logs appear with stable keys and expected messages in plan/download flows.
* [ ] Example inputs confirm deterministic normalization in outputs (e.g., default ports omitted, http→https upgrades visible when applicable).

---

If you want, I can turn this checklist into a PR description you can paste directly into GitHub, with each checkbox mapped to a file/function touchpoint in your tree.
