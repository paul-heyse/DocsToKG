Perfect—here’s a **granular, code-free implementation plan** for **PR 3** to **rewrite `validate_url_security` internals** while **keeping every call-site unchanged**. I’ll anchor the steps to your current call paths and config so an AI agent can implement directly.

---

# North-star (what “done” looks like)

* `validate_url_security(url, http_config)` **keeps the exact signature and return type** (returns a normalized, policy-compliant URL string), with no call-site edits required. It is invoked by planner metadata probes and download execution before any HTTP I/O, including redirects.
* `_populate_plan_metadata` already passes the **full** HTTP configuration (not just an allowlist) and stores the secure URL back into the plan—this remains unchanged.
* The implementation enforces:

  * **RFC-3986 parsing & normalization** (strict scheme/authority parsing),
  * **IDN → ASCII (punycode)** canonicalization,
  * **registrable-domain allowlisting** (PSL-based, with wildcard suffix support),
  * **port controls** (global and per-host),
  * **TLS policy** (HTTP→HTTPS upgrading by default for allowlisted hosts, with explicit config to allow plain HTTP),
  * **DNS classification** (public vs private/loopback/multicast/reserved), with **strict/lenient** modes,
  * **safe redirects** via your redirect-audit helper, which already re-invokes the validator on each hop.

---

# Where to implement (module location & imports)

* **Edit the function where it is currently defined/exported** in your OntologyDownload package (today this lives under the “io/network” surface and is imported by planner and downloader code). Keep the name and import surface stable so `ontology_download.py` and `planning.py` continue to call it without changes.

> Context: `_populate_plan_metadata` already uses `validate_url_security(planned.plan.url, http_config)` and writes the validated URL back to the plan; don’t touch those call-sites.

---

# Inputs, outputs, and invariants

* **Input:** `url: str`, `http_config: Optional[DownloadConfiguration]`.
  Relevant config fields to honor (all already exist):

  * `allowed_hosts` (supports exact domains, wildcard suffixes, optional per-host ports, and **IP literals**),
  * `allowed_ports` (global), `allowed_port_set()` helper,
  * `allow_private_networks_for_host_allowlist` (allow private IPs when host is allowlisted),
  * `allow_plain_http_for_host_allowlist` (allow HTTP without upgrade for allowlisted hosts),
  * `strict_dns` (fail hard on DNS failure).
* **Output:** a **normalized, safe URL** string.
* **Exceptions:**

  * `PolicyError` for scheme/host/port violations or disallowed redirects,
  * `ConfigError` for DNS failures in strict mode or private-network resolution when not allowed. (These semantics already exist—preserve them.)

---

# Step-by-step rewrite plan (no call-site changes)

## 1) Parsing & canonicalization (RFC-3986, IDN, authority)

1. **Parse strictly**: construct a URI model and **reject** anything not `http` or `https` at the outset; do not tolerate missing netloc or embedded credentials (`user:pass@`).
2. **Lowercase scheme & host**; **strip trailing dot** from host.
3. **IDN canonicalization**: convert Unicode hostname to **punycode** (IDNA), but retain the normalized ASCII form for allowlist checks and DNS.
4. **Normalize path**: remove dot-segments, preserve percent-encoded octets, and avoid collapsing within query/fragment.
5. **Recompose** a normalized URL object (string form) for downstream use.

> Keep the top-level function signature and return type; all the above steps happen **inside** `validate_url_security`. Callers remain unchanged.

## 2) Host allowlist evaluation (domains, suffixes, IP literals, per-host ports)

1. **Load allowlist** via `http_config.normalized_allowed_hosts()` → returns `(exact_domains, wildcard_suffixes, per_host_ports, ip_literals)`. If `allowed_hosts` is `None`, treat as “no allowlist” (i.e., apply public-network + TLS policies only).
2. **Match logic** (registrable domain aware):

   * If host is an **IP literal**, match against `ip_literals` and, if matched, mark the URL as **allowlisted**.
   * If host is a **domain**, check either **exact** match or **suffix** match (`*.example.org`), where a suffix match means `host == suffix or host.endswith("." + suffix)`. If matched, mark as **allowlisted**. If an allowlist exists and no match, **raise `PolicyError`**.
3. **Per-host ports:** if the host appears in `per_host_ports`, use those; otherwise fall back to `allowed_port_set()` (e.g., `{80, 443}` or user-supplied list). Reject non-allowed ports with `PolicyError`.

## 3) TLS policy (HTTP→HTTPS upgrades; explicit plain-HTTP opt-in)

1. **If scheme is `http`:**

   * **If allowlisted** and `allow_plain_http_for_host_allowlist` is **false**, **upgrade** to HTTPS in place and log a warning (existing behavior).
   * **If allowlisted** and the flag is **true**, **permit HTTP**, log a warning, and **do not** upgrade.
   * **If not allowlisted**, reject non-HTTPS outright.
2. **If scheme is not `https` after the above**, and plain HTTP is not explicitly allowed, **raise `PolicyError`**.

## 4) DNS classification & private-network guards

1. **Determine “allow private networks”**:

   * If the allowlist **explicitly matched** an IP literal, set **allow_private_networks = True**.
   * Else, inherit from `allow_private_networks_for_host_allowlist` (false by default).
2. **DNS resolution:** resolve the ASCII host with a cached `getaddrinfo` helper (you already reference cache/strict behavior in tests & code).

   * If resolution **fails** and `strict_dns=False`, **warn and continue**. If `strict_dns=True`, **raise `ConfigError`**.
3. **If host is an IP literal**, examine it directly; otherwise, **evaluate resolved IPs**:

   * When **allow_private_networks is False**, **reject** loopback, link-local, private, multicast, and reserved ranges.
   * When **True**, **permit** those ranges (but still enforce your port/TLS rules above).

## 5) Redirect safety (no changes to callers)

* Your **`request_with_redirect_audit`** helper already **revalidates each redirect target** with `validate_url_security`. Keep that flow; with the stricter validator, unsafe hops will be blocked early.
* Where the call-sites pass **“already validated”** hints (e.g., `assume_url_validated=True` during probes or `url_already_validated=True` in download stream), they’ll continue to bypass redundant work while preserving end-to-end safety. **Do not change these call-sites.**

## 6) Logging & telemetry (keep existing fields)

* Reuse the established structured logging pattern with `extra={"stage": "plan" | "download", ...}` for:

  * **HTTP→HTTPS upgrade performed**,
  * **plain HTTP allowed for allowlisted host**,
  * **private network allowed because of config or IP literal**,
  * **DNS failure** (warning if not strict; error if strict).
* The planner and downloader already include rich telemetry; add/keep fields but **do not** change message keys expected by tests.

---

# Implementation sequencing (inside this PR)

1. **Normalize the inputs and parse** (scheme, host, port, userinfo).
2. **IDN → punycode** (ASCII host) and **lowercase**.
3. **Compute allowlist match** via `normalized_allowed_hosts()` and mark **allowlisted** vs not. Enforce **per-host ports** and **global allowed ports**.
4. **Apply TLS policy** (upgrade or allow HTTP per flags).
5. **Resolve and classify network** (respect `strict_dns`, allow-private toggles).
6. **Return the recomposed, normalized URL** string.

---

# Test plan (augment existing tests; keep public behavior)

Add/confirm the following unit scenarios under `tests/ontology_download/test_url_security.py`:

* **Scheme enforcement**

  * `ftp://…`, `file://…`, `data:…` → **PolicyError**.
* **IDN normalization**

  * Unicode host (e.g., contains accents / non-ASCII) → converted to punycode; allowlist works against ASCII form.
* **Allowlist behavior**

  * Exact match, wildcard suffix, and per-host port acceptance; missing match → **PolicyError**.
  * **IP literal allowlist** case: `10.0.0.7` in allowlist → permitted (and, if `http`, allowed **only** when `allow_plain_http_for_host_allowlist` is true).
* **TLS policy**

  * Allowlisted HTTP **upgrades** to HTTPS by default (assert scheme changed).
  * With `allow_plain_http_for_host_allowlist=True`, allowlisted HTTP **remains HTTP**.
* **DNS classification**

  * Public A record → allowed.
  * Private/loopback result with `allow_private_networks_for_host_allowlist=False` → **ConfigError**; with **True** → allowed.
  * **`strict_dns=True`** with synthetic resolver error → **ConfigError**; with **False** → warning and proceed.
* **Redirect audit path**

  * Planner probe and downloader redirect to disallowed host → blocked by validator (indirectly exercised in `request_with_redirect_audit` tests).

(You already added coverage for HTTP upgrade and private-resolution rejection; extend it for the new flags as above.)

---

# Documentation updates (same PR)

* In **settings docs**, ensure the semantics for:

  * `allow_private_networks_for_host_allowlist` (default **False**) and
  * `allow_plain_http_for_host_allowlist` (default **False**)
    are clearly described with examples.
* In **developer notes**, state that **redirects are individually validated** by `request_with_redirect_audit` and how the “already validated” hints suppress redundant checks.

---

# Definition of done (quick checks)

* [ ] `grep -R "validate_url_security" src/DocsToKG/OntologyDownload` shows **one authoritative implementation**; any legacy shadow versions are removed or re-export the new one.
* [ ] Planner `_populate_plan_metadata` still calls `validate_url_security(planned.plan.url, http_config)` and **persists** the returned URL; no call-site edits.
* [ ] `request_with_redirect_audit` continues to invoke the validator **per hop**; tests pass.
* [ ] Tests cover HTTP upgrade, plain-HTTP opt-in, private vs public resolution, strict DNS, IP literal allowlisting, wildcard suffix allowlisting, and port guards.
* [ ] No change to public imports or CLI behavior; planner/download telemetry still emits the expected `extra` fields.

---

## Why this specific design

* It aligns with the **fixed parameter passing** you already landed in `_populate_plan_metadata` (full `http_config`, not a subfield), so no new caller churn.
* It respects your **new flags** for **private networks** and **plain HTTP on allowlisted hosts**, which you added to strengthen defaults while offering operator escape hatches.
* It plays cleanly with your **redirect audit** path and the **“assume validated”** hints to avoid redundant validation—security where needed, speed where safe.

If you want, I can follow this with a tiny checklist mapping **each bullet** above to concrete edits (file + function) and point out any legacy helpers that can be deleted once the validator is consolidated.
