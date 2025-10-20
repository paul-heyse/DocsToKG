Amazing—here’s the **follow-up you asked for**: a **surgical checklist of concrete edits** (file + function, in execution order), plus a **delete list of helpers** that can go away once PR-3 lands. I’ve anchored each item to what’s already in your tree so an AI agent can implement without guesswork.

---

# A. Exact edits to land in PR-3 (no call-site changes)

## 1) Make `io/network.py` the **authoritative home** of URL validation

* **Keep** the public function name and signature:

  * `validate_url_security(url: str, http_config: Optional[DownloadConfiguration]) -> str`
  * Behaviour: return a **normalized, policy-compliant** URL or raise `PolicyError/ConfigError` (as today via the API wrapper).

* **Inside** `validate_url_security` (same function, updated internals):

  * Use **separate booleans** `allow_private_networks` and `allow_plain_http` (replace any old `allow_private`). This split is already reflected in recent diffs; keep that shape.
  * When a **host is allowlisted as an IP literal**, set *both*:

    * `allow_private_networks = True`
    * `allow_plain_http = True` (explicitly allows HTTP for that IP when configured).
  * For **HTTP**:

    * If `allow_plain_http` is true → **permit HTTP** (log a warning).
    * Else → **upgrade to HTTPS** (in-place scheme swap).
    * After upgrade step: if still not HTTPS and `allow_plain_http` is false → **raise**.
  * **Port policy**:

    * Default from `DownloadConfiguration.allowed_port_set()`; allow host-specific ports from `normalized_allowed_hosts()` per-host map. Reject ports not in either set. (Port checks are in the current function body; keep them and align to the new allowlist tuple shape.)
  * **DNS classification**:

    * Use `_cached_getaddrinfo` (see §2) to resolve host.
    * If result is private/loopback/multicast/reserved and `allow_private_networks` is **False** → **raise**.
    * Honour `strict_dns`: on resolution failure raise `ConfigError` when `True`, otherwise warn and continue.

> Why these mechanics: they match the refined flags shipped in `DownloadConfiguration` and the recently-merged validator behaviour/fixtures.

---

## 2) Consolidate IDN + DNS helpers under `io/network.py`

Create/keep these private helpers in **`io/network.py`** (and **remove the duplicates elsewhere**; see Delete List):

* `_enforce_idn_safety(host: str) -> None` (reject mixed-script homographs, invisible code points)

* `_rebuild_netloc(parsed, ascii_host) -> str` (rebuild `netloc` sans userinfo with bracketed IPv6)

* `_cached_getaddrinfo(host: str) -> List[Tuple]]` with a short-lived cache **and** a stub layer for tests:

  * Public test hooks:

    * `register_dns_stub(host: str, handler: Callable[[str], List[Tuple]]) -> None`
    * `clear_dns_stubs() -> None` (and evict cached stubbed lookups)
      These are already present/used in tests—keep their names and semantics.

* Ensure **test fixtures** patch the helpers on the **network** module (not `io_safe`). Your tests already moved in that direction (e.g., `network_mod.register_dns_stub`, `clear_dns_stubs`).

---

## 3) Keep API façade stable (no caller changes)

* **Do not** change imports or the public wrapper. `api.validate_url_security` must continue to call the internal network impl and normalize `PolicyError` → `ConfigError`.

---

## 4) Align with new **config flags** (already present)

* Confirm `DownloadConfiguration` exposes:

  * `allow_private_networks_for_host_allowlist: bool`
  * `allow_plain_http_for_host_allowlist: bool`
  * `normalized_allowed_hosts()` returns `(exact, suffixes, per_host_ports, ip_literals)`
  * `allowed_port_set()` returns the default/global allowed ports
* These shapes are live; ensure the validator reads and applies them as in §1.

---

## 5) Tests: confirm the moved hooks + behaviours

* The current tests already encode the expected behaviours (HTTP upgrade opt-in, private-IP blocking, DNS strictness) and call the **network-scoped** stubs. Keep them passing after consolidation.

---

# B. Delete (or fold) these helpers after PR-3

> Goal: **one** source of truth for URL validation and DNS helpers; remove shadow copies and renamed leftovers.

1. **`io_safe.py` duplicates** (delete or re-export from `io/network.py`):

   * `validate_url_security(...)` — remove; the authoritative copy lives in `io/network.py`.
   * `_cached_getaddrinfo`, `_rebuild_netloc`, `_enforce_idn_safety` — move/keep in `io/network.py`; **delete** from `io_safe.py`.
     Evidence of duplication and prior presence in `io_safe.py` is visible in earlier diffs showing these definitions; tests have already migrated to patch the **network** module.

2. **Old boolean and branch names in `validate_url_security`**:

   * Any remaining uses of `allow_private` should be removed/renamed to the explicit pair `allow_private_networks` and `allow_plain_http`. (Recent patches already reflect the new names; this is mostly a **dead-code sweep**.)

3. **Legacy allowlist tuple handling**:

   * If any code still assumes `normalized_allowed_hosts()` returns `(exact, suffixes, ports)` (3-tuple), delete/upgrade it to the 4-tuple with `ip_literals`. You have the updated 4-tuple today; remove adapters.

4. **Redundant post-request validation passes**:

   * With **`request_with_redirect_audit`** validating each hop and **download_entry** now accepting `url_already_validated=True`, purge any leftover “re-validate final URL unconditionally” paths that aren’t already guarded by the `last_validated_url` check. (The audited flow and the “validated flag” are merged; keep that and remove dup checks.)

5. **CLI / planner stray metadata pre-flights** (already mostly removed):

   * Confirm there’s no resurrected `_collect_plan_metadata` / `_extract_response_metadata` (they were deleted when `_populate_plan_metadata` became authoritative). If any stragglers exist, delete.

---

# C. Micro-checklist (agent can tick these while editing)

1. **`src/DocsToKG/OntologyDownload/io/network.py`**

   * [ ] `validate_url_security`: ensure **explicit** `allow_plain_http` and `allow_private_networks` handling; keep HTTPS upgrade logic.
   * [ ] Apply **port policy** using `allowed_port_set()` + per-host overrides.
   * [ ] Use `_cached_getaddrinfo` and raise under `strict_dns=True`; otherwise warn.
   * [ ] Keep/implement `register_dns_stub` and `clear_dns_stubs`; ensure cache eviction on clearing.

2. **`src/DocsToKG/OntologyDownload/io/__init__.py`**

   * [ ] Re-export `validate_url_security` **from** `io.network` (authoritative) so the public import surface (`from DocsToKG.OntologyDownload import io as io_mod`) continues working in tests/CLI. (Your `io/__init__` already composes network exports—maintain that.)

3. **`src/DocsToKG/OntologyDownload/api.py`**

   * [ ] **No change**; wrapper stays, still imports `validate_url_security as _validate_url_security` from `.io`. (Verifies no caller churn.)

4. **`src/DocsToKG/OntologyDownload/settings.py`**

   * [ ] Confirm flags are present and doc-strings match behaviour (HTTP opt-in is separate from private network opt-in).

5. **Tests**

   * [ ] `tests/ontology_download/test_url_security.py` imports/patches **`network_mod`**; ensure stubs still work and cache purging test passes.
   * [ ] Planner/download tests rely on `request_with_redirect_audit` + `url_already_validated=True`; confirm they **don’t** re-validate redundantly.

---

# D. “Gotchas” we’ve already de-risked in your tree

* **Redirect audit** path validates every hop and stamps `response.validated_url`; planners now **read that** instead of re-validating the `response.url`. Keep that behaviour.
* **Flags split** (HTTP vs private network) is already reflected in tests/docs; validator must continue to implement that **matrix**:

  * HTTP allowed **only** with `allow_plain_http_for_host_allowlist=True`
  * Private DNS resolution allowed **only** with `allow_private_networks_for_host_allowlist=True`
  * These are **independent** toggles.

---

## TL;DR Delete list (once PR-3 merges)

* `io_safe.py`: `validate_url_security`, `_cached_getaddrinfo`, `_rebuild_netloc`, `_enforce_idn_safety` (all superseded by `io/network.py`).
* Any lingering uses of `allow_private` → remove/rename to `allow_private_networks` + `allow_plain_http`.
* Any 3-tuple handling of `normalized_allowed_hosts()` → delete/upgrade to 4-tuple with `ip_literals`.
* Redundant post-request URL re-validation where `request_with_redirect_audit` already covers the hop and sets `validated_url`.

If you’d like, I can package this into a **single PR description** with a commit checklist (boxes you can tick in the body) and add a short **“risk & rollback”** note tailored to your CI gates.
