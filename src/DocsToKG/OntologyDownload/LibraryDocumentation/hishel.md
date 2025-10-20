Below is a **refactorer‑friendly “complete reference”** to **pluggy**, organized so an AI agent (or a human) can replace custom plugin systems with first‑class primitives from the library.

---

## 0) What pluggy is (and why to use it)

* **Purpose.** Pluggy is a minimalist, production‑ready framework for **plugin discovery, registration and hook calling**. It powers pytest, tox, devpi, kedro and more. In a host program you define **hook specifications** (APIs); plugins implement those hooks; pluggy orchestrates calling them. ([Pluggy][1])

* **Mental model.** Think of a **host ↔ plugins** contract:

  * Host defines *what can be extended* (hookspecs) and *where to call them*.
  * Plugins provide *how to extend* (hookimpls).
  * Pluggy supplies a **PluginManager**, a **HookRelay** for calling hooks, and **markers** that connect them. ([Pluggy][1])

---

## 1) Core pieces & terminology

* **`HookspecMarker(project_name)`** and **`HookimplMarker(project_name)`**
  Decorators identifying *specifications* and *implementations*. The `project_name` **must match** the `PluginManager("project_name")` used by the host. ([Pluggy][1])

* **`PluginManager(project_name)`**
  Registers plugins, loads entry‑point plugins, exposes `hook.<name>(...)` to call hooks, and offers tracing/monitoring. Key API includes: `add_hookspecs`, `register`, `unregister`, `check_pending`, `load_setuptools_entrypoints`, `enable_tracing`, `add_hookcall_monitoring`, `subset_hook_caller`, and inspection helpers like `list_name_plugin`, `get_plugins`, `get_plugin`, `get_canonical_name`. ([Pluggy][2])

* **`HookRelay` and `HookCaller`**
  `pm.hook` (a `HookRelay`) holds one `HookCaller` per defined hook. You invoke hooks as **keyword‑only** calls: `pm.hook.myhook(arg1=..., arg2=...)`. `HookCaller` also supports `call_historic`, `call_extra`, and reveals registered `HookImpl`s. ([Pluggy][2])

---

## 2) Minimal, “show me” example

```python
import pluggy

hookspec = pluggy.HookspecMarker("myproject")
hookimpl = pluggy.HookimplMarker("myproject")

class Spec:
    @hookspec
    def myhook(self, a: int, b: int):
        "Add or subtract numbers."

class P1:
    @hookimpl
    def myhook(self, a, b): return a + b

class P2:
    @hookimpl
    def myhook(self, a, b): return a - b

pm = pluggy.PluginManager("myproject")
pm.add_hookspecs(Spec)
pm.register(P1()); pm.register(P2())

# Always call with keyword args:
print(pm.hook.myhook(a=1, b=2))  # -> [ -1, 3 ] (LIFO by default)
```

Default call order is **LIFO by registration**, so `P2` is called before `P1` and results are collected in a list. ([Pluggy][1])

---

## 3) Designing your plugin API (hook specifications)

Define specs on a module or class and add them to the manager with `add_hookspecs`. Specs are validated against implementations. You can evolve specs safely using these options: ([Pluggy][1])

* **Opt‑in arguments** (compatibility): hook implementations may accept **fewer args** than the spec (never more). This lets you add new parameters to a spec without breaking existing plugins. ([Pluggy][1])

* **`firstresult=True`**: stop at the **first non‑`None`** result and return that single value (instead of a list). Useful when only one plugin needs to answer. Hook wrappers still run. ([Pluggy][1])

* **`historic=True`**: the hook may be **called before** all plugins are registered; late‑registered plugins will be “replayed” immediately on registration. Results are delivered via a `result_callback` and not returned to the caller. Do **not** combine with `firstresult`. ([Pluggy][1])

* **Deprecation nudges** (since 1.5):
  `@hookspec(warn_on_impl=Warning(...))` warns whenever the hook is implemented.
  `@hookspec(warn_on_impl_args={"arg": Warning(...)})` warns when implementations request deprecated **parameters**. Handy for staged migrations. ([Pluggy][1])

* **Validation**: by default implementations **need not** have a matching spec. If you want to enforce, call `pm.check_pending()`; alternatively mark individual implementations with `@hookimpl(optionalhook=True)` to skip strict validation. ([Pluggy][1])

* **Spec name matching**: normally by same function name; or override via `@hookimpl(specname="your_spec")`. ([Pluggy][1])

---

## 4) Implementing and **ordering** hook implementations

* **Registration targets**: a plugin is a module or object whose functions/methods are decorated with `@hookimpl`. They must be hashable. Register with `pm.register(plugin, name=None)`. You can unregister later. ([Pluggy][1])

* **Call order**: default is **LIFO** (last registered → first called). You can bias with `tryfirst=True` or `trylast=True`; within each category, order is still LIFO. ([Pluggy][1])

* **New‑style wrappers** (`wrapper=True`, Pluggy ≥1.1): write a **generator** that `yield`s once. The value sent into the `yield` is the accumulated result (or an exception is thrown into the generator). You **return** a value (or raise) and that becomes the hook’s final result. Prefer this style. ([Pluggy][1])

* **Old‑style wrappers** (`hookwrapper=True`): also a generator, but you receive a `pluggy.Result` object from the `yield`, and must use `result.get_result()`, `result.force_result(...)` or `result.force_exception(...)`. Old style cannot return values directly; use `force_result`. Pluggy warns if teardown raises; use `force_exception` to adjust exceptions. ([Pluggy][1])

---

## 5) Calling hooks & collecting results

* **Keyword‑only calls**: `pm.hook.myhook(...)` requires **keyword arguments** matching the spec. ([Pluggy][2])

* **Result shape**: by default **list of results** (non‑`None` only). With `firstresult=True`, returns the first non‑`None` value. ([Pluggy][1])

* **Exceptions**: if a hookimpl raises, **further callbacks stop**; the exception is surfaced (wrappers see it and may handle/transform). ([Pluggy][1])

* **Historic calls**: `pm.hook.myhook.call_historic(kwargs=..., result_callback=...)` records the call and replays it for **later** registered plugins; the call site receives no return value (use the callback). ([Pluggy][1])

* **Ad‑hoc methods**: `HookCaller.call_extra(methods=[...], kwargs=...)` lets you include extra functions **just for this call**. ([Pluggy][2])

* **Subset of plugins**: `pm.subset_hook_caller("hookname", remove_plugins={...})` returns a `HookCaller` that excludes certain plugins for that call. ([Pluggy][2])

---

## 6) Plugin discovery, registration, and governance

* **Direct registration**: `pm.register(plugin, name=None)` returns the assigned plugin name; raises `ValueError` if already registered. `pm.unregister(plugin_or_name)` removes one. Inspection helpers: `list_name_plugin`, `get_plugins`, `get_plugin`, `get_canonical_name`, `get_name`, `has_plugin`, `is_registered`. ([Pluggy][2])

* **Entry‑point discovery**: `pm.load_setuptools_entrypoints(group, name=None)` loads third‑party plugins advertised under a given **entry point group** in package metadata. In your plugin’s `pyproject.toml` you’d declare, e.g.:

  ```toml
  [project.entry-points."myapp.plugins"]
  my_cool_plugin = "my_package.plugin_module"
  ```

  Then, the host does: `pm.load_setuptools_entrypoints("myapp.plugins")`. (pytest uses this to find external plugins.) ([Pluggy][1])

* **Block/unblock**: prevent a plugin name from loading with `pm.set_blocked(name)`; check with `pm.is_blocked(name)`; reverse with `pm.unblock(name)`. Useful for policy‑based allow/deny lists. ([Pluggy][1])

* **Spec enforcement**: run `pm.check_pending()` to fail if any registered implementations don’t match a spec and aren’t marked `optionalhook`. ([Pluggy][2])

---

## 7) Tracing, monitoring, and inspection

* **Turnkey tracing**: `pm.enable_tracing()` + `pm.trace.root.setwriter(print)` prints **who** was called and **with what**. Returns an undo function. ([Pluggy][1])

* **Custom monitoring**: `pm.add_hookcall_monitoring(before, after)` installs two callbacks for every hook call; you get the hook name, participating `HookImpl`s, kwargs, and a `Result` wrapper after the call. Also returns an undo function. ([Pluggy][1])

* **Low‑level introspection**: `pm.get_hookcallers(plugin)` returns the hook callers a specific plugin participates in. ([Pluggy][2])

---

## 8) Common refactors from bespoke systems → pluggy

> Below are patterns to translate typical in‑house plugin registries or callback systems.

1. **Dict of callback lists → hookspec + hookimpls**

   * Define a spec with the final signature you want (add extra args later—impls can accept fewer).
   * Replace `callbacks["event"].append(fn)` with `pm.register(MyPlugin())`. Call with `pm.hook.event_kw(name=...)`. ([Pluggy][1])

2. **First handler wins → `firstresult=True`**

   * If your dispatcher stops at the first handler that returns a value, mark the spec with `@hookspec(firstresult=True)` and call as usual. ([Pluggy][1])

3. **Around‑advice/middleware → wrappers**

   * Convert “before/after pipeline” code into **new‑style wrappers** (`wrapper=True`): write a single‑`yield` generator that can inspect/replace the result or propagate/transform exceptions. Prefer new‑style; keep old‑style (`hookwrapper=True`) only for older pluggy versions. ([Pluggy][1])

4. **Late plugin loading → `historic=True`**

   * If your system caches events for late listeners, mark the spec as historic and use `call_historic`. Connect newly loaded plugins via `pm.register(...)` and their implementations will be called immediately with the cached kwargs. ([Pluggy][1])

5. **Feature flags/tenant‑specific disabling → subset callers / blocking**

   * For per‑call exclusions: `pm.subset_hook_caller("hook", remove_plugins={...})`.
   * For policy‑level deny lists: `pm.set_blocked("name")`. ([Pluggy][2])

6. **Ad‑hoc one‑off callbacks → `call_extra`**

   * When you need to include a function just for one call (e.g., script‑level customizations), use `call_extra(methods=[...], kwargs=...)`. ([Pluggy][2])

---

## 9) Full worked example (host + third‑party plugin)

**Host library (`eggsample`)**

```python
# eggsample/hookspecs.py
import pluggy
hookspec = pluggy.HookspecMarker("eggsample")

@hookspec
def eggsample_add_ingredients(ingredients: tuple) -> list: ...
@hookspec
def eggsample_prep_condiments(condiments: dict) -> str | None: ...

# eggsample/__init__.py
import pluggy
hookimpl = pluggy.HookimplMarker("eggsample")  # re-export for plugin authors

# eggsample/host.py
import pluggy
from eggsample import hookspecs
pm = pluggy.PluginManager("eggsample")
pm.add_hookspecs(hookspecs)
pm.load_setuptools_entrypoints("eggsample")  # discover external plugins
def run():
    tray = {"steak sauce": 4}
    adds = pm.hook.eggsample_add_ingredients(ingredients=("egg",))
    comments = pm.hook.eggsample_prep_condiments(condiments=tray)
    return adds, tray, [c for c in comments if c]
```

**Third‑party plugin (`eggsample-spam`)** declares an entry point:

```toml
# pyproject.toml
[project]
name = "eggsample-spam"
dependencies = ["eggsample"]

[project.entry-points."eggsample"]
spam = "eggsample_spam"

# eggsample_spam.py
import eggsample
@eggsample.hookimpl
def eggsample_add_ingredients(ingredients):
    return ["lovely spam"]
@eggsample.hookimpl
def eggsample_prep_condiments(condiments):
    condiments["spam sauce"] = 42
    return "Condiments upgraded!"
```

The host’s `load_setuptools_entrypoints("eggsample")` picks this up automatically. ([Pluggy][1])

---

## 10) Ordering, result shaping & wrappers—by example

* **Bias order**:

  ```python
  @hookimpl(tryfirst=True)  # run before others
  def transform(...): ...
  @hookimpl(trylast=True)   # run after others
  def transform(...): ...
  ```

  Order is still LIFO within each category. ([Pluggy][1])

* **First result wins**:

  ```python
  @hookspec(firstresult=True)
  def choose_handler(req): ...
  ```

  Returns a single value (first non‑None). ([Pluggy][1])

* **New‑style wrapper**:

  ```python
  @hookimpl(wrapper=True)
  def transform(data):
      try:
          result = yield          # run inner impls, get result or raise
          return result + ["wrap"]
      except Exception as e:
          raise
  ```

  Prefer this on Pluggy ≥1.1. ([Pluggy][1])

* **Old‑style wrapper**:

  ```python
  @hookimpl(hookwrapper=True)
  def transform(data):
      outcome = yield
      items = outcome.get_result()
      outcome.force_result(items + ["wrap"])
  ```

  Use `force_result/force_exception`; don’t `return` values from old‑style wrappers. ([Pluggy][1])

---

## 11) Error handling & robustness

* If any implementation raises an exception, **remaining implementations are skipped**. Wrappers are given a chance to observe/transform the error; then it is raised to the caller. ([Pluggy][1])

* Old‑style wrappers that raise in teardown cause a **`PluggyTeardownRaisedWarning`** (added in 1.4); prefer new‑style wrappers or use `Result.force_exception`. ([Pluggy][3])

* The library enforces correct calling: **positional args are not allowed**—call hooks with keywords only. Violations raise `HookCallError`. ([Pluggy][2])

---

## 12) Discovery & packaging checklist

* **Define a unique entry point group** for your ecosystem (e.g., `"myapp.plugins"`).
* **Advertise plugins** via `pyproject.toml` under `[project.entry-points."<group>"]`.
* **Load them** with `pm.load_setuptools_entrypoints("<group>")`.
* **Control policy** with `set_blocked()/unblock()` to allow/deny by name. ([Python Packaging][4])

---

## 13) Observability for AI agents (instrumentation hooks)

* **Built‑in tracing**:

  ```python
  pm.trace.root.setwriter(print)
  undo = pm.enable_tracing()
  ```

  Produces structured events about hook calls. ([Pluggy][1])

* **Custom monitoring**:

  ```python
  def before(name, impls, kwargs): ...
  def after(result, name, impls, kwargs): ...
  undo = pm.add_hookcall_monitoring(before, after)
  ```

  `result` is a `pluggy.Result` wrapper; this is ideal for telemetry or replay logs. ([Pluggy][1])

---

## 14) Version awareness & typing

* **Recent changes** (stable docs):

  * **1.6.0 (2025‑05‑15):** dropped Python **3.8**; bug fixes to wrappers/results.
  * **1.5.0 (2024‑04‑19):** `warn_on_impl_args` for **parameter‑level deprecations**.
  * **1.4.0 (2024‑01‑24):** `PluginManager.unblock`, warning for old‑style wrapper teardown exceptions.
  * **1.3.0 (2023‑08‑26):** exported typing stubs (`HookRelay`, `HookCaller`, `Result`, `HookImpl`, `HookimplOpts`, `HookspecOpts`).
  * **1.2.0:** new‑style wrappers must be **explicitly** marked `wrapper=True`.
  * **1.1.0 (yanked):** introduced new‑style wrappers. ([Pluggy][3])

*(Rule of thumb: target Pluggy ≥1.5 if you want parameter‑level deprecation warnings; ≥1.2 for explicit new‑style wrappers; ≥1.6 if you’re on Python 3.9+.)*

---

## 15) Quick API lookup (copy/paste friendly)

* **Markers**: `HookspecMarker`, `HookimplMarker`. ([Pluggy][2])
* **Manager**: `PluginManager(project_name)`, `add_hookspecs`, `register`, `unregister`, `check_pending`, `load_setuptools_entrypoints`, `set_blocked/is_blocked/unblock`, `list_name_plugin/get_plugins/get_plugin/get_canonical_name/get_name`, `subset_hook_caller`, `enable_tracing`, `add_hookcall_monitoring`. ([Pluggy][2])
* **Calling**: `pm.hook.<name>(**kwargs)`, returns `[results]` or single value with `firstresult=True`; `call_historic(result_callback=..., kwargs=...)`; `call_extra(methods=[...], kwargs=...)`. ([Pluggy][2])
* **Wrapper tools**: `Result.get_result()`, `Result.force_result()`, `Result.force_exception()` (old‑style); return from the generator (new‑style). ([Pluggy][2])

---

## 16) Migration playbook (from custom systems)

1. **Define a spec interface** from your existing callback names. Keep docstrings; use **opt‑in arguments** so you can add parameters later without breaking plugins. ([Pluggy][1])
2. **Implement wrappers** instead of ad‑hoc “before/after” stacks. Prefer **new‑style** (`wrapper=True`). ([Pluggy][1])
3. **Turn “first handler wins” logic** into `@hookspec(firstresult=True)`. ([Pluggy][1])
4. **Late listeners** → `historic=True` + `call_historic`. ([Pluggy][1])
5. **Swap custom discovery** for **entry points**; add a group, document it, and call `load_setuptools_entrypoints`. ([Pluggy][1])
6. **Enforce quality** with `check_pending()` (or mark rare backward‑compat hooks `optionalhook=True`). Add **warn_on_impl** / **warn_on_impl_args** to guide plugin authors through deprecations. ([Pluggy][1])
7. **Instrument** with `enable_tracing` or `add_hookcall_monitoring` so your CI (or AI agent) can verify call order and results during refactor. ([Pluggy][1])

---

## 17) Gotchas & best practices

* **Always call hooks with keyword args**—positional calls raise. ([Pluggy][2])
* **Register specs before plugins** to validate early (you *can* add later, but then validation is delayed). ([Pluggy][1])
* **Avoid name collisions** by picking a **unique** `project_name` and an entry point group like `"yourapp.plugins"`. ([Pluggy][2])
* **Order management**: Combine `tryfirst/trylast` with **registration order** for precise pipelines; when you need per‑call exclusions use `subset_hook_caller`. ([Pluggy][1])
* **Wrappers in a mixed world**: New- and old-style wrappers interoperate, but use **one style per plugin** for clarity. ([Pluggy][1])

---

## DocsToKG integration

* `DocsToKG.OntologyDownload.net` wraps the shared HTTPX client with Hishel using `CacheTransport` + `FileStorage` under `${CACHE_DIR}/http/ontology`. This gives ontology downloads RFC‑9111 caching (validators, revalidation) without bespoke ETag plumbing.
* Requests that need polite headers or per-call overrides should pass `request.extensions["ontology_headers"]`; the shared hooks merge polite headers, correlation IDs, and service-specific metadata before sending.
* Tests rely on `use_mock_http_client` to swap in `httpx.MockTransport` instances. The helper calls `configure_http_client` so Hishel metadata still flows, and `reset_http_client()` restores the default cache-backed client afterwards.

---

### References (select)

* **User guide** (concepts, ordering, wrappers, firstresult, historic calls, discovery, blocking, inspection, monitoring). ([Pluggy][1])
* **API reference** (complete list of public classes/functions and their semantics). ([Pluggy][2])
* **Changelog** (what’s new in 1.6/1.5/1.4/1.3/1.2/1.1/1.0). ([Pluggy][3])
* **Packaging entry points** (how to advertise/discover plugins via `pyproject.toml`). ([Python Packaging][4])

---

If you share a snippet of your existing plugin/dispatcher code, I can map each piece to **pluggy** (specs, impls, wrappers, discovery) and output a ready‑to‑apply patch.

[1]: https://pluggy.readthedocs.io/en/latest/ "pluggy — pluggy 1.6.1.dev38+g37d3aa319 documentation"
[2]: https://pluggy.readthedocs.io/en/latest/api_reference.html "API Reference — pluggy 1.6.1.dev38+g37d3aa319 documentation"
[3]: https://pluggy.readthedocs.io/en/stable/changelog.html "Changelog — pluggy 0.1.dev96+gfd08ab5 documentation"
[4]: https://packaging.python.org/guides/creating-and-discovering-plugins/?utm_source=chatgpt.com "Creating and discovering plugins"
