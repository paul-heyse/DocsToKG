Below is a practical, **end‑to‑end guide to `yaml` (PyYAML)**—the de‑facto YAML parser/emitter for Python—written for an AI programming agent refactoring from custom config/serialization code. It focuses on safe loading, performance, extensibility (constructors/representers), YAML features that matter in production (anchors, merge keys, multi‑doc streams), and common footguns.

> **What library is this?**
> The Python package is **PyYAML** (installed as `PyYAML`, imported as `yaml`). As of **Sep 25, 2025** the latest version is **6.0.3** (Python 3.8+). PyYAML implements a complete **YAML 1.1** parser and emitter. ([PyPI][1])

---

## Install & quick import

```bash
pip install PyYAML
```

```python
import yaml
```

**Optional C speedups.** If your environment has **libyaml** available, PyYAML exposes C‑accelerated classes (`CLoader`, `CSafeLoader`, `CDumper`, `CSafeDumper`). These are drop‑in equivalents to the Python loaders/dumpers and can be 5–10× faster on large files.

---

## Core mental model

PyYAML has two halves:

* **Loaders**: parse YAML → Python objects (choose how *rich* and how *safe*).
* **Dumpers**: serialize Python objects → YAML text (choose formatting & tags).

You use *functions* that pick a loader/dumper for you (`safe_load`, `dump`, etc.) or you pass explicit classes (`Loader=…`, `Dumper=…`) for full control. Under the hood PyYAML also has resolvers (how scalars get types), constructors (turn a node into a Python object), and representers (turn a Python object into a YAML node).

---

## Safe defaults you should adopt (copy/paste)

For untrusted or external config, **never** use `yaml.load` without care. Use `safe_load`; when dumping, pick human‑readable emit options:

```python
from yaml import CSafeLoader as SafeLoader, CSafeDumper as SafeDumper  # falls back to SafeLoader/SafeDumper if C not installed

def load_yaml(text_or_stream):
    return yaml.load(text_or_stream, Loader=SafeLoader)  # safe

def dump_yaml(obj):
    return yaml.dump(
        obj,
        Dumper=SafeDumper,
        sort_keys=False,            # preserve insertion order
        default_flow_style=False,   # block style (multiline-friendly)
        allow_unicode=True,
    )
```

Why: `safe_load` avoids Python object construction (a historic code‑execution risk), and `sort_keys=False` + block style produce stable, diff‑friendly files for ops. PyYAML explicitly documents the loader choices and the security implications. ([GitHub][2])

---

## Loaders you’ll actually use

* **`SafeLoader`** / `yaml.safe_load(...)`: loads only standard YAML tags → Python base types; **recommended for untrusted input**.
* **`FullLoader`** / `yaml.full_load(...)`: loads the full YAML language and Python types such as timestamps; more permissive but **not for untrusted data** (historic exploits existed—prefer `SafeLoader` unless you explicitly need richer types). ([GitHub][2])
* **`UnsafeLoader`** / `yaml.unsafe_load(...)`: can construct arbitrary Python objects (`!!python/object`, `!!python/object/apply`, etc.); only for trusted inputs and special cases. The long‑standing guidance is to avoid it for external data.
* **C‑accelerated** variants: `CLoader`, `CSafeLoader`, `CBaseLoader` (parser) and `CDumper`, `CSafeDumper` (emitter).

> **Deprecation note.** Calling `yaml.load(...)` **without** explicitly choosing a loader was deprecated in PyYAML 5.1+ because it was unsafe on untrusted input. If you ever see `YAMLLoadWarning`, update the code to pass `Loader=...` explicitly or switch to `safe_load`. ([GitHub][2])

---

## Dumpers & common formatting switches

* **`yaml.dump(obj, Dumper=...)`** (or `yaml.safe_dump`) serializes Python objects to YAML.
* Key options you’ll want:

  * `sort_keys=False` – keep insertion order (PyYAML sorts by default). ([Stack Overflow][3])
  * `default_flow_style=False` – force **block style** lists/maps (more readable). ([pyyaml.org][4])
  * `allow_unicode=True` – emit non‑ASCII characters.
  * `width=…`, `indent=…` – line length and indentation control.

---

## YAML features that matter in configs

### 1) Multi‑document streams (useful for “bundles”)

YAML can contain several docs in one file separated by `---`. Load or dump them with the `*_all` APIs:

```python
docs = list(yaml.safe_load_all(open("bundle.yaml")))
yaml.safe_dump_all(docs, open("bundle.out.yaml", "w"), explicit_start=True, explicit_end=False)
```

`explicit_start=True` writes `---` before each document (handy for concatenation).

### 2) Anchors, aliases & **merge keys**

* **Anchors (`&`)** & **aliases (`*`)** let you re‑use structures without repeating.
* **Merge key** (`<<`) merges mappings; later keys override earlier ones:

```yaml
defaults: &base
  retries: 3
  timeout: 10
service:
  <<: *base
  timeout: 30   # override
```

The merge key is specified in YAML 1.1 (widely supported, including PyYAML). ([pyyaml.org][4])

> FYI: The merge key is a YAML 1.1 extension; not part of YAML 1.2. Most tools still support it; just be aware if you interop with strict 1.2 parsers.

### 3) Tags, resolvers & surprising scalars

PyYAML follows YAML 1.1’s implicit typing rules. That means bare scalars like `yes`, `no`, `on`, `off`, `y`, `n` are booleans, and certain date/timestamp shapes get auto‑typed. If you want strings, **quote them** (e.g., `"on"`), or disable specific implicit resolvers with a custom loader. ([PyPI][1])

**Keep timestamps from auto‑becoming `datetime`:**

```python
class NoDatesSafeLoader(yaml.SafeLoader):
    @classmethod
    def remove_implicit_resolver(cls, tag):
        for ch, patterns in list(cls.yaml_implicit_resolvers.items()):
            cls.yaml_implicit_resolvers[ch] = [
                (t, r) for (t, r) in patterns if t != tag
            ] or None
NoDatesSafeLoader.remove_implicit_resolver('tag:yaml.org,2002:timestamp')
data = yaml.load(open("config.yaml"), Loader=NoDatesSafeLoader)
```

This pattern removes the implicit resolver for timestamps so those tokens load as strings. ([Stack Overflow][5])

---

## Extending: custom tags, constructors & representers

Use **constructors** to turn a tagged node into a Python value at load time, and **representers** to control how Python types dump.

### Example: environment variable expansion (`!ENV`)

```python
import os, yaml

class EnvLoader(yaml.SafeLoader): pass

def env_constructor(loader, node):
    value = loader.construct_scalar(node)
    return os.environ.get(value, "")

yaml.add_constructor("!ENV", env_constructor, Loader=EnvLoader)

cfg = yaml.load("api_key: !ENV WRITER_API_KEY", Loader=EnvLoader)
```

You can also add **representers**—e.g., to emit multiline strings using the literal `|` style:

```python
def str_presenter(dumper, data):
    style = '|' if '\n' in data else None
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=style)

yaml.add_representer(str, str_presenter, Dumper=yaml.SafeDumper)
```

These hooks are part of PyYAML’s public API (`add_constructor`, `add_representer`, `represent_scalar`, etc.).

---

## Streaming & large files

* Use `yaml.safe_load_all(stream)` to iterate documents without loading everything at once.
* For very large inputs or custom pipelines, PyYAML also exposes **scanner/parser/composer** APIs (`scan`, `parse`, `compose`) so you can process tokens/events/nodes progressively.

---

## Interop and “round‑trip” editing

PyYAML **does not** preserve comments, original key order formatting decisions, or anchors on re‑dump (it will materialize merge keys). If you need “read‑modify‑write without changing formatting/comments,” use **`ruamel.yaml`** (a YAML 1.2 loader/dumper that round‑trips comments and styles). Keep PyYAML for simple read/write, use ruamel when you must preserve the human‑authored form. ([PyPI][6])

---

## Common patterns for AI/agent codebases

### 1) Read configuration safely (with include‑ish anchors)

```python
with open("agent.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
# Use anchors & merges in YAML to avoid duplication (see §anchors).
```

### 2) Compose environment or secrets via custom tags

Add a `!ENV` constructor (above) to keep secrets out of files yet project a full config at load time.

### 3) Emit diff‑friendly YAML to disk

```python
yaml.safe_dump(config, open("agent.out.yaml","w"), sort_keys=False, default_flow_style=False, allow_unicode=True)
```

Produces stable order and block style; easier for code reviews. ([pyyaml.org][4])

### 4) Validate after load

After `safe_load`, validate with `pydantic`/`jsonschema`—especially useful because YAML’s implicit typing can surprise you (booleans/timestamps). ([pyyaml.org][4])

---

## Footguns (and how to avoid them)

1. **Never `yaml.load` untrusted input** without an explicit safe/full loader choice; prefer `safe_load`. This avoids Python object construction paths like `!!python/object/apply` that historically led to code execution in old codebases. ([GitHub][2])
2. **Implicit types in YAML 1.1**: unquoted `on`, `off`, `yes`, `no` → bool; strings like `2025-10-20` may become `datetime`. Quote them or remove resolvers if you truly need raw strings. ([GitHub][7])
3. **Merge keys materialize on dump**: After loading, `<<` merges are flattened when you dump—anchors/aliases used only for plain aliasing might be preserved, but merge‑driven structure is expanded. Don’t expect re‑emission to reconstruct the same merge syntax. ([GitHub][8])
4. **Key sorting**: `yaml.dump` sorts keys by default; set `sort_keys=False` to keep insertion order (Python 3.7+ dicts are ordered). ([Stack Overflow][3])
5. **Tabs**: YAML is indentation‑sensitive; best rule is **don’t use tabs**.

---

## API crib (what you’ll reach for most)

* **Loaders (string or file‑like → Python):**
  `yaml.safe_load`, `yaml.safe_load_all`, `yaml.full_load`, `yaml.unsafe_load`, plus `Loader=` variants (`SafeLoader`, `FullLoader`, `UnsafeLoader`, `BaseLoader`). C‑accelerated: `CSafeLoader`, `CLoader`.
* **Dumpers (Python → YAML text or file):**
  `yaml.dump`, `yaml.safe_dump`, `yaml.dump_all` with `Dumper=` (`SafeDumper`, `Dumper`) or C‑accelerated (`CSafeDumper`, `CDumper`); key options: `sort_keys`, `default_flow_style`, `allow_unicode`, `width`, `indent`.
* **Extensibility:**
  `yaml.add_constructor`, `yaml.add_multi_constructor`, `yaml.add_representer`, `yaml.add_multi_representer`, `yaml.add_implicit_resolver`.

---

## Mapping your custom code → PyYAML

| If your in‑house code does this…              | Use this in PyYAML                                                              |
| --------------------------------------------- | ------------------------------------------------------------------------------- |
| Parse untrusted config safely                 | `yaml.safe_load` (or `Loader=SafeLoader`)                                       |
| Load many docs from one file                  | `yaml.safe_load_all`                                                            |
| Keep YAML small & readable when writing       | `yaml.dump(..., sort_keys=False, default_flow_style=False, allow_unicode=True)` |
| Share config fragments without duplication    | Anchors/aliases and `<<` merge keys (YAML 1.1)                                  |
| Expand env vars at load time                  | Custom tag + `add_constructor` (e.g., `!ENV`)                                   |
| Reuse existing C bindings for speed           | Import `CSafeLoader` / `CLoader` / `CSafeDumper`                                |
| Prevent timestamps or yes/no being auto‑typed | Quote them **or** remove implicit resolvers for those tags                      |

(See the sections above for examples and caveats.)

---

## Notes on YAML versions (1.1 vs 1.2)

PyYAML targets **YAML 1.1**, which includes implicit boolean forms like `yes/no` and `on/off`. YAML 1.2 tightened this (JSON‑compatible) and drops those forms. If you require strict YAML 1.2 parsing or **round‑trip (comment‑preserving)** editing, consider **`ruamel.yaml`**. ([PyPI][1])

---

## Security background (the short version)

Older code often used `yaml.load(s)` with the default loader, which could construct arbitrary Python objects from specially tagged YAML and execute code during construction. PyYAML 5.1+ **deprecated** that pattern; always choose a loader (`safe_load` for untrusted input). Multiple advisories and write‑ups document this history. ([GitHub][2])

---

## Appendix: quick recipes

**Load with C speedups (fallback to Python):**

```python
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader
data = yaml.load(open("cfg.yml"), Loader=SafeLoader)
```

C‑accelerated loaders/dumpers are available when PyYAML is built with libyaml.

**Emit multiple docs with `---`:**

```python
yaml.safe_dump_all(docs, open("out.yml","w"), explicit_start=True, sort_keys=False)
```

**Custom representer for dataclasses:**

```python
from dataclasses import is_dataclass, asdict
def dataclass_representer(dumper, obj):
    return dumper.represent_mapping('tag:yaml.org,2002:map', asdict(obj))
yaml.add_multi_representer(type, dataclass_representer, Dumper=yaml.SafeDumper)
```

Representer/constructor APIs are part of the documented surface.

---

## References (authoritative)

* **PyPI: current version & Python support** (6.0.3, Python 3.8+). ([PyPI][1])
* **Official PyYAML docs** (loaders/dumpers, C variants, resolvers/constructors/representers, low‑level APIs, anchors/aliases).
* **`yaml.load` deprecation note** (why, and which loader to choose). ([GitHub][2])
* **YAML merge key spec** (`<<` behavior). ([yaml.org][9])
* **YAML 1.1 vs 1.2 booleans** (yes/no/on/off behavior and gotchas). ([GitHub][7])
* **ruamel.yaml** for round‑trip/comment‑preserving editing and YAML 1.2 loader. ([PyPI][6])

---

If you share a snippet of your current YAML utilities (how you load/dump, any custom expansion, formatting expectations, and whether files come from trusted sources), I’ll translate them 1‑for‑1 into a small, safe wrapper module with C‑accelerated fallbacks and the formatting defaults your repo expects.

[1]: https://pypi.org/project/PyYAML/ "PyYAML · PyPI"
[2]: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load%28input%29-Deprecation "PyYAML yaml.load(input) Deprecation · yaml/pyyaml Wiki · GitHub"
[3]: https://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order?utm_source=chatgpt.com "Can PyYAML dump dict items in non-alphabetical order?"
[4]: https://pyyaml.org/wiki/PyYAMLDocumentation?utm_source=chatgpt.com "PyYAML Documentation"
[5]: https://stackoverflow.com/questions/34667108/ignore-dates-and-times-while-parsing-yaml?utm_source=chatgpt.com "python - Ignore dates and times while parsing YAML"
[6]: https://pypi.org/project/ruamel.yaml/?utm_source=chatgpt.com "ruamel.yaml"
[7]: https://github.com/yaml/pyyaml/issues/613?utm_source=chatgpt.com "PyYAML interprets yes, no, on, off as Boolean #613"
[8]: https://github.com/yaml/pyyaml/issues/151?utm_source=chatgpt.com "Inconsistent rendering of aliases · Issue #151 · yaml/pyyaml"
[9]: https://yaml.org/type/merge.html?utm_source=chatgpt.com "Merge Key Language-Independent Type for ..."
