Below is a **refactorer‑friendly “complete reference”** to **`jsonlines`** for Python—aimed at an AI (or human) programming agent replacing ad‑hoc JSONL/NDJSON reading/writing loops with the library’s first‑class primitives.

---

## 0) What `jsonlines` is (and when to reach for it)

* **Purpose.** `jsonlines` is a small, production‑ready library for the *JSON Lines* format (a.k.a. **NDJSON**): **one valid JSON value per line, UTF‑8 encoded**. It wraps the dull bits—newline handling, bytes vs. text, validation, clear errors—so your code doesn’t have to. ([Jsonlines][1])
* **Format, briefly.** The NDJSON spec requires **one JSON text followed by `\n`** (CRLF accepted on input); the whole file is a sequence of JSON texts, not a single JSON array/document. This is why simple “`for line in f: json.loads(line)`” works—and what `jsonlines` formalizes and hardens. ([GitHub][2])

---

## 1) Big picture: core building blocks

* **`jsonlines.open(path, mode=...)`** → returns a **`Reader`** (for `'r'`) or a **`Writer`** (for `'w'|'a'|'x'`), and can be used as a context manager. Extra kwargs are forwarded to the underlying class. ([Jsonlines][1])
* **`Reader(file_or_iterable, *, loads=...)`** → iterate values, or call `read()`/`iter()` with type & validation flags. Accepts **text or bytes** sources (files, sockets, `BytesIO`, or *any iterable of lines*). ([Jsonlines][1])
* **`Writer(fp, *, compact=False, sort_keys=False, flush=False, dumps=...)`** → write one value (`write`) or many (`write_all`) per line to **text or bytes** sinks; returns the number of chars/bytes written. ([Jsonlines][1])
* **Exceptions.** `InvalidLineError` (subclasses `ValueError`) carries `.line` and `.lineno` for precise diagnostics; base class is `jsonlines.Error`. ([Jsonlines][1])

**Supported Python:** 3.8+ (current published version: **4.0.0**, released 2023‑09‑01). ([PyPI][3])

---

## 2) Reader: consuming JSON Lines safely & fast

**Construction & inputs**

```python
import jsonlines, gzip, sys, io

# From a path (returns a Reader)
with jsonlines.open("events.jsonl") as reader:
    for obj in reader:
        ...

# From any file-like or iterable (text or bytes)
reader = jsonlines.Reader(sys.stdin)           # text
reader = jsonlines.Reader(sys.stdin.buffer)    # bytes
reader = jsonlines.Reader(gzip.open("x.jsonl.gz", "rb"))  # bytes
reader = jsonlines.Reader(["1", "2", "3"])
```

**Reading API**

* **Iterate directly**: `for obj in reader:` (same as `reader.iter()` with no flags).
* **Type & validity checks**:
  `reader.read(type=dict, allow_none=False, skip_empty=False)` — raises `InvalidLineError` if the decoded value’s type is not one of the supported built‑ins (`dict`, `list`, `str`, `int`, `float`, `bool`), or a `null` is seen when `allow_none=False`. ([Jsonlines][1])
* **Robust iteration**:
  `reader.iter(type=dict, skip_invalid=True, allow_none=True, skip_empty=True)`—iterate all lines, **silently skipping invalid/empty lines**, and letting explicit `null` become `None`. ([Jsonlines][1])

**JSON backend**

* Out of the box, `jsonlines` supports multiple JSON libs and will **use `orjson` or `ujson` for reading if they’re installed** (falls back to stdlib `json`). You can also **override** with `loads=...`. ([Jsonlines][1])

**Why Reader instead of “hand‑rolled” loops?**
It normalizes **bytes/text**, ignores UTF‑8 BOM if present, uses **standards‑compliant line breaking**, provides **clear exceptions** with line numbers, and offers **skip/validate** knobs—things ad‑hoc loops often forget. ([Jsonlines][1])

---

## 3) Writer: producing clean, deterministic JSON Lines

**Construction & outputs**

```python
import jsonlines, io, orjson

# Path helper
with jsonlines.open("out.jsonl", mode="w") as writer:
    writer.write({"ok": True})

# File-like sink (bytes OR text)
buf = io.BytesIO()
with jsonlines.Writer(buf, compact=True, sort_keys=True, flush=True,
                      dumps=orjson.dumps) as writer:
    writer.write_all([{"b":2,"a":1}, {"x": [1,2,3]}])

# Appending or exclusive create:
with jsonlines.open("log.jsonl", mode="a") as w: w.write({"evt":"start"})
with jsonlines.open("new.jsonl", mode="x") as w: w.write({"first": 1})
```

**Output controls**

* **`compact=True`** → minimal whitespace; **`sort_keys=True`** → deterministic key order.
* **`flush=True`** → call `fp.flush()` after each line (useful for logs/long‑running jobs).
* **`dumps=`** → full control; may return **`str` or `bytes`** (e.g., `orjson.dumps`). Both are supported. **Return values** of `write`/`write_all` are the number of chars/bytes emitted. ([Jsonlines][1])

---

## 4) Convenience: `jsonlines.open(...)`

```python
# Reader (default mode='r'); Writer for 'w'/'a'/'x'
with jsonlines.open("data.jsonl") as reader:
    for row in reader: ...
with jsonlines.open("data.jsonl", mode="w", compact=True, sort_keys=True) as writer:
    writer.write({"id": 1})
```

* Returns a `Reader` or `Writer` based on **`mode`**, forwards extra kwargs to them, and **closes the file for you** when the context exits. ([Jsonlines][1])

---

## 5) Errors & strictness

* **`InvalidLineError(message, line, lineno)`** — raised when a line is not parseable JSON *or* when the `type=` constraint is violated; you get the **exact line and number** for triage. Example:

```python
try:
    with jsonlines.open("in.jsonl") as r:
        for obj in r.iter(type=dict):  # require objects
            ...
except jsonlines.InvalidLineError as e:
    print("bad line", e.lineno, repr(e.line))
```

(Also see base `jsonlines.Error`.) ([Jsonlines][1])

---

## 6) End‑to‑end recipes (drop‑in refactors)

### A) Replace `for line in f: json.loads(line)` with validation + resilience

```python
import jsonlines

with jsonlines.open("input.jsonl") as reader:
    for obj in reader.iter(type=dict, allow_none=False, skip_empty=True, skip_invalid=True):
        process(obj)
```

– Reads **text/bytes** correctly, skips blank/garbage lines, and keeps going. ([Jsonlines][1])

### B) Stream from `stdin` or a socket

```python
import sys, jsonlines
reader = jsonlines.Reader(sys.stdin)          # text
# reader = jsonlines.Reader(sys.stdin.buffer) # bytes
for obj in reader:
    handle(obj)
```

– Reader accepts **any iterable** yielding lines. ([Jsonlines][1])

### C) Read/write **compressed** files (gzip)

```python
import gzip, jsonlines
with gzip.open("data.jsonl.gz", "rb") as f:          # or "rt"
    for obj in jsonlines.Reader(f):
        ...

with gzip.open("out.jsonl.gz", "wb") as f:
    w = jsonlines.Writer(f, compact=True)
    w.write_all(records); w.close()
```

– Works because both **bytes and text** streams are supported. ([Jsonlines][1])

### D) High‑throughput writes with `orjson`

```python
import orjson, jsonlines
with jsonlines.open("fast.jsonl", "w", dumps=orjson.dumps, compact=True) as w:
    w.write_all(big_iterable)
```

– `orjson` returns **bytes** and is automatically supported; `jsonlines` also prefers `orjson`/`ujson` for reading when installed. ([Jsonlines][1])

### E) Append or “create only” files

```python
jsonlines.open("audit.jsonl", "a").close()   # append
jsonlines.open("newfile.jsonl", "x").close() # exclusive create
```

– `'a'` and `'x'` are supported modes. ([Jsonlines][1])

---

## 7) Behavior details that matter in refactors

* **Line breaks & encoding.** The format is **one JSON value per line**, terminated with `\n` (CRLF accepted on input), and **UTF‑8** throughout. `jsonlines` handles **UTF‑8 BOM** and uses **standards‑compliant line‑breaking**, avoiding common `splitlines()` pitfalls. ([GitHub][2])
* **Types enforced at the edge.** Use `type=` and `allow_none` to assert schema‑level expectations without writing custom guards. `skip_invalid` lets you keep the pipeline alive while you log/inspect error lines. ([Jsonlines][1])
* **Deterministic output.** `sort_keys=True` ensures stable line‑by‑line diffs; `compact=True` reduces size; `flush=True` is ideal for logging/streaming where durability is preferred over throughput. ([Jsonlines][1])
* **Custom JSON codecs.** Swap in `loads`/`dumps` (e.g., `orjson`, `ujson`, or `json.dumps` with a custom `default=`) to support datetimes, `Decimal`, pydantic/dataclass objects, etc. (Note: with a custom `dumps`, **`compact/sort_keys` are ignored**.) ([Jsonlines][1])
* **Context managers.** `Reader`/`Writer` **do not** close file‑like objects you pass in (you own them). `jsonlines.open(...)` **does** close the file it opens. ([Jsonlines][1])

---

## 8) Migration playbook (from hand‑rolled JSONL)

1. **Replace raw loops**

   * Before:

     ```python
     with open("in.jsonl") as f:
         for line in f:
             obj = json.loads(line)
     ```

   * After:

     ```python
     with jsonlines.open("in.jsonl") as r:
         for obj in r: ...
     ```

   Add `type=...`, `skip_invalid`, `allow_none` per your needs. ([Jsonlines][1])

2. **Centralize serialization**

   * If you were calling `json.dumps(..., default=...)` or `orjson.dumps(...)`, pass it once as `dumps=` to the `Writer` (and `loads=` for `Reader` if you need symmetry). ([Jsonlines][1])

3. **Make failures actionable**

   * Swap generic `ValueError` handling for `InvalidLineError` and **log `.lineno` + a snippet of `.line`**. Optionally switch to `skip_invalid=True` in pipelines that must not halt. ([Jsonlines][1])

4. **Adopt append/exclusive modes**

   * Use `'a'` for logs and `'x'` to avoid clobbering files—both supported in `jsonlines.open`. ([Jsonlines][1])

5. **Performance**

   * Install `orjson` for faster loads (picked up automatically for reading in **4.0.0**), and consider `compact=True` for output size. ([Jsonlines][1])

---

## 9) Gotchas & best practices

* **The file is not one big JSON document.** Tools expecting an array will choke; handle it as a **stream** of values. (That’s the whole point of NDJSON/JSON Lines.) ([GitHub][2])
* **Only specific built‑ins are allowed in `type=`.** As of 3.0.0, `numbers.Number` is **not** accepted—use concrete types (`int`, `float`). ([Jsonlines][1])
* **Don’t rely on `splitlines()` semantics.** Let `Reader` handle line breaks correctly (e.g., Windows CRLF, lone CRs). ([Jsonlines][1])
* **Bytes vs. text.** Both are supported in inputs and outputs; choose based on your I/O stack (e.g., `gzip.open(..., 'rb')` + `Writer(fp=...)`). ([Jsonlines][1])
* **Flushing every line hurts throughput.** Use `flush=True` only when you need low‑latency durability (logs, tail‑ing). ([Jsonlines][1])

---

## 10) Version awareness (helpful when pinning / upgrading)

* **4.0.0 (2023‑09‑01):** prefers **`orjson`/`ujson`** for reading when available; Python **3.8+**.
* **3.1.0:** `write()`/`write_all()` return **count of chars/bytes**; `open(mode='x')` supported.
* **3.0.0:** type annotations; **UTF‑8 BOM** ignored; `dumps` may return **bytes**; basic **RFC 7464** (JSON text sequences) support; removed `numbers.Number` from `type=`. ([Jsonlines][1])

---

## 11) API lookup (copy/paste quick sheet)

* **Open:** `jsonlines.open(path, mode='r'|'w'|'a'|'x', *, loads=..., dumps=..., compact=..., sort_keys=..., flush=...)` → `Reader`|`Writer`. ([Jsonlines][1])
* **Reader:**

  * `Reader(file_or_iterable, *, loads=...)`
  * `for obj in reader:` — iterate
  * `reader.read(type=..., allow_none=False, skip_empty=False)`
  * `reader.iter(type=..., allow_none=..., skip_empty=..., skip_invalid=...)`
  * `reader.close()` ([Jsonlines][1])
* **Writer:**

  * `Writer(fp, *, compact=False, sort_keys=False, flush=False, dumps=...)`
  * `writer.write(obj) -> int`, `writer.write_all(iterable) -> int`
  * `writer.close()` ([Jsonlines][1])
* **Errors:** `jsonlines.Error`, `jsonlines.InvalidLineError(message, line, lineno)` ([Jsonlines][1])

---

## 12) Appendix: “What exactly is a line?” (interop note)

Per **NDJSON spec**, each JSON text **must** be followed by `\n` (0x0A). Parsers **must accept** LF or CRLF as delimiters. Using `jsonlines.Writer` ensures your output matches these expectations so downstream tools (e.g., shell pipelines, log shippers) work reliably. ([GitHub][2])

---

### References (select)

* **Official docs & user guide** (features, API, examples, context‑manager behavior). ([Jsonlines][1])
* **Reader & Writer API details** (type checks, `skip_invalid`, `compact/sort_keys/flush`, `write*_` return values). ([Jsonlines][1])
* **Version & Python support** (4.0.0, Python ≥3.8). ([PyPI][3])
* **NDJSON format spec** (line delimiter, encoding). ([GitHub][2])

---

If you paste a snippet of your current JSONL I/O (e.g., custom loops, ad‑hoc error handling, gzip usage), I’ll map it one‑for‑one to `jsonlines` (choosing `loads`/`dumps`, flags, and modes) and output a ready‑to‑apply patch.

[1]: https://jsonlines.readthedocs.io/ "jsonlines — jsonlines  documentation"
[2]: https://github.com/ndjson/ndjson-spec "GitHub - ndjson/ndjson-spec: Specification"
[3]: https://pypi.org/project/jsonlines/ "jsonlines · PyPI"
