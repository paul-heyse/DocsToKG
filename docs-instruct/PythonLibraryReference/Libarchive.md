Below is a **refactorer‑friendly “complete reference”** to **`libarchive` (the Python bindings)**—written so an AI agent (or human) can replace ad‑hoc tar/zip/7z code with the library’s first‑class, streaming primitives.

> **Which Python package is this?**
> There are **two** bindings commonly seen on PyPI:
>
> 1. **`libarchive-c`** (actively maintained; module is imported as `libarchive`).
> 2. **`python-libarchive`** (a SWIG wrapper with a different API surface).
>    This guide focuses on **`libarchive-c`** (import name `libarchive`), whose README/API cover extraction, reading, and writing via context managers and a streaming model. I’ll call out differences where helpful. ([GitHub][1])

---

## 0) What `libarchive` (bindings) are—and aren’t

* **Purpose.** Thin Python bindings for the C **libarchive** library: a robust, high‑performance, *streaming* engine that **auto‑detects archive formats and compression filters** and reads/writes many formats (tar, cpio, zip, iso9660, etc.). Because the C library is *stream‑oriented*, you iterate entries serially; there’s **no in‑place modification nor random access** to archives. If you need to change an archive, write a new one. ([libarchive.org][2])

* **When to choose it.** You want a single API to **read & write multiple formats**, including archives that have been additionally compressed (e.g., `.tar.gz`, `.tar.zst`), and you care about correctness and security (e.g., safe extraction flags). ([libarchive.org][2])

---

## 1) Installation & runtime requirements

* **Install the Python binding**: `pip install libarchive-c` (import as `libarchive`). On macOS, the system `libarchive` may be old; install a current one (`brew install libarchive`) and point the binding to the library with `LIBARCHIVE=/path/to/libarchive.dylib`. ([GitHub][1])

* **Supported Python (as tested upstream):** The project’s README states it is *currently tested* with Python **3.12 & 3.13**. ([GitHub][1])

* **Conda**: packaged as `python-libarchive-c` on conda‑forge for Linux/macOS/Windows. Good option on Windows because it also pulls in the native `libarchive` runtime. ([Anaconda][3])

> **Windows gotcha:** If importing `libarchive` fails with something like *“specified module could not be found”*, it usually means the **C runtime (`libarchive` DLL)** isn’t installed or discoverable. Installing via **conda‑forge** or ensuring the DLL is on the PATH resolves this. ([Stack Overflow][4])

---

## 2) The mental model (hosted by `libarchive-c`)

* **Extractors**: `extract_file`, `extract_fd`, `extract_memory` — high‑level, secure by default (see §7). Write to disk from an archive in one call. ([GitHub][1])

* **Readers** (stream in): `file_reader`, `fd_reader`, `memory_reader`, `stream_reader`, `custom_reader`. Iterate **entries** and pull **data blocks** from each entry. Great for *list‑only* operations, filtering, or custom copy pipelines. ([GitHub][1])

* **Writers** (stream out): `file_writer`, `fd_writer`, `memory_writer`, `custom_writer`. Add files from disk (`add_files`) or from memory (`add_file_from_memory`), select a **container format** and **compression filter**. ([GitHub][1])

* **Formats & filters**: The second (required) writer argument is a **format** (e.g., `'ustar'`, `'zip'`), the optional third is a **filter** (e.g., `'gzip'`, `'xz'`, `'zstd'`). The set supported at runtime depends on the **installed C libarchive build**; query via `libarchive.ffi.WRITE_FORMATS` and `libarchive.ffi.WRITE_FILTERS`. ([GitHub][1])

* **Streaming orientation**: Readers sequentially deliver entries; **no random seek** or in‑place edit; to modify an archive, write a new stream. This is a libarchive design invariant. ([Arch Manual Pages][5])

---

## 3) Quickstart (show me)

### A) Extract safely to a *target directory*

```python
import os, libarchive
os.chdir("/path/to/target")     # extraction root
libarchive.extract_file("bundle.zip")  # secure flags on by default
```

The binding’s extractors pass secure defaults (see §7) unless you’re extracting at filesystem root. ([GitHub][1])

### B) List entries (without extracting)

```python
import libarchive
with libarchive.file_reader("input.7z") as a:
    for entry in a:
        print(entry.pathname, entry.size)       # iterate entries
        # Or stream data:
        for block in entry.get_blocks():
            ...  # process bytes
```

Use `.get_blocks()` to stream file contents; see `libarchive.entry.ArchiveEntry` for attributes. ([GitHub][1])

### C) Create a `tar.gz` (add from disk, then from memory)

```python
from libarchive.entry import FileType
import libarchive

with libarchive.file_writer("out.tar.gz", "ustar", "gzip") as w:
    w.add_files("src_dir/", "README.md")   # recursive for dirs
    data = b"hello\n"
    w.add_file_from_memory("hello.txt", len(data), data)
```

Formats/filters are validated against those compiled into your libarchive build. ([GitHub][1])

### D) Track progress while reading

```python
from tqdm import tqdm
import os, libarchive

path = "big.tar.zst"
with tqdm(total=os.stat(path).st_size, unit="bytes") as bar, \
     libarchive.file_reader(path) as a:
    for entry in a:
        for _ in entry.get_blocks():
            pass
        bar.update(a.bytes_read - bar.n)
```

`bytes_read` lets you build a progress bar in O(1). ([GitHub][1])

---

## 4) Extractors (disk‑write helpers)

**APIs**: `extract_file(path, *, flags=None)`, `extract_fd(fd, *, flags=None)`, `extract_memory(buf, *, flags=None)`. These create files, directories, symlinks, etc., on disk using **libarchive’s `archive_write_disk`** machinery. The `flags` you pass are forwarded to `archive_write_disk_set_options(3)`. Use constants from `libarchive.extract` (`EXTRACT_*`). ([GitHub][1])

**Defaults (since 5.3):** when `flags=None`, the binding now passes **secure defaults**: `SECURE_NODOTDOT`, `SECURE_NOABSOLUTEPATHS`, `SECURE_SYMLINKS` (unless the current directory is `/`). This blocks path traversal & other unsafe writes. ([GitHub][6])

> **Target directory**: extractors write relative to the **current working directory**—`chdir` to where you want files to land (see the README example). ([GitHub][1])

---

## 5) Readers (stream in)

```python
with libarchive.file_reader("archive.any") as a:
    for entry in a:
        # metadata
        p = entry.pathname     # decoded with header_codec (see §9)
        sz = entry.size        # may be None if unknown
        # data
        for block in entry.get_blocks():
            consume(block)
```

**Variations**

* `memory_reader(bytes_or_buffer)` – read from a Python buffer/bytes.
* `fd_reader(fd)` – read from a POSIX file descriptor.
* `stream_reader(obj)` – read from an object that supports **`readinto`** (e.g., a custom stream).
* `custom_reader(read_cb, skip_cb, seek_cb, close_cb)` – supply C‑style callbacks for exotic sources.
  The binding **auto‑detects format and filter** (e.g., tar.gz, tar.zst). ([GitHub][1])

> **Design caveat:** libarchive readers are **sequential**; no random access to an entry’s tail without consuming intervening data, and no “open by name” without scanning. Plan pipelines accordingly. ([Arch Manual Pages][5])

---

## 6) Writers (stream out)

```python
with libarchive.file_writer("out.warc.gz", "warc", "gzip") as w:
    w.add_files("docs/")   # recursive; preserves symlinks by default

# Or build entries from memory (full control over metadata):
from libarchive.entry import FileType
data = b"bytes"
with libarchive.file_writer("out.zip", "zip") as w:
    w.add_file_from_memory(
        "path/in/archive.txt", len(data), data,
        filetype=FileType.REGULAR, permission=0o644,
        uid=1000, gid=1000,
        mtime=(int(1_702_000_000), 0),  # seconds, nanoseconds
    )
```

* **Format vs filter**: “Format” is the container (`'ustar'`, `'zip'`, `'cpio'`, `'iso9660'`, `'warc'`, `'xar'`, …). “Filter” is compression (`'gzip'`, `'bzip2'`, `'xz'`, `'zstd'`, `'lz4'`, …). Retrieve the **runtime‑supported** sets from `libarchive.ffi.WRITE_FORMATS` / `WRITE_FILTERS` (varies by how the C library was built). ([GitHub][1])

* **Symlink policy**: `add_files(..., symlink_mode='logical')` follows links and archives targets; default behavior is to **preserve** symlinks. If you ask to follow links and a link is broken, an `ArchiveError` is raised. ([GitHub][1])

* **Memory output**: `memory_writer` writes an archive to an in‑memory buffer (useful for HTTP responses, tests, or piping). ([GitHub][1])

> **Format coverage note:** libarchive’s **read** support is broader than its **write** support (and depends on how the native library was compiled). Always check what your runtime supports by inspecting `WRITE_FORMATS`/`WRITE_FILTERS`. The authoritative “supported formats” docs live in the C library’s man pages. ([manpages.debian.org][7])

---

## 7) Security: safe extraction by default

* `libarchive-c` **now passes secure disk‑write flags by default** (5.3, May 22, 2025):
  `SECURE_NODOTDOT`, `SECURE_NOABSOLUTEPATHS`, `SECURE_SYMLINKS`. These mitigate directory traversal (e.g., `../evil`) and symlink escapes. You can still customize `flags` if you need different behavior (e.g., to allow absolute paths in a controlled sandbox). ([GitHub][6])

* Flags map to the C API **`archive_write_disk_set_options(3)`**; see its man page for the full matrix of behaviors. ([Linux Documentation][8])

---

## 8) Performance & progress

* The binding exposes **`archive.bytes_read`** while streaming; combine with `os.stat(path).st_size` to render accurate progress bars (see quickstart). ([GitHub][1])

* libarchive is designed with a **zero‑copy, streaming architecture** and robust automatic format detection; it will decompress + parse in one pass (e.g., `tar.gz`, `tar.zst`). ([libarchive.org][2])

---

## 9) Metadata encoding (filenames, user/group names)

By default the binding uses **UTF‑8** to read/write **archive headers** (pathnames, link names, user/group names). If your archives use a different legacy encoding, pass `header_codec="..."` to the `*_reader`/`*_writer` factories. (Example in README shows `cp037`). ([GitHub][1])

---

## 10) What to do about encrypted archives?

libarchive’s C API supports passing **passphrases** to readers (e.g., for encrypted ZIP/others) via `archive_read_add_passphrase()` / `archive_read_set_passphrase_callback()`. Whether this is available in *your* Python binding depends on the binding and the underlying libarchive build; if you need decryption, verify support in your environment. (These are C‑level functions; exact Python exposure varies.) ([manpages.debian.org][9])

---

## 11) Migration playbook (from bespoke tar/zip code)

1. **Replace bespoke loop → streaming reader**
   Before:

   ```python
   import zipfile
   with zipfile.ZipFile(p) as z:
       for n in z.namelist(): ...
   ```

   After:

   ```python
   import libarchive
   with libarchive.file_reader(p) as a:
       for entry in a:
           use(entry.pathname, entry.size)
   ```

   You now work **format‑agnostically** and can point the same loop at `.zip`, `.7z`, `.tar.*`, `.iso`, etc. (subject to libarchive support in your runtime). ([manpages.debian.org][7])

2. **Extraction → use the high‑level helpers with secure defaults**
   Swap custom extraction code for `extract_file` (or `extract_fd`/`extract_memory`) and let the library handle the security flags. **Remember to `chdir`** to the target directory first. ([GitHub][1])

3. **Repack/convert** (e.g., tar → zip) **without hitting disk**
   When practical, read each entry’s data and write into a new writer:

   ```python
   import libarchive
   from libarchive.entry import FileType

   def copy_archive(src_path, dst_path, dst_format, dst_filter=None):
       with libarchive.file_reader(src_path) as r, \
            libarchive.file_writer(dst_path, dst_format, dst_filter) as w:
           for e in r:
               # Drain entry data to memory (stream blocks)
               buf = bytearray()
               for b in e.get_blocks():
                   buf.extend(b)
               # Re-emit with basic metadata; add more fields as needed
               w.add_file_from_memory(e.pathname, len(buf), bytes(buf),
                                      filetype=FileType.REGULAR)
   ```

   (If you need to carry across full metadata—uid/gid, perms, times—pass those explicitly in `add_file_from_memory`. This is still usually simpler than juggling multiple stdlib modules.) ([GitHub][1])

4. **Writing entirely from memory** (e.g., to send over HTTP)
   Use `memory_writer` to produce a bytes buffer instead of a file; pick `zip`, `ustar`, etc., and an optional filter (`gzip`, `xz`, `zstd`). ([GitHub][1])

5. **Filenames in legacy encodings** → set `header_codec="..."` on reader/writer. ([GitHub][1])

---

## 12) Choosing formats & filters (at runtime)

The exact *write* formats and *filters* depend on how `libarchive` was compiled on your system (e.g., whether zstd or lz4 were enabled). Program defensively:

```python
import libarchive
assert "zip" in libarchive.ffi.WRITE_FORMATS
assert "gzip" in libarchive.ffi.WRITE_FILTERS
```

For **read** behavior and corner cases (e.g., ZIP central directory vs streaming limitations), see the **libarchive‑formats(5)** man page—ZIP support is streaming but some self‑modifying/self‑extracting variants require central directory access that a streaming reader can’t always provide. ([manpages.debian.org][7])

---

## 13) Troubleshooting

* **“Works for tar but not for zip/7z”** → confirm your **libarchive** build includes those format modules/filters; inspect `ffi.WRITE_FORMATS`/`WRITE_FILTERS`. ([GitHub][1])
* **Windows import errors** → install via conda‑forge or ensure the `libarchive` DLL is available; otherwise `ctypes` can’t load it. ([Stack Overflow][4])
* **Non‑UTF8 metadata** → pass `header_codec=` on both read/write. ([GitHub][1])
* **Need decryption** → verify C library has crypto, and your binding exposes passphrase APIs; otherwise consider a format‑specific lib for that case. ([manpages.debian.org][9])

---

## 14) API quick lookup (copy/paste sheet)

* **Extract**:
  `libarchive.extract_file(path, flags=None)` · `extract_fd(fd, flags=None)` · `extract_memory(buf, flags=None)`
  *Flags mapped to* `archive_write_disk_set_options(3)`; secure defaults enabled since **5.3**. ([Linux Documentation][8])

* **Read**:
  `file_reader(path, header_codec=None)` · `memory_reader(buf, header_codec=None)` · `fd_reader(fd, ...)` · `stream_reader(obj_with_readinto, ...)` · `custom_reader(...)`
  Iterate entries; per entry: `.pathname`, `.size`, `.get_blocks()`. Track **`bytes_read`** for progress. ([GitHub][1])

* **Write**:
  `file_writer(path, format, filter=None, header_codec=None)` · `memory_writer(...)` · `fd_writer(...)` · `custom_writer(...)`
  Add content with `add_files(*paths, symlink_mode='preserve'|'logical')` and `add_file_from_memory(pathname, size, data, **metadata)`; valid formats/filters: `libarchive.ffi.WRITE_FORMATS` / `WRITE_FILTERS`. ([GitHub][1])

---

## 15) Differences vs the stdlib (`tarfile`/`zipfile`)

* **One API, many formats** (including `.7z` read, `.iso`, `.xar`, `.warc`, etc.) instead of juggling separate modules. Actual support depends on the C build + what you ask libarchive to enable. ([manpages.debian.org][7])
* **Streaming** by design (great for very large archives or sockets) vs some stdlib operations that assume random access (notably parts of `zipfile`). ([Arch Manual Pages][5])
* **Security**: `libarchive-c` now defaults to **secure extraction flags** that guard against traversal/symlink attacks, whereas stdlib extractors historically needed extra care. ([GitHub][6])

---

## 16) Version awareness (binding)

* **`libarchive-c` latest (as of May 22, 2025): 5.3.** Notable change: **secure extract flags enabled by default**; timestamp fields return `None` when unset (API cleanup). See release notes. ([GitHub][6])

---

## 17) Worked patterns (end‑to‑end)

### Pattern 1 — “Extract to a sandbox dir; reject absolute paths/symlinks”

```python
import os, libarchive
os.makedirs("/srv/sandbox/run-42", exist_ok=True)
os.chdir("/srv/sandbox/run-42")
libarchive.extract_file("/uploads/job42.tar.zst")  # secure by default
```

Secure defaults map to `archive_write_disk_set_options`; override `flags=` if you *intentionally* need different behavior. ([Linux Documentation][8])

### Pattern 2 — “List JSON files inside any archive without extracting”

```python
import libarchive, fnmatch
with libarchive.file_reader("any.format") as a:
    for e in a:
        if fnmatch.fnmatch(e.pathname, "**/*.json"):
            print(e.pathname, e.size)
```

Format‑agnostic listing & filtering. ([GitHub][1])

### Pattern 3 — “Repack to zip; carry over names, inject generated content”

```python
import libarchive
from libarchive.entry import FileType

with libarchive.file_reader("in.tar.xz") as r, \
     libarchive.file_writer("out.zip", "zip") as w:
    for e in r:                          # copy selected files
        buf = bytearray()
        for b in e.get_blocks(): buf.extend(b)
        w.add_file_from_memory(e.pathname, len(buf), bytes(buf),
                               filetype=FileType.REGULAR)

    # add an extra manifest
    manifest = b'{"ok": true}\n'
    w.add_file_from_memory("MANIFEST.json", len(manifest), manifest)
```

Use `.get_blocks()` to stream each file body, then re‑emit with `add_file_from_memory`. Add more metadata if you need exact preservation. ([GitHub][1])

### Pattern 4 — “Archive a directory tree from disk with specific encoding”

```python
with libarchive.file_writer("out.tar", "ustar", header_codec="cp037") as w:
    w.add_files("legacy-cp037-dir/")
```

The `header_codec` applies to pathnames & user/group names. ([GitHub][1])

---

## 18) One‑paragraph note on the **other** binding (`python-libarchive`)

If your environment uses **`python-libarchive`** (SWIG wrapper), you’ll see modules like `libarchive.public` and APIs closer to the C surface (plus “compat layers” for `zipfile`/`tarfile`). It is a different project, last released **2022‑09‑27**; verify its security posture if you use its extract helpers. If you’re starting fresh, prefer **`libarchive-c`** for a compact, Pythonic surface. ([PyPI][10])

---

### Sources & further reading

* **`libarchive-c` README** (install, usage, readers/writers, header codec, symlink policy, progress). ([GitHub][1])
* **Release notes 5.3** (secure extract flags default). ([GitHub][6])
* **libarchive (C)** overview & features; **streaming/no random access** design. ([libarchive.org][2])
* **Man pages**: `archive_write_disk(3)` (extract flags), `libarchive-formats(5)` (format notes), index of libarchive manpages. ([Linux Documentation][8])
* **Conda‑forge package** (cross‑platform binaries for Windows/macOS/Linux). ([Anaconda][3])

---

If you paste a snippet of your current archiving/extraction code (or name the formats you must support), I can map each part onto the exact `libarchive` reader/writer/extractor calls and produce a ready‑to‑apply refactor.

[1]: https://github.com/Changaco/python-libarchive-c "GitHub - Changaco/python-libarchive-c: Python interface to libarchive"
[2]: https://www.libarchive.org/?utm_source=chatgpt.com "libarchive - C library and command-line tools for reading and ..."
[3]: https://anaconda.org/conda-forge/python-libarchive-c?utm_source=chatgpt.com "Python Libarchive C"
[4]: https://stackoverflow.com/questions/37165002/using-libarchive-in-python-on-windows?utm_source=chatgpt.com "Using libarchive in python on Windows"
[5]: https://man.archlinux.org/man/libarchive.3.en?utm_source=chatgpt.com "libarchive(3) - Arch manual pages"
[6]: https://github.com/Changaco/python-libarchive-c/releases?utm_source=chatgpt.com "Releases · Changaco/python-libarchive-c"
[7]: https://manpages.debian.org/testing/libarchive-dev/libarchive-formats.5.en.html "libarchive-formats(5) — libarchive-dev — Debian testing — Debian Manpages"
[8]: https://linux.die.net/man/3/archive_write_disk?utm_source=chatgpt.com "archive_write_disk(3) - Linux man page"
[9]: https://manpages.debian.org/testing/libarchive-dev/archive_read_add_passphrase.3.en.html?utm_source=chatgpt.com "archive_read_add_passphrase(3) — libarchive-dev"
[10]: https://pypi.org/project/python-libarchive/?utm_source=chatgpt.com "python-libarchive"
