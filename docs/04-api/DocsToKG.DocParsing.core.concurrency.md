# 1. Module: concurrency

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core.concurrency``.

## 1. Overview

Concurrency utilities shared across DocParsing stages.

## 2. Functions

### `acquire_lock(path, timeout)`

Acquire an advisory lock using ``.lock`` sentinel files.

### `_pid_is_running(pid)`

Return ``True`` if a process with the given PID appears to be alive.

### `_read_lock_owner(lock_path)`

Read the PID stored in ``lock_path`` and return both the parsed and raw forms.

### `_evict_stale_lock(lock_path)`

Remove a stale lock file after a brief jitter to avoid thundering herds.

### `set_spawn_or_warn(logger)`

Ensure the multiprocessing start method is set to ``spawn``.

### `_bind_reserved_socket(host, port)`

Bind a socket to the specified host and port, returning None on failure.

### `find_free_port(start, span)`

Reserve an available TCP port and return a context manager guarding it.

The returned context manager keeps the port reserved by holding an open
listening socket. Callers should release the reservation (either by exiting
the context or invoking :meth:`ReservedPort.close`) only when the consumer
of the port is ready to bind and accept connections.

### `__enter__(self)`

Enter the context manager and return self.

### `__exit__(self, exc_type, exc, tb)`

Exit the context manager and close the socket.

### `socket(self)`

Return the underlying socket that keeps the reservation alive.

### `port(self)`

Return the port reserved by this context.

### `host(self)`

Return the host interface the reservation is bound to.

### `close(self)`

Release the reservation if it is currently held.

## 3. Classes

### `ReservedPort`

Context manager representing a reserved TCP port.
