"""Cooperative cancellation primitives shared by long-running download tasks.

The ontology downloader runs concurrent planners, resolvers, and validators.
This module offers the light-weight :class:`CancellationToken` used to stop
those routines gracefully, along with :class:`CancellationTokenGroup` for
broadcast-style cancellation across related tasks (for example, all resolvers
within a batch).  The implementation intentionally avoids thread interruption
in favour of explicit checks so that cleanup code can persist manifests,
diagnostics, and partial downloads predictably.
"""

from __future__ import annotations

import threading


class CancellationToken:
    """Thread-safe cancellation token for cooperative task cancellation.

    This token allows tasks to be cancelled cooperatively by checking
    the token's status periodically rather than relying on thread
    interruption mechanisms.

    Examples:
        >>> token = CancellationToken()
        >>> # In a task
        >>> if token.is_cancelled():
        ...     return  # Exit gracefully
        >>> # From another thread
        >>> token.cancel()
    """

    def __init__(self) -> None:
        """Initialize a new cancellation token."""
        self._is_cancelled = threading.Event()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Signal that cancellation has been requested."""
        with self._lock:
            self._is_cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancellation has been requested, False otherwise.
        """
        return self._is_cancelled.is_set()

    def reset(self) -> None:
        """Reset the cancellation token to its initial state.

        This should only be used for testing or when reusing tokens
        in controlled scenarios.
        """
        with self._lock:
            self._is_cancelled.clear()


class CancellationTokenGroup:
    """A group of cancellation tokens that can be cancelled together.

    This is useful when you need to cancel multiple related tasks
    simultaneously, such as all tasks in a batch operation.
    """

    def __init__(self) -> None:
        """Initialize an empty token group."""
        self._tokens: list[CancellationToken] = []
        self._lock = threading.Lock()
        self._cancelled = False

    def add_token(self, token: CancellationToken) -> None:
        """Add a token to this group.

        Args:
            token: The cancellation token to add to the group.
        """
        with self._lock:
            self._tokens.append(token)
            if self._cancelled:
                token.cancel()

    def create_token(self) -> CancellationToken:
        """Create a new token and add it to this group.

        Returns:
            A new cancellation token that is part of this group.
        """
        token = CancellationToken()
        self.add_token(token)
        return token

    def remove_token(self, token: CancellationToken) -> None:
        """Remove ``token`` from this group if it is present.

        This allows callers to release tokens once work associated with them
        has completed, keeping the group size accurate for monitoring and
        follow-up operations.
        """

        with self._lock:
            try:
                self._tokens.remove(token)
            except ValueError:
                # Token may have already been removed or was never part of the
                # group; treat as a no-op for convenience.
                pass

    def cancel_all(self) -> None:
        """Cancel all tokens in this group."""
        with self._lock:
            self._cancelled = True
            for token in self._tokens:
                token.cancel()

    def is_any_cancelled(self) -> bool:
        """Check if any token in the group has been cancelled.

        Returns:
            True if any token in the group has been cancelled.
        """
        with self._lock:
            return any(token.is_cancelled() for token in self._tokens)

    def __len__(self) -> int:
        """Return the number of tokens in this group."""
        with self._lock:
            return len(self._tokens)
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.cancellation",
#   "purpose": "Provide cooperative cancellation tokens shared by planners, resolvers, and validators",
#   "sections": [
#     {"id": "token", "name": "CancellationToken", "anchor": "TOK", "kind": "api"},
#     {"id": "group", "name": "CancellationTokenGroup", "anchor": "GRP", "kind": "api"}
#   ]
# }
# === /NAVMAP ===
