# 1. Module: cancellation

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.cancellation``.

## 1. Overview

Cancellation token implementation for cooperative task cancellation.

## 2. Functions

### `cancel(self)`

Signal that cancellation has been requested.

### `is_cancelled(self)`

Check if cancellation has been requested.

Returns:
True if cancellation has been requested, False otherwise.

### `reset(self)`

Reset the cancellation token to its initial state.

This should only be used for testing or when reusing tokens
in controlled scenarios.

### `add_token(self, token)`

Add a token to this group.

Args:
token: The cancellation token to add to the group.

### `create_token(self)`

Create a new token and add it to this group.

Returns:
A new cancellation token that is part of this group.

### `remove_token(self, token)`

Remove ``token`` from this group if it is present.

This allows callers to release tokens once work associated with them
has completed, keeping the group size accurate for monitoring and
follow-up operations.

### `cancel_all(self)`

Cancel all tokens in this group.

### `is_any_cancelled(self)`

Check if any token in the group has been cancelled.

Returns:
True if any token in the group has been cancelled.

### `__len__(self)`

Return the number of tokens in this group.

## 3. Classes

### `CancellationToken`

Thread-safe cancellation token for cooperative task cancellation.

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

### `CancellationTokenGroup`

A group of cancellation tokens that can be cancelled together.

This is useful when you need to cancel multiple related tasks
simultaneously, such as all tasks in a batch operation.
