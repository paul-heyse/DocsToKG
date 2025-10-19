# 1. Module: args

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.args``.

## 1. Overview

Command-line argument parsing for DocsToKG content downloads.

Provides helpers that transform CLI inputs into structured configuration
objects used by the downloader. The utilities here manage resolver bootstrap,
run directory preparation, and OpenAlex query assembly without triggering any
network activity at import time.

## 2. Functions

### `bootstrap_run_environment(resolved)`

Initialise directories required for a resolved download run.

### `build_parser()`

Create and return the CLI argument parser.

### `parse_args(parser, argv)`

Parse CLI arguments using ``parser`` and optional argv override.

### `resolve_config(args, parser, resolver_factory)`

Validate arguments, resolve configuration, and prepare run state.

### `_parse_size(value)`

Parse human-friendly size strings (e.g., ``10MB``) into bytes.

### `_parse_domain_interval(value)`

Parse ``DOMAIN=SECONDS`` CLI arguments for domain throttling.

Args:
value: Argument provided via ``--domain-min-interval``.

Returns:
Tuple containing the normalized domain name and interval seconds.

Raises:
argparse.ArgumentTypeError: If the argument is malformed or negative.

### `_parse_domain_token_bucket(value)`

Parse ``DOMAIN=RPS[:capacity=X]`` specifications into bucket configs.

### `build_query(args)`

Build a pyalex Works query based on CLI arguments.

Args:
args: Parsed command-line arguments.

Returns:
Configured Works query object ready for iteration.

### `_lookup_topic_id(topic_text)`

Cached helper to resolve an OpenAlex topic identifier.

### `resolve_topic_id_if_needed(topic_text)`

Resolve a textual topic label into an OpenAlex topic identifier.

Args:
topic_text: Free-form topic text supplied via CLI.

Returns:
OpenAlex topic identifier string if resolved, else None.

### `_expand_path(value)`

*No documentation available.*

### `_normalise_order(order)`

*No documentation available.*

## 3. Classes

### `ResolvedConfig`

Immutable configuration derived from CLI arguments.

The dataclass is frozen to prevent callers from mutating configuration at
runtime. Any operational side effects (filesystem initialisation, telemetry
bootstrapping, etc.) must be performed explicitly via helper functions such
as :func:`bootstrap_run_environment` rather than during configuration
resolution.
