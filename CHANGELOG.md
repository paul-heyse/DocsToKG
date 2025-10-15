# Changelog

## Unreleased

### Added
- Migration guide for the modular resolver architecture, including new
  configuration defaults.
- Developer documentation for adding custom resolver providers and extending the
  registry.
- Property-based tests covering retry backoff, conditional request headers, and
  dedupe behaviour to lift branch coverage.

### Changed
- Resolver namespace now emits deprecation warnings when importing legacy
  ``time`` or ``requests`` aliases.
- ``ResolverPipeline`` captures resolver exceptions consistently across
  sequential and concurrent execution paths.
- README and API docs updated to highlight Zenodo/Figshare resolvers, bounded
  concurrency, and HEAD pre-check filtering.

### Fixed
- HTTP retry helper now validates parameters early and exposes clearer error
  messages for invalid usage.
- Conditional request helper rejects negative content lengths and validates
  response shapes before processing HTTP 304 outcomes.
