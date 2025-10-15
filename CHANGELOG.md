# Changelog

All notable changes to DocsToKG are documented in this file.

## [Unreleased]

### Added
- Zenodo and Figshare resolvers integrated into the modular content download pipeline with defensive error handling and metadata-rich events.
- Bounded intra-work concurrency controls and HEAD pre-check filtering options documented in the README and resolver guides.
- Migration and developer guides covering resolver import paths, configuration, and extensibility patterns.
- Architecture diagram illustrating the resolver pipeline, conditional request helper, and logging flow.

### Changed
- Centralised HTTP retry helper now logs timeout/connection issues separately and emits warnings when retries are exhausted.
- Conditional request helper surfaces detailed error messages when cache metadata is incomplete.
- All resolver providers emit structured error events for HTTP, timeout, connection, and JSON failures.

### Fixed
- Tests cover HEAD pre-check redirects, resolver concurrency error isolation, and configuration validation edge cases.
- Property-based tests ensure conditional request headers and deduplication utilities behave correctly across arbitrary inputs.
