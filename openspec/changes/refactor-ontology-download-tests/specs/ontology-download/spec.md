## ADDED Requirements

### Requirement: Testing Harness Executes Real Pipeline
The ontology download package MUST expose a testing harness that drives the same download, storage, and validation pipeline used in production without patching internal modules.

#### Scenario: Loopback HTTP download
- **WHEN** `DocsToKG.OntologyDownload.testing.TestingEnvironment` is used with an ontology fixture
- **THEN** the harness serves the fixture over a loopback HTTP endpoint and returns a `ResolvedConfig` wired for that endpoint
- **AND** calling `core.fetch_one` retrieves the artifact, writes manifests, and records validation results using the production pipeline.

### Requirement: Runtime Resolver Registration API
The package MUST provide public functions to register and unregister resolvers (and validators) at runtime with automatic restoration.

#### Scenario: Temporary resolver registration
- **WHEN** a test calls `temporary_resolver("fixture", resolver_instance)`
- **THEN** `core.plan_all` can resolve specifications using the new resolver
- **AND** after the context exits, the resolver registry is restored to its prior state without manual cleanup.

### Requirement: Deterministic Test Isolation
The testing harness MUST guarantee isolated caches, storage, and rate limiter state per invocation so tests cannot leak state across runs.

#### Scenario: Independent harness invocations
- **WHEN** two `TestingEnvironment` contexts run sequentially within the same process
- **THEN** each receives unique cache and ontology directories
- **AND** token bucket / session pool state is reset between runs, ensuring downloads in the second context do not reuse artifacts created by the first.
