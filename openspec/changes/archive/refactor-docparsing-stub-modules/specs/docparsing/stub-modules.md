## ADDED Requirements
### Requirement: Static DocParsing Fake Dependencies
The docparsing integration test suite MUST load its optional dependency fakes from importable Python modules instead of on-the-fly `ModuleType` injections.

#### Scenario: MyPy Scans Test Fakes
- **GIVEN** the `pre-commit` MyPy hook runs across the repository
- **WHEN** it inspects the docparsing fake dependency package
- **THEN** no `attr-defined` errors are emitted because each fake module exposes explicit attributes.

#### Scenario: Tests Request Dependency Stubs
- **GIVEN** a test calls `tests.docparsing.stubs.dependency_stubs()`
- **WHEN** the helper executes
- **THEN** the static fake dependency package is inserted on `sys.path`
- **AND** subsequent imports resolve to the fake modules without relying on dynamic module registration.

#### Scenario: Extending Fake Dependencies
- **GIVEN** a contributor needs to add behaviour to a fake dependency used by docparsing tests
- **WHEN** they add or modify code under `tests/docparsing/fake_deps/`
- **THEN** the change preserves the module layout that mirrors the production dependency namespace
- **AND** updated fakes continue to be type-checkable by MyPy.
- **AND** the contributor updates the fake package documentation (README / migration notes) listing the new exports.
