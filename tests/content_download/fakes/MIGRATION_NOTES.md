# Extending the Content Download Fakes

1. Prefer reusing the doc parsing fake modules when a dependency is shared.
   Call :func:`tests.docparsing.stubs.dependency_stubs` before registering any
   new module names so the shared tree is present.
2. Add new fake modules under ``tests/content_download/fakes`` when the content
   download suite touches a dependency that doc parsing does not already cover.
   Keep exports minimal but typed.
3. Update :mod:`tests.content_download.stubs` to register the new module names
   and document the addition in this file so future contributors know why the
   shim exists.
4. Refresh the mypy baseline (`openspec/changes/refactor-content-download-mypy/mypy-baseline.md`)
   whenever the fakes expand to track the expected diagnostics.
