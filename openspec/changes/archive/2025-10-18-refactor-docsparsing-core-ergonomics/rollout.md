# Rollout Notes

**Audience:** DocParsing operators and downstream automation owners

- Core helpers now live under `DocsToKG.DocParsing.core.*`. The top-level
  `DocsToKG.DocParsing.core` module continues to re-export the stable API, so
  no import changes are required, but new code should target the focused
  submodules when contributing internals.
- Structural marker loaders are public (`DocsToKG.DocParsing.config.load_yaml_markers`
  and `load_toml_markers`) with structured error messages. Update any internal
  tooling that previously reached into underscored helpers.
- CLI invocation failures now exit with code `2` and emit `[stage] --option: ...`
  messages. Replace traceback parsing in automation scripts with exit-code
  handling.
- Embedding runtime no longer imports `sentence_transformers` / `vllm` eagerly;
  install the optional extras only when using SPLADE or Qwen embeddings.

**Recommended communication:**

```
Hi DocParsing folks,

We just merged the core ergonomics refactor:
- `DocsToKG.DocParsing.core` is now a facade backed by discovery/http/manifest/
  planning/cli_utils submodules.
- Structural marker loaders are public helpers with better error messages.
- CLI validation exits with `[stage] --option: message` instead of tracebacks.
- Embedding imports for SPLADE/Qwen are lazy; you only need the heavy deps when
  running those stages.

No import changes are required; existing automation keeps working. Please
update any scripts that scraped tracebacks to rely on exit codes instead.
```
