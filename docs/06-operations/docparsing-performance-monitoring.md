# DocParsing performance monitoring

This guide documents the workflow introduced by `docparse perf run`. It covers
fixture preparation, artifact layout, baseline rotation, and automation
recommendations for CI or nightly builds.

## Fixture selection

`docparse perf run` bootstraps a deterministic HTML fixture under
`${DOCSTOKG_DATA_ROOT}/Profiles/<name>/HTML/` and reuses the same directory for
DocTags, chunk, and BM25 embedding outputs. The default `synthetic-html` fixture
creates five HTML files with repeated prose so each run operates on identical
inputs. Override the document count via `--documents` when you need a larger
sample for regression detection.

All stage commands honour the standard DocParsing manifests, so rerunning the
profiling command with `--resume` is safe—the CLI will skip work when manifests
already contain successful records. Pass `--no-profile` to disable cProfile when
only timing data is required.

## Artifact layout

Each invocation writes into `${DOCSTOKG_DATA_ROOT}/Profiles/<fixture>/runs/<timestamp>/` by default
(override with `--output-dir`). The directory contains:

- `summary.json` – aggregated metrics for every executed stage. The file is
  consumable by CI jobs and the `docparse perf compare` helper.
- `<stage>.metrics.json` – per-stage wall time, CPU time, RSS, and exit code.
- `<stage>.stdout.log` / `<stage>.stderr.log` – captured command output.
- `<stage>.pstats` (optional) – raw cProfile statistics when profiling is enabled.
- `<stage>.collapsed.txt` (optional) – collapsed stacks derived from the
  cProfile dump for flamegraph tooling (`inferno`, `speedscope`, etc.).

Older runs can be trimmed by deleting their timestamped directories. Keep at
least one historical run per hardware tier so baseline comparisons remain
meaningful.

## Baseline rotation and regression budgets

Store a canonical baseline (for example `Data/Profiles/baseline.json`) with the
`summary.json` payload captured from a known-good run. Subsequent executions can
consume that baseline with `docparse perf run --baseline Data/Profiles/baseline.json`.
The CLI exposes fractional thresholds via `--wall-threshold`, `--cpu-threshold`,
and `--rss-threshold`; values represent acceptable regressions relative to the
baseline. The default 15 % (wall/CPU) and 20 % (RSS) serve as guardrails for
nightly monitoring and can be tightened when the pipeline stabilises.

When regressions are detected the command exits with status code 2. CI pipelines
should treat this as a failure and upload the corresponding run directory for
inspection. Improvements and stable runs are surfaced on stdout for operator
awareness but do not change the exit code.

## CI and cron integration

A minimal GitHub Actions job that runs nightly might look like:

```yaml
name: docparse-perf
on:
  schedule:
    - cron: "30 6 * * *"  # 06:30 UTC
jobs:
  perf:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: ./scripts/bootstrap_env.sh
      - run: direnv exec . python -m DocsToKG.DocParsing.cli perf run \
            --output-dir Data/Profiles/nightly \
            --baseline Data/Profiles/baseline.json \
            --documents 10
      - name: Upload artifacts on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: docparse-perf-nightly
          path: Data/Profiles/nightly/*
```

Rotate the baseline whenever a material improvement lands. Commit the new
`summary.json` under `Data/Profiles/baseline.json` (or your chosen location) so
future runs compare against the improved numbers.

## Troubleshooting

- **Missing optional dependencies** – the perf command executes the real stage
  CLIs; ensure Docling, vLLM, and other extras are installed in environments
  where those stages are profiled. Use `--stage` to limit runs to the available
  components.
- **Flamegraph tooling** – the collapsed stack files follow the standard
  "stack count" format with `/` separators. Convert them to SVG with
  `inferno --input <stage>.collapsed.txt --output flame.svg` or import them into
  [speedscope](https://www.speedscope.app/) after replacing `/` with `;`.
- **False positives** – adjust thresholds or increase `--documents` if natural
  variance triggers frequent regressions. Consider pinning CPU frequency or
  reserving dedicated runners for consistent baselines.

For additional context see the high-level summary in the
[DocParsing README](../../src/DocsToKG/DocParsing/README.md#performance-monitoring).
