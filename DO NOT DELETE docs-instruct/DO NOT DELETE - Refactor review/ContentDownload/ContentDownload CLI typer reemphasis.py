# PR #4 — CLI polish (Typer), config introspection, and operator UX

> Paste this whole document into `docs/pr4-cli.md` (or your PR description).
> It includes a compiling file tree, ready-to-copy code, and tests.

---

## Goals

1. Ship a clean **Typer CLI**: `run`, `print-config`, `validate-config`, `explain`.
2. Respect **merged config precedence**: file ⊕ env ⊕ CLI.
3. Provide **introspection** (effective resolver order, enabled/disabled, key policies).
4. Prepare for PR #5 by delegating to a seam: `run_from_config(cfg)` (wired fully in PR #5).

**Assumptions:** PRs #1–#3 are merged (telemetry plumbing, atomic writer, Pydantic v2 config+registry, canonical API types).

---

## New/updated file tree

```text
src/DocsToKG/ContentDownload/
  cli/
    __init__.py            # NEW: exports `app`
    app.py                 # NEW: Typer application with commands
  config/
    loader.py              # (existing) load/merge file/env/CLI -> ContentDownloadConfig
    models.py              # (existing) Pydantic v2 models
  resolvers/
    __init__.py            # (existing) registry helpers (get_registry, build_resolvers)
  bootstrap.py             # NEW: thin seam (run_from_config) that PR#5 will flesh out
tests/
  contentdownload/
    test_cli_basic.py      # NEW: run/print/validate/explain tests via CliRunner
```

---

## 1) `src/DocsToKG/ContentDownload/cli/app.py`

```python
# src/DocsToKG/ContentDownload/cli/app.py
from __future__ import annotations

import json
from typing import Optional, Dict, Any, List

import typer

from DocsToKG.ContentDownload.config.loader import load_config
from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
from DocsToKG.ContentDownload.resolvers import get_registry
from DocsToKG.ContentDownload.bootstrap import run_from_config  # seam for PR#5


app = typer.Typer(help="DocsToKG ContentDownload CLI")


def _comma_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value == "":
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _cli_overrides(
    resolver_order: Optional[str],
    no_robots: bool,
    no_atomic: bool,
    chunk_size: Optional[int],
) -> Dict[str, Any]:
    """Build a nested dict compatible with ContentDownloadConfig from simple flags."""
    overrides: Dict[str, Any] = {}

    if resolver_order:
        overrides.setdefault("resolvers", {})["order"] = _comma_list(resolver_order)

    if no_robots:
        overrides.setdefault("robots", {})["enabled"] = False

    if no_atomic:
        overrides.setdefault("download", {})["atomic_write"] = False

    if chunk_size is not None:
        overrides.setdefault("download", {})["chunk_size_bytes"] = int(chunk_size)

    return overrides


@app.command("run")
def run(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML/JSON config (file precedence)."
    ),
    resolver_order: Optional[str] = typer.Option(
        None,
        "--resolver-order",
        help='Override resolver order as a comma list, e.g. "unpaywall,crossref,landing"',
    ),
    no_robots: bool = typer.Option(False, "--no-robots", help="Disable robots.txt checks."),
    no_atomic: bool = typer.Option(False, "--no-atomic-write", help="Disable atomic writes (debug)."),
    chunk_size: Optional[int] = typer.Option(
        None, "--chunk-size", help="Override download chunk size (bytes)."
    ),
):
    """
    Run the ContentDownload pipeline using the effective config (file ⊕ env ⊕ CLI).
    Delegates to `run_from_config(cfg)` (implemented fully in PR#5).
    """
    overrides = _cli_overrides(resolver_order, no_robots, no_atomic, chunk_size)
    cfg: ContentDownloadConfig = load_config(config, cli_overrides=overrides)
    run_from_config(cfg)


@app.command("print-config")
def print_config(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty-print JSON."),
):
    """
    Print the effective merged configuration as JSON (stable, machine-readable).
    """
    cfg = load_config(config)
    payload = cfg.model_dump(mode="json")
    text = json.dumps(payload, indent=2 if pretty else None, sort_keys=True)
    typer.echo(text)


@app.command("validate-config")
def validate_config(
    config: Optional[str] = typer.Option(..., "--config", "-c", help="Path to config to validate."),
):
    """
    Validate a config file (or fail with a descriptive error).
    """
    try:
        _ = load_config(config)
        typer.echo("OK")
    except Exception as e:  # pydantic ValidationError surfaces messages cleanly
        typer.secho(f"INVALID: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("explain")
def explain(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    show_policies: bool = typer.Option(
        True, "--show-policies/--no-policies", help="Print key per-resolver policies."
    ),
):
    """
    Explain the effective resolver plan and (optionally) per-resolver policies.
    """
    cfg = load_config(config)
    reg = get_registry()

    order = list(cfg.resolvers.order)
    typer.secho("Resolver order:", fg=typer.colors.CYAN, bold=True)
    typer.echo("  " + ", ".join(order))

    missing = [name for name in order if name not in reg]
    if missing:
        typer.secho("Missing / unregistered resolvers:", fg=typer.colors.YELLOW)
        for name in missing:
            typer.echo(f"  - {name}")

    disabled = [name for name in order if not getattr(cfg.resolvers, name).enabled]
    if disabled:
        typer.secho("Disabled resolvers:", fg=typer.colors.YELLOW)
        for name in disabled:
            typer.echo(f"  - {name}")

    if show_policies:
        typer.secho("\nPolicies:", fg=typer.colors.CYAN, bold=True)
        for name in order:
            rcfg = getattr(cfg.resolvers, name, None)
            if rcfg is None:
                continue
            typer.echo(f"  [{name}] enabled={rcfg.enabled}")
            rl = rcfg.rate_limit
            rp = rcfg.retry
            typer.echo(
                f"    rate_limit: capacity={rl.capacity}, refill_per_sec={rl.refill_per_sec}, burst={rl.burst}"
            )
            typer.echo(
                f"    retry: max_attempts={rp.max_attempts}, statuses={rp.retry_statuses}, base_delay_ms={rp.base_delay_ms}, max_delay_ms={rp.max_delay_ms}"
            )
            if rcfg.timeout_read_s is not None:
                typer.echo(f"    timeout_read_s: {rcfg.timeout_read_s}")

    # Also show a few global knobs operators care about
    typer.secho("\nGlobal:", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  robots.enabled: {cfg.robots.enabled} (ttl={cfg.robots.ttl_seconds}s)")
    typer.echo(
        f"  download: atomic_write={cfg.download.atomic_write}, verify_content_length={cfg.download.verify_content_length}, chunk_size={cfg.download.chunk_size_bytes}"
    )
    typer.echo(f"  http.user_agent: {cfg.http.user_agent}")
    if cfg.http.proxies:
        typer.echo(f"  http.proxies: {cfg.http.proxies}")
```

---

## 2) `src/DocsToKG/ContentDownload/cli/__init__.py`

```python
# src/DocsToKG/ContentDownload/cli/__init__.py
from .app import app

__all__ = ["app"]
```

---

## 3) `src/DocsToKG/ContentDownload/bootstrap.py`

```python
# src/DocsToKG/ContentDownload/bootstrap.py
from __future__ import annotations

from DocsToKG.ContentDownload.config.models import ContentDownloadConfig


def run_from_config(cfg: ContentDownloadConfig) -> None:
    """
    Thin seam used by the CLI to start a run.

    PR#4: This can call a legacy runner if needed (to keep behavior unchanged),
          or simply be a placeholder.
    PR#5: Wire resolvers, telemetry sinks, HTTP client, and pipeline here, driven by `cfg`.
    """
    # --- Placeholder for PR#4 ---
    # Replace this with the actual bootstrapping in PR#5.
    # For now, we do nothing or call an existing legacy entrypoint if present, e.g.:
    #
    # from DocsToKG.ContentDownload.legacy_runner import main
    # main(cfg)
    #
    # Keeping it a no-op avoids coupling PR#4 to the PR#5 wiring changes.
    return None
```

---

## 4) Packaging entry point (`pyproject.toml`)

```toml
# pyproject.toml
[project.scripts]
contentdownload = "DocsToKG.ContentDownload.cli:app"
```

> Typer provides `--help` and `--install-completion` out of the box.

---

## 5) Tests — `tests/contentdownload/test_cli_basic.py`

```python
# tests/contentdownload/test_cli_basic.py
from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner
from DocsToKG.ContentDownload.cli.app import app

runner = CliRunner()


def test_print_config_smoke(monkeypatch, tmp_path):
    cfg_text = """
    run_id: "test-run"
    http:
      user_agent: "UA-Test"
    resolvers:
      order: ["unpaywall","landing"]
      unpaywall: { enabled: true }
      landing: { enabled: true }
    """
    cfg_file = tmp_path / "cd.yaml"
    cfg_file.write_text(cfg_text)

    result = runner.invoke(app, ["print-config", "-c", str(cfg_file)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["run_id"] == "test-run"
    assert data["http"]["user_agent"] == "UA-Test"
    assert data["resolvers"]["order"] == ["unpaywall", "landing"]


def test_validate_config_ok(tmp_path):
    cfg_file = tmp_path / "cd.json"
    cfg_file.write_text(json.dumps({"resolvers": {"order": ["unpaywall"]}}))
    result = runner.invoke(app, ["validate-config", "-c", str(cfg_file)])
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_validate_config_fail(tmp_path):
    # Unknown extra field triggers pydantic "extra=forbid" error
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("resolvers: { order: [unpaywall], unknown_key: 1 }")
    result = runner.invoke(app, ["validate-config", "-c", str(cfg_file)])
    assert result.exit_code != 0
    assert "INVALID:" in result.stdout


def test_explain_order_and_policies(tmp_path):
    cfg_text = """
    resolvers:
      order: ["unpaywall","landing"]
      unpaywall:
        enabled: true
        rate_limit: { capacity: 2, refill_per_sec: 0.5, burst: 1 }
        retry: { max_attempts: 3, retry_statuses: [429,500,503], base_delay_ms: 100, max_delay_ms: 1000 }
      landing:
        enabled: false
    """
    cfg_file = tmp_path / "cd.yaml"
    cfg_file.write_text(cfg_text)
    result = runner.invoke(app, ["explain", "-c", str(cfg_file)])
    assert result.exit_code == 0
    # Order printed
    assert "Resolver order:" in result.stdout
    assert "unpaywall, landing" in result.stdout
    # Disabled printed
    assert "Disabled resolvers:" in result.stdout
    assert "landing" in result.stdout


def test_run_invokes_bootstrap(monkeypatch, tmp_path):
    cfg_file = tmp_path / "cd.json"
    cfg_file.write_text(json.dumps({"resolvers": {"order": ["unpaywall"]}}))

    called: dict[str, Any] = {"count": 0}

    # Monkeypatch bootstrap.run_from_config to observe invocation
    import DocsToKG.ContentDownload.bootstrap as bootstrap

    def fake_run_from_config(cfg):
        called["count"] += 1

    monkeypatch.setattr(bootstrap, "run_from_config", fake_run_from_config, raising=True)
    result = runner.invoke(app, ["run", "-c", str(cfg_file), "--resolver-order", "unpaywall,landing"])
    assert result.exit_code == 0
    assert called["count"] == 1
```

---

## Optional: `config-schema` command

If you want an editor-friendly JSON Schema for your config:

```python
# Append to src/DocsToKG/ContentDownload/cli/app.py
@app.command("config-schema")
def config_schema():
    from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
    typer.echo(json.dumps(ContentDownloadConfig.model_json_schema(), indent=2, sort_keys=True))
```

---

## How to try it locally

```bash
# 1) Install your package in editable mode
pip install -e .

# 2) See commands
contentdownload --help

# 3) Print merged config (file ⊕ env ⊕ CLI)
contentdownload print-config -c cd.yaml

# 4) Validate a config
contentdownload validate-config -c cd.yaml

# 5) Explain resolver plan
contentdownload explain -c cd.yaml

# 6) Run (delegates to run_from_config, fully wired in PR#5)
contentdownload run -c cd.yaml --resolver-order unpaywall,landing --no-robots --chunk-size 2097152
```

---

## Acceptance checklist

* [ ] `contentdownload --help` shows `run`, `print-config`, `validate-config`, `explain`.
* [ ] `print-config` prints merged config (file ⊕ env ⊕ CLI).
* [ ] `validate-config` returns 0 on valid, non-zero with clear error on invalid.
* [ ] `explain` prints resolver order, missing/disabled resolvers, and core policies.
* [ ] `run` calls `run_from_config(cfg)` (seam for PR #5).
* [ ] Tests (CliRunner) pass locally/CI.
* [ ] Entry point added to `pyproject.toml` and works in a fresh venv.

---

## Risks & mitigations

* **Coupling to PR #5**: kept minimal via `run_from_config` seam.
* **Config drift/typos**: Pydantic `extra="forbid"` + `validate-config`.
* **Unknown resolvers**: surfaced by `explain` before a run.

---

## Follow-up (PR #5)

Implement `bootstrap.run_from_config(cfg)` to:

1. Instantiate telemetry sinks (`cfg.telemetry`).
2. Build HTTP session from `cfg.http` (timeouts, TLS, proxies).
3. Materialize resolvers via registry (`build_resolvers(cfg.resolvers.order, cfg)`).
4. Construct and run `ResolverPipeline(session, telemetry, run_id=cfg.run_id)` over work items.
5. Respect `cfg.download` & `cfg.robots` in download execution.
