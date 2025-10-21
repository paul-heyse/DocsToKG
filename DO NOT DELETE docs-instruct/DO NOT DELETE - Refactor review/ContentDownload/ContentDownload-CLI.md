Absolutely — here’s a **single PR doc** you can paste into your repo or PR description for **PR #4 — CLI polish**. It contains the *why*, a compiling **file tree**, **drop-in code** for the CLI (Typer), and **tests** using `CliRunner`. It assumes PR#1–#3 are in flight/merged (telemetry plumbing, atomic writer, config models+registry, canonical API types). It does **not** force “wire-everything” yet—that’s PR#5.

---

# PR #4 — CLI polish (Typer), config introspection, and operator UX

## Goals

1. Ship a clean **Typer CLI**: `run`, `print-config`, `validate-config`, `explain`.
2. Respect **merged config precedence**: file ⊕ env ⊕ CLI.
3. Provide **introspection** (effective resolver order, enabled/disabled, key policies).
4. Prepare for PR#5 by calling a single `run_from_config(cfg)` bootstrap hook (which you’ll implement/finish in PR#5).

---

## New/Updated file tree

```text
src/DocsToKG/ContentDownload/
  cli/
    __init__.py            # NEW: exports `app`
    app.py                 # NEW: Typer application with commands
  config/
    loader.py              # (existing from PR#1/2) used here
    models.py              # (existing) ContentDownloadConfig
  resolvers/
    __init__.py            # (existing from PR#2) registry helpers
  bootstrap.py             # NEW: thin seam that PR#5 will flesh out (run_from_config)
tests/
  contentdownload/
    test_cli_basic.py      # NEW: run/print/validate/explain tests via CliRunner
```

> `bootstrap.py` is intentionally tiny: it holds a single `run_from_config(cfg)` function. In this PR, it may be a no-op or call your legacy runner. In **PR#5**, we wire it to the new pipeline/config end-to-end.

---

## 1) `src/DocsToKG/ContentDownload/cli/app.py` (NEW)

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
    # In PR#4 we keep bootstrapping minimal; PR#5 wires actual pipeline & sinks.
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

## 2) `src/DocsToKG/ContentDownload/cli/__init__.py` (NEW)

```python
# src/DocsToKG/ContentDownload/cli/__init__.py
from .app import app

__all__ = ["app"]
```

---

## 3) `src/DocsToKG/ContentDownload/bootstrap.py` (NEW, seam for PR#5)

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

## 4) Packaging entry point (pyproject.toml)

Add a console entry so users can run `contentdownload` from shell.

```toml
# pyproject.toml
[project.scripts]
contentdownload = "DocsToKG.ContentDownload.cli:app"
```

> Typer will expose `--help` and `--install-completion` automatically.

---

## 5) Tests — `tests/contentdownload/test_cli_basic.py` (NEW)

```python
# tests/contentdownload/test_cli_basic.py
from __future__ import annotations

import json
from typing import Any
import types

import pytest
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


def test_validate_config_ok(monkeypatch, tmp_path):
    cfg_file = tmp_path / "cd.json"
    cfg_file.write_text(json.dumps({"resolvers": {"order": ["unpaywall"]}}))
    result = runner.invoke(app, ["validate-config", "-c", str(cfg_file)])
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_validate_config_fail(monkeypatch, tmp_path):
    # Unknown extra field triggers pydantic "extra=forbid" error
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("resolvers: { order: [unpaywall], unknown_key: 1 }")
    result = runner.invoke(app, ["validate-config", "-c", str(cfg_file)])
    assert result.exit_code != 0
    assert "INVALID:" in result.stdout


def test_explain_order_and_policies(monkeypatch, tmp_path):
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

## 6) Developer UX (optional niceties)

* **Shell completion**: Operators can run `contentdownload --install-completion` (Typer) to enable completion.
* **Colors**: We used `typer.secho` to keep important sections readable.
* **JSON schema** (optional): You can add a `config-schema` command that prints `ContentDownloadConfig.model_json_schema()` for editor/IDE validation.

Example (optional) command:

```python
@app.command("config-schema")
def config_schema():
    from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
    import json
    typer.echo(json.dumps(ContentDownloadConfig.model_json_schema(), indent=2, sort_keys=True))
```

---

## 7) Acceptance checklist

* [ ] `contentdownload --help` shows `run`, `print-config`, `validate-config`, `explain`.
* [ ] `print-config` prints merged config (resolves file, env, and CLI overrides).
* [ ] `validate-config` returns 0 on valid, non-zero with a human-readable error on invalid.
* [ ] `explain` prints resolver order, missing/disabled resolvers, and core policies.
* [ ] `run` calls `run_from_config(cfg)` (seam for PR#5).
* [ ] Tests (CliRunner) pass locally/CI.
* [ ] Entry point added to `pyproject.toml` and works in a fresh venv.

---

## 8) Risk & mitigation

* **Coupling to PR#5**: `run_from_config` is a seam; PR#4 doesn’t wire the pipeline, so it won’t block.
* **Config drift**: `print-config` and `validate-config` give immediate feedback; “extra=forbid” prevents silent typos.
* **Missing resolvers**: `explain` surfaces missing/disabled names before a run.

---

## 9) Follow-ups (PR#5 “wire config end-to-end”)

In the next PR, fill `bootstrap.run_from_config(cfg)` to:

1. Instantiate telemetry sinks from `cfg.telemetry`.
2. Build HTTP session from `cfg.http` (timeouts, TLS, proxies).
3. Materialize resolvers via registry `build_resolvers(cfg.resolvers.order, cfg)`.
4. Construct `ResolverPipeline(session, telemetry, run_id=cfg.run_id)` and process work items.
5. Respect `cfg.download` and `cfg.robots` in download execution (already covered by PR#1).

---

If you want this split into **patch files** (one per file) or a **single markdown** you can drop at `docs/pr4-cli.md`, say the word and I’ll output exactly that format.
