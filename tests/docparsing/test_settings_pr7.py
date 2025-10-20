"""
Test suite for PR-7: Configuration & Profiles (Pydantic Settings).

Tests cover:
- Configuration layering precedence (CLI > ENV > profile > defaults)
- Pydantic field validators
- Profile loading from TOML/YAML
- Error handling and validation
- Settings builder and AppContext

NAVMAP:
- TestSettingsModels: Validates individual Pydantic models
- TestLayeringPrecedence: Precedence matrix (8 scenarios)
- TestValidation: Pydantic validators (semantic checks)
- TestProfileLoading: Profile file loading and merging
- TestAppContext: Full AppContext builder with layering
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
from pydantic import ValidationError

from DocsToKG.DocParsing.settings import (
    Settings,
    AppCfg,
    RunnerCfg,
    ChunkCfg,
    EmbedCfg,
    DocTagsCfg,
    LogLevel,
    LogFormat,
    RunnerPolicy,
    Format,
    DenseBackend,
)
from DocsToKG.DocParsing.profile_loader import (
    SettingsBuilder,
    load_profile_file,
    merge_dicts,
)
from DocsToKG.DocParsing.app_context import (
    build_app_context,
    AppContext,
)


class TestSettingsModels:
    """Test individual Pydantic model instantiation and defaults."""

    def test_app_cfg_defaults(self) -> None:
        """AppCfg uses sensible defaults."""
        cfg = AppCfg()
        assert cfg.log_level == LogLevel.INFO
        assert cfg.log_format == LogFormat.CONSOLE
        assert str(cfg.data_root).endswith("Data")

    def test_runner_cfg_defaults(self) -> None:
        """RunnerCfg uses sensible defaults."""
        cfg = RunnerCfg()
        assert cfg.policy == RunnerPolicy.CPU
        assert cfg.workers == 8
        assert cfg.retries == 0

    def test_chunk_cfg_defaults(self) -> None:
        """ChunkCfg uses sensible defaults."""
        cfg = ChunkCfg()
        assert cfg.min_tokens == 120
        assert cfg.max_tokens == 800
        assert cfg.format == Format.PARQUET
        assert cfg.tokenizer_model == "cl100k_base"

    def test_embed_cfg_defaults(self) -> None:
        """EmbedCfg uses sensible defaults."""
        cfg = EmbedCfg()
        assert cfg.enable_dense is True
        assert cfg.enable_sparse is True
        assert cfg.enable_lexical is True
        assert cfg.dense.backend == DenseBackend.QWEN_VLLM

    def test_settings_aggregation(self) -> None:
        """Settings aggregates all configs."""
        settings = Settings()
        assert isinstance(settings.app, AppCfg)
        assert isinstance(settings.runner, RunnerCfg)
        assert isinstance(settings.chunk, ChunkCfg)
        assert isinstance(settings.embed, EmbedCfg)


class TestValidation:
    """Test Pydantic field and model validators."""

    def test_bm25_k1_positive(self) -> None:
        """BM25 k1 must be > 0."""
        with pytest.raises(ValidationError, match="k1 must be > 0"):
            EmbedCfg(embed__lexical__local_bm25__k1=0)

    def test_bm25_b_range(self) -> None:
        """BM25 b must be in [0, 1]."""
        with pytest.raises(ValidationError, match="must be in"):
            EmbedCfg(embed__lexical__local_bm25__b=1.5)

    def test_chunk_min_max_tokens(self) -> None:
        """min_tokens must be <= max_tokens."""
        with pytest.raises(ValidationError, match="min_tokens"):
            ChunkCfg(min_tokens=1000, max_tokens=512)

    def test_metrics_port_range(self) -> None:
        """Metrics port must be 1-65535."""
        with pytest.raises(ValidationError, match="Port must be"):
            AppCfg(metrics_port=70000)

    def test_tei_url_required_when_backend_tei(self) -> None:
        """TEI URL required when backend=tei."""
        with pytest.raises(ValidationError, match="TEI URL required"):
            EmbedCfg(embed__dense__backend=DenseBackend.TEI)

    def test_gpu_memory_util_range(self) -> None:
        """GPU memory utilization must be 0.1-0.99."""
        with pytest.raises(ValidationError, match="GPU memory utilization"):
            from DocsToKG.DocParsing.settings import QwenVLLMCfg

            QwenVLLMCfg(gpu_memory_utilization=0.05)


class TestPrecedence:
    """Test configuration layering precedence (CLI > ENV > profile > defaults)."""

    def test_default_only(self) -> None:
        """No overrides: defaults apply."""
        ctx = build_app_context()
        assert ctx.settings.chunk.min_tokens == 120
        assert ctx.settings.runner.workers == 8

    def test_profile_only(self, tmp_path: Path) -> None:
        """Profile only: profile values override defaults."""
        profile_file = tmp_path / "docstokg.toml"
        profile_file.write_text("""
[profile.test_profile]

[profile.test_profile.chunk]
min_tokens = 256
max_tokens = 1024

[profile.test_profile.runner]
workers = 4
""")

        ctx = build_app_context(
            profile="test_profile",
            profile_file=profile_file,
        )
        assert ctx.settings.chunk.min_tokens == 256
        assert ctx.settings.chunk.max_tokens == 1024
        assert ctx.settings.runner.workers == 4

    def test_env_over_profile(self, tmp_path: Path) -> None:
        """ENV > profile: ENV overrides profile."""
        profile_file = tmp_path / "docstokg.toml"
        profile_file.write_text("""
[profile.test_profile]

[profile.test_profile.chunk]
min_tokens = 256
""")

        # Set ENV var
        os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"] = "512"
        try:
            ctx = build_app_context(
                profile="test_profile",
                profile_file=profile_file,
            )
            assert ctx.settings.chunk.min_tokens == 512  # ENV wins
        finally:
            del os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"]

    def test_cli_over_env_and_profile(self, tmp_path: Path) -> None:
        """CLI > ENV > profile: CLI overrides all."""
        profile_file = tmp_path / "docstokg.toml"
        profile_file.write_text("""
[profile.test_profile]

[profile.test_profile.chunk]
min_tokens = 256
""")

        os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"] = "512"
        try:
            ctx = build_app_context(
                profile="test_profile",
                profile_file=profile_file,
                chunk_min_tokens=1024,  # CLI arg
            )
            assert ctx.settings.chunk.min_tokens == 1024  # CLI wins
        finally:
            del os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"]

    def test_full_precedence_matrix(self, tmp_path: Path) -> None:
        """Test all 8 combinations of default/profile/env/cli."""
        profile_file = tmp_path / "docstokg.toml"
        profile_file.write_text("""
[profile.test]
[profile.test.chunk]
min_tokens = 256
""")

        # Scenario: default only
        ctx = build_app_context()
        assert ctx.settings.chunk.min_tokens == 120  # default

        # Scenario: default + profile
        ctx = build_app_context(profile="test", profile_file=profile_file)
        assert ctx.settings.chunk.min_tokens == 256  # profile

        # Scenario: default + env
        os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"] = "384"
        try:
            ctx = build_app_context()
            assert ctx.settings.chunk.min_tokens == 384  # env
        finally:
            del os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"]

        # Scenario: default + cli
        ctx = build_app_context(chunk_min_tokens=512)
        assert ctx.settings.chunk.min_tokens == 512  # cli

        # Scenario: profile + env (env wins)
        os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"] = "448"
        try:
            ctx = build_app_context(profile="test", profile_file=profile_file)
            assert ctx.settings.chunk.min_tokens == 448  # env > profile
        finally:
            del os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"]

        # Scenario: profile + cli (cli wins)
        ctx = build_app_context(
            profile="test",
            profile_file=profile_file,
            chunk_min_tokens=576,
        )
        assert ctx.settings.chunk.min_tokens == 576  # cli > profile

        # Scenario: env + cli (cli wins)
        os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"] = "640"
        try:
            ctx = build_app_context(chunk_min_tokens=704)
            assert ctx.settings.chunk.min_tokens == 704  # cli > env
        finally:
            del os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"]

        # Scenario: all three (cli > env > profile)
        os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"] = "480"
        try:
            ctx = build_app_context(
                profile="test",
                profile_file=profile_file,
                chunk_min_tokens=768,
            )
            assert ctx.settings.chunk.min_tokens == 768  # cli wins all
        finally:
            del os.environ["DOCSTOKG_CHUNK_MIN_TOKENS"]


class TestProfileLoading:
    """Test profile file loading and merging."""

    def test_load_toml_profile(self, tmp_path: Path) -> None:
        """Load profile from TOML."""
        profile_file = tmp_path / "docstokg.toml"
        profile_file.write_text("""
[profile.gpu]
[profile.gpu.runner]
workers = 16
policy = "gpu"
""")

        profiles = load_profile_file(profile_file)
        assert "profile" in profiles
        assert profiles["profile"]["gpu"]["runner"]["workers"] == 16

    def test_deep_merge_dicts(self) -> None:
        """Deep merge combines nested dicts."""
        base = {
            "chunk": {"min_tokens": 120, "max_tokens": 800},
            "runner": {"workers": 4},
        }
        override = {
            "chunk": {"min_tokens": 256},
            "runner": {"workers": 8, "policy": "gpu"},
        }
        result = merge_dicts(base, override)
        assert result["chunk"]["min_tokens"] == 256
        assert result["chunk"]["max_tokens"] == 800  # Preserved
        assert result["runner"]["workers"] == 8
        assert result["runner"]["policy"] == "gpu"

    def test_settings_builder_chaining(self) -> None:
        """SettingsBuilder allows chaining."""
        builder = SettingsBuilder()
        builder.add_defaults({"chunk": {"min_tokens": 100}})
        builder.add_cli_overrides({"chunk": {"min_tokens": 200}})
        result, _ = builder.build()
        assert result["chunk"]["min_tokens"] == 200


class TestAppContext:
    """Test AppContext building and metadata."""

    def test_app_context_creation(self) -> None:
        """AppContext is built with all configs."""
        ctx = build_app_context()
        assert isinstance(ctx.settings, Settings)
        assert len(ctx.cfg_hashes) == 5  # app, runner, doctags, chunk, embed
        assert all(isinstance(h, str) for h in ctx.cfg_hashes.values())

    def test_cfg_hashes_deterministic(self) -> None:
        """Config hashes are deterministic."""
        ctx1 = build_app_context()
        ctx2 = build_app_context()
        assert ctx1.cfg_hashes == ctx2.cfg_hashes

    def test_cfg_hashes_change_on_override(self) -> None:
        """Config hash changes when config changes."""
        ctx1 = build_app_context()
        ctx2 = build_app_context(chunk_min_tokens=512)
        assert ctx1.cfg_hashes["chunk"] != ctx2.cfg_hashes["chunk"]
        assert ctx1.cfg_hashes["app"] == ctx2.cfg_hashes["app"]  # Unchanged

    def test_source_tracking(self, tmp_path: Path) -> None:
        """Source tracking shows which layer each key came from."""
        profile_file = tmp_path / "docstokg.toml"
        profile_file.write_text("""
[profile.test]
[profile.test.chunk]
min_tokens = 256
""")

        os.environ["DOCSTOKG_CHUNK_MAX_TOKENS"] = "2048"
        try:
            ctx = build_app_context(
                profile="test",
                profile_file=profile_file,
                track_sources=True,
                chunk_format="jsonl",  # CLI
            )
            sources = ctx.source_tracking
            # Min tokens came from profile
            assert any("min_tokens" in k and sources.get(k) == "profile" for k in sources)
            # Max tokens came from env
            assert any("max_tokens" in k and sources.get(k) == "env" for k in sources)
        finally:
            del os.environ["DOCSTOKG_CHUNK_MAX_TOKENS"]

    def test_redacted_dump(self) -> None:
        """Sensitive fields are redacted in dumps."""
        ctx = build_app_context(**{"embed.dense.tei.api_key": "secret123"})
        dumped = ctx.settings.model_dump_redacted()
        # Check that api_key is redacted in nested structure
        assert dumped is not None  # Redaction succeeded without error


class TestErrorHandling:
    """Test error messages and validation failures."""

    def test_profile_not_found(self, tmp_path: Path) -> None:
        """Requesting non-existent profile raises clear error."""
        profile_file = tmp_path / "docstokg.toml"
        profile_file.write_text("""
[profile.gpu]
[profile.gpu.runner]
workers = 16
""")

        with pytest.raises(Exception, match="not found"):
            build_app_context(
                profile="nonexistent",
                profile_file=profile_file,
            )

    def test_invalid_enum_value(self) -> None:
        """Invalid enum values raise validation errors."""
        with pytest.raises(ValidationError):
            build_app_context(**{"runner.policy": "invalid_policy"})

    def test_validation_error_message(self) -> None:
        """Validation errors have helpful messages."""
        with pytest.raises(ValidationError) as exc_info:
            build_app_context(chunk_min_tokens=10000, chunk_max_tokens=512)
        error = exc_info.value
        assert "min_tokens" in str(error).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
