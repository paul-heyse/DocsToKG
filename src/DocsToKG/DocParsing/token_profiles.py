#!/usr/bin/env python3
"""Print simple token ratio stats for DocTags samples."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

from DocsToKG.DocParsing.config import StageConfigBase
from DocsToKG.DocParsing.core import (
    CLIOption,
    DEFAULT_TOKENIZER,
    build_subcommand,
)
from DocsToKG.DocParsing.env import data_doctags, detect_data_root
from DocsToKG.DocParsing.io import iter_doctags
from DocsToKG.DocParsing.logging import get_logger, log_event
from DocsToKG.DocParsing.doctags import add_data_root_option

__all__ = ["TokenProfilesCfg", "build_parser", "parse_args", "main"]

DEFAULT_COMPARATORS = [
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base-v2",
]

DEFAULT_DATA_ROOT = detect_data_root()
DEFAULT_DOCTAGS_DIR = data_doctags(DEFAULT_DATA_ROOT)

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


@dataclass
class TokenProfilesCfg(StageConfigBase):
    """Structured configuration for tokenizer ratio profiling."""

    log_level: str = "INFO"
    data_root: Optional[Path] = None
    doctags_dir: Path = DEFAULT_DOCTAGS_DIR
    sample_size: int = 20
    max_chars: int = 4000
    baseline: str = DEFAULT_TOKENIZER
    tokenizers: Tuple[str, ...] = ()
    window_min: int = 256
    window_max: int = 512

    ENV_VARS: ClassVar[Dict[str, str]] = {
        "log_level": "DOCSTOKG_TOKPROF_LOG_LEVEL",
        "data_root": "DOCSTOKG_TOKPROF_DATA_ROOT",
        "doctags_dir": "DOCSTOKG_TOKPROF_DOCTAGS_DIR",
        "sample_size": "DOCSTOKG_TOKPROF_SAMPLE_SIZE",
        "max_chars": "DOCSTOKG_TOKPROF_MAX_CHARS",
        "baseline": "DOCSTOKG_TOKPROF_BASELINE",
        "tokenizers": "DOCSTOKG_TOKPROF_TOKENIZERS",
        "window_min": "DOCSTOKG_TOKPROF_WINDOW_MIN",
        "window_max": "DOCSTOKG_TOKPROF_WINDOW_MAX",
        "config": "DOCSTOKG_TOKPROF_CONFIG",
    }

    FIELD_PARSERS: ClassVar[Dict[str, Callable[[Any, Optional[Path]], Any]]] = {
        "config": StageConfigBase._coerce_optional_path,
        "log_level": StageConfigBase._coerce_str,
        "data_root": StageConfigBase._coerce_optional_path,
        "doctags_dir": StageConfigBase._coerce_path,
        "sample_size": StageConfigBase._coerce_int,
        "max_chars": StageConfigBase._coerce_int,
        "baseline": StageConfigBase._coerce_str,
        "tokenizers": StageConfigBase._coerce_str_tuple,
        "window_min": StageConfigBase._coerce_int,
        "window_max": StageConfigBase._coerce_int,
    }

    @classmethod
    def from_env(cls, defaults: Optional[Dict[str, Any]] = None) -> "TokenProfilesCfg":
        """Instantiate configuration using environment overlays."""

        cfg = cls(**(defaults or {}))
        cfg.apply_env()
        if cfg.data_root is None:
            fallback_root = os.getenv("DOCSTOKG_DATA_ROOT")
            if fallback_root:
                cfg.data_root = StageConfigBase._coerce_optional_path(fallback_root, None)
        cfg.finalize()
        return cfg

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> "TokenProfilesCfg":
        """Layer CLI arguments, config files, and env vars into a configuration."""

        cfg = cls.from_env(defaults=defaults)
        config_path = getattr(args, "config", None)
        if config_path:
            cfg.update_from_file(Path(config_path))
        cfg.apply_args(args)
        cfg.finalize()
        return cfg

    def finalize(self) -> None:
        """Normalise derived attributes after overlays are applied."""

        if self.data_root is not None:
            self.data_root = StageConfigBase._coerce_optional_path(self.data_root, None)
        base_dir = self.data_root
        self.doctags_dir = StageConfigBase._coerce_path(self.doctags_dir, base_dir)
        if self.config is not None:
            self.config = StageConfigBase._coerce_optional_path(self.config, None)
        baseline = str(self.baseline or "").strip()
        self.baseline = baseline or DEFAULT_TOKENIZER
        tokens = tuple(token for token in self.tokenizers if token)
        if not tokens and not self.is_overridden("tokenizers"):
            tokens = tuple(DEFAULT_COMPARATORS)
        self.tokenizers = tuple(dict.fromkeys(tokens))
        self.log_level = str(self.log_level or "INFO").upper()
        if self.sample_size < 0:
            self.sample_size = 0
        if self.max_chars < 0:
            self.max_chars = 0
        if self.window_min < 0:
            self.window_min = 0
        if self.window_max < 0:
            self.window_max = 0

    from_sources = from_args

    def tokenizer_ids(self) -> List[str]:
        """Return the ordered tokenizer identifiers to profile."""

        ordered: List[str] = []
        for name in (self.baseline, *self.tokenizers):
            if not name:
                continue
            if name not in ordered:
                ordered.append(name)
        return ordered


TOKEN_PROFILE_CLI_OPTIONS: Tuple[CLIOption, ...] = (
    CLIOption(
        ("--config",),
        {"type": Path, "default": None, "help": "Optional path to JSON/YAML/TOML config."},
    ),
    CLIOption(
        ("--doctags-dir",),
        {"type": Path, "default": DEFAULT_DOCTAGS_DIR, "help": "Directory containing DocTags files."},
    ),
    CLIOption(
        ("--sample-size",),
        {"type": int, "default": 20, "help": "Number of DocTags files to sample (<=0 means all)."},
    ),
    CLIOption(
        ("--max-chars",),
        {"type": int, "default": 4000, "help": "Trim samples to this many characters (<=0 keeps full text)."},
    ),
    CLIOption(
        ("--baseline",),
        {"type": str, "default": DEFAULT_TOKENIZER, "help": "Tokenizer treated as baseline for ratios."},
    ),
    CLIOption(
        ("--tokenizer",),
        {
            "dest": "tokenizers",
            "action": "append",
            "default": None,
            "metavar": "NAME",
            "help": "Additional tokenizer identifier to profile (repeatable).",
        },
    ),
    CLIOption(
        ("--window-min",),
        {"type": int, "default": 256, "help": "Reference min tokens scaled by observed ratios."},
    ),
    CLIOption(
        ("--window-max",),
        {"type": int, "default": 512, "help": "Reference max tokens scaled by observed ratios."},
    ),
    CLIOption(
        ("--log-level",),
        {
            "type": lambda value: str(value).upper(),
            "default": "INFO",
            "choices": ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
            "help": "Logging verbosity for structured output.",
        },
    ),
)


def _clean_text(text: str, max_chars: Optional[int]) -> str:
    """Strip DocTags markup and collapse whitespace."""

    stripped = TAG_RE.sub(" ", text)
    collapsed = SPACE_RE.sub(" ", stripped).strip()
    if max_chars and max_chars > 0:
        return collapsed[:max_chars]
    return collapsed


def _load_samples(root: Path, sample_size: int, max_chars: int) -> List[str]:
    """Read DocTags files from ``root`` and return cleaned text samples."""

    samples: List[str] = []
    limit = sample_size if sample_size > 0 else None
    max_len = max_chars if max_chars > 0 else None
    for path in iter_doctags(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        cleaned = _clean_text(text, max_len)
        if not cleaned:
            continue
        samples.append(cleaned)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _count_tokens(name: str, texts: Sequence[str]) -> List[int]:
    """Return token counts for ``texts`` using the HuggingFace tokenizer ``name``."""

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    return [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_ratio(candidate: Sequence[int], baseline: Sequence[int]) -> Optional[float]:
    ratios = [cand / base for cand, base in zip(candidate, baseline) if base]
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def _scale(window: int, ratio: Optional[float]) -> str:
    return "n/a" if ratio is None else str(int(round(window * ratio)))


def _render_table(
    tokenizer_ids: Sequence[str],
    counts: Dict[str, Sequence[int]],
    baseline_name: str,
    baseline_counts: Sequence[int],
    window_min: int,
    window_max: int,
) -> str:
    """Render a table summarising token statistics."""

    header = (
        "tokenizer".ljust(45)
        + "mean_tokens".rjust(12)
        + "mean_ratio".rjust(12)
        + "scaled_min".rjust(12)
        + "scaled_max".rjust(12)
    )
    lines = [header, "-" * len(header)]
    for name in tokenizer_ids:
        mean_tokens = _mean(counts[name])
        ratio = 1.0 if name == baseline_name else _mean_ratio(counts[name], baseline_counts)
        ratio_text = f"{ratio:.3f}" if ratio is not None else "n/a"
        lines.append(
            f"{name.ljust(45)}"
            f"{mean_tokens:12.1f}"
            f"{ratio_text:>12}"
            f"{_scale(window_min, ratio):>12}"
            f"{_scale(window_max, ratio):>12}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for tokenizer profiling."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_option(parser)
    build_subcommand(parser, TOKEN_PROFILE_CLI_OPTIONS)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for tokenizer profiling."""

    return build_parser().parse_args(argv)


def main(args: argparse.Namespace | Sequence[str] | None = None) -> int:
    """Entry point for the tokenizer profiling CLI."""

    parser = build_parser()
    if args is None:
        namespace = parser.parse_args()
    elif isinstance(args, argparse.Namespace):
        namespace = argparse.Namespace(**vars(args))
    else:
        namespace = parser.parse_args(list(args))

    cfg = TokenProfilesCfg.from_args(namespace)
    tokenizer_ids = cfg.tokenizer_ids()
    logger = get_logger(
        __name__,
        level=cfg.log_level,
        base_fields={
            "stage": "token_profiles",
            "baseline": cfg.baseline,
            "sample_size": cfg.sample_size,
            "window_min": cfg.window_min,
            "window_max": cfg.window_max,
        },
    )

    samples = _load_samples(cfg.doctags_dir, cfg.sample_size, cfg.max_chars)
    if not samples:
        log_event(
            logger,
            "warning",
            "No DocTags samples located for profiling",
            stage="token_profiles",
            doc_id="__aggregate__",
            input_hash=None,
            error_code="NO_INPUT_FILES",
            doctags_dir=str(cfg.doctags_dir),
        )
        print("No DocTags samples found; nothing to profile.")
        return 1

    logger.info(
        "Profiling tokenizers",
        extra={
            "extra_fields": {
                "doctags_dir": str(cfg.doctags_dir),
                "tokenizers": tokenizer_ids,
                "max_chars": cfg.max_chars,
                "samples": len(samples),
            }
        },
    )

    counts: Dict[str, List[int]] = {name: _count_tokens(name, samples) for name in tokenizer_ids}
    baseline_counts = counts.get(cfg.baseline)
    if baseline_counts is None:
        log_event(
            logger,
            "error",
            "Baseline tokenizer missing from counts",
            stage="token_profiles",
            doc_id="__aggregate__",
            input_hash=None,
            error_code="BASELINE_MISSING",
            baseline=cfg.baseline,
        )
        print(f"Baseline tokenizer {cfg.baseline!r} was not profiled.")
        return 1

    table = _render_table(tokenizer_ids, counts, cfg.baseline, baseline_counts, cfg.window_min, cfg.window_max)
    print(f"Sampled {len(samples)} DocTags from {cfg.doctags_dir}")
    print(f"Baseline tokenizer: {cfg.baseline}")
    print("")
    print(table)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
