#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "scripts.calibrate_tokenizers",
#   "purpose": "Utility script for calibrate tokenizers workflows",
#   "sections": [
#     {
#       "id": "_load_sample_texts",
#       "name": "_load_sample_texts",
#       "anchor": "LST",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "MAIN",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Calibrate tokenizer discrepancies between BERT and Qwen models."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import List

from transformers import AutoTokenizer

from DocsToKG.DocParsing._common import get_logger, iter_doctags


def _load_sample_texts(doctags_dir: Path, sample_size: int) -> List[str]:
    """Read up to ``sample_size`` DocTags files as raw text samples."""

    texts: List[str] = []
    for index, path in enumerate(iter_doctags(doctags_dir)):
        if index >= sample_size:
            break
        texts.append(path.read_text(encoding="utf-8", errors="replace")[:5000])
    return texts


def main() -> None:
    """Entry point for tokenizer calibration CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctags-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    logger = get_logger(__name__)
    texts = _load_sample_texts(args.doctags_dir, args.sample_size)

    if not texts:
        logger.warning(
            "No DocTags samples located for calibration",
            extra={"extra_fields": {"doctags_dir": str(args.doctags_dir)}},
        )
        return

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B", use_fast=True)

    bert_counts = [len(bert_tokenizer.encode(sample, add_special_tokens=False)) for sample in texts]
    qwen_counts = [len(qwen_tokenizer.encode(sample, add_special_tokens=False)) for sample in texts]

    ratios = [q / max(b, 1) for b, q in zip(bert_counts, qwen_counts)]
    mean_ratio = statistics.mean(ratios)
    std_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 0.0

    logger.info(
        "Calibration results",
        extra={
            "extra_fields": {
                "samples": len(texts),
                "mean_ratio": round(mean_ratio, 3),
                "std_ratio": round(std_ratio, 3),
                "bert_mean_tokens": round(statistics.mean(bert_counts), 1),
                "qwen_mean_tokens": round(statistics.mean(qwen_counts), 1),
            }
        },
    )

    if mean_ratio > 1.1:
        logger.info(
            "Qwen tokenizer produces more tokens than BERT",
            extra={
                "extra_fields": {
                    "recommendation": "Increase --min-tokens proportionally",
                    "ratio_delta_percent": round((mean_ratio - 1.0) * 100, 1),
                }
            },
        )
    elif mean_ratio < 0.9:
        logger.info(
            "Qwen tokenizer produces fewer tokens than BERT",
            extra={
                "extra_fields": {
                    "recommendation": "Decrease --min-tokens proportionally",
                    "ratio_delta_percent": round((1.0 - mean_ratio) * 100, 1),
                }
            },
        )
    else:
        logger.info(
            "Tokenizers are closely aligned; no adjustments recommended",
            extra={"extra_fields": {"recommendation": "No action"}},
        )


if __name__ == "__main__":
    main()
