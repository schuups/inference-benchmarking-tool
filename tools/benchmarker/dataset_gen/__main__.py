"""CLI: python -m tools.benchmarker.dataset_gen <benchmark.yaml> --out DIR

Resolves the tokenizer per §11.6 (the target model's, unless
dataset_config.tokenizer_id overrides; `--tokenizer word` for offline smoke runs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.common.config import SCENARIOS_DIR, load_benchmark_config

from .generator import generate
from .sources import DatasetSourceError
from .tokenizers import load_tokenizer


def _resolve_tokenizer_id(cfg, override: str | None) -> str:
    if override:
        return override
    if cfg.dataset_config.tokenizer_id:
        return cfg.dataset_config.tokenizer_id
    models = {d.model for d in cfg.deployments}
    if len(models) > 1:
        raise SystemExit(
            f"deployments use multiple models {sorted(models)}: set "
            "dataset_config.tokenizer_id explicitly (§11.6 — the target tokenizer "
            "is authoritative and one pool serves one tokenizer)"
        )
    return models.pop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the prompt pool (M1)")
    parser.add_argument("yaml_path", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--tokenizer", default=None, help="override tokenizer id ('word' = offline)")
    parser.add_argument("--registry", type=Path, default=SCENARIOS_DIR)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.yaml_path)
    tokenizer = load_tokenizer(_resolve_tokenizer_id(cfg, args.tokenizer))
    try:
        manifest = generate(cfg, tokenizer, args.out, args.registry)
    except DatasetSourceError as exc:
        print(f"ABORT (§11.1 source failure): {exc}")
        return 1
    shares = ", ".join(f"{m['scenario']}={m['expected_request_share']:.0%}" for m in manifest["mix"])
    print(f"OK: pool written to {args.out} — request shares: {shares}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
