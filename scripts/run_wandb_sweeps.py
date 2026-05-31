"""
Create and run multiple W&B sweeps sequentially.

By default this runs the large epistemic sweep first and the large aleatoric
sweep second. Each sampled W&B config is still logged as a separate W&B run by
scripts/sweep_uq_sd.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import wandb
import yaml

from scripts.sweep_uq_sd import run_sweep


DEFAULT_CONFIGS = [
    Path("config/wandb_sweep_epistemic_50steps_large.yaml"),
    Path("config/wandb_sweep_aleatoric_50steps_large.yaml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W&B sweep configs sequentially")
    parser.add_argument(
        "configs",
        type=Path,
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Sweep YAML files. Defaults to the large epistemic and aleatoric sweeps.",
    )
    parser.add_argument("--device", default="cuda", help="Device passed to sweep_uq_sd.py")
    parser.add_argument(
        "--count-per-sweep",
        type=int,
        default=None,
        help="Maximum agent runs per sweep. Defaults to each sweep config run_cap.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "diffusion-uq-sweep"),
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="Optional W&B entity/team",
    )
    return parser.parse_args()


def load_sweep_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    config.pop("program", None)
    return config


def main() -> None:
    args = parse_args()
    for config_path in args.configs:
        sweep_config = load_sweep_config(config_path)
        run_cap = sweep_config.get("run_cap", "unlimited")
        print(f"\n[run-wandb-sweeps] creating sweep from {config_path}")
        print(f"[run-wandb-sweeps] project={args.project} entity={args.entity or '-'} run_cap={run_cap}")
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=args.project,
            entity=args.entity,
        )
        print(f"[run-wandb-sweeps] running sweep_id={sweep_id}")
        wandb.agent(
            sweep_id=sweep_id,
            function=lambda: run_sweep(device_override=args.device),
            project=args.project,
            entity=args.entity,
            count=args.count_per_sweep,
        )
        print(f"[run-wandb-sweeps] finished sweep_id={sweep_id}")


if __name__ == "__main__":
    main()
