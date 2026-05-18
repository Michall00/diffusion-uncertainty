"""
Prepare a Stable Diffusion sweep config from Recap-COCO prompts.

The generated config is compatible with scripts/run_uq_sweep.py. Optionally,
the script also saves the matching real COCO images into a reference directory
so aggregate FID can compare generated images against real images.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_CONFIG = REPO_ROOT / "config" / "stable_diffusion_uq_sweep.yaml"
DEFAULT_OUT_CONFIG = REPO_ROOT / "config" / "stable_diffusion_recap_coco_sweep.yaml"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "data" / "recap_coco_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Recap-COCO prompts for the SD UQ sweep")
    parser.add_argument("--dataset-id", default="UCSC-VLAA/Recap-COCO-30K")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-column", choices=["caption", "recaption"], default="recaption")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Seed used for dataset shuffling")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-config", type=Path, default=DEFAULT_OUT_CONFIG)
    parser.add_argument("--sweep-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--seed-mode", choices=["cartesian", "per-prompt"], default="cartesian")
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--no-reference-images", action="store_true")
    parser.add_argument("--allow-duplicate-images", action="store_true")
    return parser.parse_args()


def normalize_prompt(value: Any) -> str:
    prompt = str(value or "").strip()
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt


def load_base_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def save_reference_image(image: Any, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path, quality=95)
        return
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            import io

            Image.open(io.BytesIO(image["bytes"])).convert("RGB").save(path, quality=95)
            return
        if image.get("path") is not None:
            Image.open(image["path"]).convert("RGB").save(path, quality=95)
            return
    raise TypeError(f"Unsupported image value for reference export: {type(image)!r}")


def main() -> None:
    args = parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required. Install requirements.txt first.") from exc

    dataset = load_dataset(args.dataset_id, split=args.split, streaming=True)
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    prompts: list[str] = []
    rows: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()

    for row in dataset:
        prompt = normalize_prompt(row.get(args.prompt_column))
        if not prompt:
            continue

        image_id = str(row.get("image_id", len(prompts)))
        if not args.allow_duplicate_images and image_id in seen_image_ids:
            continue

        seen_image_ids.add(image_id)
        prompts.append(prompt)
        rows.append(row)

        if len(prompts) >= args.num_prompts:
            break

    if len(prompts) < args.num_prompts:
        raise RuntimeError(f"Only collected {len(prompts)} prompts, requested {args.num_prompts}")

    config = load_base_config(args.base_config)
    config["project_name"] = "diffusion-uncertainty-recap-coco"
    config["seeds"] = args.sweep_seeds
    config["prompts"] = prompts
    if args.seed_mode == "per-prompt":
        base_seed = args.sweep_seeds[0]
        config["prompt_seed_pairs"] = [
            {"prompt": prompt, "seed": base_seed + idx}
            for idx, prompt in enumerate(prompts)
        ]
    config["recap_coco"] = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "prompt_column": args.prompt_column,
        "num_prompts": args.num_prompts,
        "shuffle_seed": args.seed,
        "seed_mode": args.seed_mode,
        "reference_dir": None if args.no_reference_images else str(args.reference_dir),
    }

    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    with args.out_config.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    if not args.no_reference_images:
        args.reference_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.reference_dir / "manifest.csv"
        with manifest_path.open("w") as manifest:
            manifest.write("index,image_id,coco_url,prompt\n")
            for idx, row in enumerate(rows):
                image_id = str(row.get("image_id", idx))
                image_path = args.reference_dir / f"{idx:06d}_{image_id}.jpg"
                save_reference_image(row["image"], image_path)
                coco_url = str(row.get("coco_url", ""))
                prompt = prompts[idx].replace('"', '""')
                manifest.write(f'{idx},{image_id},"{coco_url}","{prompt}"\n')

    print(f"[recap-coco] wrote config: {args.out_config}")
    if not args.no_reference_images:
        print(f"[recap-coco] wrote reference images: {args.reference_dir}")


if __name__ == "__main__":
    main()
