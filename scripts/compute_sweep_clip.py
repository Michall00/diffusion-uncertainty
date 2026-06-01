"""
Compute CLIPScore for all generated method images in a run_uq_sweep.py output.

This is much faster than running evaluate_uq_sd.py with CLIP enabled for every
single prompt result, because the CLIP model is loaded once and evaluated in
batches.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute aggregate CLIPScore for a UQ sweep")
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--only-trial-index", type=int, default=None,
                        help="Compute CLIPScore only for one trial index from the sweep")
    parser.add_argument("--only-trial-dir", type=Path, default=None,
                        help="Compute CLIPScore only for one trial directory")
    return parser.parse_args()


def trial_dirs(sweep_dir: Path) -> list[Path]:
    return sorted(p for p in sweep_dir.iterdir() if (p / "trial_config.json").exists())


def selected_trial_dirs(args: argparse.Namespace) -> list[tuple[int, Path]]:
    trials = list(enumerate(trial_dirs(args.sweep_dir)))
    if args.only_trial_index is not None:
        trials = [
            (trial_index, trial_dir)
            for trial_index, trial_dir in trials
            if trial_index == args.only_trial_index
        ]
    if args.only_trial_dir is not None:
        target = args.only_trial_dir.resolve()
        trials = [
            (trial_index, trial_dir)
            for trial_index, trial_dir in trials
            if trial_dir.resolve() == target
        ]
    return trials


def prompt_result_dirs(trial_dir: Path) -> list[Path]:
    return sorted(p for p in trial_dir.iterdir() if p.is_dir() and (p / "results.npz").exists())


def load_items(args: argparse.Namespace) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for trial_index, trial_dir in selected_trial_dirs(args):
        for result_dir in prompt_result_dirs(trial_dir):
            data = np.load(result_dir / "results.npz", allow_pickle=True)
            prompt = str(data["prompt"])
            seed = int(data["seed"])
            methods = [str(method) for method in list(data["methods"])]
            for method in methods:
                image_path = result_dir / f"{method}.png"
                if image_path.exists():
                    items.append({
                        "trial_index": trial_index,
                        "group": trial_dir.name,
                        "result_dir": str(result_dir),
                        "prompt": prompt,
                        "seed": seed,
                        "method": method,
                        "image_path": image_path,
                    })
    return items


def compute_scores(items: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(args.model_id, use_safetensors=True).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_id)
    model.eval()

    rows: list[dict[str, object]] = []
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
        prompts = [str(item["prompt"]) for item in batch]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(args.device)

        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        scores = logits.diag().detach().float().cpu().tolist()

        for item, score in zip(batch, scores):
            row = {key: value for key, value in item.items() if key != "image_path"}
            row["clip_score"] = f"{score:.6f}"
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    out_csv = args.out_csv or (args.sweep_dir / "clip_results.csv")
    items = load_items(args)
    rows = compute_scores(items, args) if items else []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["trial_index", "group", "result_dir", "prompt", "seed", "method", "clip_score"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[clip] wrote {out_csv}")


if __name__ == "__main__":
    main()
