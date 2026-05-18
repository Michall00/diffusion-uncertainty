"""
Create overview image grids for a completed run_uq_sweep.py output.

Rows are prompt/seed pairs, columns are methods. This is intended for W&B and
quick visual inspection without opening many per-prompt folders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD_ORDER = ["baseline", "aleatoric", "original", "gradient", "resampling"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build method overview grids for a UQ sweep")
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=12)
    parser.add_argument("--thumb-size", type=int, default=192)
    parser.add_argument("--include-prompts", action="store_true")
    return parser.parse_args()


def trial_dirs(sweep_dir: Path) -> list[Path]:
    return sorted(p for p in sweep_dir.iterdir() if (p / "trial_config.json").exists())


def trial_method(trial_dir: Path) -> str:
    config_path = trial_dir / "trial_config.json"
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
        methods = config.get("methods", [])
        if len(methods) == 1:
            return str(methods[0])
    name = trial_dir.name
    for method in METHOD_ORDER:
        if f"fixed_{method}" in name:
            return method
    return name


def collect_images(sweep_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Path]], dict[tuple[str, str], str]]:
    images: dict[tuple[str, str], dict[str, Path]] = {}
    prompts: dict[tuple[str, str], str] = {}

    for trial_dir in trial_dirs(sweep_dir):
        method = trial_method(trial_dir)
        for result_dir in sorted(p for p in trial_dir.iterdir() if p.is_dir() and (p / "results.npz").exists()):
            data = np.load(result_dir / "results.npz", allow_pickle=True)
            prompt = str(data["prompt"])
            seed = str(int(data["seed"]))
            key = (prompt, seed)
            image_path = result_dir / f"{method}.png"
            if image_path.exists():
                images.setdefault(key, {})[method] = image_path
                prompts[key] = prompt
    return images, prompts


def short_prompt(prompt: str, max_len: int = 70) -> str:
    prompt = " ".join(prompt.split())
    if len(prompt) <= max_len:
        return prompt
    return prompt[: max_len - 3].rstrip() + "..."


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.sweep_dir / "overview")
    out_dir.mkdir(parents=True, exist_ok=True)

    images_by_key, prompts = collect_images(args.sweep_dir)
    keys = list(images_by_key)[: args.max_rows]
    methods = [method for method in METHOD_ORDER if any(method in images_by_key[key] for key in keys)]
    if not keys or not methods:
        print("[grid] no images found")
        return

    from PIL import Image, ImageDraw, ImageFont

    label_h = 28
    prompt_w = 360 if args.include_prompts else 0
    thumb = args.thumb_size
    width = prompt_w + len(methods) * thumb
    height = label_h + len(keys) * thumb

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col, method in enumerate(methods):
        x = prompt_w + col * thumb
        draw.text((x + 6, 8), method, fill="black", font=font)

    for row, key in enumerate(keys):
        y = label_h + row * thumb
        prompt, seed = key
        if args.include_prompts:
            draw.text((6, y + 6), f"seed={seed}", fill="black", font=font)
            draw.text((6, y + 22), short_prompt(prompt), fill="black", font=font)
        for col, method in enumerate(methods):
            x = prompt_w + col * thumb
            path = images_by_key[key].get(method)
            if path is None:
                draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), fill=(245, 245, 245), outline=(220, 220, 220))
                draw.text((x + 8, y + 8), "missing", fill="gray", font=font)
                continue
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb, thumb))
                tile = Image.new("RGB", (thumb, thumb), "white")
                paste_x = (thumb - image.width) // 2
                paste_y = (thumb - image.height) // 2
                tile.paste(image, (paste_x, paste_y))
            canvas.paste(tile, (x, y))

    out_path = out_dir / "method_overview.png"
    canvas.save(out_path)
    print(f"[grid] wrote {out_path}")


if __name__ == "__main__":
    main()
