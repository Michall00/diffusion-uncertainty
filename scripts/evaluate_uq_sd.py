"""
scripts/evaluate_uq_sd.py
--------------------------
Evaluate UQ guidance comparison results saved by compare_uq_sd.py.

Loads results.npz and the original images, computes:
    - CLIPScore  : alignment between generated image and prompt (needs transformers or torchmetrics)
    - Mean γ²    : mean uncertainty value across spatial dims
    - P95 γ²     : 95th-percentile γ² (high-uncertainty pixel intensity)

Outputs:
    - Printed Markdown table to stdout
    - evaluation_results.csv saved to the same output directory

Usage:
    uv run --extra cpu python scripts/evaluate_uq_sd.py \\
        --result-dir outputs/compare_uq/a_photo_of_a_cat_seed42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate UQ comparison results")
    p.add_argument("--result-dir", type=str, required=True,
                   help="Path to the output directory created by compare_uq_sd.py")
    p.add_argument("--no-clip", action="store_true",
                   help="Skip CLIPScore computation (when transformers not available)")
    return p.parse_args()


def compute_clip_score(image_np: np.ndarray, prompt: str) -> float | None:
    """Compute CLIPScore via transformers.  Returns None if unavailable."""
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        device = "cpu"
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", use_safetensors=True
        ).to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        pil_image = Image.fromarray(image_np)
        inputs = processor(text=[prompt], images=pil_image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        return float(logits[0, 0].cpu().numpy())
    except ImportError:
        return None
    except Exception as e:
        print(f"[warn] CLIPScore failed: {e}")
        return None


def umap_stats(umap: np.ndarray | None) -> tuple[float, float]:
    """Returns (mean_gamma2, p95_gamma2) or (0, 0) if map is None."""
    if umap is None:
        return 0.0, 0.0
    return float(umap.mean()), float(np.percentile(umap, 95))


def format_table(rows: list[dict[str, object]]) -> str:
    """Render a simple Markdown table from a list of dicts (same keys)."""
    if not rows:
        return ""
    headers = list(rows[0].keys())
    widths = [max(len(str(h)), max(len(str(r[h])) for r in rows)) for h in headers]

    header_line = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    sep_line = " | ".join("-" * w for w in widths)
    data_lines = [
        " | ".join(f"{str(r[h]):<{widths[i]}}" for i, h in enumerate(headers))
        for r in rows
    ]
    return "\n".join(["| " + header_line + " |", "| " + sep_line + " |"] + ["| " + l + " |" for l in data_lines])


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    npz_path = result_dir / "results.npz"

    if not npz_path.exists():
        print(f"[error] results.npz not found in {result_dir}")
        return

    data = np.load(npz_path, allow_pickle=True)
    prompt = str(data["prompt"])
    methods_raw: list[str] = list(data["methods"])

    print(f"[evaluate_uq_sd] prompt: {prompt!r}")
    print(f"[evaluate_uq_sd] methods: {methods_raw}")
    print()

    rows: list[dict[str, object]] = []

    for method in methods_raw:
        image_key = f"{method}_image"
        umap_key = f"{method}_umap"

        image_np: np.ndarray | None = data[image_key] if image_key in data else None
        umap_np: np.ndarray | None = data[umap_key] if umap_key in data else None

        clip_score: float | None = None
        if not args.no_clip and image_np is not None:
            print(f"[evaluate_uq_sd] Computing CLIPScore for {method}...")
            clip_score = compute_clip_score(image_np, prompt)

        mean_g2, p95_g2 = umap_stats(umap_np)
        display_label = {
            "none": "Baseline",
            "original": "Authors Aleatoric",
            "aleatoric": "Reimpl Aleatoric MC",
            "gradient": "Epistemic Gradient",
            "resampling": "Epistemic Resampling",
        }.get(method, method)

        rows.append({
            "Method ID": method,
            "Method": display_label,
            "CLIPScore": f"{clip_score:.4f}" if clip_score is not None else "N/A",
            "Mean γ²": f"{mean_g2:.6f}",
            "P95 γ²": f"{p95_g2:.6f}",
        })

    print(format_table(rows))
    print()

    csv_path = result_dir / "evaluation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[evaluate_uq_sd] CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
