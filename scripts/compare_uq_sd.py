"""
scripts/compare_uq_sd.py
-------------------------
Compare 4 UQ guidance modes for Stable Diffusion v1.5 side-by-side.

Methods:
    baseline     — standard DDIM, no guidance
    aleatoric    — MC variance → Bayesian posterior update on pred_eps
    gradient     — LLLA Laplace γ² → Bayesian posterior update on pred_eps
    resampling   — LLLA Laplace γ² → local noise injection after DDIM step

Output (per run):
    <out_dir>/<slug>_seed<N>/
        baseline.png
        aleatoric.png           + aleatoric_uncertainty.png
        epistemic_gradient.png  + epistemic_gradient_umap.png
        epistemic_resampling.png + epistemic_resampling_umap.png
        comparison_grid.png
        results.npz

Usage (MPS / CPU):
    uv run --extra cpu python scripts/compare_uq_sd.py \\
        --prompt "a photo of a cat" --seed 42 --num-steps 10 --device mps

Usage (CUDA VM):
    uv run --extra cu118 python scripts/compare_uq_sd.py \\
        --prompt "a photo of a cat" --seed 42 --num-steps 30 --device cuda
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare UQ guidance modes for SD v1.5")
    p.add_argument("--prompt", type=str, default="a photo of a cat sitting on a chair")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "aleatoric", "gradient", "resampling"],
        choices=["baseline", "aleatoric", "gradient", "resampling"],
        help="Methods to run",
    )
    p.add_argument("--guidance-start-step", type=int, default=0)
    p.add_argument("--guidance-n-steps", type=int, default=20,
                   help="Number of steps to apply guidance (aleatoric/epistemic)")
    p.add_argument("--percentile", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--n-ref", type=int, default=3,
                   help="Reference latents for Laplace fitting")
    p.add_argument("--n-pairs", type=int, default=50,
                   help="Regression pairs for Laplace fitting")
    p.add_argument("--num-mc", type=int, default=5,
                   help="MC forward passes for aleatoric variance")
    p.add_argument("--device", type=str, default=None,
                   help="Device: cuda | mps | cpu (auto-detect if not set)")
    p.add_argument("--model-id", type=str, default=MODEL_ID)
    p.add_argument("--out-dir", type=str, default="outputs/compare_uq")
    return p.parse_args()


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "_", text)[:60]


def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) float32 → (H, W, 3) uint8."""
    arr = t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def save_png(image_np: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(image_np).save(path)


def main() -> None:
    args = parse_args()
    device = args.device or auto_device()
    print(f"[compare_uq_sd] device={device}  model={args.model_id}")
    print(f"[compare_uq_sd] prompt: {args.prompt!r}  seed={args.seed}")

    from diffusion_uncertainty.pipeline_uncertainty.pipeline_stable_diffusion_epistemic_guided import (
        StableDiffusionPipelineUQComparison,
        UQComparisonOutput,
    )
    from diffusion_uncertainty.uq_laplace.plotting import save_comparison_grid, save_heatmap_png

    slug = slugify(args.prompt)
    out_dir = Path(args.out_dir) / f"{slug}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compare_uq_sd] output → {out_dir}")

    dtype = torch.float32  # float32 for MPS and CPU; float16 on CUDA for speed
    if device == "cuda":
        dtype = torch.float16

    print(f"[compare_uq_sd] Loading pipeline ({dtype})...")
    pipe = StableDiffusionPipelineUQComparison.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    # Shared Laplace — fitted once for gradient+resampling modes
    shared_laplace = None

    images_for_grid: list[np.ndarray] = []
    umaps_for_grid: list = []
    labels_for_grid: list[str] = []
    npz_data: dict[str, np.ndarray] = {}

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"[compare_uq_sd] Running method: {method}")
        print(f"{'='*60}")

        guidance_mode = method if method in {"aleatoric", "gradient", "resampling"} else "none"

        result: UQComparisonOutput = pipe(
            prompt=args.prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            height=args.height,
            width=args.width,
            guidance_mode=guidance_mode,
            guidance_start_step=args.guidance_start_step,
            guidance_n_steps=args.guidance_n_steps,
            percentile=args.percentile,
            lr=args.lr,
            n_ref_latents=args.n_ref,
            n_laplace_pairs=args.n_pairs,
            num_mc_samples=args.num_mc,
            pre_fitted_laplace=shared_laplace if method in {"resampling"} else None,
        )

        # Cache fitted Laplace after first epistemic run
        if result.fitted_laplace is not None and shared_laplace is None:
            shared_laplace = result.fitted_laplace

        img_np = tensor_to_numpy_image(result.image)
        label = {
            "none": "Baseline",
            "aleatoric": "Aleatoric MC",
            "gradient": "Epistemic Gradient",
            "resampling": "Epistemic Resampling",
        }[guidance_mode]

        # Save individual image
        save_png(img_np, out_dir / f"{method}.png")

        # Save uncertainty map
        umap = result.uncertainty_map
        if umap is not None:
            save_heatmap_png(umap, out_dir / f"{method}_umap.png")

        if result.u_proj is not None:
            save_heatmap_png(result.u_proj, out_dir / f"{method}_uproj.png")

        # Collect for grid
        images_for_grid.append(img_np.astype(np.float32) / 255.0)
        umaps_for_grid.append(umap)
        labels_for_grid.append(label)

        # Collect for NPZ
        npz_data[f"{method}_image"] = img_np
        if umap is not None:
            npz_data[f"{method}_umap"] = umap.cpu().numpy()
        if result.u_proj is not None:
            npz_data[f"{method}_uproj"] = result.u_proj.cpu().numpy()

        print(f"[compare_uq_sd] Saved {method}.png")

    # Save comparison grid
    save_comparison_grid(
        images=images_for_grid,
        labels=labels_for_grid,
        uncertainty_maps=umaps_for_grid,
        path=out_dir / "comparison_grid.png",
        title=f"UQ Guidance Comparison | seed={args.seed}\n{args.prompt}",
    )
    print(f"\n[compare_uq_sd] Grid saved → {out_dir / 'comparison_grid.png'}")

    # Save NPZ
    npz_data["prompt"] = np.array(args.prompt)
    npz_data["seed"] = np.array(args.seed)
    npz_data["methods"] = np.array(args.methods)
    np.savez_compressed(out_dir / "results.npz", **npz_data)
    print(f"[compare_uq_sd] NPZ saved → {out_dir / 'results.npz'}")
    print(f"\n[compare_uq_sd] All done. Results in {out_dir}")


if __name__ == "__main__":
    main()
