"""
scripts/sweep_uq_sd.py
-----------------------
W&B sweep agent for UQ guidance hyperparameter search.

Sweeps over: lr, percentile, guidance_start_step, guidance_n_steps,
             n_pairs, laplace_mode

Logs per run:
    - CLIPScore for baseline / gradient / resampling
    - ΔCLIP = gradient_clip - baseline_clip  (primary signal)
    - mean γ² and P95 γ² for each epistemic method
    - Generated images (baseline, gradient, resampling, heatmaps)
    - All hyperparameters

Usage:
    # 1. Create the sweep (once):
    uv run --extra cpu python scripts/sweep_uq_sd.py --create-sweep

    # 2. Start agent(s) — each agent picks up runs from the sweep:
    uv run --extra cpu python scripts/sweep_uq_sd.py --sweep-id <ID>

    # Or use wandb CLI directly:
    wandb sweep config/wandb_sweep.yaml
    wandb agent <entity>/<project>/<sweep-id>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import wandb


PROJECT = os.environ.get("WANDB_PROJECT", "diffusion-uq-sweep")
ENTITY = os.environ.get("WANDB_ENTITY", None)  # set via env or wandb login


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--create-sweep", action="store_true",
                   help="Create a W&B sweep from the given config file and print the ID")
    p.add_argument("--config", type=str, default=None,
                   help="Path to sweep YAML config (required with --create-sweep)")
    p.add_argument("--sweep-id", type=str, default=None,
                   help="Run as agent for an existing sweep ID")
    p.add_argument("--count", type=int, default=None,
                   help="Max runs per agent (default: unlimited)")
    p.add_argument("--device", type=str, default=None,
                   help="Device override: cuda | mps | cpu")
    return p.parse_args()


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_clip_score(image_np: np.ndarray, prompt: str, device: str) -> float | None:
    try:
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", use_safetensors=True
        ).to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        pil_image = Image.fromarray(image_np)
        inputs = processor(
            text=[prompt], images=pil_image, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        return float(logits[0, 0].cpu())
    except Exception as e:
        print(f"[warn] CLIPScore failed: {e}")
        return None


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) → (H, W, 3) uint8."""
    arr = t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def umap_to_uint8(umap: torch.Tensor | None) -> np.ndarray | None:
    """(1, C, H, W) uncertainty map → (H, W) uint8 grey."""
    if umap is None:
        return None
    u = umap.squeeze(0).mean(0).float().cpu().numpy()
    u_min, u_max = u.min(), u.max()
    if u_max - u_min < 1e-8:
        return np.zeros(u.shape, dtype=np.uint8)
    return ((u - u_min) / (u_max - u_min) * 255).astype(np.uint8)


def umap_stats(uncertainty_map: "torch.Tensor | None") -> tuple[float, float]:
    if uncertainty_map is None:
        return 0.0, 0.0
    u = uncertainty_map.float().cpu().numpy()
    return float(u.mean()), float(np.percentile(u, 95))


EVAL_PROMPTS: list[tuple[str, int]] = [
    ("a golden retriever in a sunny park", 42),
    ("a red double-decker bus on a rainy London street", 7),
    ("an astronaut riding a horse on the moon", 123),
    ("a bowl of ramen with steam rising", 99),
    ("a snowy mountain village at night with lights", 17),
    ("a portrait of an old fisherman with a weathered face", 55),
    ("a futuristic cityscape at sunset with flying cars", 200),
    ("a cat sitting on a windowsill watching rain", 33),
    ("a field of sunflowers under a dramatic stormy sky", 88),
    ("a medieval castle on a cliff above the ocean", 11),
]


def run_sweep() -> None:
    """Single sweep run — called by wandb agent.

    Expects cfg.method to be "gradient", "resampling", or "aleatoric".
    Loops over EVAL_PROMPTS and reports mean ΔCLIP as the primary metric.
    """
    with wandb.init() as run:
        cfg = wandb.config
        device = auto_device()
        method: str = cfg.method

        print(f"\n[sweep] run={run.name}  method={method}  device={device}")
        print(f"[sweep] lr={getattr(cfg, 'lr', '-')}  "
              f"percentile={getattr(cfg, 'percentile', '-')}")
        print(f"[sweep] guidance_start_step={cfg.guidance_start_step}  "
              f"guidance_n_steps={cfg.guidance_n_steps}")
        if method in {"gradient", "resampling"}:
            print(f"[sweep] n_pairs={cfg.n_pairs}  laplace_mode={cfg.laplace_mode}")

        from diffusion_uncertainty.pipeline_uncertainty.pipeline_stable_diffusion_epistemic_guided import (
            StableDiffusionPipelineUQComparison,
            UQComparisonOutput,
        )

        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipelineUQComparison.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        delta_clips: list[float] = []
        image_table = wandb.Table(columns=["prompt", "baseline", method, "delta_clip"])

        for i, (prompt, seed) in enumerate(EVAL_PROMPTS):
            print(f"\n[sweep] prompt {i+1}/{len(EVAL_PROMPTS)}: {prompt[:50]}…")

            base_kwargs = dict(
                prompt=prompt,
                num_inference_steps=cfg.num_steps,
                guidance_scale=cfg.guidance_scale,
                seed=seed,
                guidance_start_step=cfg.guidance_start_step,
                guidance_n_steps=cfg.guidance_n_steps,
            )
            method_kwargs: dict = {}
            if method in {"gradient", "resampling"}:
                method_kwargs = dict(
                    percentile=cfg.percentile,
                    lr=cfg.lr,
                    n_ref_latents=cfg.n_ref,
                    n_laplace_pairs=cfg.n_pairs,
                    num_mc_samples=cfg.num_mc,
                    laplace_mode=cfg.laplace_mode,
                    n_mc_subnet=cfg.n_mc_subnet,
                    subnet_max_params=cfg.subnet_max_params,
                )
            elif method == "aleatoric":
                method_kwargs = dict(
                    num_mc_samples=cfg.num_mc,
                )

            baseline_out: UQComparisonOutput = pipe(
                **base_kwargs, **method_kwargs, guidance_mode="none"
            )
            method_out: UQComparisonOutput = pipe(
                **base_kwargs, **method_kwargs, guidance_mode=method
            )

            baseline_img = tensor_to_uint8(baseline_out.image)
            method_img = tensor_to_uint8(method_out.image)

            baseline_clip = compute_clip_score(baseline_img, prompt, device="cpu") or 0.0
            method_clip = compute_clip_score(method_img, prompt, device="cpu") or 0.0
            delta = method_clip - baseline_clip
            delta_clips.append(delta)

            image_table.add_data(
                prompt,
                wandb.Image(baseline_img, caption=f"baseline | {prompt[:30]}"),
                wandb.Image(method_img, caption=f"{method} Δ{delta:+.2f}"),
                delta,
            )
            print(f"[sweep] prompt {i+1} ΔCLIP={delta:+.4f}")

        mean_delta = float(np.mean(delta_clips))
        std_delta = float(np.std(delta_clips))

        wandb.log({
            "delta_clip": mean_delta,
            "delta_clip_std": std_delta,
            "images/per_prompt": image_table,
        })
        print(f"\n[sweep] DONE  mean ΔCLIP={mean_delta:+.4f} ± {std_delta:.4f}")


def create_sweep(config_path: Path) -> str:
    """Create W&B sweep from a YAML config file. Returns sweep ID."""
    import yaml

    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    # 'program' key is for CLI use only — remove before passing to API
    sweep_config.pop("program", None)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=PROJECT,
        entity=ENTITY,
    )
    print(f"\n[sweep] Created sweep from {config_path.name}: {sweep_id}")
    print(f"[sweep] Start agent with:")
    print(f"  uv run --extra cpu python scripts/sweep_uq_sd.py --sweep-id {sweep_id}")
    return sweep_id


def main() -> None:
    args = parse_args()

    if args.create_sweep:
        if args.config is None:
            print("[error] --create-sweep requires --config <path-to-yaml>")
            return
        create_sweep(Path(args.config))
        return

    if args.sweep_id is None:
        print("[error] Provide --create-sweep --config <yaml> to create a sweep, "
              "or --sweep-id <ID> to run agent.")
        return

    wandb.agent(
        sweep_id=args.sweep_id,
        function=run_sweep,
        project=PROJECT,
        entity=ENTITY,
        count=args.count,
    )


if __name__ == "__main__":
    main()
