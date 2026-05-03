"""
scripts/run_batch_experiments.py
---------------------------------
Uruchom wszystkie metody UQ dla wielu promptów/seedów automatycznie,
a potem zebierz wyniki w jeden plik CSV.

Obsługa:
  - listy promptów (z pliku YAML lub inline --prompts)
  - wielu seedów per prompt
  - automatycznego wznowienia (--skip-existing)
  - zbiorczego raportu z CLIPScore + statystyk γ²

Przykłady:
  # Szybki test na MPS (10 kroków, 2 prompty, 1 seed)
  uv run --extra cpu python scripts/run_batch_experiments.py \
    --model-id CompVis/stable-diffusion-v1-4 \
    --device mps --num-steps 10 \
    --prompts "a cat on a chair" "a misty forest at dawn" \
    --seeds 42 \
    --methods baseline aleatoric gradient resampling \
    --num-mc 2 --n-ref 1 --n-pairs 10 \
    --out-dir outputs/batch_quick

  # Pełny eksperyment na CUDA
  uv run --extra cu118 python scripts/run_batch_experiments.py \
    --model-id CompVis/stable-diffusion-v1-4 \
    --device cuda --num-steps 30 \
    --config config/stable_diffusion_uq_comparison.yaml \
    --seeds 42 123 777 \
    --out-dir outputs/batch_full \
    --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch UQ comparison runner")

    # Prompt sources
    g = p.add_mutually_exclusive_group()
    g.add_argument("--prompts", nargs="+", default=None,
                   help="Prompts to evaluate (inline)")
    g.add_argument("--config", type=str, default=None,
                   help="YAML config file with prompts list (uses config['prompts'])")

    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="Random seeds per prompt")
    p.add_argument("--methods", nargs="+",
                   default=["baseline", "aleatoric", "gradient", "resampling"],
                   choices=["baseline", "aleatoric", "gradient", "resampling"])

    # Pipeline params
    p.add_argument("--model-id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--guidance-start-step", type=int, default=0)
    p.add_argument("--guidance-n-steps", type=int, default=20)
    p.add_argument("--percentile", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--num-mc", type=int, default=5)
    p.add_argument("--n-ref", type=int, default=3)
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # Batch control
    p.add_argument("--out-dir", type=str, default="outputs/batch")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip prompt+seed combos whose NPZ already exists")
    p.add_argument("--no-clip", action="store_true",
                   help="Skip CLIPScore in the summary (faster)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be run, do not execute")

    return p.parse_args()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "_", text)[:60]


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        return args.prompts
    if args.config:
        if not HAS_YAML:
            print("[batch] PyYAML not installed. Use --prompts or: uv add pyyaml")
            sys.exit(1)
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        prompts = cfg.get("prompts", [])
        if not prompts:
            print(f"[batch] No 'prompts' key found in {args.config}")
            sys.exit(1)
        return prompts
    # Default fallback
    return [
        "a cat on a velvet chair",
        "a serene mountain lake at sunset",
        "a golden retriever in autumn leaves",
        "a gothic cathedral interior with stained glass",
        "a ceramic teapot on a wooden table",
    ]


def auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def run_compare(prompt: str, seed: int, args: argparse.Namespace,
                out_dir: Path) -> bool:
    """Run compare_uq_sd.py for one prompt+seed. Returns True on success."""
    cmd = [
        sys.executable, "scripts/compare_uq_sd.py",
        "--prompt", prompt,
        "--seed", str(seed),
        "--num-steps", str(args.num_steps),
        "--device", args.device,
        "--methods", *args.methods,
        "--model-id", args.model_id,
        "--guidance-scale", str(args.guidance_scale),
        "--guidance-start-step", str(args.guidance_start_step),
        "--guidance-n-steps", str(args.guidance_n_steps),
        "--percentile", str(args.percentile),
        "--lr", str(args.lr),
        "--num-mc", str(args.num_mc),
        "--n-ref", str(args.n_ref),
        "--n-pairs", str(args.n_pairs),
        "--height", str(args.height),
        "--width", str(args.width),
        "--out-dir", str(out_dir),
    ]
    if args.dry_run:
        print("[dry-run]", " ".join(cmd))
        return True

    print(f"\n{'='*65}")
    print(f"[batch] prompt={prompt!r}  seed={seed}")
    print(f"{'='*65}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_evaluate(result_dir: Path, args: argparse.Namespace) -> list[dict] | None:
    """Run evaluate_uq_sd.py on a result dir. Returns parsed rows or None."""
    if not (result_dir / "results.npz").exists():
        return None

    cmd = [
        sys.executable, "scripts/evaluate_uq_sd.py",
        "--result-dir", str(result_dir),
    ]
    if args.no_clip:
        cmd.append("--no-clip")

    if args.dry_run:
        print("[dry-run]", " ".join(cmd))
        return []

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        return None

    csv_path = result_dir / "evaluation_results.csv"
    if not csv_path.exists():
        return None

    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    args.device = args.device or auto_device()

    prompts = load_prompts(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[batch] device={args.device}  model={args.model_id}")
    print(f"[batch] {len(prompts)} prompts × {len(args.seeds)} seeds × {len(args.methods)} methods")
    print(f"[batch] output → {out_dir}\n")

    # ── Run compare for all prompt × seed ──────────────────────────────────
    all_rows: list[dict] = []
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for prompt in prompts:
        for seed in args.seeds:
            slug = slugify(prompt)
            run_dir = out_dir / f"{slug}_seed{seed}"

            if args.skip_existing and (run_dir / "results.npz").exists():
                print(f"[batch] SKIP {slug}_seed{seed} (already exists)")
                n_skip += 1
            else:
                success = run_compare(prompt, seed, args, out_dir)
                if success:
                    n_ok += 1
                else:
                    print(f"[batch] FAILED: {slug}_seed{seed}")
                    n_fail += 1
                    continue

            # ── Evaluate ────────────────────────────────────────────────────
            rows = run_evaluate(run_dir, args)
            if rows:
                for row in rows:
                    row["prompt"] = prompt
                    row["seed"] = seed
                all_rows.extend(rows)

    # ── Write aggregated CSV ────────────────────────────────────────────────
    if all_rows and not args.dry_run:
        agg_path = out_dir / "all_results.csv"
        fieldnames = ["prompt", "seed"] + [
            k for k in all_rows[0] if k not in ("prompt", "seed")
        ]
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[batch] Aggregated CSV → {agg_path}")

        # ── Print per-method summary ────────────────────────────────────────
        print("\n[batch] Per-method summary (averaged across all prompt×seed runs):")
        from collections import defaultdict
        method_stats: dict[str, list[dict]] = defaultdict(list)
        for row in all_rows:
            method_stats[row["Method"]].append(row)

        print(f"{'Method':<25} {'CLIPScore':>12} {'Mean γ²':>12} {'P95 γ²':>12}")
        print("-" * 63)
        for method, rows_m in sorted(method_stats.items()):
            clips = [float(r["CLIPScore"]) for r in rows_m if r["CLIPScore"] != "N/A"]
            mg2s = [float(r["Mean γ²"]) for r in rows_m if float(r["Mean γ²"]) > 0]
            p95s = [float(r["P95 γ²"]) for r in rows_m if float(r["P95 γ²"]) > 0]
            clip_str = f"{sum(clips)/len(clips):.3f}" if clips else "N/A"
            mg2_str  = f"{sum(mg2s)/len(mg2s):.4f}" if mg2s else "N/A"
            p95_str  = f"{sum(p95s)/len(p95s):.4f}" if p95s else "N/A"
            print(f"  {method:<23} {clip_str:>12} {mg2_str:>12} {p95_str:>12}")

    print(f"\n[batch] Done: {n_ok} OK, {n_skip} skipped, {n_fail} failed.")


if __name__ == "__main__":
    main()
