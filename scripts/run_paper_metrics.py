"""
Run paper-like ImageNet/CIFAR metrics for uncertainty guidance.

This wrapper runs the two protocols used in the original project:
  1. FID for generated baseline vs uncertainty-guided samples.
  2. AUSE/AURG for reconstruction from halfway-noised real images.

It is intended for dataset/class-conditioned experiments, not text-prompt
Stable Diffusion sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
MODELS = REPO_ROOT / "models"
DATA = REPO_ROOT / "data"

MODEL_REQUIREMENTS = {
    "imagenet64": (
        MODELS / "64x64_diffusion.pt",
        "python scripts/download_guided_diffusion_models.py --dataset imagenet64",
    ),
    "imagenet128": (
        MODELS / "128x128_diffusion.pt",
        "python scripts/download_guided_diffusion_models.py --dataset imagenet128",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FID + AUSE/AURG paper-like metrics")
    p.add_argument("--dataset", default="imagenet128",
                   choices=["cifar10", "imagenet64", "imagenet128"])
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-index", type=int, default=0)

    p.add_argument("--scheduler", "--scheduler-type", dest="scheduler_type",
                   default="uncertainty_centered")
    p.add_argument("--samplers", nargs="+", default=["ddim"],
                   choices=["ddim", "ddpm", "all"],
                   help="Sampling schedulers for FID/guidance. Use 'all' for ddim ddpm.")
    p.add_argument("--guidance-types", nargs="+", default=["posterior"],
                   choices=["gradient", "posterior", "second_order", "all"],
                   help="Guidance variants to evaluate. Use 'all' for gradient posterior second_order.")
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--start-step-guidance", type=int, default=0)
    p.add_argument("--num-steps-guidance", type=int, default=20)
    p.add_argument("-M", "--num-uncertainty-samples", type=int, default=5, dest="M")
    p.add_argument("--percentile", type=float, default=0.95)
    p.add_argument("--lambda-update", type=float, default=1.0)
    p.add_argument("--gradient-wrt", default="input", choices=["input", "score"])
    p.add_argument("--gradient-direction", default="descend", choices=["ascend", "descend"])
    p.add_argument("--threshold-type", default="higher", choices=["higher", "lower"])
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--skip-ause", action="store_true")
    p.add_argument("--invert-uncertainty", action="store_true")
    p.add_argument("--ensure-starting-data", action="store_true",
                   help="Generate missing diffusion starting points before FID guidance")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip local file checks before running commands")
    p.add_argument("--preflight-only", action="store_true",
                   help="Only check required local files and exit")

    p.add_argument("--out-dir", type=Path, default=Path("outputs/paper_metrics"))
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="diffusion-uncertainty-paper-metrics")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def preflight(args: argparse.Namespace) -> None:
    missing: list[str] = []
    hints: list[str] = []

    model_req = MODEL_REQUIREMENTS.get(args.dataset)
    if model_req is not None:
        model_path, hint = model_req
        if not model_path.exists():
            missing.append(str(model_path))
            hints.append(hint)

    if not args.skip_fid:
        fid_dir = RESULTS / "score_dataset_pytorch_fid" / args.dataset
        for name in ["m.pt", "s.pt"]:
            path = fid_dir / name
            if not path.exists():
                missing.append(str(path))
        if not fid_dir.exists() or not (fid_dir / "m.pt").exists() or not (fid_dir / "s.pt").exists():
            dataset_dir = DATA / args.dataset
            hints.append(
                "Prepare FID stats, e.g. after making the dataset available under "
                f"{dataset_dir}:\n"
                f"python scripts/compute_dataset_fid.py {dataset_dir} "
                f"--dataset-name {args.dataset} --device {args.device} --batch-size {args.batch_size}"
            )

    if not args.skip_ause:
        dataset_dir = DATA / args.dataset
        if args.dataset != "cifar10" and not dataset_dir.exists():
            missing.append(str(dataset_dir))
            hints.append(
                f"Make the {args.dataset} dataset available under {dataset_dir} "
                "or create the expected symlink in data/."
            )

    data_dir = RESULTS / "diffusion_starting_points" / args.dataset
    if not args.ensure_starting_data:
        for name in ["X_T.pth", "y.pth"]:
            path = data_dir / name
            if not path.exists():
                missing.append(str(path))
        if not (data_dir / "X_T.pth").exists() or not (data_dir / "y.pth").exists():
            hints.append(
                "Generate starting data or add --ensure-starting-data:\n"
                f"python scripts/generate_diffusion_starting_data.py --datasets {args.dataset} "
                f"--num-samples {args.start_index + args.num_samples} --extra-samples 0"
            )

    if missing:
        lines = ["Preflight failed. Missing required files/directories:"]
        lines.extend(f"  - {item}" for item in dict.fromkeys(missing))
        if hints:
            lines.append("")
            lines.append("Suggested fixes:")
            lines.extend(f"  - {hint}" for hint in dict.fromkeys(hints))
        raise FileNotFoundError("\n".join(lines))


def run_command(cmd: list[str], log_file, dry_run: bool) -> int:
    print("$", " ".join(cmd), flush=True)
    log_file.write("$ " + " ".join(cmd) + "\n")
    log_file.flush()
    if dry_run:
        return 0
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(proc.stdout, end="")
    log_file.write(proc.stdout)
    log_file.flush()
    return proc.returncode


def maybe_generate_starting_data(args: argparse.Namespace, log_file) -> None:
    data_dir = RESULTS / "diffusion_starting_points" / args.dataset
    x_t = data_dir / "X_T.pth"
    y = data_dir / "y.pth"
    if x_t.exists() and y.exists():
        return
    if not args.ensure_starting_data:
        raise FileNotFoundError(
            f"Missing {x_t} or {y}. Re-run with --ensure-starting-data, or run "
            "scripts/generate_diffusion_starting_data.py manually."
        )
    cmd = [
        sys.executable,
        "scripts/generate_diffusion_starting_data.py",
        "--datasets",
        args.dataset,
        "--num-samples",
        str(args.start_index + args.num_samples),
        "--extra-samples",
        "0",
    ]
    code = run_command(cmd, log_file, args.dry_run)
    if code != 0:
        raise RuntimeError(f"Starting data generation failed with code {code}")


def read_latest_guidance_result(before_count: int) -> dict[str, Any]:
    path = RESULTS / "uncertainty_guidance" / "results.json"
    if not path.exists():
        return {}
    with path.open() as f:
        rows = json.load(f)
    if len(rows) <= before_count:
        return {}
    return rows[-1]


def guidance_results_count() -> int:
    path = RESULTS / "uncertainty_guidance" / "results.json"
    if not path.exists():
        return 0
    with path.open() as f:
        return len(json.load(f))


def selected_guidance_types(args: argparse.Namespace) -> list[str]:
    if "all" in args.guidance_types:
        return ["posterior", "gradient", "second_order"]
    return args.guidance_types


def selected_samplers(args: argparse.Namespace) -> list[str]:
    if "all" in args.samplers:
        return ["ddim", "ddpm"]
    return args.samplers


def run_fid_guidance(args: argparse.Namespace, guidance_type: str, sampler: str, log_file) -> dict[str, Any]:
    before_count = guidance_results_count()
    cmd = [
        sys.executable,
        "scripts/generate_images_with_uncertainty_threshold.py",
        "--dataset",
        args.dataset,
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--num-steps",
        str(args.num_steps),
        "--start-step-guidance",
        str(args.start_step_guidance),
        "--num-steps-guidance",
        str(args.num_steps_guidance),
        "--start-index",
        str(args.start_index),
        "--seed",
        str(args.seed),
        "--scheduler",
        args.scheduler_type,
        "--guidance-type",
        guidance_type,
        "--sampler",
        sampler,
        "-M",
        str(args.M),
        "--percentile",
        str(args.percentile),
        "--lambda-update",
        str(args.lambda_update),
        "--gradient-wrt",
        args.gradient_wrt,
        "--gradient-direction",
        args.gradient_direction,
        "--threshold-type",
        args.threshold_type,
        "--use-percentile",
    ]
    if args.device == "cpu":
        cmd.append("--on-cpu")
    if args.skip_fid:
        cmd.append("--skip-fid")
    code = run_command(cmd, log_file, args.dry_run)
    if code != 0:
        raise RuntimeError(f"FID guidance command failed with code {code}")
    return {} if args.dry_run else read_latest_guidance_result(before_count)


def run_ause(args: argparse.Namespace, log_file) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/compute_ause.py",
        "--dataset",
        args.dataset,
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--scheduler",
        args.scheduler_type,
        "-M",
        str(args.M),
        "--start-step-uc",
        str(args.start_step_guidance),
        "--num-steps-uc",
        str(args.num_steps_guidance),
        "--seed",
        str(args.seed),
    ]
    if args.invert_uncertainty:
        cmd.append("--invert-uncertainty")
    code = run_command(cmd, log_file, args.dry_run)
    if code != 0:
        raise RuntimeError(f"AUSE command failed with code {code}")
    if args.dry_run:
        return {}
    suffix = "_inverted" if args.invert_uncertainty else ""
    path = RESULTS / "ause" / args.dataset / f"results_{args.scheduler_type}{suffix}.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def flatten_metrics(fid_result: dict[str, Any], ause_result: dict[str, Any], guidance_type: str, sampler: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics["guidance_type"] = guidance_type
    metrics["sampler"] = sampler
    for key in [
        "fid_score",
        "fid_score_guidance",
        "pixel_mean_abs_diff",
        "pixel_max_abs_diff",
        "pixel_changed_fraction",
    ]:
        if key in fid_result:
            metrics[key] = fid_result[key]
    if "fid_score" in metrics and "fid_score_guidance" in metrics:
        metrics["fid_improvement"] = float(metrics["fid_score"]) - float(metrics["fid_score_guidance"])
    if "mean_ause" in ause_result:
        metrics["mean_ause"] = float(ause_result["mean_ause"])
    if "mean_aurg" in ause_result:
        metrics["mean_aurg"] = float(ause_result["mean_aurg"])
    return metrics


def write_summary(out_dir: Path, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "metrics": metrics,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(payload, f, indent=2)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def write_summary_rows(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": rows,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(payload, f, indent=2)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def log_wandb(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    if not args.wandb or args.dry_run:
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=f"{args.dataset}_{args.scheduler_type}_{'-'.join(selected_samplers(args))}_{'-'.join(selected_guidance_types(args))}_{args.num_samples}",
        config=vars(args),
    )
    for row in rows:
        wandb.log(row)
    run.finish()


def main() -> None:
    args = parse_args()
    if not args.skip_preflight:
        preflight(args)
    if args.preflight_only:
        print("[paper-metrics] preflight ok")
        return
    run_name = datetime.now().strftime("paper_metrics_%Y%m%d_%H%M%S")
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with (out_dir / "run.log").open("w") as log_file:
        maybe_generate_starting_data(args, log_file)
        ause_result = {} if args.skip_ause else run_ause(args, log_file)
        for sampler in selected_samplers(args):
            for guidance_type in selected_guidance_types(args):
                fid_result = {} if args.skip_fid else run_fid_guidance(args, guidance_type, sampler, log_file)
                rows.append(flatten_metrics(fid_result, ause_result, guidance_type, sampler))
    write_summary_rows(out_dir, args, rows)
    log_wandb(args, rows)
    print(json.dumps(rows, indent=2))
    print(f"[paper-metrics] wrote {out_dir}")


if __name__ == "__main__":
    main()
