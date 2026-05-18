"""
Run Stable Diffusion UQ guidance sweeps from a YAML config.

This script is an orchestration layer over:
  - scripts/compare_uq_sd.py
  - scripts/evaluate_uq_sd.py

It is intended for long unattended runs: each trial gets its own output
directory, command file, optional log file, aggregated CSV rows, and optional
Weights & Biases image/metric logging.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("PyYAML is required. Run through uv with the project extras.") from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "stable_diffusion_uq_sweep.yaml"

COMPARE_KEYS = {
    "model_id": "--model-id",
    "device": "--device",
    "num_steps": "--num-steps",
    "guidance_scale": "--guidance-scale",
    "guidance_start_step": "--guidance-start-step",
    "guidance_n_steps": "--guidance-n-steps",
    "percentile": "--percentile",
    "lr": "--lr",
    "num_mc": "--num-mc",
    "n_ref": "--n-ref",
    "n_pairs": "--n-pairs",
    "height": "--height",
    "width": "--width",
    "laplace_mode": "--laplace-mode",
    "n_mc_subnet": "--n-mc-subnet",
    "subnet_max_params": "--subnet-max-params",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Stable Diffusion UQ guidance sweep")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/uq_sweep"))
    p.add_argument("--device", type=str, default=None, help="Override config device")
    p.add_argument("--model-id", type=str, default=None, help="Override config model_id")
    p.add_argument("--only-groups", nargs="+", default=None,
                   help="Run only selected sweep group names")
    p.add_argument("--include-disabled", action="store_true",
                   help="Also run groups with enabled: false")
    p.add_argument("--max-trials", type=int, default=None,
                   help="Limit expanded hyperparameter trials for quick checks")
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Limit prompts for quick checks")
    p.add_argument("--max-seeds", type=int, default=None,
                   help="Limit seeds for quick checks")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip prompt/seed/trial runs with an existing results.npz")
    p.add_argument("--no-clip", action="store_true",
                   help="Skip CLIPScore evaluation")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running them")
    p.add_argument("--stream-logs", action="store_true",
                   help="Stream child process logs instead of writing run.log files")

    p.add_argument("--wandb", action="store_true",
                   help="Log aggregated metrics and images to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--wandb-log-images", choices=["none", "grid", "all"],
                   default="grid")

    p.add_argument("--compute-fid", action="store_true",
                   help="Compute aggregate FID per trial after the sweep")
    p.add_argument("--fid-reference-dir", type=Path, default=None,
                   help="Optional real/reference image directory for FID")
    p.add_argument("--fid-batch-size", type=int, default=32)
    p.add_argument("--fid-min-images", type=int, default=8)
    p.add_argument("--fid-device", type=str, default=None,
                   help="FID device override; defaults to --device")
    return p.parse_args()


def slugify(text: str, max_len: int = 90) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s.-]+", "", text)
    text = re.sub(r"[\s_.-]+", "_", text)
    return text.strip("_")[:max_len]


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return cfg


def expand_group(group: dict[str, Any], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    axes: dict[str, list[Any]] = {}
    fixed: dict[str, Any] = {}
    for key, value in group.items():
        if key in {"name", "enabled"}:
            continue
        values = as_list(value)
        if len(values) > 1 and key != "methods":
            axes[key] = values
        else:
            fixed[key] = value

    if not axes:
        trial = defaults | fixed
        trial["group"] = group["name"]
        return [trial]

    trials: list[dict[str, Any]] = []
    keys = list(axes)
    for combo in itertools.product(*(axes[key] for key in keys)):
        trial = defaults | fixed | dict(zip(keys, combo))
        trial["group"] = group["name"]
        trials.append(trial)
    return trials


def expand_trials(cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    defaults = dict(cfg.get("defaults", {}))
    if "model_id" in cfg:
        defaults["model_id"] = cfg["model_id"]
    if args.device:
        defaults["device"] = args.device
    if args.model_id:
        defaults["model_id"] = args.model_id

    trials: list[dict[str, Any]] = []
    for group in cfg.get("sweep", []):
        name = group.get("name")
        if not name:
            raise ValueError("Every sweep group needs a name")
        if args.only_groups and name not in args.only_groups:
            continue
        if not group.get("enabled", True) and not args.include_disabled:
            continue
        trials.extend(expand_group(group, defaults))

    if args.max_trials is not None:
        trials = trials[:args.max_trials]
    return trials


def trial_slug(trial: dict[str, Any], index: int) -> str:
    parts = [
        f"{index:04d}",
        trial["group"],
        f"lm_{trial.get('laplace_mode', 'na')}",
        f"steps_{trial.get('num_steps')}",
        f"gwin_{trial.get('guidance_n_steps')}",
        f"lr_{trial.get('lr')}",
        f"p_{trial.get('percentile')}",
        f"pairs_{trial.get('n_pairs')}",
        f"mc_{trial.get('num_mc')}",
    ]
    return slugify("__".join(str(part) for part in parts))


def prompt_run_dir(out_dir: Path, prompt: str, seed: int) -> Path:
    return out_dir / f"{slugify(prompt, 60)}_seed{seed}"


def build_compare_cmd(prompt: str, seed: int, trial: dict[str, Any], out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/compare_uq_sd.py",
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--methods",
        *[str(method) for method in trial["methods"]],
        "--out-dir",
        str(out_dir),
    ]
    for key, flag in COMPARE_KEYS.items():
        if key in trial and trial[key] is not None:
            cmd.extend([flag, str(trial[key])])
    return cmd


def build_evaluate_cmd(result_dir: Path, no_clip: bool) -> list[str]:
    cmd = [sys.executable, "scripts/evaluate_uq_sd.py", "--result-dir", str(result_dir)]
    if no_clip:
        cmd.append("--no-clip")
    return cmd


def run_command(cmd: list[str], log_path: Path | None, dry_run: bool) -> int:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    if log_path is None:
        return subprocess.run(cmd, cwd=REPO_ROOT).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_f:
        log_f.write(f"\n$ {' '.join(cmd)}\n")
        log_f.flush()
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        ).returncode


def print_log_tail(log_path: Path | None, n_lines: int = 80) -> None:
    if log_path is None or not log_path.exists():
        return
    print(f"[sweep] Last {n_lines} log lines from {log_path}:")
    with log_path.open(errors="replace") as f:
        for line in deque(f, maxlen=n_lines):
            print(line, end="")
    print()


def read_evaluation_rows(result_dir: Path) -> list[dict[str, str]]:
    csv_path = result_dir / "evaluation_results.csv"
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> float | None:
    if value in {"", "N/A", None}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(rows[0])
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def init_wandb(args: argparse.Namespace, cfg: dict[str, Any], run_name: str):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed in this environment") from exc

    project = args.wandb_project or cfg.get("project_name") or "diffusion-uncertainty-uq"
    return wandb.init(
        project=project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_name,
        mode=args.wandb_mode,
        config={
            "config_path": str(args.config),
            "out_dir": str(args.out_dir),
            "no_clip": args.no_clip,
            "sweep_config": cfg,
        },
    )


def wandb_log_result(
    wandb_run: Any,
    args: argparse.Namespace,
    result_dir: Path,
    trial: dict[str, Any],
    prompt: str,
    seed: int,
    trial_index: int,
    rows: list[dict[str, Any]],
    step: int,
) -> None:
    if wandb_run is None:
        return
    import wandb

    for row in rows:
        clip_score = parse_float(row.get("CLIPScore", "N/A"))
        mean_g2 = parse_float(row.get("Mean gamma2", row.get("Mean γ²", "N/A")))
        p95_g2 = parse_float(row.get("P95 gamma2", row.get("P95 γ²", "N/A")))
        payload: dict[str, Any] = {
            "trial/index": trial_index,
            "trial/group": trial["group"],
            "trial/method": row.get("Method"),
            "trial/seed": seed,
            "trial/prompt": prompt,
            "params/lr": trial.get("lr"),
            "params/percentile": trial.get("percentile"),
            "params/guidance_n_steps": trial.get("guidance_n_steps"),
            "params/laplace_mode": trial.get("laplace_mode"),
            "params/n_pairs": trial.get("n_pairs"),
            "params/n_ref": trial.get("n_ref"),
            "params/num_mc": trial.get("num_mc"),
        }
        if clip_score is not None:
            payload["metrics/clip_score"] = clip_score
        if mean_g2 is not None:
            payload["metrics/mean_gamma2"] = mean_g2
        if p95_g2 is not None:
            payload["metrics/p95_gamma2"] = p95_g2
        wandb.log(payload, step=step)

    if args.wandb_log_images == "none":
        return

    images: dict[str, Any] = {}
    grid_path = result_dir / "comparison_grid.png"
    if grid_path.exists():
        images["images/comparison_grid"] = wandb.Image(
            str(grid_path),
            caption=f"{trial['group']} | seed={seed} | {prompt}",
        )
    if args.wandb_log_images == "all":
        for method in trial["methods"]:
            image_path = result_dir / f"{method}.png"
            if image_path.exists():
                images[f"images/{method}"] = wandb.Image(str(image_path))
            umap_path = result_dir / f"{method}_umap.png"
            if umap_path.exists():
                images[f"uncertainty/{method}_umap"] = wandb.Image(str(umap_path))
    if images:
        wandb.log(images, step=step)


def compute_fid(args: argparse.Namespace, out_root: Path) -> Path:
    fid_csv = out_root / "fid_results.csv"
    cmd = [
        sys.executable,
        "scripts/compute_sweep_fid.py",
        "--sweep-dir",
        str(out_root),
        "--out-csv",
        str(fid_csv),
        "--batch-size",
        str(args.fid_batch_size),
        "--device",
        str(args.fid_device or args.device or "cuda"),
        "--min-images",
        str(args.fid_min_images),
    ]
    if args.fid_reference_dir is not None:
        cmd.extend(["--reference-dir", str(args.fid_reference_dir)])
    code = run_command(cmd, None, args.dry_run)
    if code != 0:
        print(f"[sweep] WARN FID failed, returncode={code}")
    return fid_csv


def wandb_log_fid(wandb_run: Any, fid_csv: Path) -> None:
    if wandb_run is None or not fid_csv.exists():
        return
    import wandb

    with fid_csv.open() as f:
        rows = list(csv.DictReader(f))
    for idx, row in enumerate(rows):
        fid = parse_float(row.get("fid", "N/A"))
        payload: dict[str, Any] = {
            "fid/trial_index": int(row["trial_index"]),
            "fid/method": row["method"],
            "fid/reference": row["reference"],
            "fid/n_images": int(row["n_images"]),
            "fid/status": row["status"],
        }
        if fid is not None:
            payload["fid/value"] = fid
        wandb.log(payload, step=10_000_000 + idx)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    prompts = list(cfg.get("prompts", []))
    seeds = list(cfg.get("seeds", [42]))
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    if args.max_seeds is not None:
        seeds = seeds[:args.max_seeds]
    if not prompts:
        raise ValueError("No prompts configured")

    trials = expand_trials(cfg, args)
    if not trials:
        raise ValueError("No sweep trials selected")

    run_name = datetime.now().strftime("uq_sweep_%Y%m%d_%H%M%S")
    out_root = args.out_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "expanded_trials.json").open("w") as f:
        json.dump(trials, f, indent=2)

    total_jobs = len(trials) * len(prompts) * len(seeds)
    print(f"[sweep] config={args.config}")
    print(f"[sweep] output={out_root}")
    print(f"[sweep] {len(trials)} trials × {len(prompts)} prompts × {len(seeds)} seeds = {total_jobs} jobs")
    print(f"[sweep] dry_run={args.dry_run} skip_existing={args.skip_existing}")

    wandb_run = init_wandb(args, cfg, run_name)
    aggregate_csv = out_root / "all_sweep_results.csv"
    failures_csv = out_root / "failures.csv"

    step = 0
    ok = 0
    skipped = 0
    failed = 0

    for trial_index, trial in enumerate(trials):
        t_slug = trial_slug(trial, trial_index)
        trial_dir = out_root / t_slug
        trial_dir.mkdir(parents=True, exist_ok=True)
        with (trial_dir / "trial_config.json").open("w") as f:
            json.dump(trial, f, indent=2)

        for prompt, seed in itertools.product(prompts, seeds):
            result_dir = prompt_run_dir(trial_dir, prompt, seed)
            log_path = None if args.stream_logs else result_dir / "run.log"

            print(f"\n[sweep] trial={trial_index + 1}/{len(trials)} group={trial['group']} seed={seed}")
            print(f"[sweep] prompt={prompt!r}")
            print(f"[sweep] result_dir={result_dir}")

            if args.skip_existing and (result_dir / "results.npz").exists():
                print("[sweep] SKIP existing results.npz")
                skipped += 1
            else:
                cmd = build_compare_cmd(prompt, seed, trial, trial_dir)
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "command.txt").write_text(" ".join(cmd) + "\n")
                code = run_command(cmd, log_path, args.dry_run)
                if code != 0:
                    failed += 1
                    append_csv(failures_csv, [{
                        "trial_index": trial_index,
                        "group": trial["group"],
                        "prompt": prompt,
                        "seed": seed,
                        "returncode": code,
                        "result_dir": str(result_dir),
                    }])
                    print(f"[sweep] FAILED compare, returncode={code}")
                    print_log_tail(log_path)
                    continue
                ok += 1

            eval_cmd = build_evaluate_cmd(result_dir, args.no_clip)
            eval_code = run_command(eval_cmd, log_path, args.dry_run)
            if eval_code != 0:
                print(f"[sweep] WARN evaluate failed, returncode={eval_code}")
                print_log_tail(log_path)
                continue

            rows_raw = [] if args.dry_run else read_evaluation_rows(result_dir)
            rows: list[dict[str, Any]] = []
            for row in rows_raw:
                full_row = {
                    "trial_index": trial_index,
                    "group": trial["group"],
                    "prompt": prompt,
                    "seed": seed,
                    "result_dir": str(result_dir),
                    **{f"param_{k}": v for k, v in trial.items() if k != "methods"},
                    "param_methods": " ".join(str(m) for m in trial["methods"]),
                    **row,
                }
                rows.append(full_row)

            append_csv(aggregate_csv, rows)
            wandb_log_result(
                wandb_run, args, result_dir, trial, prompt, seed, trial_index, rows, step
            )
            step += 1

    fid_csv = None
    if args.compute_fid:
        fid_csv = compute_fid(args, out_root)
        wandb_log_fid(wandb_run, fid_csv)

    if wandb_run is not None:
        wandb_run.summary["jobs_ok"] = ok
        wandb_run.summary["jobs_skipped"] = skipped
        wandb_run.summary["jobs_failed"] = failed
        if aggregate_csv.exists():
            import wandb
            artifact = wandb.Artifact("uq_sweep_results", type="results")
            artifact.add_file(str(aggregate_csv))
            if fid_csv is not None and fid_csv.exists():
                artifact.add_file(str(fid_csv))
            if failures_csv.exists():
                artifact.add_file(str(failures_csv))
            wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print(f"\n[sweep] Done: {ok} OK, {skipped} skipped, {failed} failed.")
    print(f"[sweep] Aggregated CSV: {aggregate_csv}")
    if failures_csv.exists():
        print(f"[sweep] Failures CSV: {failures_csv}")


if __name__ == "__main__":
    main()
