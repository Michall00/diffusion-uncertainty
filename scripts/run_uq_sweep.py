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
    p.add_argument("--resume-dir", type=Path, default=None,
                   help="Resume/rebuild an existing outputs/uq_sweep/<run> directory")
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
    p.add_argument("--compute-clip", action="store_true",
                   help="Compute and log aggregate CLIPScore after every trial")
    p.add_argument("--clip-batch-size", type=int, default=32)
    p.add_argument("--clip-device", type=str, default="cpu")
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
    p.add_argument("--wandb-run-per-config", "--wandb-run-per-trial",
                   action="store_true", dest="wandb_run_per_config",
                   help="Create a separate W&B run for every expanded sweep config")
    p.add_argument("--wandb-log-images", choices=["none", "grid", "all"],
                   default="grid")
    p.add_argument("--overview-max-rows", type=int, default=12,
                   help="Number of prompt rows in the final method overview image")

    p.add_argument("--compute-fid", action="store_true",
                   help="Compute and log aggregate FID after every trial")
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


def load_prompt_seed_pairs(cfg: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, int]]:
    if cfg.get("prompt_seed_pairs"):
        pairs = [
            (str(item["prompt"]), int(item["seed"]))
            for item in cfg["prompt_seed_pairs"]
        ]
        if args.max_prompts is not None:
            pairs = pairs[:args.max_prompts]
        return pairs

    prompts = list(cfg.get("prompts", []))
    seeds = list(cfg.get("seeds", [42]))
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    if args.max_seeds is not None:
        seeds = seeds[:args.max_seeds]
    return list(itertools.product(prompts, seeds))


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


def run_command(cmd: list[str], log_path: Path | None, dry_run: bool, stream: bool = False) -> int:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    if log_path is None and not stream:
        return subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if stream:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        log_f = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("a")
            log_f.write(f"\n$ {' '.join(cmd)}\n")
            log_f.flush()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end="")
                if log_f is not None:
                    log_f.write(line)
        finally:
            if log_f is not None:
                log_f.flush()
                log_f.close()
        return proc.wait()
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backup_file(path: Path, suffix: str) -> None:
    if not path.exists():
        return
    backup_path = path.with_name(f"{path.name}.{suffix}.bak")
    counter = 1
    while backup_path.exists():
        backup_path = path.with_name(f"{path.name}.{suffix}.{counter}.bak")
        counter += 1
    path.replace(backup_path)


def prepare_resume_outputs(out_root: Path, run_name: str) -> None:
    suffix = f"resume_{run_name}"
    for path in [
        out_root / "all_sweep_results.csv",
        out_root / "failures.csv",
        out_root / "fid_results.csv",
        out_root / "clip_results.csv",
        out_root / "method_summary.csv",
        out_root / "per_prompt_metrics.csv",
    ]:
        backup_file(path, suffix)


def init_wandb(args: argparse.Namespace, cfg: dict[str, Any], run_name: str):
    if not args.wandb or args.dry_run:
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
        wandb_run.log(payload, step=step)

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
        wandb_run.log(images, step=step)


def compute_fid(
    args: argparse.Namespace,
    out_root: Path,
    out_csv: Path | None = None,
    trial_dir: Path | None = None,
) -> Path:
    fid_csv = out_csv or (out_root / "fid_results.csv")
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
    if trial_dir is not None:
        cmd.extend(["--only-trial-dir", str(trial_dir)])
    if args.fid_reference_dir is not None:
        cmd.extend(["--reference-dir", str(args.fid_reference_dir)])
    code = run_command(cmd, None, args.dry_run)
    if code != 0:
        print(f"[sweep] WARN FID failed, returncode={code}")
    return fid_csv


def compute_clip(
    args: argparse.Namespace,
    out_root: Path,
    out_csv: Path | None = None,
    trial_dir: Path | None = None,
) -> Path:
    clip_csv = out_csv or (out_root / "clip_results.csv")
    cmd = [
        sys.executable,
        "scripts/compute_sweep_clip.py",
        "--sweep-dir",
        str(out_root),
        "--out-csv",
        str(clip_csv),
        "--batch-size",
        str(args.clip_batch_size),
        "--device",
        args.clip_device,
    ]
    if trial_dir is not None:
        cmd.extend(["--only-trial-dir", str(trial_dir)])
    code = run_command(cmd, None, args.dry_run)
    if code != 0:
        print(f"[sweep] WARN aggregate CLIP failed, returncode={code}")
    return clip_csv


def summarize_methods(args: argparse.Namespace, out_root: Path) -> Path:
    summary_csv = out_root / "method_summary.csv"
    per_prompt_csv = out_root / "per_prompt_metrics.csv"
    cmd = [
        sys.executable,
        "scripts/summarize_uq_sweep.py",
        "--sweep-dir",
        str(out_root),
        "--out-csv",
        str(summary_csv),
        "--out-per-prompt-csv",
        str(per_prompt_csv),
    ]
    code = run_command(cmd, None, args.dry_run)
    if code != 0:
        print(f"[sweep] WARN method summary failed, returncode={code}")
    return summary_csv


def make_method_grid(args: argparse.Namespace, out_root: Path) -> Path:
    overview_dir = out_root / "overview"
    cmd = [
        sys.executable,
        "scripts/make_uq_method_grid.py",
        "--sweep-dir",
        str(out_root),
        "--out-dir",
        str(overview_dir),
        "--max-rows",
        str(args.overview_max_rows),
        "--include-prompts",
    ]
    code = run_command(cmd, None, args.dry_run)
    if code != 0:
        print(f"[sweep] WARN method overview grid failed, returncode={code}")
    return overview_dir / "method_overview.png"


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
        wandb_run.log(payload, step=10_000_000 + idx)


def wandb_log_csv_table(wandb_run: Any, csv_path: Path, table_name: str) -> None:
    if wandb_run is None or not csv_path.exists():
        return
    import wandb

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    table = wandb.Table(columns=list(rows[0]))
    for row in rows:
        table.add_data(*[row.get(column, "") for column in table.columns])
    wandb_run.log({table_name: table})


def wandb_log_overview(wandb_run: Any, overview_path: Path) -> None:
    if wandb_run is None or not overview_path.exists():
        return
    import wandb

    wandb_run.log({"images/method_overview": wandb.Image(str(overview_path))})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def group_rows_by_trial(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            trial_index = int(row.get("trial_index", ""))
        except ValueError:
            continue
        grouped.setdefault(trial_index, []).append(row)
    return grouped


def row_method_id(row: dict[str, str]) -> str:
    return row.get("Method ID") or row.get("method") or row.get("Method", "").lower().replace(" ", "_")


def mean_value(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std_value(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def set_payload_value(payload: dict[str, Any], key: str, value: float | int | str | None) -> None:
    if value is None:
        return
    payload[key] = value


def summarize_trial_for_wandb(
    trial_index: int,
    per_prompt_rows: list[dict[str, str]],
    fid_rows: list[dict[str, str]],
    job_counts: dict[str, int],
    clip_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trial/index": trial_index,
        "jobs/ok": job_counts.get("ok", 0),
        "jobs/skipped": job_counts.get("skipped", 0),
        "jobs/failed": job_counts.get("failed", 0),
    }
    by_method: dict[str, list[dict[str, str]]] = {}
    baseline_clip: dict[tuple[str, str], float] = {}
    for row in per_prompt_rows:
        method = row_method_id(row)
        by_method.setdefault(method, []).append(row)
        clip = parse_float(row.get("CLIPScore", "N/A"))
        if method == "baseline" and clip is not None:
            baseline_clip[(row.get("prompt", ""), row.get("seed", ""))] = clip

    guided_delta_means: list[float] = []
    for method, method_rows in sorted(by_method.items()):
        clip_values: list[float] = []
        delta_values: list[float] = []
        mean_gamma2_values: list[float] = []
        p95_gamma2_values: list[float] = []
        for row in method_rows:
            clip = parse_float(row.get("CLIPScore", "N/A"))
            if clip is not None:
                clip_values.append(clip)
                baseline = baseline_clip.get((row.get("prompt", ""), row.get("seed", "")))
                if method != "baseline" and baseline is not None:
                    delta_values.append(clip - baseline)
            mean_g2 = parse_float(row.get("Mean gamma2", row.get("Mean γ²", "N/A")))
            p95_g2 = parse_float(row.get("P95 gamma2", row.get("P95 γ²", "N/A")))
            if mean_g2 is not None:
                mean_gamma2_values.append(mean_g2)
            if p95_g2 is not None:
                p95_gamma2_values.append(p95_g2)

        prefix = f"metrics/{method}"
        set_payload_value(payload, f"{prefix}/n_eval", len(method_rows))
        set_payload_value(payload, f"{prefix}/clip_mean", mean_value(clip_values))
        set_payload_value(payload, f"{prefix}/clip_std", std_value(clip_values))
        set_payload_value(payload, f"{prefix}/mean_gamma2", mean_value(mean_gamma2_values))
        set_payload_value(payload, f"{prefix}/p95_gamma2", mean_value(p95_gamma2_values))
        if delta_values:
            delta_mean = mean_value(delta_values)
            guided_delta_means.append(delta_mean or 0.0)
            set_payload_value(payload, f"{prefix}/delta_clip_vs_baseline_mean", delta_mean)
            set_payload_value(payload, f"{prefix}/delta_clip_vs_baseline_std", std_value(delta_values))
            set_payload_value(
                payload,
                f"{prefix}/delta_clip_positive_rate",
                sum(value > 0 for value in delta_values) / len(delta_values),
            )

    if guided_delta_means:
        set_payload_value(payload, "metrics/best_guided_delta_clip", max(guided_delta_means))

    if clip_rows:
        clip_by_method: dict[str, list[dict[str, str]]] = {}
        baseline_aggregate_clip: dict[tuple[str, str], float] = {}
        for row in clip_rows:
            method = row.get("method", "")
            if not method:
                continue
            clip_by_method.setdefault(method, []).append(row)
            clip = parse_float(row.get("clip_score", "N/A"))
            if method == "baseline" and clip is not None:
                baseline_aggregate_clip[(row.get("prompt", ""), row.get("seed", ""))] = clip

        aggregate_delta_means: list[float] = []
        for method, method_rows in sorted(clip_by_method.items()):
            clip_values: list[float] = []
            delta_values: list[float] = []
            for row in method_rows:
                clip = parse_float(row.get("clip_score", "N/A"))
                if clip is None:
                    continue
                clip_values.append(clip)
                baseline = baseline_aggregate_clip.get((row.get("prompt", ""), row.get("seed", "")))
                if method != "baseline" and baseline is not None:
                    delta_values.append(clip - baseline)

            prefix = f"clip/{method}"
            set_payload_value(payload, f"{prefix}/mean", mean_value(clip_values))
            set_payload_value(payload, f"{prefix}/std", std_value(clip_values))
            set_payload_value(payload, f"{prefix}/n_images", len(clip_values))
            if delta_values:
                delta_mean = mean_value(delta_values)
                aggregate_delta_means.append(delta_mean or 0.0)
                set_payload_value(payload, f"{prefix}/delta_vs_baseline_mean", delta_mean)
                set_payload_value(payload, f"{prefix}/delta_vs_baseline_std", std_value(delta_values))
                set_payload_value(
                    payload,
                    f"{prefix}/delta_positive_rate",
                    sum(value > 0 for value in delta_values) / len(delta_values),
                )
        if aggregate_delta_means:
            set_payload_value(payload, "clip/best_delta_vs_baseline_mean", max(aggregate_delta_means))

    fid_by_method: dict[str, float] = {}
    for row in fid_rows:
        method = row.get("method", "")
        fid = parse_float(row.get("fid", "N/A"))
        if not method or fid is None:
            continue
        fid_by_method[method] = fid
        payload[f"fid/{method}/value"] = fid
        payload[f"fid/{method}/status"] = row.get("status", "")
        set_payload_value(payload, f"fid/{method}/n_images", parse_float(row.get("n_images", "N/A")))
        if row.get("reference"):
            payload["fid/reference"] = row["reference"]

    baseline_fid = fid_by_method.get("baseline")
    if baseline_fid is not None:
        for method, fid in fid_by_method.items():
            if method != "baseline":
                payload[f"fid/{method}/delta_vs_baseline"] = fid - baseline_fid
    return payload


def wandb_trial_run_name(base_name: str, trial: dict[str, Any], trial_index: int) -> str:
    return f"{base_name}__{trial_index:04d}_{slugify(str(trial['group']), 50)}"


def wandb_trial_config(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    run_name: str,
    out_root: Path,
    trial: dict[str, Any],
    trial_index: int,
    prompt_seed_pairs: list[tuple[str, int]],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "config_path": str(args.config),
        "sweep_run_name": run_name,
        "sweep_out_dir": str(out_root),
        "trial_index": trial_index,
        "prompt_seed_pairs_count": len(prompt_seed_pairs),
        "no_clip": args.no_clip,
        "compute_clip": args.compute_clip,
        "compute_fid": args.compute_fid,
        "fid_reference_dir": str(args.fid_reference_dir) if args.fid_reference_dir else None,
        "sweep_project_name": cfg.get("project_name"),
    }
    config.update(trial)
    if isinstance(config.get("methods"), list):
        config["methods"] = " ".join(str(method) for method in config["methods"])
    return config


def init_wandb_trial_run(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    run_name: str,
    out_root: Path,
    trial: dict[str, Any],
    trial_index: int,
    prompt_seed_pairs: list[tuple[str, int]],
):
    if not args.wandb or args.dry_run:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed in this environment") from exc

    project = args.wandb_project or cfg.get("project_name") or "diffusion-uncertainty-uq"
    base_name = args.wandb_run_name or run_name
    wandb_run = wandb.init(
        project=project,
        entity=args.wandb_entity,
        name=wandb_trial_run_name(base_name, trial, trial_index),
        mode=args.wandb_mode,
        reinit=True,
        config=wandb_trial_config(
            args, cfg, run_name, out_root, trial, trial_index, prompt_seed_pairs
        ),
    )
    wandb_run.log({"trial/index": trial_index, "trial/started": 1})
    return wandb_run


def wandb_log_trial_images(wandb_run: Any, args: argparse.Namespace, trial_dir: Path, trial: dict[str, Any]) -> None:
    if args.wandb_log_images == "none":
        return
    import wandb

    result_dirs = sorted(
        path for path in trial_dir.iterdir()
        if path.is_dir() and (path / "comparison_grid.png").exists()
    )
    images = [
        wandb.Image(
            str(result_dir / "comparison_grid.png"),
            caption=f"{trial['group']} | {result_dir.name}",
        )
        for result_dir in result_dirs[:args.overview_max_rows]
    ]
    if images:
        wandb_run.log({"images/comparison_grids": images})


def wandb_log_per_config_runs(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    run_name: str,
    out_root: Path,
    trials: list[dict[str, Any]],
    prompt_seed_pairs: list[tuple[str, int]],
    trial_job_counts: dict[int, dict[str, int]],
    fid_csv: Path | None,
    clip_csv: Path | None,
    failures_csv: Path,
    trial_wandb_runs: dict[int, Any] | None = None,
) -> None:
    if not args.wandb or args.dry_run:
        return
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed in this environment") from exc

    project = args.wandb_project or cfg.get("project_name") or "diffusion-uncertainty-uq"
    base_name = args.wandb_run_name or run_name
    per_prompt_csv = out_root / "per_prompt_metrics.csv"
    aggregate_csv = out_root / "all_sweep_results.csv"
    per_prompt_rows = read_csv(per_prompt_csv if per_prompt_csv.exists() else aggregate_csv)
    fid_rows = read_csv(fid_csv) if fid_csv is not None else []
    clip_rows = read_csv(clip_csv) if clip_csv is not None else []
    failure_rows = read_csv(failures_csv)

    per_prompt_by_trial = group_rows_by_trial(per_prompt_rows)
    fid_by_trial = group_rows_by_trial(fid_rows)
    clip_by_trial = group_rows_by_trial(clip_rows)
    failures_by_trial = group_rows_by_trial(failure_rows)

    for trial_index, trial in enumerate(trials):
        t_slug = trial_slug(trial, trial_index)
        trial_dir = out_root / t_slug
        trial_wandb_dir = trial_dir / "wandb"
        trial_wandb_dir.mkdir(parents=True, exist_ok=True)
        trial_per_prompt_csv = trial_wandb_dir / "per_prompt_metrics.csv"
        trial_fid_csv = trial_wandb_dir / "fid_results.csv"
        trial_clip_csv = trial_wandb_dir / "clip_results.csv"
        trial_failures_csv = trial_wandb_dir / "failures.csv"
        write_csv(trial_per_prompt_csv, per_prompt_by_trial.get(trial_index, []))
        write_csv(trial_fid_csv, fid_by_trial.get(trial_index, []))
        write_csv(trial_clip_csv, clip_by_trial.get(trial_index, []))
        write_csv(trial_failures_csv, failures_by_trial.get(trial_index, []))

        wandb_run = (trial_wandb_runs or {}).get(trial_index)
        if wandb_run is None:
            wandb_run = wandb.init(
                project=project,
                entity=args.wandb_entity,
                name=wandb_trial_run_name(base_name, trial, trial_index),
                mode=args.wandb_mode,
                reinit=True,
                config=wandb_trial_config(
                    args, cfg, run_name, out_root, trial, trial_index, prompt_seed_pairs
                ),
            )
        payload = summarize_trial_for_wandb(
            trial_index,
            per_prompt_by_trial.get(trial_index, []),
            fid_by_trial.get(trial_index, []),
            trial_job_counts.get(trial_index, {}),
            clip_rows=clip_by_trial.get(trial_index, []),
        )
        wandb_run.log(payload)
        for key, value in payload.items():
            wandb_run.summary[key] = value

        wandb_log_csv_table(wandb_run, trial_per_prompt_csv, "tables/per_prompt_metrics")
        if fid_by_trial.get(trial_index):
            wandb_log_csv_table(wandb_run, trial_fid_csv, "tables/fid_results")
        if clip_by_trial.get(trial_index):
            wandb_log_csv_table(wandb_run, trial_clip_csv, "tables/clip_results")
        if failures_by_trial.get(trial_index):
            wandb_log_csv_table(wandb_run, trial_failures_csv, "tables/failures")
        wandb_log_trial_images(wandb_run, args, trial_dir, trial)

        artifact = wandb.Artifact(
            f"uq_sweep_results_{trial_index:04d}_{slugify(str(trial['group']), 50)}",
            type="results",
        )
        artifact.add_file(str(trial_dir / "trial_config.json"))
        artifact.add_file(str(trial_per_prompt_csv))
        if fid_by_trial.get(trial_index):
            artifact.add_file(str(trial_fid_csv))
        if clip_by_trial.get(trial_index):
            artifact.add_file(str(trial_clip_csv))
        if failures_by_trial.get(trial_index):
            artifact.add_file(str(trial_failures_csv))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


def compute_and_log_trial_fid(
    args: argparse.Namespace,
    out_root: Path,
    trial_dir: Path,
    trial_index: int,
    aggregate_fid_csv: Path,
    wandb_run: Any,
    job_counts: dict[str, int],
) -> None:
    trial_wandb_dir = trial_dir / "wandb"
    trial_wandb_dir.mkdir(parents=True, exist_ok=True)
    trial_fid_csv = trial_wandb_dir / "fid_results.csv"
    compute_fid(args, out_root, out_csv=trial_fid_csv, trial_dir=trial_dir)
    fid_rows = read_csv(trial_fid_csv)
    if not fid_rows:
        return
    append_csv(aggregate_fid_csv, fid_rows)
    if wandb_run is None:
        return
    payload = summarize_trial_for_wandb(
        trial_index,
        per_prompt_rows=[],
        fid_rows=fid_rows,
        job_counts=job_counts,
    )
    wandb_run.log(payload)
    for key, value in payload.items():
        wandb_run.summary[key] = value
    wandb_log_csv_table(wandb_run, trial_fid_csv, "tables/fid_results")


def compute_and_log_trial_clip(
    args: argparse.Namespace,
    out_root: Path,
    trial_dir: Path,
    trial_index: int,
    aggregate_clip_csv: Path,
    wandb_run: Any,
    job_counts: dict[str, int],
) -> None:
    trial_wandb_dir = trial_dir / "wandb"
    trial_wandb_dir.mkdir(parents=True, exist_ok=True)
    trial_clip_csv = trial_wandb_dir / "clip_results.csv"
    compute_clip(args, out_root, out_csv=trial_clip_csv, trial_dir=trial_dir)
    clip_rows = read_csv(trial_clip_csv)
    if not clip_rows:
        return
    append_csv(aggregate_clip_csv, clip_rows)
    if wandb_run is None:
        return
    payload = summarize_trial_for_wandb(
        trial_index,
        per_prompt_rows=[],
        fid_rows=[],
        job_counts=job_counts,
        clip_rows=clip_rows,
    )
    wandb_run.log(payload)
    for key, value in payload.items():
        wandb_run.summary[key] = value
    wandb_log_csv_table(wandb_run, trial_clip_csv, "tables/clip_results")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    prompt_seed_pairs = load_prompt_seed_pairs(cfg, args)
    if not prompt_seed_pairs:
        raise ValueError("No prompts configured")

    trials = expand_trials(cfg, args)
    if not trials:
        raise ValueError("No sweep trials selected")

    if args.resume_dir is not None:
        out_root = args.resume_dir
        run_name = out_root.name
        if not out_root.exists():
            raise FileNotFoundError(f"--resume-dir does not exist: {out_root}")
        prepare_resume_outputs(out_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        run_name = datetime.now().strftime("uq_sweep_%Y%m%d_%H%M%S")
        out_root = args.out_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "expanded_trials.json").open("w") as f:
        json.dump(trials, f, indent=2)

    total_jobs = len(trials) * len(prompt_seed_pairs)
    print(f"[sweep] config={args.config}")
    print(f"[sweep] output={out_root}")
    if args.resume_dir is not None:
        print(f"[sweep] resume_dir={args.resume_dir}")
    print(f"[sweep] {len(trials)} trials × {len(prompt_seed_pairs)} prompt/seed pairs = {total_jobs} jobs")
    print(f"[sweep] dry_run={args.dry_run} skip_existing={args.skip_existing}")

    wandb_run = None if args.wandb_run_per_config else init_wandb(args, cfg, run_name)
    aggregate_csv = out_root / "all_sweep_results.csv"
    failures_csv = out_root / "failures.csv"
    trial_job_counts: dict[int, dict[str, int]] = {
        index: {"ok": 0, "skipped": 0, "failed": 0}
        for index in range(len(trials))
    }
    trial_wandb_runs: dict[int, Any] = {}
    fid_csv = out_root / "fid_results.csv" if args.compute_fid else None
    clip_csv = out_root / "clip_results.csv" if args.compute_clip else None

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
        trial_wandb_run = None
        if args.wandb_run_per_config:
            trial_wandb_run = init_wandb_trial_run(
                args, cfg, run_name, out_root, trial, trial_index, prompt_seed_pairs
            )
            if trial_wandb_run is not None:
                trial_wandb_runs[trial_index] = trial_wandb_run

        for prompt, seed in prompt_seed_pairs:
            result_dir = prompt_run_dir(trial_dir, prompt, seed)
            log_path = result_dir / "run.log"

            print(f"\n[sweep] trial={trial_index + 1}/{len(trials)} group={trial['group']} seed={seed}")
            print(f"[sweep] prompt={prompt!r}")
            print(f"[sweep] result_dir={result_dir}")

            if args.skip_existing and (result_dir / "results.npz").exists():
                print("[sweep] SKIP existing results.npz")
                skipped += 1
                trial_job_counts[trial_index]["skipped"] += 1
            else:
                cmd = build_compare_cmd(prompt, seed, trial, trial_dir)
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "command.txt").write_text(" ".join(cmd) + "\n")
                code = run_command(cmd, log_path, args.dry_run, stream=args.stream_logs)
                if code != 0:
                    failed += 1
                    trial_job_counts[trial_index]["failed"] += 1
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
                trial_job_counts[trial_index]["ok"] += 1

            eval_cmd = build_evaluate_cmd(result_dir, args.no_clip)
            eval_code = run_command(eval_cmd, log_path, args.dry_run, stream=args.stream_logs)
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
                trial_wandb_run or wandb_run,
                args,
                result_dir,
                trial,
                prompt,
                seed,
                trial_index,
                rows,
                step,
            )
            step += 1

        if args.compute_fid and fid_csv is not None:
            print(f"[sweep] computing FID after trial {trial_index + 1}/{len(trials)}")
            compute_and_log_trial_fid(
                args,
                out_root,
                trial_dir,
                trial_index,
                fid_csv,
                trial_wandb_run or wandb_run,
                trial_job_counts[trial_index],
            )
        if args.compute_clip and clip_csv is not None:
            print(f"[sweep] computing CLIPScore after trial {trial_index + 1}/{len(trials)}")
            compute_and_log_trial_clip(
                args,
                out_root,
                trial_dir,
                trial_index,
                clip_csv,
                trial_wandb_run or wandb_run,
                trial_job_counts[trial_index],
            )

    if args.compute_fid:
        fid_csv = out_root / "fid_results.csv"
    if args.compute_clip:
        clip_csv = out_root / "clip_results.csv"
    summary_csv = summarize_methods(args, out_root)
    overview_path = make_method_grid(args, out_root)
    if wandb_run is not None:
        wandb_log_csv_table(wandb_run, summary_csv, "tables/method_summary")
        per_prompt_csv = out_root / "per_prompt_metrics.csv"
        wandb_log_csv_table(
            wandb_run,
            per_prompt_csv if per_prompt_csv.exists() else aggregate_csv,
            "tables/per_prompt_metrics",
        )
        if fid_csv is not None:
            wandb_log_csv_table(wandb_run, fid_csv, "tables/fid_results")
        if clip_csv is not None:
            wandb_log_csv_table(wandb_run, clip_csv, "tables/clip_results")
        wandb_log_overview(wandb_run, overview_path)

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
            if clip_csv is not None and clip_csv.exists():
                artifact.add_file(str(clip_csv))
            if summary_csv.exists():
                artifact.add_file(str(summary_csv))
            per_prompt_csv = out_root / "per_prompt_metrics.csv"
            if per_prompt_csv.exists():
                artifact.add_file(str(per_prompt_csv))
            if overview_path.exists():
                artifact.add_file(str(overview_path))
            if failures_csv.exists():
                artifact.add_file(str(failures_csv))
            wandb_run.log_artifact(artifact)
        wandb_run.finish()

    if args.wandb_run_per_config:
        wandb_log_per_config_runs(
            args,
            cfg,
            run_name,
            out_root,
            trials,
            prompt_seed_pairs,
            trial_job_counts,
            fid_csv,
            clip_csv,
            failures_csv,
            trial_wandb_runs,
        )

    print(f"\n[sweep] Done: {ok} OK, {skipped} skipped, {failed} failed.")
    print(f"[sweep] Aggregated CSV: {aggregate_csv}")
    if summary_csv.exists():
        print(f"[sweep] Method summary CSV: {summary_csv}")
    per_prompt_csv = out_root / "per_prompt_metrics.csv"
    if per_prompt_csv.exists():
        print(f"[sweep] Per-prompt metrics CSV: {per_prompt_csv}")
    if overview_path.exists():
        print(f"[sweep] Method overview image: {overview_path}")
    if failures_csv.exists():
        print(f"[sweep] Failures CSV: {failures_csv}")


if __name__ == "__main__":
    main()
