"""
Build method-level metrics for a completed run_uq_sweep.py output directory.

Inputs:
  - all_sweep_results.csv: per prompt/seed evaluation rows
  - fid_results.csv: aggregate FID rows from compute_sweep_fid.py

Output:
  - method_summary.csv: one row per method
  - per_prompt_metrics.csv: merged per prompt/seed rows with aggregate CLIPScore
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


LABEL_TO_METHOD = {
    "Baseline": "baseline",
    "Authors Aleatoric": "original",
    "Reimpl Aleatoric MC": "aleatoric",
    "Epistemic Gradient": "gradient",
    "Epistemic Resampling": "resampling",
}

METHOD_CATEGORY = {
    "baseline": "baseline",
    "original": "aleatoric_authors",
    "aleatoric": "aleatoric",
    "gradient": "epistemic",
    "resampling": "epistemic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stable Diffusion UQ sweep by method")
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-per-prompt-csv", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
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


def parse_float(value: Any) -> float | None:
    if value in {None, "", "N/A", "nan", "None"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def method_id(row: dict[str, str]) -> str:
    raw = row.get("Method ID")
    if raw:
        return raw
    label = row.get("Method", "")
    if label in LABEL_TO_METHOD:
        return LABEL_TO_METHOD[label]
    group = row.get("group", "")
    if group.startswith("fixed_"):
        return group.removeprefix("fixed_")
    return label.lower().replace(" ", "_")


def merge_clip_rows(eval_rows: list[dict[str, str]], clip_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not clip_rows:
        return eval_rows

    clip_by_key = {
        (row.get("result_dir", ""), row.get("method", "")): row.get("clip_score", "")
        for row in clip_rows
    }
    merged: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    for row in eval_rows:
        method = method_id(row)
        key = (row.get("result_dir", ""), method)
        seen_keys.add(key)
        out = dict(row)
        clip_score = clip_by_key.get(key)
        if clip_score:
            out["CLIPScore"] = clip_score
        if "Method ID" not in out:
            out["Method ID"] = method
        merged.append(out)

    for row in clip_rows:
        key = (row.get("result_dir", ""), row.get("method", ""))
        if key in seen_keys:
            continue
        merged.append({
            "trial_index": row.get("trial_index", ""),
            "group": row.get("group", ""),
            "prompt": row.get("prompt", ""),
            "seed": row.get("seed", ""),
            "result_dir": row.get("result_dir", ""),
            "Method ID": row.get("method", ""),
            "Method": row.get("method", ""),
            "CLIPScore": row.get("clip_score", ""),
            "Mean gamma2": "N/A",
            "P95 gamma2": "N/A",
        })
    return merged


def add_prompt_stats(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        prompt = row.get("prompt", "")
        row["prompt_n_chars"] = str(len(prompt))
        row["prompt_n_words"] = str(len(prompt.split()))
    return rows


def mean_or_na(values: list[float]) -> str:
    return f"{mean(values):.6f}" if values else "N/A"


def std_or_na(values: list[float]) -> str:
    return f"{stdev(values):.6f}" if len(values) > 1 else "N/A"


def summarize_eval(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    baseline_clip: dict[tuple[str, str, str, str], float] = {}

    for row in rows:
        method = method_id(row)
        by_method[method].append(row)
        clip = parse_float(row.get("CLIPScore"))
        if method == "baseline" and clip is not None:
            baseline_clip[(
                row.get("trial_index", ""),
                row.get("group", ""),
                row.get("prompt", ""),
                row.get("seed", ""),
            )] = clip

    summary: dict[str, dict[str, Any]] = {}
    for method, method_rows in by_method.items():
        clip_values: list[float] = []
        delta_values: list[float] = []
        mean_gamma2_values: list[float] = []
        p95_gamma2_values: list[float] = []
        prompt_word_values: list[float] = []
        prompt_char_values: list[float] = []

        for row in method_rows:
            clip = parse_float(row.get("CLIPScore"))
            if clip is not None:
                clip_values.append(clip)
                base = baseline_clip.get((
                    row.get("trial_index", ""),
                    row.get("group", ""),
                    row.get("prompt", ""),
                    row.get("seed", ""),
                ))
                if base is not None and method != "baseline":
                    delta_values.append(clip - base)

            mean_g = parse_float(row.get("Mean gamma2", row.get("Mean γ²")))
            p95_g = parse_float(row.get("P95 gamma2", row.get("P95 γ²")))
            if mean_g is not None:
                mean_gamma2_values.append(mean_g)
            if p95_g is not None:
                p95_gamma2_values.append(p95_g)
            prompt_words = parse_float(row.get("prompt_n_words"))
            prompt_chars = parse_float(row.get("prompt_n_chars"))
            if prompt_words is not None:
                prompt_word_values.append(prompt_words)
            if prompt_chars is not None:
                prompt_char_values.append(prompt_chars)

        summary[method] = {
            "method": method,
            "category": METHOD_CATEGORY.get(method, "other"),
            "n_eval": len(method_rows),
            "clip_mean": mean_or_na(clip_values),
            "clip_std": std_or_na(clip_values),
            "delta_clip_vs_baseline_mean": mean_or_na(delta_values),
            "delta_clip_vs_baseline_std": std_or_na(delta_values),
            "mean_gamma2_mean": mean_or_na(mean_gamma2_values),
            "p95_gamma2_mean": mean_or_na(p95_gamma2_values),
            "prompt_words_mean": mean_or_na(prompt_word_values),
            "prompt_chars_mean": mean_or_na(prompt_char_values),
        }
    return summary


def summarize_fid(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "")
        if method:
            by_method[method].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for method, method_rows in by_method.items():
        fid_values = [
            value for value in (parse_float(row.get("fid")) for row in method_rows)
            if value is not None
        ]
        statuses = sorted({row.get("status", "") for row in method_rows if row.get("status")})
        n_images_values = [
            int(float(row["n_images"]))
            for row in method_rows
            if parse_float(row.get("n_images")) is not None
        ]
        summary[method] = {
            "fid_mean": mean_or_na(fid_values),
            "fid_std": std_or_na(fid_values),
            "fid_n_images": max(n_images_values) if n_images_values else 0,
            "fid_status": "; ".join(statuses) if statuses else "N/A",
        }
    return summary


def main() -> None:
    args = parse_args()
    eval_rows = read_csv(args.sweep_dir / "all_sweep_results.csv")
    fid_rows = read_csv(args.sweep_dir / "fid_results.csv")
    clip_rows = read_csv(args.sweep_dir / "clip_results.csv")
    eval_rows = merge_clip_rows(eval_rows, clip_rows)
    eval_rows = add_prompt_stats(eval_rows)
    per_prompt_csv = args.out_per_prompt_csv or (args.sweep_dir / "per_prompt_metrics.csv")
    write_rows(per_prompt_csv, eval_rows)

    eval_summary = summarize_eval(eval_rows)
    fid_summary = summarize_fid(fid_rows)
    methods = sorted(set(eval_summary) | set(fid_summary), key=lambda m: (
        ["baseline", "aleatoric", "original", "gradient", "resampling"].index(m)
        if m in ["baseline", "aleatoric", "original", "gradient", "resampling"]
        else 99,
        m,
    ))

    rows: list[dict[str, Any]] = []
    for method in methods:
        row = {
            "method": method,
            "category": METHOD_CATEGORY.get(method, "other"),
            "n_eval": 0,
            "clip_mean": "N/A",
            "clip_std": "N/A",
            "delta_clip_vs_baseline_mean": "N/A",
            "delta_clip_vs_baseline_std": "N/A",
            "fid_mean": "N/A",
            "fid_std": "N/A",
            "fid_n_images": 0,
            "fid_status": "N/A",
            "mean_gamma2_mean": "N/A",
            "p95_gamma2_mean": "N/A",
            "prompt_words_mean": "N/A",
            "prompt_chars_mean": "N/A",
        }
        row.update(eval_summary.get(method, {}))
        row.update(fid_summary.get(method, {}))
        rows.append(row)

    out_csv = args.out_csv or (args.sweep_dir / "method_summary.csv")
    fieldnames = [
        "method",
        "category",
        "n_eval",
        "clip_mean",
        "clip_std",
        "delta_clip_vs_baseline_mean",
        "delta_clip_vs_baseline_std",
        "fid_mean",
        "fid_std",
        "fid_n_images",
        "fid_status",
        "mean_gamma2_mean",
        "p95_gamma2_mean",
        "prompt_words_mean",
        "prompt_chars_mean",
    ]
    write_rows(out_csv, rows, fieldnames=fieldnames)

    print(f"[summary] wrote {out_csv}")
    print(f"[summary] wrote {per_prompt_csv}")


if __name__ == "__main__":
    main()
