"""
Compute FID for completed run_uq_sweep.py outputs.

FID is computed over a set of generated images, not for a single prompt/seed.
By default each method is compared to the baseline images from the same trial.
Pass --reference-dir to compare every method against an external image folder.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute FID over sweep image sets")
    p.add_argument("--sweep-dir", type=Path, required=True,
                   help="Path to outputs/uq_sweep/<timestamp>")
    p.add_argument("--reference-dir", type=Path, default=None,
                   help="Optional real/reference image directory")
    p.add_argument("--compare-to", type=str, default="baseline",
                   help="Method used as reference when --reference-dir is not set")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dims", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--min-images", type=int, default=8,
                   help="Skip FID for sets with fewer images")
    return p.parse_args()


def trial_dirs(sweep_dir: Path) -> list[Path]:
    return sorted(p for p in sweep_dir.iterdir() if (p / "trial_config.json").exists())


def prompt_result_dirs(trial_dir: Path) -> list[Path]:
    return sorted(p for p in trial_dir.iterdir() if p.is_dir() and (p / "results.npz").exists())


def method_images(trial_dir: Path) -> dict[str, list[Path]]:
    images: dict[str, list[Path]] = {}
    for result_dir in prompt_result_dirs(trial_dir):
        for path in sorted(result_dir.glob("*.png")):
            if path.name.endswith("_umap.png") or path.name.endswith("_uproj.png"):
                continue
            if path.name == "comparison_grid.png":
                continue
            method = path.stem
            images.setdefault(method, []).append(path)
    return images


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_image_set(root: Path, method: str, paths: list[Path]) -> Path:
    out = root / method
    out.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(paths):
        link_or_copy(src, out / f"{idx:06d}_{src.name}")
    return out


def compute_fid(path_a: Path, path_b: Path, args: argparse.Namespace) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths

    return float(calculate_fid_given_paths(
        [str(path_a), str(path_b)],
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims,
        num_workers=args.num_workers,
    ))


def main() -> None:
    args = parse_args()
    out_csv = args.out_csv or (args.sweep_dir / "fid_results.csv")
    rows: list[dict[str, object]] = []
    fid_root = args.sweep_dir / "_fid_sets"
    fid_root.mkdir(parents=True, exist_ok=True)

    for trial_index, trial_dir in enumerate(trial_dirs(args.sweep_dir)):
        images_by_method = method_images(trial_dir)
        if args.reference_dir is None and args.compare_to not in images_by_method:
            rows.append({
                "trial_index": trial_index,
                "trial_dir": str(trial_dir),
                "method": "",
                "reference": args.compare_to,
                "n_images": 0,
                "fid": "N/A",
                "status": f"missing reference method: {args.compare_to}",
            })
            continue

        trial_fid_root = fid_root / trial_dir.name
        method_dirs = {
            method: materialize_image_set(trial_fid_root, method, paths)
            for method, paths in images_by_method.items()
        }

        if args.reference_dir is not None:
            reference_dir = args.reference_dir
            reference_name = str(args.reference_dir)
            reference_count = len(list(reference_dir.glob("*")))
        else:
            reference_dir = method_dirs[args.compare_to]
            reference_name = args.compare_to
            reference_count = len(images_by_method[args.compare_to])

        for method, paths in sorted(images_by_method.items()):
            n_images = len(paths)
            row = {
                "trial_index": trial_index,
                "trial_dir": str(trial_dir),
                "method": method,
                "reference": reference_name,
                "n_images": n_images,
                "fid": "N/A",
                "status": "ok",
            }
            if args.reference_dir is None and method == args.compare_to:
                row["fid"] = "0.0"
                row["status"] = "reference"
            elif n_images < args.min_images or reference_count < args.min_images:
                row["status"] = (
                    f"too few images: method={n_images}, reference={reference_count}, "
                    f"min={args.min_images}"
                )
            else:
                try:
                    row["fid"] = f"{compute_fid(reference_dir, method_dirs[method], args):.6f}"
                except Exception as exc:  # pragma: no cover - runtime diagnostic
                    row["status"] = f"error: {exc}"
            rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fieldnames = ["trial_index", "trial_dir", "method", "reference", "n_images", "fid", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[fid] wrote {out_csv}")


if __name__ == "__main__":
    main()
