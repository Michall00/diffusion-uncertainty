"""
Download OpenAI guided-diffusion checkpoints expected by this repository.

The ImageNet64/ImageNet128 experiments load files directly from ./models, for
example models/128x128_diffusion.pt. This script downloads those exact files
from a Hugging Face mirror or from the original OpenAI blob storage URLs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_HF_REPO = "devilops/openai-guided-diffusion"
OPENAI_BASE_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021"


@dataclass(frozen=True)
class Checkpoint:
    filename: str
    datasets: tuple[str, ...]
    is_classifier: bool = False

    @property
    def openai_url(self) -> str:
        return f"{OPENAI_BASE_URL}/{self.filename}"


CHECKPOINTS: dict[str, Checkpoint] = {
    "64x64_diffusion.pt": Checkpoint("64x64_diffusion.pt", ("imagenet64",)),
    "128x128_diffusion.pt": Checkpoint("128x128_diffusion.pt", ("imagenet128",)),
    "64x64_classifier.pt": Checkpoint("64x64_classifier.pt", ("imagenet64",), is_classifier=True),
    "128x128_classifier.pt": Checkpoint("128x128_classifier.pt", ("imagenet128",), is_classifier=True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download guided-diffusion checkpoints")
    parser.add_argument(
        "--dataset",
        choices=["imagenet64", "imagenet128", "all"],
        default="imagenet128",
        help="Dataset checkpoint set to download.",
    )
    parser.add_argument(
        "--include-classifiers",
        action="store_true",
        help="Also download classifier checkpoints. Not needed for run_paper_metrics.py.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "hf", "openai"],
        default="auto",
        help="Download source. 'auto' tries Hugging Face first, then OpenAI.",
    )
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading.")
    return parser.parse_args()


def selected_checkpoints(args: argparse.Namespace) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    for checkpoint in CHECKPOINTS.values():
        if checkpoint.is_classifier and not args.include_classifiers:
            continue
        if args.dataset == "all" or args.dataset in checkpoint.datasets:
            checkpoints.append(checkpoint)
    return checkpoints


def download_from_hf(checkpoint: Checkpoint, args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install requirements.txt or use --source openai."
        ) from exc

    kwargs = {
        "repo_id": args.hf_repo,
        "filename": checkpoint.filename,
        "local_dir": str(args.models_dir),
    }
    try:
        return Path(hf_hub_download(**kwargs, local_dir_use_symlinks=False))
    except TypeError:
        return Path(hf_hub_download(**kwargs))


def download_from_openai(checkpoint: Checkpoint, args: argparse.Namespace) -> Path:
    args.models_dir.mkdir(parents=True, exist_ok=True)
    destination = args.models_dir / checkpoint.filename
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(checkpoint.openai_url, headers={"User-Agent": "diffusion-uncertainty"})

    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as f:
        total = int(response.headers.get("Content-Length", "0"))
        try:
            from tqdm import tqdm

            progress = tqdm(total=total or None, unit="B", unit_scale=True, desc=checkpoint.filename)
        except ImportError:
            progress = None

        try:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                if progress is not None:
                    progress.update(len(chunk))
        finally:
            if progress is not None:
                progress.close()

    partial.replace(destination)
    return destination


def ensure_regular_file(path: Path, destination: Path) -> None:
    if path == destination:
        if destination.is_symlink():
            target = destination.resolve()
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(target, temporary)
            temporary.replace(destination)
        return
    if path.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def download_checkpoint(checkpoint: Checkpoint, args: argparse.Namespace) -> Path:
    destination = args.models_dir / checkpoint.filename
    if destination.exists() and not args.force:
        print(f"[skip] {destination} already exists")
        return destination
    if args.dry_run:
        print(f"[dry-run] would download {checkpoint.filename} to {destination}")
        return destination

    args.models_dir.mkdir(parents=True, exist_ok=True)

    if args.source in {"auto", "hf"}:
        try:
            hf_path = download_from_hf(checkpoint, args)
            ensure_regular_file(hf_path, destination)
            print(f"[ok] {checkpoint.filename} from Hugging Face -> {destination}")
            return destination
        except Exception as exc:
            if args.source == "hf":
                raise
            print(f"[warn] Hugging Face download failed for {checkpoint.filename}: {exc}", file=sys.stderr)
            print("[warn] Falling back to original OpenAI URL", file=sys.stderr)

    path = download_from_openai(checkpoint, args)
    print(f"[ok] {checkpoint.filename} from OpenAI -> {path}")
    return path


def main() -> None:
    args = parse_args()
    checkpoints = selected_checkpoints(args)
    if not checkpoints:
        raise ValueError("No checkpoints selected")

    for checkpoint in checkpoints:
        download_checkpoint(checkpoint, args)


if __name__ == "__main__":
    main()
