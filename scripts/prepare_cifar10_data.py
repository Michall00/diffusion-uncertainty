"""
Download CIFAR-10 and export it to the directory layout expected by this repo.

The local CIFAR loader expects:

    data/cifar10/images/train/Airplane/*.png
    data/cifar10/images/test/Airplane/*.png
    ...

torchvision stores CIFAR-10 in Python batch files, so this script converts it
once to class folders with PNG images.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CLASS_DIR_NAMES = {
    "airplane": "Airplane",
    "automobile": "Automobile",
    "bird": "Bird",
    "cat": "Cat",
    "deer": "Deer",
    "dog": "Dog",
    "frog": "Frog",
    "horse": "Horse",
    "ship": "Ship",
    "truck": "Truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 image folders")
    parser.add_argument("--out-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/_torchvision"))
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--limit-per-split", type=int, default=None, help="Optional quick-test limit")
    parser.add_argument("--no-download", action="store_true", help="Use an existing torchvision download")
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files")
    return parser.parse_args()


def export_split(split: str, args: argparse.Namespace) -> int:
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to prepare CIFAR-10. Install the project "
            "environment with the matching torch/torchvision build first."
        ) from exc

    train = split == "train"
    dataset = CIFAR10(root=str(args.raw_dir), train=train, download=not args.no_download)
    classes = list(dataset.classes)
    root = args.out_dir / "images" / split
    root.mkdir(parents=True, exist_ok=True)

    count = 0
    total = len(dataset) if args.limit_per_split is None else min(args.limit_per_split, len(dataset))
    for index in range(total):
        image, label = dataset[index]
        class_name = CLASS_DIR_NAMES[classes[int(label)]]
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        path = class_dir / f"{index:05d}.png"
        if path.exists() and not args.force:
            count += 1
            continue
        image.save(path)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    for split in args.splits:
        count = export_split(split, args)
        print(f"[cifar10] {split}: wrote/found {count} images under {args.out_dir / 'images' / split}")


if __name__ == "__main__":
    main()
