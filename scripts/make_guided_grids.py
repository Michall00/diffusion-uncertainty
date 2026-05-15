import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image


def main():
    parser = argparse.ArgumentParser(description='Create baseline, guided, and diff grids for an uncertainty-guided run.')
    parser.add_argument('run', nargs='?', type=Path, help='Path to a results/uncertainty_guidance/... run directory.')
    parser.add_argument('--nrow', type=int, default=8)
    args = parser.parse_args()

    run = args.run or latest_guidance_run()
    if run is None:
        raise FileNotFoundError('No uncertainty_guidance run found.')

    guided = load_images(run / 'gen_images_threshold.pth')
    save_image(guided[:64] / 255, run / 'grid_guided.png', nrow=args.nrow)
    print(run / 'grid_guided.png')

    baseline_path = run / 'gen_images.pth'
    if baseline_path.exists():
        baseline = load_images(baseline_path)
        save_image(baseline[:64] / 255, run / 'grid_baseline.png', nrow=args.nrow)
        save_diff_grids(baseline, guided, run, args.nrow)
        print(run / 'grid_baseline.png')
        print(run / 'grid_abs_diff.png')
        print(run / 'grid_abs_diff_x10.png')
        print(run / 'grid_abs_diff_norm.png')


def latest_guidance_run() -> Path | None:
    root = Path('results/uncertainty_guidance')
    runs = sorted(root.glob('imagenet*/*'), key=lambda path: path.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def load_images(path: Path) -> torch.Tensor:
    return torch.load(path, map_location='cpu').float()


def save_diff_grids(baseline: torch.Tensor, guided: torch.Tensor, run: Path, nrow: int):
    abs_diff = (baseline - guided).abs()
    max_diff = abs_diff.max().clamp_min(1.0)
    save_image((abs_diff / 255)[:64], run / 'grid_abs_diff.png', nrow=nrow)
    save_image(((abs_diff * 10) / 255).clamp(0, 1)[:64], run / 'grid_abs_diff_x10.png', nrow=nrow)
    save_image((abs_diff / max_diff)[:64], run / 'grid_abs_diff_norm.png', nrow=nrow)


if __name__ == '__main__':
    main()
