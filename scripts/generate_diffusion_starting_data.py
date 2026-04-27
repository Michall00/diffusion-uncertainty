"""
Generates X_T.pth files for different datasets.

This script generates X_T.pth files for a list of datasets. Each dataset is defined by its name, width, height, and number of channels.
The generated files are saved in the X_T folder.

"""
import sys
from path import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import argparse
import torch
from diffusion_uncertainty.paths import DIFFUSION_STARTING_POINTS
from dataclasses import dataclass


@dataclass
class Dataset:
    name: str
    width: int
    height: int
    num_channels: int
    num_classes: int


DATASETS: dict[str, Dataset] = {
    'imagenet64': Dataset('imagenet64', 64, 64, 3, 1000),
    'imagenet128': Dataset('imagenet128', 128, 128, 3, 1000),
    'imagenet128_uvit': Dataset('imagenet128_uvit', 2**(7-3), 2**(7-3), 4, 1000),
    'imagenet256': Dataset('imagenet256', 2**(8-3), 2**(8-3), 4, 1000),
    'imagenet512': Dataset('imagenet512', 2**(9-3), 2**(9-3), 4, 1000),
    'cifar10': Dataset('cifar10', 32, 32, 3, 10),
}


@torch.no_grad()
def main():
    args = parse_args()
    num_samples = args.num_samples
    device = 'cpu'
    extra_samples = args.extra_samples
    seed = args.seed
    total_samples = num_samples + extra_samples

    for dataset_metadata in [DATASETS[name] for name in args.datasets]:
        print('Generating for', dataset_metadata.name)
        dest_folder = DIFFUSION_STARTING_POINTS / dataset_metadata.name
        if not dest_folder.exists():
            dest_folder.mkdir()

        x_t_path = dest_folder / 'X_T.pth'
        y_path = dest_folder / 'y.pth'
        if not args.force and x_t_path.exists() and y_path.exists():
            y_existing = torch.load(y_path, map_location='cpu')
            if y_existing.shape[0] >= total_samples:
                print(f'Skipping {dataset_metadata.name}; existing files contain {y_existing.shape[0]} samples.')
                seed += 1
                continue
            print(f'Regenerating {dataset_metadata.name}; existing files contain only {y_existing.shape[0]} samples.')

        generator = torch.Generator(device='cpu').manual_seed(seed)
        gen_data = torch.randn(total_samples, dataset_metadata.num_channels, dataset_metadata.height, dataset_metadata.width, device=device, generator=generator)

        y = torch.randint(0, dataset_metadata.num_classes, (total_samples,), device=device, generator=generator)

        print("Stats of gen_data:")  
        print("\tMean:", gen_data.mean().item()) 
        print("\tStandard Deviation:", gen_data.std().item())
        print("\tMin:", gen_data.min().item())
        print("\tMax:", gen_data.max().item())

        
        torch.save(gen_data, x_t_path)
        torch.save(y, y_path)
        print('Using seed:', seed)
        print(f"Saved {dataset_metadata.name} to {x_t_path}")
        print(f"Saved {dataset_metadata.name} to {y_path}")

        seed += 1
    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help='Datasets for which X_T.pth and y.pth should be generated.',
    )
    parser.add_argument('--num-samples', type=int, default=60_000, dest='num_samples')
    parser.add_argument('--extra-samples', type=int, default=1_000, dest='extra_samples')
    parser.add_argument('--seed', type=int, default=49394)
    parser.add_argument('--force', action='store_true', help='Overwrite existing starting-point tensors.')
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError('--num-samples must be positive')
    if args.extra_samples < 0:
        raise ValueError('--extra-samples must be non-negative')
    return args


if __name__ == '__main__':
    main()
