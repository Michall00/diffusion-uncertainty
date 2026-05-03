"""
diffusion_uncertainty/uq_laplace/plotting.py
---------------------------------------------
Visualization helpers for UQ comparison results.

Functions:
    uncertainty_to_gray         - collapse (C, H, W) or (1, C, H, W) to 2D
    save_heatmap_png            - save a 2D uncertainty map as a magma heatmap PNG
    save_attention_overlay_png  - overlay normalized γ² heatmap on top of an image
    save_comparison_grid        - 2-row grid: generated images + uncertainty maps
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def uncertainty_to_gray(var_map: torch.Tensor | np.ndarray) -> np.ndarray:
    """Collapse a latent-space uncertainty map to a single 2D channel.

    Args:
        var_map: (C, H, W) or (1, C, H, W) or (H, W) tensor/array

    Returns:
        gray: (H, W) float32 numpy array
    """
    if isinstance(var_map, torch.Tensor):
        arr = var_map.detach().float().cpu().numpy()
    else:
        arr = np.asarray(var_map, dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim == 3:
        return arr.mean(axis=0).astype(np.float32)
    return arr.astype(np.float32)


def _normalize_0_1(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def save_heatmap_png(var_map: torch.Tensor | np.ndarray, path: Path | str) -> None:
    """Save a 2D uncertainty map as a magma colormap PNG.

    Args:
        var_map: uncertainty map (any shape accepted by uncertainty_to_gray)
        path:    output PNG file path
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")

    gray = uncertainty_to_gray(var_map)
    gray_norm = _normalize_0_1(gray)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(gray_norm, cmap="magma", vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _as_rgb_float(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """Normalize any image to (H, W, 3) float32 [0, 1]."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().float().cpu().numpy()
    else:
        arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size and finite.max() > 1.0:
        arr = arr / 255.0
    return np.nan_to_num(np.clip(arr, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)


def save_attention_overlay_png(
    image: np.ndarray | torch.Tensor,
    var_map: torch.Tensor | np.ndarray,
    path: Path | str,
    alpha: float = 0.45,
) -> None:
    """Save an image with a semi-transparent uncertainty heatmap overlay.

    Args:
        image:   source image (H, W, 3) float [0,1] or (3, H, W) tensor
        var_map: uncertainty map (any shape accepted by uncertainty_to_gray)
        path:    output PNG path
        alpha:   transparency of the heatmap overlay [0 = invisible, 1 = opaque]
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    import torch.nn.functional as F

    img_arr = _as_rgb_float(image)
    gray = uncertainty_to_gray(var_map)

    gray_t = torch.from_numpy(gray).view(1, 1, *gray.shape).float()
    gray_up = F.interpolate(
        gray_t, size=img_arr.shape[:2], mode="bilinear", align_corners=False
    ).squeeze().numpy()
    gray_norm = _normalize_0_1(gray_up)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img_arr)
    ax.imshow(gray_norm, cmap="magma", alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_comparison_grid(
    images: list[np.ndarray | torch.Tensor],
    labels: list[str],
    uncertainty_maps: list[torch.Tensor | np.ndarray | None],
    path: Path | str,
    title: str = "",
) -> None:
    """Save a 2-row comparison grid: images (top) and uncertainty maps (bottom).

    Args:
        images:           list of N images as (H, W, 3) float [0,1] arrays or tensors
        labels:           list of N method names (shown as column titles)
        uncertainty_maps: list of N uncertainty maps (or None if not available)
        path:             output PNG path
        title:            optional super-title for the whole figure
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (img, label, umap) in enumerate(zip(images, labels, uncertainty_maps)):
        if isinstance(img, torch.Tensor):
            img_np = img.detach().float().cpu().permute(1, 2, 0).numpy()
        else:
            img_np = np.asarray(img, dtype=np.float32)
        img_np = np.clip(img_np, 0.0, 1.0)

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(label, fontsize=10)
        axes[0, col].axis("off")

        if umap is not None:
            gray = uncertainty_to_gray(umap)
            gray_norm = _normalize_0_1(gray)
            axes[1, col].imshow(gray_norm, cmap="magma", vmin=0, vmax=1)
            axes[1, col].set_title(f"γ² / uncertainty", fontsize=8)
        else:
            axes[1, col].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[1, col].transAxes)
            axes[1, col].set_facecolor("#111111")
        axes[1, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)
