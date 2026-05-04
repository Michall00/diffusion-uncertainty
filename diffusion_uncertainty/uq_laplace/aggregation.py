"""
diffusion_uncertainty/uq_laplace/aggregation.py
------------------------------------------------
Uncertainty-map aggregation and attention-weighting helpers.

Functions:
    uncertainty_stats   — mean / sum / p95 of a variance map
    normalize_attention_map — normalize any 2D attention tensor to a PDF
    attention_weighted_scores — attention-weighted mean and global mean of γ²
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def uncertainty_stats(var_map: np.ndarray) -> dict[str, float]:
    """Return basic statistics for a variance map.

    Args:
        var_map: any-shape float array (e.g. (C, H, W) or (H, W))

    Returns:
        dict with keys "mean", "sum", "p95"
    """
    var = np.asarray(var_map, dtype=np.float32)
    return {
        "mean": float(var.mean()),
        "sum": float(var.sum()),
        "p95": float(np.percentile(var, 95)),
    }


def normalize_attention_map(
    attention_map: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Normalize an attention map to sum to 1 (convert to a discrete PDF).

    Args:
        attention_map: 2D tensor or array of any spatial size

    Returns:
        (H, W) float32 array summing to 1 (or uniform if all-zero)
    """
    if isinstance(attention_map, torch.Tensor):
        attn = attention_map.detach().float().cpu().numpy()
    else:
        attn = np.asarray(attention_map, dtype=np.float32)
    attn = np.clip(attn, 0, None)
    total = float(attn.sum())
    if total > 0:
        attn = attn / total
    else:
        attn = np.full_like(attn, 1.0 / max(attn.size, 1))
    return attn.astype(np.float32)


def attention_weighted_scores(
    var_map: np.ndarray | torch.Tensor,
    attention_map: np.ndarray | torch.Tensor,
) -> tuple[float, float]:
    """Compute attention-weighted and global mean uncertainty.

    Bilinearly resizes the attention map to match var_map if needed.

    Args:
        var_map:       (C, H, W) or (H, W) variance / γ² map (latent space)
        attention_map: (H', W') cross-attention map (any resolution)

    Returns:
        Tuple (weighted_mean, global_mean):
            weighted_mean — Σ(var_gray * attn_normalized)
            global_mean   — mean(var_gray)
    """
    if isinstance(var_map, torch.Tensor):
        var_np = var_map.float().cpu().numpy()
    else:
        var_np = np.asarray(var_map, dtype=np.float32)

    if var_np.ndim == 3:
        var_gray = var_np.mean(axis=0)
    else:
        var_gray = var_np

    attn = normalize_attention_map(attention_map)

    if attn.shape != var_gray.shape:
        attn_t = torch.from_numpy(attn).view(1, 1, *attn.shape).float()
        attn = F.interpolate(
            attn_t,
            size=var_gray.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
        attn = normalize_attention_map(attn)

    weighted_mean = float((var_gray * attn).sum())
    global_mean = float(var_gray.mean())
    return weighted_mean, global_mean
