"""
diffusion_uncertainty/uq_laplace/gamma2.py
-------------------------------------------
Per-pixel epistemic variance (γ²) computation from the diagonal Laplace posterior.

Functions:
    compute_gamma2_llla      - γ² from last-layer Conv2d Laplace (MPS-safe, no vmap)
    ddim_transport_factors   - (a_t, b_t) coefficients for FLARE accumulation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def compute_gamma2_llla(
    features: torch.Tensor,
    posterior_var: torch.Tensor,
    conv: nn.Conv2d,
) -> torch.Tensor:
    """Per-pixel epistemic variance from diagonal LLLA for a Conv2d last layer.

    Uses the im2col (F.unfold) identity:  γ²_{c,l} = Σ_j p²_{j,l} · W_var_{c,j} + b_var_c

    where p is the unfolded receptive-field patch at spatial position l.

    Args:
        features:      (B, C_in, H, W) input tensor to conv_out captured by FeatureCapture
        posterior_var: (n_params,) diagonal posterior variance from ManualDiagLaplace
        conv:          the Conv2d layer (unet.conv_out)

    Returns:
        gamma2: (B, C_out, H, W)  per-pixel epistemic variance, clamped ≥ 0
    """
    features = features.to(dtype=torch.float32)
    var = posterior_var.to(device=features.device, dtype=torch.float32)

    C_out, C_in, kH, kW = conv.weight.shape
    patches = F.unfold(features, (kH, kW), padding=conv.padding)  # (B, C_in*kH*kW, L)
    B, CkK, L = patches.shape

    n_w = C_out * CkK
    W_var = var[:n_w].view(C_out, CkK)  # (C_out, C_in*kH*kW)

    # γ²[b,c,l] = Σ_j p²[b,j,l] · W_var[c,j]
    gamma2 = torch.einsum("bjl,cj->bcl", patches.pow(2), W_var)

    if conv.bias is not None:
        b_var = var[n_w : n_w + C_out]
        gamma2 = gamma2 + b_var.view(1, C_out, 1)

    H_out, W_out = features.shape[2], features.shape[3]
    return gamma2.view(B, C_out, H_out, W_out).clamp_min(0.0)


def ddim_transport_factors(
    abar: torch.Tensor,
    timesteps: list[torch.Tensor],
    step_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DDIM FLARE transport factors (a_t, b_t) for step step_idx.

    The FLARE projection formula is:
        u_proj += cum_a² · b_t² · γ²_t
        cum_a  *= a_t

    where a_t and b_t encode how the epistemic variance at step t propagates to x_0.

    Args:
        abar:       (T,) cumulative alpha schedule from scheduler.alphas_cumprod
        timesteps:  list of integer timestep tensors from scheduler.timesteps
        step_idx:   current step index (0 = first denoising step, T-noisy)

    Returns:
        (a, b): scalar tensors on the same device as abar
    """
    t_int = int(timesteps[step_idx].item())
    t_next = int(timesteps[step_idx + 1].item()) if step_idx < len(timesteps) - 1 else 0

    ab_t = abar[t_int]
    ab_next = abar[t_next] if t_next > 0 else torch.ones_like(ab_t)

    a = torch.sqrt(ab_next / ab_t.clamp_min(1e-12))
    b = torch.sqrt(1.0 - ab_next) - torch.sqrt(
        ab_next * (1.0 - ab_t) / ab_t.clamp_min(1e-12)
    )
    return a, b
