"""
diffusion_uncertainty/uq_laplace/guidance.py
---------------------------------------------
Epistemic guidance functions for SD v1.5 denoising steps.

All functions are MPS-safe (no vmap, no in-place grad ops on tensors
that require grad).

Functions:
    apply_gradient_guidance    - posterior update on pred_eps using Laplace γ²
    apply_resampling_guidance  - local noise injection on latents using γ²
"""

from __future__ import annotations

import torch


def _percentile_threshold(
    tensor: torch.Tensor,
    percentile: float,
) -> torch.Tensor:
    """Compute per-sample percentile threshold and return a binary mask.

    Args:
        tensor:     (B, C, H, W) float32 uncertainty map
        percentile: fraction in [0, 1], e.g. 0.95

    Returns:
        mask: (B, C, H, W) float32 binary mask, 1 where tensor > threshold
    """
    B = tensor.shape[0]
    flat = tensor.flatten(1)  # (B, C*H*W)
    threshold = torch.quantile(
        flat.to(torch.float32), percentile, dim=1, keepdim=True
    ).view(B, *([1] * (tensor.ndim - 1)))
    return (tensor > threshold).to(torch.float32)


def apply_gradient_guidance(
    pred_eps: torch.Tensor,
    gamma2: torch.Tensor,
    alpha_hat_t: torch.Tensor,
    percentile: float = 0.95,
    lr: float = 1.0,
) -> torch.Tensor:
    """Epistemic gradient guidance via Bayesian posterior update on pred_eps.

    In high-γ² (epistemically uncertain) regions, blends pred_eps toward a
    precision-weighted posterior estimate.  The formula mirrors the aleatoric
    posterior guidance (use_posterior=True) with γ² replacing MC variance:

        inv_var  = 1 / γ²
        post_var = 1 / (inv_var + 1 / α_hat_t)
        post_eps = post_var · inv_var · pred_eps  (precision-weighted toward prior)
        out_eps  = pred_eps + lr · mask · (post_eps − pred_eps)

    In confident regions (low γ²), post_eps ≈ pred_eps (likelihood-dominated).
    In uncertain regions (high γ²), post_eps shrinks toward zero (prior-dominated).

    NOTE on calibration: raw γ² scale depends on prior_prec and conv_out activation
    magnitudes, so it is normalized per-step before the posterior formula:
        γ²_norm = (γ² / γ²_ref) · α̂_t
    where γ²_ref = p(percentile+0.02) of γ².  This maps the high-uncertainty
    pixels to γ²_norm ≈ α̂_t so that the posterior gives ~50% shrinkage there,
    making the correction magnitude independent of prior_prec choice.

    Args:
        pred_eps:    (B, C, H, W) predicted noise from CFG
        gamma2:      (B, C, H, W) per-pixel epistemic variance from Laplace
        alpha_hat_t: scalar ᾱ_t from scheduler.alphas_cumprod[t]
        percentile:  fraction of pixels to apply guidance to (highest γ²)
        lr:          guidance strength in [0, 1]  (1.0 = full correction)

    Returns:
        corrected pred_eps with same shape and dtype as input
    """
    orig_dtype = pred_eps.dtype
    pred_f = pred_eps.float()
    g2 = gamma2.float().clamp_min(1e-10)

    alpha_scalar = max(
        float(alpha_hat_t.item()) if alpha_hat_t.numel() == 1 else 0.5, 1e-3
    )

    # Normalize γ² so that the (percentile+2%) pixel maps to alpha_hat_t.
    # This calibrates the posterior formula to be meaningful regardless of
    # the absolute scale of posterior_variance (i.e., prior_prec choice).
    ref_q = min(percentile + 0.02, 0.99)
    g2_ref = torch.quantile(g2.flatten().float(), ref_q).clamp_min(1e-8)
    g2_n = (g2 / g2_ref) * alpha_scalar  # high-uncertainty pixels ≈ alpha_hat_t

    inv_var = 1.0 / g2_n
    post_var = 1.0 / (inv_var + 1.0 / alpha_scalar)
    post_eps = post_var * inv_var * pred_f

    mask = _percentile_threshold(g2, percentile)
    corrected = pred_f + lr * mask * (post_eps - pred_f)
    return corrected.to(orig_dtype)


def apply_resampling_guidance(
    latent: torch.Tensor,
    gamma2: torch.Tensor,
    percentile: float = 0.95,
    lr: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Epistemic resampling guidance via local noise injection on latents.

    After the DDIM denoising step, injects additional noise proportional to
    sqrt(γ²) in high-uncertainty spatial regions.  This implements a local
    "epistemic reset" — uncertain positions are pushed back toward stochastic
    exploration while confident positions remain unchanged.

    Analogous to the DDPM aleatoric noise term β̃_t in BayesDiff, but applied
    to the latent after each denoising step rather than during diffusion.

        mask    = γ² > percentile_threshold(γ²)
        noise   ~ N(0, I)
        noise_scale = sqrt(γ² / γ²_ref).clamp(0, 1)   # p95 pixel → scale=1.0
        latent += lr · noise_scale · noise · mask

    Normalizing by γ²_ref (p95) ensures the injection magnitude is always
    lr at the most uncertain pixels, regardless of prior_prec absolute scale.

    Args:
        latent:     (B, C, H, W) latent after DDIM step
        gamma2:     (B, C, H, W) per-pixel epistemic variance from Laplace
        percentile: fraction of pixels to apply guidance to
        lr:         noise injection strength (amplitude at p95 uncertainty pixels)
        generator:  optional torch.Generator for reproducibility

    Returns:
        latent with added epistemic noise in uncertain regions, same dtype
    """
    orig_dtype = latent.dtype
    g2 = gamma2.float().clamp_min(0.0)

    # Normalize so the percentile-th pixel gets noise_scale = 1.0
    g2_ref = torch.quantile(g2.flatten().float(), percentile).clamp_min(1e-8)
    noise_scale = torch.sqrt(g2 / g2_ref).clamp(0.0, 1.0)

    mask = _percentile_threshold(g2, percentile)
    noise = torch.randn(latent.shape, device=latent.device, dtype=torch.float32, generator=generator)
    injection = lr * noise_scale * noise * mask
    return (latent.float() + injection).to(orig_dtype)
