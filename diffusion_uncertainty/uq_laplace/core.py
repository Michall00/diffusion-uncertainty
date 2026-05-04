"""
diffusion_uncertainty/uq_laplace/core.py
-----------------------------------------
Diagonal Laplace approximation utilities for SD v1.5 UNet.

Classes:
    FeatureCapture       - forward hook to capture conv_out input features
    ManualDiagLaplace    - diagonal GGN-Laplace for a single Conv2d layer

Functions:
    generate_reference_latents - DDIM generation of z0 latents for Laplace fitting
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class FeatureCapture:
    """Captures the input tensor to a Conv2d module via a forward hook.

    Usage:
        cap = FeatureCapture(unet.conv_out)
        unet(z_t, t, encoder_hidden_states=emb)  # hook fires
        feats = cap.features  # (B, C_in, H, W)
        cap.remove()          # deregister hook when done
    """

    def __init__(self, module: nn.Module) -> None:
        self.features: torch.Tensor | None = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        self.features = inp[0].detach()

    def remove(self) -> None:
        self._handle.remove()


class ManualDiagLaplace:
    """Diagonal GGN-Laplace for a single Conv2d layer.

    Approximates the Hessian as the diagonal of the Generalized Gauss-Newton
    matrix via the im2col trick (F.unfold).  For a Conv2d layer with input
    patches p and C_out output channels:

        H_diag[w_{c,j}] = (1/N) Σ_i Σ_l p²_{i,j,l}

    posterior_precision = H_diag + prior_prec
    posterior_variance  = 1 / posterior_precision

    Args:
        conv:       the Conv2d layer to fit Laplace on (typically unet.conv_out)
        prior_prec: isotropic prior precision (weight decay strength)
    """

    def __init__(self, conv: nn.Conv2d, prior_prec: float = 1.0) -> None:
        self.conv = conv
        self.prior_prec = prior_prec
        n_params = sum(p.numel() for p in conv.parameters())
        self.H_diag: torch.Tensor = torch.zeros(n_params, dtype=torch.float32)
        self.n_data: int = 0

    @torch.no_grad()
    def accumulate(self, features: torch.Tensor) -> None:
        """Accumulate diagonal GGN from one batch of captured features.

        Args:
            features: (B, C_in, H, W) input tensor to conv_out
        """
        conv = self.conv
        features = features.float()
        C_out, C_in, kH, kW = conv.weight.shape
        patches = F.unfold(features, (kH, kW), padding=conv.padding)
        B, CkK, L = patches.shape

        p2 = patches.pow(2).sum(dim=(0, 2))
        H_w = p2.unsqueeze(0).expand(C_out, -1).reshape(-1)

        if conv.bias is not None:
            H_b = torch.full((C_out,), float(B * L), device=features.device)
            H_diag = torch.cat([H_w, H_b])
        else:
            H_diag = H_w

        self.H_diag = self.H_diag.to(H_diag.device) + H_diag
        self.n_data += B * L

    def fit(
        self,
        unet: nn.Module,
        feat_cap: FeatureCapture,
        z0_set: torch.Tensor,
        abar: torch.Tensor,
        cond_emb: torch.Tensor,
        device: str,
        T: int = 1000,
        n_pairs: int = 50,
    ) -> None:
        """Fit the diagonal Laplace by running forward diffusion + UNet passes.

        Args:
            unet:      the UNet model
            feat_cap:  FeatureCapture hook registered on unet.conv_out
            z0_set:    reference clean latents (N, C, H, W) for adding noise
            abar:      cumulative alpha schedule (T,)
            cond_emb:  text conditioning embedding (1, 77, 768)
            device:    device string
            T:         diffusion timesteps (1000 for SD)
            n_pairs:   number of (z_t, eps) regression pairs to accumulate
        """
        n_z0 = z0_set.shape[0]
        latent_dtype = next(unet.parameters()).dtype
        unet.eval()

        for _ in tqdm(range(n_pairs), desc="Laplace fit", leave=False):
            idx = int(torch.randint(0, n_z0, (1,)).item())
            z0 = z0_set[idx : idx + 1].to(device, dtype=latent_dtype)
            t = torch.randint(0, T, (1,), device=device)
            abar_t = abar[t.long()].view(1, 1, 1, 1).float()
            eps = torch.randn_like(z0, dtype=torch.float32)
            z_t = (
                torch.sqrt(abar_t) * z0.float() + torch.sqrt(1.0 - abar_t) * eps
            ).to(dtype=latent_dtype)

            with torch.no_grad():
                unet(z_t, t, encoder_hidden_states=cond_emb)

            if feat_cap.features is not None:
                self.accumulate(feat_cap.features)

        if self.n_data > 0:
            self.H_diag = self.H_diag / float(self.n_data)

    def optimize_prior(self) -> None:
        """Empirical Bayes prior: prior_prec = P / Σ θ²."""
        theta = torch.cat([p.detach().reshape(-1) for p in self.conv.parameters()])
        theta_sq_sum = float(theta.float().pow(2).sum().item())
        self.prior_prec = float(theta.numel()) / (theta_sq_sum + 1e-8)

    @property
    def posterior_variance(self) -> torch.Tensor:
        """Returns (n_params,) diagonal posterior variance vector."""
        prec = self.H_diag + self.prior_prec
        return (1.0 / prec.clamp_min(1e-12)).to(dtype=torch.float32)


@torch.no_grad()
def generate_reference_latents(
    unet: nn.Module,
    scheduler,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    latent_shape: tuple[int, int, int],
    n_ref: int,
    n_steps: int,
    guidance_scale: float,
    device: str,
    base_seed: int,
) -> torch.Tensor:
    """Generate n_ref reference z0 latents with DDIM for Laplace fitting.

    Args:
        unet:          UNet model
        scheduler:     DDIMScheduler (will call set_timesteps internally)
        cond_emb:      conditional text embedding (1, 77, 768)
        uncond_emb:    unconditional text embedding (1, 77, 768)
        latent_shape:  (C, H, W) e.g. (4, 64, 64) for 512×512
        n_ref:         number of reference latents to generate
        n_steps:       DDIM denoising steps
        guidance_scale: CFG scale
        device:        device string
        base_seed:     base random seed (each sample uses base_seed + i)

    Returns:
        z0_set: (n_ref, C, H, W) tensor of clean latents
    """
    latent_dtype = next(unet.parameters()).dtype
    cfg_emb = torch.cat([uncond_emb, cond_emb], dim=0)
    z0_list: list[torch.Tensor] = []

    for i in range(n_ref):
        scheduler.set_timesteps(n_steps)
        gen = torch.Generator(device=device).manual_seed(base_seed + i)
        z = torch.randn(1, *latent_shape, device=device, dtype=latent_dtype, generator=gen)

        for t in tqdm(scheduler.timesteps, desc=f"ref z0 [{i+1}/{n_ref}]", leave=False):
            z_in = torch.cat([z, z])
            pred = unet(z_in, t, encoder_hidden_states=cfg_emb).sample
            eps_u, eps_c = pred.chunk(2)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            z = scheduler.step(eps, t, z, generator=gen).prev_sample.to(dtype=latent_dtype)

        z0_list.append(z)

    return torch.cat(z0_list, dim=0)
