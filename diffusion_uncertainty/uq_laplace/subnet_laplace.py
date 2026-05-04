"""
diffusion_uncertainty/uq_laplace/subnet_laplace.py
---------------------------------------------------
Diagonal Laplace approximation on a random subset of UNet weight parameters.

Unlike ManualDiagLaplace (which covers only conv_out via im2col), SubnetLaplace:
  1. Selects weight tensors from broader UNet blocks (default: up_blocks)
  2. Accumulates diagonal Fisher via stochastic gradient projection:
       v ~ N(0, I_output),  g = d(v^T eps_pred)/dθ  →  H_diag += g²
     This is an unbiased estimator of diag(J^T J) for MSE loss.
  3. Estimates gamma2 at inference via MC weight perturbations:
       δk ~ N(0, posterior_var),  run n_mc extra forward passes,
       gamma2 = Var_k(eps_pred_k)

Advantages over last-layer approach:
  - Uncertainty reflects broader model computation, not just the final linear map
  - No im2col needed — analytical formula replaced by clean MC variance
  - Naturally handles any layer type (conv, linear, attention projections)

Cost: n_mc extra UNet forward passes per guidance step.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


def _get_param(module: nn.Module, name: str) -> nn.Parameter:
    """Access a nested parameter by dotted name path."""
    obj: nn.Module | nn.Parameter = module
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


class SubnetLaplace:
    """Diagonal Laplace on a random subset of UNet weight parameters.

    Fitting accumulates diagonal Fisher via stochastic projection.
    Gamma2 estimation uses MC weight perturbation — no analytical im2col required.
    """

    def __init__(
        self,
        unet: nn.Module,
        prior_prec: float = 1e-3,
        max_params: int = 20_000,
        target_blocks: Sequence[str] = ("up_blocks",),
        seed: int = 0,
    ) -> None:
        self.prior_prec = prior_prec
        self._unet_ref = unet

        rng = torch.Generator()
        rng.manual_seed(seed)

        candidates = [
            (name, param)
            for name, param in unet.named_parameters()
            if param.requires_grad
            and any(blk in name for blk in target_blocks)
            and "weight" in name
        ]

        perm = torch.randperm(len(candidates), generator=rng).tolist()
        selected: list[tuple[str, Tensor]] = []
        total = 0

        for ci in perm:
            if total >= max_params:
                break
            name, param = candidates[ci]
            n_avail = min(param.numel(), max_params - total)
            idx = torch.randperm(param.numel(), generator=rng)[:n_avail].sort().values.cpu()
            selected.append((name, idx))
            total += n_avail

        self._selected = selected
        self.total_params = total
        self._H_flat = torch.zeros(total, dtype=torch.float32)
        self._n_data = 0

        print(
            f"[SubnetLaplace] {total} params from {len(selected)} tensors "
            f"in blocks: {list(target_blocks)}"
        )

    def _apply_delta(self, delta: Tensor, device: torch.device) -> None:
        offset = 0
        for name, idx in self._selected:
            param = _get_param(self._unet_ref, name)
            n = idx.numel()
            param.data.view(-1)[idx.to(device)] += delta[offset : offset + n]
            offset += n

    def _restore(self, saved: list[tuple[str, Tensor, Tensor]], device: torch.device) -> None:
        for name, idx, orig in saved:
            _get_param(self._unet_ref, name).data.view(-1)[idx.to(device)] = orig

    def fit(
        self,
        unet: nn.Module,
        z0_set: Tensor,
        abar: Tensor,
        cond_emb: Tensor,
        device: torch.device,
        T: int = 1000,
        n_pairs: int = 20,
        n_projections: int = 2,
    ) -> None:
        """Accumulate diagonal Fisher via stochastic gradient projection.

        For each calibration pair (z0, t):
          z_t = sqrt(ᾱ_t)·z0 + sqrt(1−ᾱ_t)·ε
          v ~ N(0, I_output)
          g = d(v^T eps_pred) / dθ_subnet   →   H_diag += g²

        Args:
            z0_set:        reference clean latents (N, C, H, W)
            abar:          alphas_cumprod from scheduler (T,)
            cond_emb:      conditional text embeddings (1, 77, 768)
            n_pairs:       (z_t, t) pairs per reference latent
            n_projections: random projections per pair (more = less variance)
        """
        params_list = [_get_param(unet, name) for name, _ in self._selected]
        H = torch.zeros(self.total_params, dtype=torch.float32, device=device)
        n_total = 0

        unet.eval()
        ref_dtype = cond_emb.dtype

        for z0 in z0_set:
            z0u = z0.unsqueeze(0).to(device, dtype=ref_dtype)
            for _ in range(n_pairs):
                t_idx = int(torch.randint(0, T, (1,)).item())
                t_tensor = torch.tensor([t_idx], device=device, dtype=torch.long)
                abt = abar[t_idx]

                noise = torch.randn_like(z0u)
                z_t = torch.sqrt(abt) * z0u + torch.sqrt(1.0 - abt) * noise

                for _ in range(n_projections):
                    for p in params_list:
                        if p.grad is not None:
                            p.grad = None

                    with torch.enable_grad():
                        pred = unet(z_t, t_tensor, encoder_hidden_states=cond_emb).sample
                        v = torch.randn_like(pred)
                        grads = torch.autograd.grad(
                            (pred * v).sum(),
                            params_list,
                            retain_graph=False,
                            create_graph=False,
                        )

                    offset = 0
                    for (_, idx), g in zip(self._selected, grads):
                        g_flat = g.detach().view(-1)[idx.to(device)]
                        H[offset : offset + idx.numel()] += g_flat ** 2
                        offset += idx.numel()
                    n_total += 1

        self._H_flat = (H / max(n_total, 1)).cpu()
        self._n_data = n_total
        print(
            f"[SubnetLaplace] Fit done. n_data={n_total}, "
            f"H_diag mean={self._H_flat.mean():.3e}, "
            f"post_var mean={self.posterior_variance.mean():.3e}"
        )

    @property
    def posterior_variance(self) -> Tensor:
        """(total_params,) posterior variance for all selected param indices."""
        return 1.0 / (self._H_flat + self.prior_prec)

    @torch.no_grad()
    def compute_gamma2(
        self,
        unet: nn.Module,
        z_t: Tensor,
        t: Tensor,
        cfg_emb: Tensor,
        guidance_scale: float,
        n_mc: int = 5,
        device: torch.device | None = None,
    ) -> Tensor:
        """Per-pixel epistemic variance via MC weight perturbations.

        Samples n_mc perturbations δk ~ N(0, posterior_var) for the selected
        subnet, runs n_mc CFG forward passes, returns Var_k(eps_pred_k).

        Args:
            z_t:            noisy latent (1, C, H, W)
            t:              current timestep tensor
            cfg_emb:        concatenated [uncond, cond] embeddings (2, 77, 768)
            guidance_scale: CFG guidance scale
            n_mc:           number of MC weight perturbation samples

        Returns:
            gamma2: (1, C, H, W) float32 per-pixel epistemic variance
        """
        if device is None:
            device = z_t.device

        post_std = torch.sqrt(self.posterior_variance.to(device))
        z_in = torch.cat([z_t, z_t])  # (2, C, H, W) for CFG

        saved: list[tuple[str, Tensor, Tensor]] = []
        for name, idx in self._selected:
            param = _get_param(unet, name)
            saved.append((name, idx, param.data.view(-1)[idx.to(device)].clone()))

        preds: list[Tensor] = []
        for _ in range(n_mc):
            delta = torch.randn(self.total_params, device=device) * post_std
            self._apply_delta(delta, device)

            pred = unet(z_in, t, encoder_hidden_states=cfg_emb).sample
            eps_u, eps_c = pred.chunk(2)
            preds.append((eps_u + guidance_scale * (eps_c - eps_u)).float())

            self._restore(saved, device)

        return torch.var(torch.stack(preds, dim=0), dim=0).clamp_min(0.0)
