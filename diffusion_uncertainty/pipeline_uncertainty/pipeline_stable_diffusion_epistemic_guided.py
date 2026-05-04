"""
diffusion_uncertainty/pipeline_uncertainty/pipeline_stable_diffusion_epistemic_guided.py
------------------------------------------------------------------------------------------
Stable Diffusion v1.5 pipeline supporting 4 UQ guidance modes for comparison:

    "none"        — standard DDIM, no guidance (baseline)
    "aleatoric"   — MC variance → Bayesian posterior update on pred_eps
    "gradient"    — LLLA Laplace γ² → Bayesian posterior update on pred_eps
    "resampling"  — LLLA Laplace γ² → local noise injection on latents after DDIM step

Usage:
    pipe = StableDiffusionPipelineUQComparison.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("mps")  # or "cuda"

    result = pipe(
        prompt="a photo of a cat",
        seed=42,
        guidance_mode="gradient",
        num_inference_steps=20,
        guidance_n_steps=10,
    )
    result.image.save("out.png")
    result.uncertainty_map  # (1, 4, 64, 64) γ² from last step
    result.u_proj            # (1, 4, 64, 64) FLARE-accumulated projection

    # Reuse fitted Laplace for a second run (same prompt, different mode)
    result2 = pipe(
        prompt="a photo of a cat",
        seed=42,
        guidance_mode="resampling",
        pre_fitted_laplace=result.fitted_laplace,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

from diffusion_uncertainty.uq_laplace.core import (
    FeatureCapture,
    ManualDiagLaplace,
    generate_reference_latents,
)
from diffusion_uncertainty.uq_laplace.subnet_laplace import SubnetLaplace
from diffusion_uncertainty.uq_laplace.gamma2 import (
    compute_gamma2_llla,
    ddim_transport_factors,
)
from diffusion_uncertainty.uq_laplace.guidance import (
    apply_gradient_guidance,
    apply_resampling_guidance,
)


@dataclass
class UQComparisonOutput:
    image: torch.Tensor          # (1, 3, H, W) float32 [0, 1]
    uncertainty_map: torch.Tensor | None  # (1, C, H, W) γ² at last guidance step
    u_proj: torch.Tensor | None  # (1, C, H, W) FLARE-accumulated epistemic projection
    fitted_laplace: ManualDiagLaplace | SubnetLaplace | None  # reusable for subsequent runs
    method: str
    laplace_mode: str = "last_layer"


GuidanceMode = Literal["none", "aleatoric", "gradient", "resampling"]


class StableDiffusionPipelineUQComparison(StableDiffusionPipeline):
    """SD v1.5 pipeline with unified UQ guidance modes.

    Inherits all model components (UNet, VAE, tokenizer, text_encoder,
    scheduler) from StableDiffusionPipeline via from_pretrained.

    The __call__ signature extends the base pipeline with UQ parameters.
    The scheduler is always replaced with DDIMScheduler for determinism.
    """

    def _resolve_device(self) -> torch.device:
        return torch.device(self._execution_device)

    def _encode_text(
        self, prompt: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cond_emb, uncond_emb), each (1, 77, 768)."""
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_inputs = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            cond_emb = self.text_encoder(text_inputs.input_ids.to(device))[0]
            uncond_emb = self.text_encoder(uncond_inputs.input_ids.to(device))[0]
        return cond_emb, uncond_emb

    @torch.no_grad()
    def _compute_mc_variance(
        self,
        pred_eps: torch.Tensor,
        latent: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        uncond_emb: torch.Tensor,
        alpha_hat_t: torch.Tensor,
        guidance_scale: float,
        num_mc: int,
    ) -> torch.Tensor:
        """MC variance of pred_eps by re-noising x0 and re-predicting eps.

        Returns mc_var: (1, C, H, W) float32 aleatoric variance map.
        """
        latent_dtype = latent.dtype
        cfg_emb = torch.cat([uncond_emb, cond_emb], dim=0)
        aht = alpha_hat_t.float().view(1, 1, 1, 1)

        pred_x0 = (latent.float() - torch.sqrt(1.0 - aht) * pred_eps.float()) / torch.sqrt(aht).clamp_min(1e-8)
        mc_list: list[torch.Tensor] = []

        for _ in range(num_mc):
            noise = torch.randn_like(pred_x0)
            x_hat = (torch.sqrt(aht) * pred_x0 + torch.sqrt(1.0 - aht) * noise).to(dtype=latent_dtype)
            x_hat_in = torch.cat([x_hat, x_hat])
            pred = self.unet(x_hat_in, t, encoder_hidden_states=cfg_emb).sample
            eps_u, eps_c = pred.chunk(2)
            mc_list.append((eps_u + guidance_scale * (eps_c - eps_u)).float())

        mc_var = torch.var(torch.stack(mc_list, dim=0), dim=0).clamp_min(0.0)
        return mc_var

    def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        height: int = 512,
        width: int = 512,
        guidance_mode: GuidanceMode = "none",
        guidance_start_step: int = 0,
        guidance_n_steps: int = 20,
        percentile: float = 0.95,
        lr: float = 1.0,
        n_ref_latents: int = 3,
        n_laplace_pairs: int = 50,
        num_mc_samples: int = 5,
        laplace_mode: Literal["last_layer", "subnet"] = "last_layer",
        n_mc_subnet: int = 5,
        subnet_max_params: int = 20_000,
        pre_fitted_laplace: ManualDiagLaplace | SubnetLaplace | None = None,
    ) -> UQComparisonOutput:
        """Run SD v1.5 with the specified UQ guidance mode.

        Args:
            prompt:              text prompt
            num_inference_steps: DDIM denoising steps
            guidance_scale:      CFG scale (7.5 default)
            seed:                random seed for reproducibility
            height, width:       output image size in pixels
            guidance_mode:       "none" | "aleatoric" | "gradient" | "resampling"
            guidance_start_step: first step index to apply guidance (0-indexed)
            guidance_n_steps:    number of consecutive steps with guidance active
            percentile:          fraction of highest-uncertainty pixels to correct
            lr:                  guidance correction strength [0, 1]
            n_ref_latents:       reference z0 latents for Laplace fitting
            n_laplace_pairs:     (z_t, eps) pairs for Laplace Hessian accumulation
            num_mc_samples:      MC forward passes for aleatoric mode
            pre_fitted_laplace:  reuse a previously fitted ManualDiagLaplace object
                                 (skips fitting, valid only for gradient/resampling)

        Returns:
            UQComparisonOutput with image, uncertainty_map, u_proj, fitted_laplace, method
        """
        device = self._resolve_device()
        latent_dtype = next(self.unet.parameters()).dtype

        # ── 1. Text encoding ────────────────────────────────────────────────
        cond_emb, uncond_emb = self._encode_text(prompt, device)
        cfg_emb = torch.cat([uncond_emb, cond_emb], dim=0)

        # ── 2. Initial latent ───────────────────────────────────────────────
        latent_shape = (self.unet.config.in_channels, height // 8, width // 8)
        gen = torch.Generator(device=device).manual_seed(seed)
        latents = randn_tensor(
            (1, *latent_shape), generator=gen, device=device, dtype=latent_dtype
        )

        # ── 3. Scheduler setup ──────────────────────────────────────────────
        if not isinstance(self.scheduler, DDIMScheduler):
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = list(self.scheduler.timesteps)
        latents = latents * self.scheduler.init_noise_sigma
        abar = self.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

        # ── 4. Laplace fitting (gradient / resampling modes only) ───────────
        feat_cap: FeatureCapture | None = None
        laplace: ManualDiagLaplace | SubnetLaplace | None = None
        fitted_laplace: ManualDiagLaplace | SubnetLaplace | None = None

        if guidance_mode in {"gradient", "resampling"}:
            # feat_cap only needed for last_layer mode (hooks into conv_out)
            if laplace_mode == "last_layer":
                feat_cap = FeatureCapture(self.unet.conv_out)

            if pre_fitted_laplace is not None:
                laplace = pre_fitted_laplace
                print(f"[UQ] Reusing pre-fitted Laplace ({type(laplace).__name__}, "
                      f"prior_prec={laplace.prior_prec:.3e})")
            else:
                print(f"[UQ] Generating {n_ref_latents} reference latents...")
                z0_ref = generate_reference_latents(
                    unet=self.unet,
                    scheduler=self.scheduler,
                    cond_emb=cond_emb,
                    uncond_emb=uncond_emb,
                    latent_shape=latent_shape,
                    n_ref=n_ref_latents,
                    n_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    device=device,
                    base_seed=seed + 1000,
                )
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = list(self.scheduler.timesteps)

                print(f"[UQ] Fitting Laplace ({laplace_mode}, {n_laplace_pairs} pairs)...")
                if laplace_mode == "last_layer":
                    laplace = ManualDiagLaplace(self.unet.conv_out, prior_prec=1e-3)
                    laplace.fit(
                        unet=self.unet,
                        feat_cap=feat_cap,
                        z0_set=z0_ref,
                        abar=abar,
                        cond_emb=cond_emb,
                        device=device,
                        T=1000,
                        n_pairs=n_laplace_pairs,
                    )
                else:  # subnet
                    laplace = SubnetLaplace(
                        self.unet, prior_prec=1e-3, max_params=subnet_max_params
                    )
                    laplace.fit(
                        unet=self.unet,
                        z0_set=z0_ref,
                        abar=abar,
                        cond_emb=cond_emb,
                        device=device,
                        T=1000,
                        n_pairs=n_laplace_pairs,
                    )
                print(f"[UQ] Laplace done. prior_prec={laplace.prior_prec:.3e}")
                fitted_laplace = laplace

        # ── 5. DDIM loop ────────────────────────────────────────────────────
        guidance_end_step = guidance_start_step + guidance_n_steps
        u_proj = torch.zeros(1, *latent_shape, device=device, dtype=torch.float32)
        cum_a2 = torch.ones(1, 1, 1, 1, device=device, dtype=torch.float32)
        last_gamma2: torch.Tensor | None = None
        laplace_gen = torch.Generator(device=device).manual_seed(seed + 99999)

        for i, t in enumerate(timesteps):
            latent_input = torch.cat([latents, latents])
            latent_input = self.scheduler.scale_model_input(latent_input, t)

            with torch.no_grad():
                pred = self.unet(latent_input, t, encoder_hidden_states=cfg_emb).sample
            eps_u, eps_c = pred.chunk(2)
            noise_pred = eps_u + guidance_scale * (eps_c - eps_u)

            in_guidance_window = guidance_start_step <= i < guidance_end_step
            alpha_hat_t = abar[t.long() if t.dim() > 0 else t]

            # ── Compute uncertainty signal ──────────────────────────────────
            if in_guidance_window and guidance_mode != "none":
                if guidance_mode == "aleatoric":
                    gamma2 = self._compute_mc_variance(
                        noise_pred, latents, t, cond_emb, uncond_emb,
                        alpha_hat_t, guidance_scale, num_mc_samples,
                    )
                else:
                    assert laplace is not None
                    if laplace_mode == "last_layer":
                        assert feat_cap is not None
                        cond_feats = feat_cap.features[1:2] if feat_cap.features is not None and feat_cap.features.shape[0] > 1 else feat_cap.features
                        if cond_feats is not None:
                            gamma2 = compute_gamma2_llla(cond_feats, laplace.posterior_variance, self.unet.conv_out)
                        else:
                            gamma2 = torch.zeros_like(noise_pred, dtype=torch.float32)
                    else:  # subnet: MC weight perturbation
                        assert isinstance(laplace, SubnetLaplace)
                        gamma2 = laplace.compute_gamma2(
                            self.unet, latents, t, cfg_emb,
                            guidance_scale, n_mc=n_mc_subnet, device=device,
                        )

                last_gamma2 = gamma2

                # ── Apply gradient guidance to pred_eps ─────────────────────
                if guidance_mode in {"aleatoric", "gradient"}:
                    noise_pred = apply_gradient_guidance(
                        noise_pred, gamma2, alpha_hat_t, percentile=percentile, lr=lr
                    )

                # ── FLARE accumulation ───────────────────────────────────────
                if guidance_mode in {"gradient", "resampling"}:
                    a_t, b_t = ddim_transport_factors(abar, timesteps, i)
                    u_proj = u_proj + cum_a2 * (b_t ** 2) * gamma2.float()
                    cum_a2 = cum_a2 * (a_t ** 2)

            # ── Scheduler step ───────────────────────────────────────────────
            with torch.no_grad():
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # ── Resampling guidance applied after DDIM step ─────────────────
            if in_guidance_window and guidance_mode == "resampling" and last_gamma2 is not None:
                latents = apply_resampling_guidance(
                    latents, last_gamma2, percentile=percentile, lr=lr, generator=laplace_gen
                )

        # ── 6. Decode latent → image ────────────────────────────────────────
        if feat_cap is not None:
            feat_cap.remove()

        with torch.no_grad():
            image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        image = (image / 2.0 + 0.5).clamp(0.0, 1.0).float()

        u_proj_out = u_proj if guidance_mode in {"gradient", "resampling"} else None

        return UQComparisonOutput(
            image=image.cpu(),
            uncertainty_map=last_gamma2.cpu() if last_gamma2 is not None else None,
            u_proj=u_proj_out.cpu() if u_proj_out is not None else None,
            fitted_laplace=fitted_laplace,
            method=guidance_mode,
            laplace_mode=laplace_mode,
        )
