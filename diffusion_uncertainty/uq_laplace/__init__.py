"""
diffusion_uncertainty/uq_laplace
---------------------------------
Epistemic uncertainty quantification for Stable Diffusion via diagonal Laplace
approximation on the UNet's last Conv2d layer (conv_out).

Supports three guidance modes for comparison:
    - "aleatoric"  : MC variance → posterior update on pred_eps
    - "gradient"   : LLLA Laplace γ² → posterior update on pred_eps
    - "resampling" : LLLA Laplace γ² → local noise injection on latents

Public API:
    ManualDiagLaplace
    SubnetLaplace
    FeatureCapture
    generate_reference_latents
    compute_gamma2_llla
    ddim_transport_factors
    apply_gradient_guidance
    apply_resampling_guidance
    uncertainty_to_gray
    save_heatmap_png
    save_attention_overlay_png
    save_comparison_grid
    uncertainty_stats
    attention_weighted_scores
"""

from diffusion_uncertainty.uq_laplace.aggregation import (
    attention_weighted_scores,
    uncertainty_stats,
)
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
from diffusion_uncertainty.uq_laplace.plotting import (
    save_attention_overlay_png,
    save_comparison_grid,
    save_heatmap_png,
    uncertainty_to_gray,
)

__all__ = [
    "FeatureCapture",
    "ManualDiagLaplace",
    "SubnetLaplace",
    "generate_reference_latents",
    "compute_gamma2_llla",
    "ddim_transport_factors",
    "apply_gradient_guidance",
    "apply_resampling_guidance",
    "uncertainty_to_gray",
    "save_heatmap_png",
    "save_attention_overlay_png",
    "save_comparison_grid",
    "uncertainty_stats",
    "attention_weighted_scores",
]
