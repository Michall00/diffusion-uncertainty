UV ?= uv
EXTRA ?= cu118
GROUP ?= --group dev
UV_RUN = $(UV) run --extra $(EXTRA) $(GROUP)

DATASET ?= imagenet128
IMAGE_SIZE ?= 128
MODEL_TYPE ?= unet
SCHEDULER ?= uncertainty_centered
N ?= 4
BATCH_SIZE ?= 1
M ?= 1
DROPOUT ?= 0.0
GENERATION_STEPS ?= 2
START_STEP_UC ?= 0
NUM_STEPS_UC ?= 1
START_INDEX ?= 0
EXTRA_SAMPLES ?= 0
SEED ?= 0
MULTI_GPU ?=
RUN ?= $(shell ls -td results/score-uncertainty/* 2>/dev/null | head -1)
GUIDANCE_RUN ?= $(shell ls -td results/uncertainty_guidance/imagenet*/* 2>/dev/null | head -1)
GUIDANCE_TYPE ?= gradient
GUIDANCE_PERCENTILE ?= 0.95
GUIDANCE_STEPS ?= 20
GUIDANCE_START_STEP ?= 0
GUIDANCE_NUM_STEPS ?= 5
GUIDANCE_LAMBDA ?= 0.1
GUIDANCE_GRADIENT_WRT ?= input
GUIDANCE_GRADIENT_DIRECTION ?= descend
GUIDANCE_THRESHOLD_TYPE ?= higher
SKIP_FID ?= --skip-fid
SKIP_DDIM ?=

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: sync-gpu
sync-gpu: ## Install the CUDA 11.8 uv environment
	$(UV) sync --extra cu118 $(GROUP)

.PHONY: sync-cpu
sync-cpu: ## Install the CPU uv environment
	$(UV) sync --extra cpu $(GROUP)

.PHONY: lock-check
lock-check: ## Check that uv.lock is up to date
	$(UV) lock --check

.PHONY: verify
verify: ## Print the selected PyTorch build for EXTRA=$(EXTRA)
	$(UV_RUN) python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

.PHONY: verify-cuda
verify-cuda: ## Print the CUDA PyTorch build
	$(UV) run --extra cu118 $(GROUP) python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

.PHONY: download-adm128
download-adm128: ## Download the ADM ImageNet 128 checkpoint
	mkdir -p models
	wget -nc -P models https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt

.PHONY: download-adm64
download-adm64: ## Download the ADM ImageNet 64 checkpoint
	mkdir -p models
	wget -nc -P models https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt

.PHONY: download-classifiers
download-classifiers: ## Download ADM ImageNet classifier checkpoints
	mkdir -p models
	wget -nc -P models https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
	wget -nc -P models https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_classifier.pt

.PHONY: download-adm
download-adm: download-adm64 download-adm128 download-classifiers ## Download ADM checkpoints and classifiers

.PHONY: download-uvit
download-uvit: ## Download U-ViT checkpoints and autoencoder
	mkdir -p models
	$(UV_RUN) gdown 1igVgRY7-A0ZV3XqdNcMGOnIGOxKr9azv -O models/imagenet64_uvit_mid.pth
	$(UV_RUN) gdown 13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u -O models/imagenet256_uvit_huge.pth
	$(UV_RUN) gdown 1uegr2o7cuKXtf2akWGAN2Vnlrtw5YKQq -O models/imagenet512_uvit_huge.pth
	$(UV_RUN) gdown 10nbEiFd4YCHlzfTkJjZf45YcSMCN34m6 -O models/autoencoder_kl_ema.pth

.PHONY: starting-data
starting-data: ## Generate diffusion starting points, e.g. make starting-data N=128 DATASET=imagenet128
	$(UV_RUN) python scripts/generate_diffusion_starting_data.py \
		--datasets $(DATASET) \
		--num-samples $(N) \
		--extra-samples $(EXTRA_SAMPLES)

.PHONY: run-imagenet
run-imagenet: ## Run ImageNet uncertainty generation with configurable variables
	$(UV_RUN) python scripts/generate_dataset_score_uncertainty_imagenet.py \
		--num-samples $(N) \
		--batch-size $(BATCH_SIZE) \
		-M $(M) \
		--dropout $(DROPOUT) \
		--scheduler $(SCHEDULER) \
		--image-size $(IMAGE_SIZE) \
		--model-type $(MODEL_TYPE) \
		--generation-steps $(GENERATION_STEPS) \
		--start-step-uc $(START_STEP_UC) \
		--num-steps-uc $(NUM_STEPS_UC) \
		--start-index $(START_INDEX) \
		--seed $(SEED) $(MULTI_GPU)

.PHONY: run-imagenet-guided
run-imagenet-guided: ## Run ImageNet generation guided by online uncertainty
	$(UV_RUN) python scripts/generate_images_with_uncertainty_threshold.py \
		--dataset $(DATASET) \
		--num-samples $(N) \
		--batch-size $(BATCH_SIZE) \
		--num-steps $(GUIDANCE_STEPS) \
		--start-step-guidance $(GUIDANCE_START_STEP) \
		--num-steps-guidance $(GUIDANCE_NUM_STEPS) \
		--start-index $(START_INDEX) \
		--seed $(SEED) \
		--guidance-type $(GUIDANCE_TYPE) \
		--percentile $(GUIDANCE_PERCENTILE) \
		--lambda-update $(GUIDANCE_LAMBDA) \
		--gradient-wrt $(GUIDANCE_GRADIENT_WRT) \
		--gradient-direction $(GUIDANCE_GRADIENT_DIRECTION) \
		--threshold-type $(GUIDANCE_THRESHOLD_TYPE) \
		--use-percentile \
		$(SKIP_FID) $(SKIP_DDIM)

.PHONY: smoke
smoke: ## End-to-end ADM128 smoke test and grid generation
	$(MAKE) download-adm128
	$(MAKE) starting-data DATASET=imagenet128 N=4 EXTRA_SAMPLES=0
	$(MAKE) run-imagenet N=4 BATCH_SIZE=1 M=1 DROPOUT=0.0 SCHEDULER=uncertainty_centered IMAGE_SIZE=128 MODEL_TYPE=unet GENERATION_STEPS=2 START_STEP_UC=0 NUM_STEPS_UC=1 START_INDEX=0
	$(MAKE) grid

.PHONY: smoke-guided
smoke-guided: ## End-to-end ADM128 smoke test with uncertainty-guided generation
	$(MAKE) download-adm128
	$(MAKE) starting-data DATASET=imagenet128 N=4 EXTRA_SAMPLES=0
	$(MAKE) run-imagenet-guided DATASET=imagenet128 N=4 BATCH_SIZE=1 GUIDANCE_STEPS=8 GUIDANCE_START_STEP=0 GUIDANCE_NUM_STEPS=1 START_INDEX=0 SKIP_FID=--skip-fid
	$(MAKE) grid-guided

.PHONY: grid
grid: ## Save grid.png for RUN=<results/score-uncertainty/...>, defaults to latest run
	@test -n "$(RUN)" || (echo "No score-uncertainty run found"; exit 1)
	$(UV_RUN) python -c 'import torch; from pathlib import Path; from torchvision.utils import save_image; run=Path("$(RUN)"); imgs=torch.load(next(run.glob("gen_images_*.pth")), map_location="cpu").float() / 255; save_image(imgs, run / "grid.png", nrow=min(5, imgs.shape[0])); print(run / "grid.png")'

.PHONY: grid-guided
grid-guided: ## Save guided and baseline grids for GUIDANCE_RUN=<results/uncertainty_guidance/...>, defaults to latest run
	@test -n "$(GUIDANCE_RUN)" || (echo "No uncertainty_guidance run found"; exit 1)
	$(UV_RUN) python -c 'import torch; from pathlib import Path; from torchvision.utils import save_image; run=Path("$(GUIDANCE_RUN)"); guided=torch.load(run / "gen_images_threshold.pth", map_location="cpu").float() / 255; save_image(guided, run / "grid_guided.png", nrow=min(5, guided.shape[0])); print(run / "grid_guided.png"); baseline_path=run / "gen_images.pth"; baseline=torch.load(baseline_path, map_location="cpu").float() / 255 if baseline_path.exists() else None; baseline is None or (save_image(baseline, run / "grid_baseline.png", nrow=min(5, baseline.shape[0])), print(run / "grid_baseline.png"))'

.PHONY: shapes
shapes: ## Print tensor shapes for RUN=<results/score-uncertainty/...>, defaults to latest run
	@test -n "$(RUN)" || (echo "No score-uncertainty run found"; exit 1)
	$(UV_RUN) python -c 'import torch; from pathlib import Path; run=Path("$(RUN)"); [print(p.name, tuple((x := torch.load(p, map_location="cpu")).shape), x.dtype) for p in sorted(run.glob("*.pth"))]'

.PHONY: sweep-small
sweep-small: ## Run a small three-scheduler ADM128 comparison
	$(MAKE) starting-data DATASET=imagenet128 N=128 EXTRA_SAMPLES=0
	for scheduler in uncertainty_centered uncertainty_zigzag_centered dpm_2_uncertainty_centered; do \
		$(MAKE) run-imagenet N=128 BATCH_SIZE=8 M=5 DROPOUT=0.5 SCHEDULER=$$scheduler IMAGE_SIZE=128 MODEL_TYPE=unet GENERATION_STEPS=20 START_STEP_UC=15 NUM_STEPS_UC=5 START_INDEX=0; \
	done
