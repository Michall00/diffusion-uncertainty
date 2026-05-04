# Status projektu — UQ w dyfuzji: porównanie metod

## Co zostało zaimplementowane

### Nowe moduły (`diffusion_uncertainty/uq_laplace/`)

| Plik | Co robi |
|------|---------|
| `core.py` | `FeatureCapture` (hook na conv_out), `ManualDiagLaplace` (diagonal GGN-Laplace na Conv2d via F.unfold), `generate_reference_latents` (DDIM z0 do fitu) |
| `subnet_laplace.py` | `SubnetLaplace` — diagonalny Laplace na losowej podsieci `up_blocks`; γ² jako `Var_k(eps_pred)` pod perturbacją wag (MC weight perturbation); alternatywa dla LLLA z semantycznie bardziej sensownymi mapami niepewności |
| `gamma2.py` | `compute_gamma2_llla` — per-pikselowa niepewność epistemiczna z Laplace'a; `ddim_transport_factors` — współczynniki FLARE do akumulacji przez kroki |
| `guidance.py` | `apply_gradient_guidance` — posterior update na pred_eps (z normalizacją γ² przez percentyl, scale-invariant); `apply_resampling_guidance` — lokalne dostrzykiwanie szumu w miejscach dużego γ² |
| `aggregation.py` | `uncertainty_stats`, `attention_weighted_scores` — mierzenie i ważenie map niepewności |
| `plotting.py` | `save_heatmap_png`, `save_comparison_grid`, `save_attention_overlay_png` |

### Pipeline (`diffusion_uncertainty/pipeline_uncertainty/`)

`pipeline_stable_diffusion_epistemic_guided.py` — `StableDiffusionPipelineUQComparison`:
- Jeden `__call__` z parametrem `guidance_mode: "none" | "aleatoric" | "gradient" | "resampling"`
- Nowe parametry: `laplace_mode: "last_layer" | "subnet"`, `n_mc_subnet`, `subnet_max_params`, `pre_fitted_laplace`
- `last_layer`: GGN-Laplace na conv_out (11.5K params), γ² = per-pikselowa wariancja aktywacji
- `subnet`: losowe 20K parametrów z up_blocks, γ² = Var(eps_pred) pod perturbacją wag (MC)
- Laplace fitowany raz (gradient), reused automatycznie (resampling) przez `pre_fitted_laplace`
- `optimize_prior()` NIE jest wywoływane — empiryczny Bayes daje `prior_prec≈860` → γ²≈0; używamy stałego `prior_prec=1e-3`
- FLARE-akumulacja `u_proj` przez wszystkie kroki przy metodach epistemicznych
- Zwraca: obraz, `uncertainty_map` (γ² z ostatniego kroku), `u_proj`, `fitted_laplace` (do reuse)

### Skrypty

| Skrypt | Co robi |
|--------|---------|
| `scripts/compare_uq_sd.py` | Uruchamia ≥1 metod, zapisuje PNG + heatmapy + grid + NPZ; flagi: `--laplace-mode`, `--n-mc-subnet`, `--subnet-max-params`, `--lr`, `--percentile` |
| `scripts/evaluate_uq_sd.py` | Wczytuje NPZ, liczy CLIPScore / mean γ² / P95 γ², drukuje tabelę Markdown, zapisuje CSV |
| `scripts/run_batch_experiments.py` | Pętla po promptach × seedach, wywołuje compare + evaluate, agreguje wyniki do `all_results.csv` |

### Konfiguracja

`config/stable_diffusion_uq_comparison.yaml` — 10 promptów, parametry per-metoda, num_steps per device.

---

## Jak uruchamiać

### Lokalnie na MPS (macOS) — szybki test

```bash
uv run --extra cpu python scripts/compare_uq_sd.py \
  --prompt "a golden retriever in a forest" \
  --seed 42 --num-steps 5 --device mps \
  --methods baseline aleatoric gradient resampling \
  --model-id CompVis/stable-diffusion-v1-4 \
  --guidance-start-step 1 --guidance-n-steps 3 \
  --num-mc 2 --n-ref 1 --n-pairs 5 \
  --lr 0.3 \
  --out-dir outputs/quick_test
```

### Lokalnie — jakościowy

```bash
uv run --extra cpu python scripts/compare_uq_sd.py \
  --prompt "a cat on a velvet chair" \
  --seed 42 --num-steps 20 --device mps \
  --methods baseline aleatoric gradient resampling \
  --model-id CompVis/stable-diffusion-v1-4 \
  --guidance-start-step 0 --guidance-n-steps 15 \
  --num-mc 5 --n-ref 3 --n-pairs 30 \
  --lr 0.3 \
  --out-dir outputs/local_qual
```

### Z SubnetLaplace (wolniejsze, lepsza mapa γ²)

```bash
uv run --extra cpu python scripts/compare_uq_sd.py \
  --prompt "a cat on a velvet chair" \
  --seed 42 --num-steps 20 --device mps \
  --methods baseline gradient resampling \
  --model-id CompVis/stable-diffusion-v1-4 \
  --guidance-start-step 0 --guidance-n-steps 15 \
  --n-ref 1 --n-pairs 10 \
  --laplace-mode subnet --n-mc-subnet 4 --subnet-max-params 20000 \
  --lr 0.3 \
  --out-dir outputs/local_subnet
```

### Na VM z CUDA (2×24GB) — pełny eksperyment

```bash
uv sync --extra cu118 --group dev

uv run --extra cu118 python scripts/compare_uq_sd.py \
  --prompt "a serene mountain lake at sunset" \
  --seed 42 --num-steps 30 --device cuda \
  --methods baseline aleatoric gradient resampling \
  --model-id CompVis/stable-diffusion-v1-4 \
  --guidance-start-step 0 --guidance-n-steps 20 \
  --num-mc 5 --n-ref 3 --n-pairs 50 \
  --lr 0.3 \
  --out-dir outputs/cuda_full
```

### Ewaluacja wyników (CLIPScore + statystyki)

```bash
uv run --extra cpu python scripts/evaluate_uq_sd.py \
  --result-dir outputs/cuda_full/a_serene_mountain_lake_seed42
# → drukuje tabelę Markdown + zapisuje evaluation_results.csv
```

### Batch po wielu promptach

```bash
uv run --extra cpu python scripts/run_batch_experiments.py \
  --prompts "a cat on a chair" "a foggy forest at dawn" \
  --seeds 42 123 \
  --methods baseline aleatoric gradient resampling \
  --laplace-mode last_layer \
  --out-dir outputs/batch
# → agreguje wszystkie CSV do outputs/batch/all_results.csv
```

---

## Znane ograniczenia obecnej implementacji

| Problem | Przyczyna | Obejście |
|---------|-----------|----------|
| `im2col` fallback na CPU przy MPS | PyTorch 2.3 nie ma im2col na MPS | działa poprawnie, tylko wolniej |
| `torch_dtype` deprecation warning | diffusers 0.31 zmienił API | ignoruj lub uaktualnij do diffusers 0.32+ |
| `last_layer` Laplace tylko na `conv_out` | 11.5K params = 0.001% UNeta | celowy LLLA; `subnet` pokrywa up_blocks |
| `subnet` fit bardzo wolny na MPS | backprop przez cały UNet 20× | zmniejsz `--n-pairs 5` lub użyj CUDA |
| Resampling reuse Laplace tylko gdy `gradient` był wcześniej | kolejność w `--methods` ma znaczenie | zawsze daj `gradient` przed `resampling` |
| Guidance subtelne przy 12 krokach DDIM | mocny CFG dominuje nad korektą | użyj `--num-steps 20 --guidance-start-step 0` |
| `optimize_prior()` nie jest używane | empiryczny Bayes daje `prior_prec≈860` → γ²≈0 | stałe `prior_prec=1e-3` zamiast optimizacji |
