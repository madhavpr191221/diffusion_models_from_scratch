# EuroSAT RGB DDPM — Detailed Implementation Plan

## Purpose

Build an unconditional image DDPM from first principles to understand its mathematical and computational mechanics. The project is educational, not a sample-quality benchmark or a generic deep-learning framework.

## Delivery method

Follow the project constitution and specification. For each stage: review the relevant requirements and acceptance criteria, make only that stage's changes, run its focused verification, record actual evidence, and pause for review. Keep changes small and preserve the user-facing distinction between a training diagnostic (noise-prediction loss) and generated-image quality.

## Stage 0 — Repository and development foundation

The workspace and Git repository already exist; use them and do not create a nested repository. Establish the project constitution, specification, design, task list, package metadata, dependency lock, and minimal React/TypeScript frontend scaffold. Keep the existing curriculum document. Put raw data and experiment artifacts under ignored `data/` and `runs/` directories. Do not add diffusion implementation code in this stage.

The current environment is Windows, Python 3.14.6, Node 24.18.1, npm 11.16.0, and uv; the NVIDIA RTX 5060 Laptop GPU has 8 GiB VRAM and driver 591.91 (reported CUDA 13.1). A dry dependency resolution succeeded for Python 3.14 with `torch==2.13.0` and `torchvision==0.28.0` from the official CUDA 13.0 wheel index. Use that pair in the project environment and record the selected index. Do not install into global Python. Stage 0 acceptance: package and frontend manifests resolve, generated dependency artifacts are reproducible, and the environment has a documented project-local setup.

## Stage 1 — EuroSAT RGB data

Use Hugging Face dataset `blanchon/EuroSAT_RGB`, pinned to commit `655ef66ef2be07f89aec61407f24c772802eb87d`, with the `datasets` library and cache under `data/huggingface/`. The source publishes train, validation, and test splits (16,200 / 5,400 / 5,400); train only on `train`, use `validation` for diagnostics, and leave `test` unused in v1. Keep labels and filenames out of the model-facing adapter and omit augmentation. Inspect the actual images. Convert RGB to float32 channel-first, scale raw `[0,255]` values to `[-1,1]`, and preserve 64×64 resolution if observed. The card reports an unknown license, so keep the downloaded copy local and do not redistribute it unless rights are confirmed.

Add a focused inspection entry point that downloads/caches the dataset, records observed counts, columns, image modes/shapes/dtypes and raw/normalized ranges, verifies a tensor-only batch shape, and writes a compact JSON report plus an original-versus-de-normalized image grid under ignored `outputs/`. Do not export or duplicate all decoded images; the pinned Hugging Face cache is the local data copy. Stage acceptance: actual split counts and image properties are recorded, batch values are finite in `[-1,1]`, and the saved grid visibly round-trips.

Record image count, sample shape, channels, dtype, raw/normalized ranges, and a batch shape. Save a compact grid showing normalized examples after display de-normalization. Acceptance: observed data matches the data contract or any necessary change is specified before proceeding.

## Stage 2 — Noise schedule and forward process

Implement a linear schedule with `T=1000`, `beta_start=1e-4`, and `beta_end=0.02`. Compute `alpha_t=1-beta_t`, cumulative `alpha_bar_t`, and the square-root/inverse quantities required by the equations. Keep schedule arrays one-dimensional and timestep-indexed; extract per-image coefficients and reshape explicitly for `[B,C,H,W]` broadcasting. Keep these values as non-learned buffers that move with the schedule.

Implement `sample_timesteps` to draw an independent zero-based timestep for each batch member, and direct noising `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps`. `q_sample` optionally accepts supplied `eps` and always returns both `x_t` and the exact noise. Do not simulate all previous forward transitions. Verify bounds, decreasing cumulative alpha, known-noise equation behavior, and output shapes. Save a reproducible same-validation-image multi-timestep visualization using fixed shared noise and a compact JSON verification report.

## Stage 2.5 — Interactive forward-noise viewer and learning log

Before building the model, make the forward process explorable in the React/TypeScript UI. Let the user page through unlabeled validation images, select one, enter a random seed, choose direct marginal or stepwise Markov mode, and scrub or play the 1,000 forward steps. Show the clean image beside the current noisy frame, one-based mathematical timestep, zero-based index where useful for code correspondence, and `alpha_bar_t`.

The direct mode calls the existing `q_sample` with one seed-derived noise tensor reused across timesteps; label it as direct marginal sampling, not a realized Markov trajectory. Add `q_sample_markov` in `diffusion.py` to start from the chosen image and apply `x_t = sqrt(alpha_t) * x_(t-1) + sqrt(beta_t) * z_t` sequentially with fresh Gaussian noise per step from a seeded generator. Return the trajectory for one image. The API caches only the current trace in memory (about 47 MiB for 1,000 float32 RGB 64×64 frames); a new trace replaces it. The UI requests a PNG frame on selection/timestep change rather than sending all frames at once.

Add FastAPI routes for paginated validation-image indices/previews, trace creation, and frame retrieval. Trace creation accepts image index, `direct` or `markov` mode, and seed; frame retrieval accepts the trace identifier and zero-based timestep, and returns a PNG with timestep and cumulative-alpha metadata. Validate image/timestep/mode inputs and return clear not-found responses when the single cached trace has been replaced. Keep labels out of all API responses. Serve the production Vite build from FastAPI and proxy `/api` to the local service during development. Keep training and data download as CLI workflows.

Create `docs/learning-log.md` for the learner. Cover foundation, EuroSAT acquisition/inspection, normalization, Stage 2 schedule/equations and evidence, the two meanings of a noise trace, runnable commands and artifact paths, and the explicit status that no model is built or trained yet. Stage acceptance: the UI can select a validation image, create reproducible traces in both modes, scrub/play frames with correct metadata, production assets build and load locally, and the learning log reflects completed evidence.

## Stage 3 — Timestep embedding and model

Implement cosine/sine alternating sinusoidal timestep features, then a small learned MLP (embedding dimension 128). Treat sinusoidal features as a practical time representation, not a required theorem. Each conditioned residual block gets its own linear projection to its channel dimension and spatially broadcasts the resulting vector.

Use a compact U-Net: input convolution, widths 32/64/128, three downsampling transitions to an 8×8 bottleneck, residual blocks with GroupNorm and SiLU, matching-resolution encoder features concatenated into the decoder, upsampling, and a final 3-channel convolution. Keep attention out. Explain representative encoder/decoder shapes and confirm the external mapping `[B,3,64,64] -> [B,3,64,64]`. Acceptance: finite output, shape agreement, and readable shape trace.

## Stage 4 — Training and measured baseline

For each training batch, load image tensors only from the published `train` split, sample `x0`, one uniform zero-based timestep per image, and Gaussian `eps`; use `q_sample` to construct `x_t`; predict noise; minimize `mean((eps - eps_theta(x_t,t))^2)`. The training split has 16,200 images. Evaluate all 5,400 validation images after each epoch, using the same seeded timesteps/noise on every pass. The test split is neither loaded nor evaluated in Stage 4. Use AdamW at `2e-4`, seed 42, batch size 8, and five epochs unless the pilot shows that the selected batch size does not fit or the runtime is impractical. Do not use labels, augmentation, mixed precision, or distributed training.

Run the fixed pilot first: select 64 training images deterministically and run 200 updates, cycling over that subset. Measure update time and peak GPU memory, and inspect the per-update loss for finite values and a downward trend. Then run the configured baseline over all training images. Commands:

```powershell
uv run --locked python -m ddpm_eurosat.train --pilot
uv run --locked python -m ddpm_eurosat.train
```

Each `runs/<run-id>/` stores the resolved configuration and device/environment, `status.json`, append-only per-update and per-epoch `metrics.jsonl`, a checkpoint manifest, and `.pt` files. Save progress snapshots at 25%, 50%, 75%, and 100% of planned optimizer updates, plus `best-val.pt` and replaceable resumable `latest.pt`. Checkpoints include model/optimizer state, update and batch position, config, and random-number-generator state; writes use atomic replacement. Resume with `uv run --locked python -m ddpm_eurosat.train --resume runs/<run-id>/checkpoints/latest.pt`.

The CLI owns training. The existing local React application adds a Training Monitor that reads run list/status, incremental metrics, and checkpoint metadata through read-only API routes. It polls while a run is active and plots raw training loss, a 100-update moving average, and full validation loss by epoch. It also shows progress, device, elapsed time, throughput, and peak GPU memory when available. There is no separate TensorBoard UI or tracking service. Interpret loss as noise-prediction performance, not generated-image quality. Stage 4 is complete when the pilot and baseline have finite losses, the baseline produces resumable checkpoints, and the React monitor displays the run artifacts.

## Stage 5 — Reverse sampling and generated evidence

Starting from Gaussian `x_T`, implement the epsilon-parameterized reverse mean and posterior variance `beta_tilde = beta_t * (1-alpha_bar_(t-1))/(1-alpha_bar_t)`. Add sampled Gaussian noise at steps above the final step; use no fresh noise at the final update. Do not substitute DDIM or another sampler. Use the complete configured 1,000-step chain; do not re-space or reduce the sampler for generated artifacts.

Verify one reverse update with supplied noise and confirm the index-0 update ignores noise. Smoke-run one complete sample, then generate eight seeded samples from the trained checkpoint. Save raw tensors, a display grid, one complete reverse trajectory, a trajectory contact sheet, and metadata under the run directory. Inspect outputs and describe observed quality separately from the training loss. The test split remains unused.

## Stage 6 — Extend the React/TypeScript local UI

Extend the Stage 2.5 React/TypeScript/Vite UI and existing local FastAPI service. The run monitor is delivered in Stage 4; Stage 6 can build on its run artifacts for dataset summaries and generation. Keep the interface local and focused on learning:

- Overview: runtime/device state, dataset summary, and available runs.
- Data view: sample grid and forward-noising display for a selected image/timestep.
- Training view: resolved run configuration, train/validation loss curves, and checkpoints.
- Generation view: select a checkpoint, count, and seed; request generation; view generated images and reverse trajectory.

The generation view uses a completed run's `best-val.pt`, defaults to eight patches and seed 42, and animates sample 0 from initial Gaussian noise to its final patch. The API runs the shared 1,000-step sampler in a background task, publishes a frame every ten updates for live playback, and saves the full trajectory and generated patch grid under a unique run artifact directory. The current baseline's visual quality is limited and must be represented honestly in the UI.

Training and dataset download remain CLI workflows in v1. Implement a small local Python API that reads recorded artifacts and calls the same sampling functions as the CLI. The API must not reimplement diffusion equations. Keep API inputs typed/validated, write generated outputs under the run-artifact directory, and bind the service to localhost. Serve the built frontend from the API service; use the Vite development server with an API proxy during development. Acceptance: the production build succeeds and one local browser session can inspect a real run and generate images from its checkpoint.

## Stage 7 ? Longer training and checkpoint inference

Use `configs/eurosat_50_epoch.toml` for a fresh 50-epoch run, keeping the 64x64 U-Net, batch size 8, learning rate, optimizer, and 1,000-step schedule unchanged. Save epoch checkpoints every 10 epochs in addition to `best-val.pt` and resumable `latest.pt`. This is approximately 101,250 optimizer updates. The first two epochs of the active run averaged about 34 updates/second, suggesting roughly 50 minutes before validation and checkpoint overhead; revise the estimate from the live monitor.

Before interpreting long-run output, inspect finite raw tensor ranges and report the fraction outside the training range `[-1,1]`. Preserve raw samples; clamp only for PNG display. Select saved checkpoints in the React Generate view and reuse a seed to compare outputs. Continue to train only on the train split, use validation loss for checkpoint selection, and leave test data untouched.

## Verification policy

Keep verification proportionate to this learning repository. Essential checks: schedule and forward equation with known noise; model output shape and finite values; a reduced reverse-sampling smoke run; dataset normalization by real inspection; and one end-to-end UI workflow. Add further tests only to address a specific requirement or observed failure. Do not build a test framework or add tests for every helper by default.

## Out of scope for v1

Conditional/class-based diffusion, multispectral or hyperspectral data, data augmentation, attention, `diffusers`, PyTorch Lightning, distributed training, classifier/text guidance, latent diffusion, DDIM, alternate samplers, remote hosting, and UI-driven training management.

## Stage 9 � Diagnose generation drift and compare EMA

First verify the existing ancestral posterior update at schedule indices 0, 499, and 999 against the configured equation with known noise. Keep the sampler unchanged if those checks pass. Log the current generation defect with its run ID, same-seed reproduction, raw value and trajectory evidence, timestep-bin validation results, and the distinction between confirmed symptoms and open root cause in `docs/bug-reports.md`.

Run one fresh 50-epoch training experiment using `configs/eurosat_50_epoch_ema.toml`. It must differ from the prior 50-epoch recipe only by EMA shadow tracking (`decay=0.999`). Train on `train`, evaluate raw and EMA weights with identical fixed validation noise/timesteps, retain resumable raw/optimizer/RNG/EMA state, and save selectable best raw and EMA checkpoints. Display both validation curves in the existing React Training Monitor.

Generate raw and EMA best-validation patches with the same seed, count, and full 1,000-step sampler. Compare raw ranges, outside-range fraction, RGB summaries, diversity, trajectory drift, and the preview grids in the existing Generate view. Append the result and any diagnosis update to BR-001 and the learning log. Consider the defect resolved only if the sample evidence improves; validation loss alone is insufficient.
