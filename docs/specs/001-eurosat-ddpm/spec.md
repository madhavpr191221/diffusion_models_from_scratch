# Specification: Unconditional EuroSAT RGB DDPM

## Status

Approved project direction; implementation is staged. Individual milestones receive their own review before work proceeds to the next one.

## User and need

The user is a statistics-oriented learner studying diffusion models. They want a first-principles implementation whose code makes the DDPM equations understandable, with an Earth-observation dataset rather than MNIST.

## Outcome

Train a small model to predict Gaussian noise for normalized EuroSAT RGB images and use ancestral DDPM sampling to generate image patches. Provide visual evidence of forward noising, generated images, and reverse denoising. Provide a local React/TypeScript interface for inspecting dataset and run artifacts and requesting generation from a saved checkpoint.

## Functional requirements

- Use EuroSAT RGB, preferably native 64×64; inspect and report observed data properties rather than assuming them.
- Load `blanchon/EuroSAT_RGB` from Hugging Face at revision `655ef66ef2be07f89aec61407f24c772802eb87d`, caching under ignored `data/huggingface/`.
- Preserve the source `train`, `validation`, and `test` splits. Use only `train` for optimization, `validation` for diagnostics, and leave `test` unused in v1.
- Ignore class labels. Do not add augmentation in the first version.
- Convert images to float32 `[B,3,H,W]` tensors normalized to `[-1,1]`.
- Implement a linear beta schedule and the direct forward equation
  `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`.
- Return the sampled noise with the noised tensor so the training target is exactly the noise used to construct it.
- Implement a cosine/sine sinusoidal timestep representation and a small learned transform.
- Implement a small timestep-conditioned U-Net mapping `[B,3,64,64]` to `[B,3,64,64]`.
- Train with mean squared error between sampled and predicted noise.
- Implement ancestral DDPM sampling with a documented fixed reverse variance and no fresh noise at the final update.
- Save training metrics, run configuration, checkpoints, sample grids, and a reverse trajectory.
- Provide a local React/TypeScript UI for dataset inspection, run/metric/checkpoint viewing, forward-noising visualization, and generation from a saved checkpoint. Training remains a CLI workflow in v1.
- Before training, provide an interactive forward-noise viewer where the user selects an unlabeled validation image and compares direct marginal samples with a seeded stepwise Markov path.
- Maintain a learner-facing log of completed stages, equations, commands, artifacts, and current model/training status.
- Record per-update training noise MSE and per-epoch validation MSE with fixed validation timesteps/noise; expose run status, metrics, and checkpoint metadata to the existing React application.
- Train from the CLI only. The React application is a read-only monitor and does not launch or control training; no separate TensorBoard UI or tracking service is added.
- Make resumable checkpoints atomic and include model, optimizer, progress, configuration, and random-number-generator state. Save progress snapshots, `best-val.pt`, and a replaceable `latest.pt`.

## Acceptance criteria

- Dataset count, image shape/channels/dtype/value range, normalized range, and batch shape are recorded from an actual load.
- The observed split counts are recorded; labels and filenames are excluded from the model-facing dataset adapter.
- Normalization/de-normalization and a small image grid are visually checked.
- Schedule coefficients are finite, satisfy `0 < alpha_t < 1`, and `alpha_bar_t` decreases with timestep.
- Forward noising matches the equation, supports supplied noise, and handles an independent timestep per image.
- The U-Net returns finite predictions with the required shape.
- A tiny debug/overfit run completes before the baseline budget is chosen.
- A saved checkpoint can be loaded for a complete sampling loop; generated tensors and figures are saved.
- Ancestral sampling uses the full configured 1,000-step schedule, posterior variance, a reproducible seed, and no added noise in the final update; one complete reverse trajectory and a small sample batch are saved.
- The UI loads real dataset/run artifacts and can request and display generation locally.
- The existing React UI can select a completed run's best-validation checkpoint, request 1–8 unconditional patches, animate sample 0 from Gaussian noise through live reverse progress, and display the completed sample grid.
- Before any model is trained, the viewer can select validation images, create repeatable traces in both forward-process modes, scrub/play frames, and display the correct timestep and cumulative-alpha metadata.
- A 64-image / 200-update pilot records loss, runtime, and peak GPU memory where available; the five-epoch baseline uses only train and validation splits, with the test split not loaded.
- While the CLI trainer runs, the React training monitor shows run state/progress, per-update training loss and moving average, per-epoch validation loss, runtime/device information, and available checkpoints.
- A dedicated 50-epoch configuration preserves the baseline model and optimizer, writes selectable checkpoints every 10 epochs, and records validation metrics without loading the held-out test split.
- The learning log accurately describes completed work and clearly distinguishes training loss from generated-image quality.

## Stage 9 — Diagnose generation drift and compare EMA

- Keep `docs/bug-reports.md` as an append-only log with reproduction steps, measurements, diagnosis, uncertainty, and resolution evidence.
- Verify reverse posterior math and timestep indexing at the first, middle, and last indices before changing sampling behavior.
- Train one separate 50-epoch run with EMA decay `0.999`, preserving the Stage 7 settings and leaving the test split unused.
- Evaluate raw and EMA validation loss with the same fixed noising inputs, save best raw and best EMA checkpoints, and display both validation curves in the existing React monitor.
- Compare same-seed samples from raw and EMA checkpoints in the React Generate view and preserve unclipped output tensors and diagnostics.

Acceptance: equation/index checks pass; an EMA checkpoint resumes with raw weights, optimizer, RNG, and EMA state intact; both best checkpoints load in the existing sampler; and the bug report records quantitative and visual comparison results without treating validation loss as proof of image quality.

## Constraints and exclusions

- Use PyTorch directly for DDPM mechanics. No `diffusers`, Lightning, distributed training, classifier guidance, text conditioning, latent diffusion, or DDIM in v1.
- Keep code small and readable. Do not optimize for maximum sample quality.
- Keep downloaded data and generated run artifacts out of Git.
- The Hugging Face dataset card reports its license as unknown. Keep the downloaded copy local and do not redistribute it unless usage rights are verified.
- Do not add a large test suite or tracking system; verify only the mathematical contracts and user-visible workflows that matter.

## Stage 10 — Original DDPM U-Net on EuroSAT

- Implement the authors' CIFAR-10 U-Net architecture at EuroSAT's native 64×64 RGB resolution. Preserve unconditional epsilon prediction and add checkpoint architecture metadata while retaining loading support for existing compact-model checkpoints.
- Train a fresh run for 800,000 optimizer updates with the paper's CIFAR training settings, using effective batch 128 through GPU-safe microbatching. Use train images only; fixed validation data selects checkpoints; never load test images.
- Show compact long-run metrics and progress in the existing React monitor, retain resumable latest state and bounded EMA milestones, and make resulting checkpoints available in the existing Generate view.

Acceptance: the model has about 35.7M parameters and predicts finite `[B,3,64,64]` tensors; the GPU pilot establishes memory use and run-rate; latest-state resume and model-only inference checkpoints work; old checkpoints still load; the completed run reaches 800,000 updates, records full validation metrics, and generates unconditional patches through the existing UI.
