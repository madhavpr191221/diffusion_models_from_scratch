# Design: Parallel VP-SDE Score Model

## Boundaries

Keep the discrete DDPM implementation in `diffusion.py`, `train.py`, and `sampling.py`. Add a separate `vp_sde.py` for continuous-time coefficients, time sampling, forward marginal sampling, epsilon-to-score conversion, and the weighted DSM identity. Never pass continuous `beta(t)` values to `DiffusionSchedule` or the DDPM reverse update.

Add a separate continuous-time model architecture ID, `vp_sde_ddpm_cifar_v1`. It shares the original DDPM U-Net backbone and learned parameter capacity. Factor only the computation that builds a time embedding so the existing `ddpm_cifar_v1` keeps its integer input and unchanged embedding semantics. The continuous model accepts float `[B]` times in `[0,1]`; map them to the original embedding's 0-to-999 phase range. Its output remains epsilon, with the score conversion visible in `vp_sde.py` and called by sampling.

## Configuration and training

Use `configs/eurosat_vp_sde_50_epoch.toml`. Keep the process-specific schedule under `[vp_sde]`, not `[diffusion]`. Match `eurosat_original_ddpm_50_epoch.toml` on seed, data revision, train-only horizontal flips, effective batch 128, microbatch 16, 6,350 updates, Adam, learning rate, warmup, gradient clipping, EMA, validation subset/interval, and checkpoint cadence. Only the diffusion process and continuous time-conditioning semantics differ.

Add `score_sde_train.py` as a separate entry point. Reuse the unlabeled dataset adapter and established run/checkpoint/JSONL conventions where practical. New runs identify their process and architecture in config, status, and checkpoints; their loss is labeled `sigma_squared_weighted_dsm_epsilon_mse`. Validation uses fixed seeded continuous times and noise so measurements are repeatable. Keep training/validation data access explicit and do not instantiate the test split.

## Sampling and UI integration

Add `score_sde_sampling.py` with a separate Euler–Maruyama sampler. On `t_i = 1 - i/N`, `N=1000`, evaluate the score at the upper endpoint of each interval and apply

$$
X_{t_i-\Delta t}=X_{t_i}-\left[-\frac{1}{2}\beta(t_i)X_{t_i}-\beta(t_i)s_\theta(X_{t_i},t_i)\right]\Delta t+\sqrt{\beta(t_i)\Delta t}\,z_i.
$$

The final interval still receives its Brownian increment; the reverse SDE is stochastic throughout. Do not query the score at exactly zero. Start from `N(0,I)` as the standard approximation to the terminal VP marginal; for the selected schedule, `Phi(1)` is about `0.0066`, so the terminal marginal is close to standard Gaussian.

Write artifacts under the selected run using the existing generation directory and filenames where compatible: raw `samples.pt`, grid PNG, full selected trajectory tensor/contact sheet, progress frames, metadata, and `summarize_images` diagnostics. Metadata records process, beta endpoints, time grid, integration steps, seed, runtime, NFE, checkpoint, trajectory order, and terminal-prior approximation.

At the API boundary, dispatch by saved process/architecture metadata to the DDPM or score-SDE sampler. Reuse run listing, checkpoint listing, generation progress, frame serving, and result routes. Extend frame metadata/UI labels so DDPM uses integer timestep `k` and VP-SDE uses continuous normalized `t`; the math modules remain separate. The Training Monitor labels score runs as weighted DSM / epsilon MSE rather than implying a new DDPM metric.

## Controlled comparison

Use the existing completed `ddpm_cifar_v1` run with 6,350 updates as the discrete reference when its local checkpoint is available; otherwise rerun its matching config. Train VP-SDE with the same seed, dataset revision, U-Net backbone, optimizer/budget, augmentation, validation subset, and EMA policy. Compare best EMA checkpoints with the same count and seed. The shared initial Gaussian tensor is the paired starting cohort; subsequent reverse updates follow different equations and stochastic paths.

Record per-run training/validation loss definitions and curves, wall time, peak memory, NFE, selected trajectories, raw range/out-of-range fraction, channel summaries, pairwise diversity, and side-by-side grids. Keep the report descriptive: different schedules and time samplers mean training curves are contextual diagnostics, not a claim that one process is better based on loss alone.
