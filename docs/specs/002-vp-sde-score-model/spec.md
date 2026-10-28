# Specification: Unconditional EuroSAT RGB VP-SDE Score Model

## Status

Approved direction; implementation is staged and reviewed one milestone at a time. The existing discrete DDPM remains a supported parallel implementation.

## User and outcome

The learner wants to understand and compare continuous-time score-based diffusion with the existing discrete DDPM on the same unconditional EuroSAT RGB patch-generation task. The code must keep the SDE equations visible and preserve the project's split, normalization, reproducibility, and artifact conventions.

## Mathematical model

Use the variance-preserving SDE

$$
dX_t=-\frac{1}{2}\beta(t)X_t\,dt+\sqrt{\beta(t)}\,dB_t,
\qquad \beta(t)=\beta_{\min}+t(\beta_{\max}-\beta_{\min}),\quad t\in[0,1],
$$

with `beta_min = 0.1` and `beta_max = 20.0`. Its integrated rate and marginal coefficients are

$$
\int_0^t\beta(s)\,ds=\beta_{\min}t+\frac{1}{2}(\beta_{\max}-\beta_{\min})t^2,
\quad \Phi(t)=\exp\!\left(-\frac{1}{2}\int_0^t\beta(s)\,ds\right),
\quad \sigma(t)=\sqrt{1-\Phi(t)^2}.
$$

The exact forward marginal is `X_t = Phi(t) X_0 + sigma(t) epsilon`. Two separate output parameterizations are supported. The epsilon model predicts noise and exposes its score as `s_theta(X_t,t) = -epsilon_theta(X_t,t) / sigma(t)`. The direct-score model predicts `s_theta(X_t,t)` itself, trained against conditional target `-epsilon/sigma(t)`. Both use the `sigma(t)^2`-weighted conditional denoising-score-matching objective; for epsilon output it is algebraically equal to epsilon MSE:

$$
\mathbb E\left[\sigma(t)^2\left\|s_\theta(X_t,t)+\frac{\varepsilon}{\sigma(t)}\right\|_2^2\right]
=\mathbb E\left[\|\varepsilon_\theta(X_t,t)-\varepsilon\|_2^2\right].
$$

Sample continuous training times uniformly from `[t_min, 1]`, with `t_min = 1e-5`, so score conversion never divides by zero. This excludes only the singular endpoint `t=0`.

## Functional requirements

- Preserve EuroSAT RGB 64×64 inputs, float32 channel-first tensors, and the existing `[-1,1]` normalization. Ignore labels and filenames.
- Train on `train` only. Use a fixed seeded subset of `validation` for evaluation and checkpoint selection. Do not load `test`.
- Keep continuous `beta(t)` and the existing discrete DDPM `beta_k` in separate configuration and code paths.
- Add a continuous-time sinusoidal/Fourier conditioning path with float `[B]` times while retaining the original integer-time DDPM path and legacy checkpoint loading.
- Reuse the original DDPM U-Net backbone, residual blocks, attention, and capacity. Give the continuous architecture its own architecture ID and checkpoint metadata.
- Implement exact forward marginal sampling with caller-supplied noise support and expose epsilon-to-score conversion as a named operation.
- Train the selected weighted DSM objective, recording its definition with the loss metrics.
- Implement the plain reverse SDE with Euler–Maruyama first, using 1,000 uniform intervals from `t=1` to `t=0`.
- Add an independent annealed Langevin sampler: start from seeded `N(0,I)`, descend across 100 uniform VP time levels from 1 to `t_min`, and make 10 updates per level using `x <- x + eta/2 * score + sqrt(eta) * z`, `eta(t)=2e-5*sigma(t)^2/sigma(1)^2`. This is a separate alternative, not a predictor-corrector coupled to the reverse SDE.
- Seed initial Gaussian noise and all reverse Brownian increments. Save raw samples, a full selected-sample trajectory, preview frames, sampler metadata, and the existing range/diversity diagnostics.
- Add completed epsilon and direct-score VP-SDE runs and checkpoints to the existing React Generate workflow. Let the user select reverse-SDE or standalone annealed-Langevin sampling and show normalized time along the trajectory.
- After the baseline works, make a controlled DDPM comparison using matched EuroSAT data, U-Net capacity, training budget, seed, and initial sampling noise where meaningful. Report loss definitions, runtime, network evaluations, trajectories, diagnostics, and qualitative grids without treating loss as image quality.

## Acceptance criteria

- Focused tests verify the schedule equations, forward marginal, score conversion, weighted DSM/epsilon-MSE identity, tensor shapes, and continuous-time conditioning.
- Discretizing the continuous integrated rate reproduces `Phi(t)^2`; an Euler approximation converges as the grid is refined. These checks do not reuse the legacy DDPM beta schedule.
- Existing DDPM tests and checkpoint loading continue to pass unchanged.
- A short GPU pilot completes with finite loss, saved metrics/checkpoints, and measured runtime/memory before any full training run.
- Seeded reverse-SDE generation repeats in the same runtime/device, returns finite `[N,3,64,64]` tensors, and saves all 1,001 trajectory states for the selected sample.
- The direct-score trainer records its objective and prediction parameterization; a seeded Langevin run repeats in the same runtime/device, returns finite `[N,3,64,64]` tensors, makes 1,000 score-network evaluations by default, and saves all 1,001 selected-sample states.
- The direct-score trainer records its objective and prediction parameterization; a seeded Langevin run repeats in the same runtime/device, returns finite `[N,3,64,64]` tensors, makes 1,000 score-network evaluations by default, and saves all 1,001 selected-sample states.
- The existing React Generate view can select a completed VP-SDE run/checkpoint, show progress and the reverse trajectory, and display the generated grid and diagnostics.
- The practical VP-SDE run matches the separate 50-epoch comparison config and never loads the test split.
- A comparison report documents matched conditions and limits, including the fact that equal seeds align initial noise but do not make reverse paths identical.

## Exclusions

No `diffusers` or training framework dependency, class/text conditioning, predictor-corrector sampler, latent diffusion, dataset changes, test-split access, or automatic VLM judging is added in this milestone. Annealed Langevin remains a standalone educational sampler.
