# VP-SDE Milestone Tasks

## Stage 1 — Specification and matched config

- [x] Record VP-SDE equations, objective choice, boundaries, and acceptance criteria.
- [x] Specify separate continuous model, trainer, sampler, API dispatch, UI, and artifact behavior.
- [x] Add a practical 50-epoch config matched to the existing original-DDPM recipe.
- [x] Review Stage 1 before implementing math or model code.

## Stage 2 — VP-SDE math and continuous-time model

- [x] Add continuous beta, integrated rate, `Phi(t)`, marginal standard deviation, continuous-time sampling, exact forward marginal, epsilon-to-score conversion, and weighted DSM identity in a separate module.
- [x] Add a continuous-time model ID sharing the original DDPM U-Net backbone; retain the discrete model's integer-time behavior and checkpoint loading.
- [x] Add focused schedule, marginal, conversion/loss identity, time-conditioning, shape, and numerical discrete-correspondence tests.
- [x] Run focused new tests and all existing DDPM tests; record evidence and pause for review.

### Stage 2 evidence

- `uv run --locked pytest tests/test_vp_sde.py tests/test_vp_sde_model.py tests/test_original_ddpm_model.py tests/test_reverse_variance.py tests/test_ema.py -q` — 16 passed.
- `git diff --check` — clean (Git reported only its normal LF-to-CRLF working-copy notice for `model.py`).

## Stage 3 — Training and durable artifacts

- [x] Add an independent score-SDE CLI trainer with fixed seeded continuous-time validation, EMA, resumption, and existing atomic artifact conventions.
- [x] Confirm train-only optimization, validation-only selection, and no test split load.
- [x] Add a 50-update GPU pilot and verify finite metrics, checkpoints, runtime, and peak memory.
- [x] Record evidence and pause before the full practical run.

### Stage 3 evidence

- `uv run --locked pytest tests/test_score_sde_train.py tests/test_score_sde_api.py tests/test_vp_sde.py tests/test_vp_sde_model.py tests/test_original_ddpm_model.py tests/test_reverse_variance.py tests/test_ema.py -q` - passed; includes config isolation, finite DSM gradients, seeded validation, checkpoint resume state, and SDE generation guards.
- `npm --prefix web run build` - passed (`tsc -b` and Vite production build).
- Pilot command: `uv run --locked python -m ddpm_eurosat.score_sde_train --config configs/eurosat_vp_sde_50_epoch.toml --pilot --device cuda`.
- Pilot run: `runs/20261019-103357-pilot-seed-42`; 50/50 updates, finite final training loss 0.38372, 100.3 seconds elapsed, peak allocated GPU memory 3516 MB. The pilot intentionally performs no validation and creates no best-validation checkpoint.
- The 6,350-update practical run was completed in Stage 5; its artifacts and final validation are recorded below.

## Stage 4 — Reverse-SDE sampler and React generation

- [x] Add seeded 1,000-step Euler–Maruyama generation with explicit reverse drift, stochastic increments through the final interval, and no score evaluation at `t=0`.
- [x] Save raw samples, one complete 1,001-state trajectory, preview frames, metadata, and existing diagnostics.
- [x] Extend API run/checkpoint discovery and sampler dispatch without mixing the DDPM and SDE schedules.
- [x] Extend the existing Generate view with VP-SDE run identification, progress, a continuous-time trajectory axis, grid, and diagnostics.
- [x] Verify repeatable seeded generation, finite outputs, artifact completeness, and API frame metadata; record evidence below.
- [ ] Verify generation through a live browser click-through.

### Stage 4 evidence

- `uv run --locked pytest -q` — 29 passed, including seeded repeatability, finite output, final-interval Brownian noise, SDE API dispatch, frame `X-Time`, and mocked background-job frame artifacts.
- `npm --prefix web run build` — passed (`tsc -b` and Vite production build).
- Pilot sampler smoke: 1 sample, 1,000 reverse steps, CUDA; saved 1,001 states and finite sample output.
- Full-run generation: 8 samples from `best-val-ema.pt`, seed 42, 1,000 steps, 1,000 network evaluations. Saved `samples.pt`, `samples.png`, `trajectory.pt` (1,001 states for sample 0), `trajectory.png`, and `metadata.json` under `runs/20261019-171309-train-seed-42/generation/seed-42/`. Samples were finite; 40.7% of raw values were outside `[-1,1]`; pairwise pixel RMSE was 1.291. This is numeric evidence, not a claim that all patches look like EuroSAT.
- Windows suspended during generation because the original 5-minute AC sleep setting had been restored. The sampler's recorded 21,896 seconds includes suspension and is not a valid active-runtime measurement. The original AC timeout was restored and verified after training.
- A live React/browser click-through was not performed; the API background-job integration is covered by mocked tests.

## Stage 5 — Controlled comparison

- [x] Train the 6,350-update VP-SDE run after the pilot gate is accepted.
- [ ] Reuse the completed local DDPM reference run when available, otherwise rerun its matching config.
- [ ] Generate matched-count, same-seed EMA cohorts with shared initial noise where possible.
- [ ] Record loss curves/definitions, runtime, memory, NFE, trajectories, numeric diagnostics, and side-by-side qualitative results.
- [ ] Update the learning log with evidence and limits; leave test data untouched.

## Stage 6 — Direct-score network and annealed Langevin

- [x] Add a separately identified continuous-time U-Net that emits score values directly, retaining the epsilon model and the same backbone/capacity.
- [x] Train the direct output against `-epsilon/sigma(t)` with explicit sigma-squared weighted DSM; preserve validation-only checkpoint selection and never load test data.
- [x] Add the matched 50-epoch direct-score config and Training Monitor labels that distinguish direct score from epsilon prediction.
- [x] Add standalone seeded annealed Langevin sampling with 100 VP time levels, 10 updates per level, eta proportional to `sigma(t)^2`, and all 1,001 selected-sample states.
- [x] Add a sampler choice to the existing React Generate view and API; direct-score and epsilon checkpoints both provide score estimates to either sampler.
- [x] Add focused tests for direct-score loss equivalence, seeded Langevin trajectories, API selection, and continuous-time trajectory metadata.
- [x] Verify focused Python tests and production React build.
- [x] Run a short direct-score GPU pilot and record loss, runtime, and memory before the full 50-epoch run.
- [ ] Use the completed direct-score checkpoint in the React Generate view and inspect the annealed-Langevin trajectory in a browser.

### Stage 6 implementation evidence

- `uv run --locked pytest -q` — 37 passed, including the full existing DDPM suite and focused direct-score/Langevin checks.
- `npm run build --prefix web` — passed (`tsc -b` and Vite production build).
- Pilot command: `uv run --locked python -m ddpm_eurosat.score_sde_train --config configs/eurosat_vp_sde_direct_score_50_epoch.toml --pilot --device cuda`.
- Pilot run `runs/20261021-054051-pilot-seed-42`: 50/50 updates, final training DSM loss 0.43125, 97.8 seconds elapsed, 3,515 MB peak allocated GPU memory. The pilot uses train images only and does not run validation or create a best-validation checkpoint.
- The full 6,350-update run and browser click-through remain outstanding. A completed full run will create the best-validation EMA checkpoint selectable in Generate.
