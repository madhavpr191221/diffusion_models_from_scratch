# Tasks: Continuous Flows on EuroSAT RGB

Stage 1 decisions are recorded. Euler/Heun, the Flow Matching CLI path, the React Flows inspection path, checkpoint/resume handling, and the run monitor are implemented. A warmed-up effective-batch pilot estimates about 1.37 hours for the 50-epoch run. Training is held for the user to review this revised estimate and explicitly approve the long run; no CNF work has started.

## Stage 0 — Repository audit

- [x] Inspect current DDPM, VP-SDE, score/Langevin paths, data split and normalization, model sizes, configs, docs, tests, API, and React UI.
- [x] Record reuse boundaries, historical hardware evidence, mathematical risks, and staged recommendation.
- [x] Save the Milestone 0 review under `docs/plans/`.
- [x] Keep this stage read-only with respect to code, dependencies, and training.

## Stage 1 — Specification review

- [x] Draft requirements, equations, architecture proposal, solver strategy, test plan, risks, and acceptance criteria.
- [x] Select centered-cell dequantization: add independent `Uniform(-1/255,1/255)` noise to normalized pixel centers; retain the normalized-coordinate `+log2(255/2)` bpd correction and leave the existing adapter unchanged.
- [x] Select the shared 2.31M-parameter full-image velocity U-Net for FM and CNF; document its capacity limitation versus the 35.7M diffusion U-Net.
- [x] Select fixed-step Euler and Heun as the initial solvers; defer adaptive-solver dependency review until the fixed solvers pass mathematical checks.
- [x] Define the blinded VLM comparison: 100 patches per model, 10 per judgment request, same recorded panel of 16 unlabeled validation patches, model-neutral rubric, and paired initial Gaussian tensors for deterministic ODE samplers where applicable.
- [x] Align mathematical conventions, VLM limitations, acceptance criteria, and milestone dependencies across the specification and design.
- [x] User reviewed the Stage 1 plan and authorized staged implementation.

## Stage 2 — ODE foundation (no training)

- [x] Implement deterministic Euler and Heun with forward/backward time grids, selected snapshots, and actual NFE accounting.
- [x] Add tests for constant/linear analytic fields, solver direction, batch handling, inverse consistency, step refinement, seeded determinism, and trajectory ordering.
- [x] Defer an adaptive solver dependency; the fixed-step methods meet the current baseline need.
- [x] Run solver checks and preserve the review gate before image-model training.

## Stage 3 — Flow-matching baseline

- [x] Add an explicit centered-cell dequantization transform without changing the shared EuroSAT adapter.
- [x] Add the continuous-time velocity model and FM endpoint/interpolation/loss utilities under separate model/config IDs.
- [x] Add focused mathematical and gradient tests; existing repository tests continue to pass.
- [x] Add the separate trainer, seeded sampler, checkpoint artifacts, API endpoints, and React Flows view while preserving train/validation/test boundaries.
- [x] Run a no-update microbatch preflight and a warmed-up effective-batch throughput pilot; record VRAM and the revised run-time estimate.
- [x] Implement best raw/EMA checkpoints, one rolling epoch-boundary resume checkpoint, and resume compatibility checks.
- [x] Expose Flow Matching training progress and loss curves in the React Flows view.
- [ ] Review the revised ~1.37-hour estimate and explicitly approve or revise the training budget before starting training.

## Stage 4 — EuroSAT flow-matching experiment

- [ ] After the resource review gate, train only on `train` and use `validation` for selection/diagnostics; do not load `test`.
- [ ] Compare Euler and Heun on matched initial seeds and report solver NFE, runtime, sensitivity, and saved trajectories.
- [ ] Inspect patch cohorts, raw ranges, channel statistics, diversity, and numerical warnings.
- [ ] Record guarantees, verified implementation behavior, observed results, and remaining uncertainty.

## Stage 5 — Likelihood-CNF feasibility gate

- [ ] Test analytic divergence and Hutchinson estimators in low-dimensional fields.
- [ ] Test Gaussian base log density, backward integration sign, dequantization cell correction, and differentiable likelihood.
- [ ] Compare direct and adjoint gradients on tractable analytic examples before image experiments.
- [ ] Request approval before a bounded image-space memory/throughput pilot.
- [ ] Stop and present evidence if full-image likelihood training is impractical; do not silently change to latent space or FM.

## Stage 6 — Approved CNF experiment

- [ ] After feasibility review, train only the approved image-space likelihood model.
- [ ] Record dequantized likelihood bound/bpd, trace variance, solver sensitivity, NFE, wall time, peak VRAM, and numerical anomalies.
- [ ] Generate image cohorts and save trajectories; keep likelihood, reconstruction, and visual quality conclusions separate.

## Stage 7 — Synthesis and optional CNF UI

- [ ] Write a research report distinguishing theory, tested identities, numerical approximations, and empirical findings.
- [ ] Compare FM/CNF with VP probability-flow and discrete DDPM only under documented conditions; do not imply universal superiority.
- [x] Add a React Flows section for the implemented FM CLI path, including trajectory inspection and fixed-cohort VLM review.
- [ ] Add CNF UI only after the likelihood path is implemented and reviewed.
