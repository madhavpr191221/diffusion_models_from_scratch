# Design: Continuous Flows on EuroSAT RGB

## Design status

Stage 1 decisions and the initial Flow Matching implementation are recorded here alongside the [specification](spec.md) and [milestone tasks](tasks.md). The Milestone 0 audit is [here](../../plans/cnf-flow-matching-eurosat.md). The fixed-step solvers, continuous velocity model, trainer, API, and React Flows inspection path are implemented. Full training remains gated on review of the preflight estimate; CNF remains unimplemented.

## Boundaries

Keep discrete DDPM, VP-SDE epsilon prediction, direct-score VP-SDE, and Langevin code unchanged. Add a separate continuous-flow model ID, configurations, trainers, samplers, and later API dispatch. Reuse `data.py`, image visualization, seed helpers, run metadata/checkpoint patterns, and selected validation-only diagnostics where their contracts match.

Implemented and planned module boundaries:

- `ode.py`: deterministic Euler/Heun fixed-step integrators, snapshots, and NFE metadata. Adaptive integration remains deferred.
- `flow_matching.py`: dequantization, endpoint/time draws, interpolation, conditional velocity, FM loss, continuous-time U-Net, and seeded sampler.
- `cnf.py`: Gaussian base log density, Hutchinson divergence, augmented likelihood state, and density correction.
- `flow_matching_train.py` / `cnf_train.py`: independent CLI loops and configuration checks.
- `api.py` and `web/src/FlowStudio.tsx`: separate Flow Matching run discovery, sampling studies, trajectories, fixed-size cohort review, and optional VLM judgment. Existing diffusion model aliases remain unchanged.

## Shared velocity architecture

Use one full-image velocity network for both new objectives so the comparison changes the objective rather than architecture. Agreed starting point: the compact U-Net topology in `model.py`, with a continuous Fourier-style time embedding replacing the integer-only embedding and an image-shaped 3-channel velocity head. Its recorded parameter count is 2,309,539. Keep the `[B,3,64,64]` input/output shape and do not initialize from DDPM/VP-SDE checkpoints.

The ODE right-hand side must be deterministic for fixed `(x,t,theta)`: no dropout, random augmentation, or internal noise. Do not clip its velocity or state. GroupNorm is per sample, but its effect on the field and gradients should be checked. The output is a velocity in normalized pixel coordinates per unit flow time; it is not bounded to `[-1,1]`.

This compact architecture preserves the 12,288-dimensional image-space problem but is not capacity-matched to the 35,746,307-parameter original-DDPM/VP-SDE U-Net. FM and CNF are matched to each other. Any comparison to existing diffusion models must report this limitation; a larger model requires a new measured feasibility decision.

## Data path

Leave `image_to_tensor` and the existing data contract unchanged. Add an explicit, seeded dequantization transform for the new flow tracks. Agreed transform treats `x=2u/255-1` as the pixel-cell center and adds independent `Uniform(-1/255,1/255)` noise to each scalar; the normalized cell width is `2/255`. Use the same transform for FM endpoints and CNF likelihood observations; record it in run configuration and metadata. Do not clip dequantized values or model states before computing losses/densities. Clip only rendered previews if necessary, while preserving raw tensors.

For standardization across experiments, use train-only updates, fixed validation subsets/seeds for diagnostics, and never load the test split. Preserve this dequantization convention because it defines the modeled continuous density and bits-per-dimension correction.

## ODE interface

Proposed solver contract accepts a deterministic field `f(t,y)`, initial tensor `y0`, a strictly monotone time grid, method/settings, optional requested save times, and returns selected states plus metadata (`NFE`, solver, interval count, direction, elapsed time). Support descending time grids for CNF inverse likelihood evaluation. Count actual field calls.

Implement Euler directly:

$$
\mathbf y_{n+1}=\mathbf y_n+h_n\mathbf f(t_n,\mathbf y_n).
$$

Implement Heun directly:

$$
\mathbf k_1=\mathbf f(t_n,\mathbf y_n),\quad
\mathbf k_2=\mathbf f(t_n+h_n,\mathbf y_n+h_n\mathbf k_1),\quad
\mathbf y_{n+1}=\mathbf y_n+\frac{h_n}{2}(\mathbf k_1+\mathbf k_2).
$$

For fixed `N` steps, Euler uses `N` evaluations and Heun uses `2N`. Keep trajectory recording separate from solver's internal stage states. Consider `torchdiffeq` for adaptive Dormand–Prince after the explicit solvers pass the analytic tests; adding that dependency is not part of this documentation milestone.

## Flow matching path

Draw independent seeded `z` and dequantized `x`; draw independent continuous `t`; form `y_t=(1-t)z+t x`, target `x-z`; call the velocity network and compute MSE averaged over scalar pixel entries. The target vector has shape `[B,3,64,64]`. This training operation does not solve an ODE or compute divergence. Sampling begins from a seeded `z` and integrates the field with Euler/Heun from 0 to 1.

Save selected states by requested mathematical time, including both endpoints. For an FM trajectory, measure and document displacement `||y(t_j)-y(t_{j-1})||` and curvature only with explicitly defined discrete diagnostics. Do not infer that conditional training line segments equal learned marginal ODE paths.

## CNF likelihood path

Start the training likelihood solve from a dequantized data image at `t=1`; integrate `(x, log-density change)` backward to `t=0`. The base density is standard Gaussian over all image scalars:

$$
\log p_0(z)=-\frac{D}{2}\log(2\pi)-\frac{1}{2}\|z\|_2^2.
$$

The density state obeys `d log p_t(y_t)/dt = -div v(y_t,t)`. If its correction starts at zero at time 1, the backward-integrated correction is `∫_0^1 div v dt`, and `log p_1(x)=log p_0(z)-correction`. This sign must be covered by an analytic linear-field test.

At each ODE function evaluation, estimate each sample's divergence with one Rademacher probe initially. Compute the VJP of `(v*xi).sum()` with respect to `x`, using `create_graph=True`, and inner-product it with `xi`. A probe is sampled once per trajectory and captured by the augmented dynamics; never resample inside solver calls. Compare Rademacher/Gaussian probes and probe counts on validation only after the baseline is correct.

The initial image-space feasibility attempt is still gated and will use the compact field, microbatch one, a small fixed step count, and direct differentiation through the solver as a correctness baseline. This may exceed the GPU memory budget because higher-order autograd retains work across solver evaluations. If so, record peak memory and failure stage, then compare adjoint gradients only in controlled small systems before reconsidering. Do not quietly substitute a latent CNF or FM objective.

For likelihood evaluation, average dequantized continuous log density and apply the cell-volume correction specified in `spec.md`. Report the dequantization variational bound in bpd, not an uncorrected density at an 8-bit grid point. Repeat probe and solver-sensitivity checks to separate Monte Carlo trace variance from ODE error.

## Relationship to VP-SDE

With diffusion time `s` (data-to-noise), reverse the clock using `t=1-s`. If `Y_t=X_{1-t}`, then its derivative is `-v_PF(Y_t,1-t)`. This is a time-reversed probability-flow path. The FM linear interpolation defines a different marginal path in general, and velocity fields are not unique for a given continuity-equation solution. Keep these claims separate in code names, documentation, and UI labels.

## Training and experiment artifacts

Use separate TOML configs with model ID, objective, time convention, data revision/dequantization, optimizer, seed, batch/microbatch, update budget, EMA, and validation protocol. Run directories reuse the current local-artifact pattern for immutable config, status, JSONL metrics, and resumable/latest/best checkpoints, but must record output meaning as `velocity` and reject incompatible checkpoint metadata.

FM and CNF losses have different units/meaning. Report each curve under its objective name; do not overlay them on one unlabeled scale. Record parameter count, optimizer updates, epochs, peak allocated VRAM, elapsed time, solver, NFE, dequantization convention, selected checkpoint, and validation/image diagnostics. Keep cohorts paired by initial Gaussian seeds when the samplers permit, while noting that solver and learned field trajectories differ.

## VLM cohort review

For a formal visual comparison, generate and judge 100 patches per model. Generation is local in chunks of up to 8; judgment sends 10 patches per request (10 requests per model). Use independent ratings, blinded randomized patch IDs, a model-neutral prompt, and the same recorded panel of 16 unlabeled validation patches for all models. Each result contains a 1–5 resemblance score, visible gap categories, and concise evidence. Retain raw structured responses, evaluator model and prompt version, reference selection, seed mapping, token usage, and aggregate summaries. Use identical initial Gaussian tensors for deterministic ODE samplers; describe stochastic cohorts as unpaired unless their sampling noise can be matched meaningfully. Treat these judgments as subjective evidence and read them alongside numeric diagnostics, not as a standalone quality metric.

## UI boundary

No frontend/API changes occur before command-line model, sampling, checkpoint, artifact, and VLM-evaluation contracts stabilize. A later React **Flows** section should show the flow clock explicitly (`0 noise → 1 data`), distinguish FM- and likelihood-trained checkpoint types, offer solver/step controls, and display NFE/runtime/trajectory/raw range. Where VLM review is exposed, it should use the same cohort and blinded-report contract above. It should not add more historical checkpoint choices to the existing two-model diffusion selector.


## Implemented Flow Matching preflight and checkpoint policy

The initial cold one-microbatch preflight was not representative of sustained training. A warmed-up effective-batch pilot then measured five optimizer updates using real training data, eight microbatches per update, Adam, clipping, and EMA. The steady median was 0.762 seconds per effective update. A validation pass over the configured 512 images took 1.672 seconds. For 6,350 updates and 50 validation passes, these measurements project approximately 1.37 hours, excluding checkpoint I/O and runtime variation. Peak allocated GPU memory was 436.7 MB on the RTX 5060 Laptop GPU. The pilot did not write weights, and the test split was not loaded. This is still an estimate, not a measured full-run duration. The full run has not started and is held for explicit approval of this revised estimate.

Training retains only `best-eurosat-flow-matching.pt` (best raw validation loss), `best-eurosat-flow-matching-ema.pt` (best EMA validation loss), and one overwrite-in-place `latest-eurosat-flow-matching.pt` recovery checkpoint. Latest is written at epoch boundaries and is removed after a successful run; if interrupted, it remains resumable. Resume validates exact config compatibility and restores model, EMA, optimizer, and RNG state at an epoch boundary. Generation defaults to the EMA best checkpoint and allows selection of the raw best checkpoint. The React Flows view includes run progress, training and validation loss, device, elapsed time, and peak VRAM. The existing two-model DDPM/VP-SDE selector is unchanged.
