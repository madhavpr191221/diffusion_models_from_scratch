# Specification: Continuous Flows on EuroSAT RGB

## Status

Stage 1 decisions are recorded. The fixed-step ODE solvers, Flow Matching model/trainer/sampler, API, and React Flows view are implemented. No 50-epoch Flow Matching training run or CNF implementation has been started; the only GPU operation so far was a one-batch preflight with no optimizer update. The current discrete DDPM and VP-SDE implementations remain parallel, supported paths.

## Purpose and research question

Study how continuous probability transport can be learned by (A) maximum-likelihood training of a continuous normalizing flow and (B) conditional flow matching, using the existing unconditional EuroSAT RGB task. The comparison concerns what the two objectives estimate, how their ODE-defined transports behave numerically, and what their trained models generate. It is not a leaderboard, and their training losses are not directly comparable.

## Data and model contract

- Use the pinned EuroSAT RGB revision already used by the repository, at native `64×64` RGB resolution.
- Preserve the existing image adapter and normalized tensor contract `[B,3,64,64]`, with values from `2u/255 - 1` for uint8 pixel value `u`.
- Train on `train` only; use `validation` for diagnostics and model selection; do not load or evaluate `test`.
- Do not pass labels or filenames to the model.
- Keep datasets, checkpoints, generated samples, and run artifacts local and ignored by Git.
- Mathematical flow time is `t∈[0,1]`, with Gaussian noise at `t=0` and data at `t=1`. This is opposite to the existing VP-SDE forward clock. Names, metadata, plots, and APIs must distinguish the clocks.
- The new field is `v_theta(x,t)`, returning a tensor shaped `[B,3,64,64]`. It predicts velocity, not a clean image, epsilon, score, or DDPM `v`-parameterization.

## Shared continuous transport

The generated state solves

$$
\frac{d\mathbf{Y}_t}{dt}=\mathbf{v}_{\boldsymbol\theta}(\mathbf{Y}_t,t),
\qquad \mathbf{Y}_0=\mathbf{Z},\quad
\mathbf{Z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_{12288}),
\qquad \widehat{\mathbf{X}}=\mathbf{Y}_1.
$$

The induced density must satisfy the continuity equation

$$
\partial_t p_t(\mathbf{x})+
\nabla_{\mathbf{x}}\cdot[p_t(\mathbf{x})\mathbf{v}_{\boldsymbol\theta}(\mathbf{x},t)]=0.
$$

The trajectory, density, vector field, and flow map are distinct objects. Under the required regularity assumptions, the learned ODE defines maps `phi_{s,t}` that compose across intermediate times and are inverse under direction reversal. Tests of numerical inverse consistency assess solver accuracy, not whether the learned distribution matches EuroSAT.

## Pixel dequantization and likelihood units

Continuous-density training and evaluation must define how the repository's uint8 observations are converted into a continuous target. The source data adapter must not be silently changed.

**Agreed Stage 1 convention:** regard each normalized pixel value `x=2u/255-1` as the center of a quantization cell with width `Δ=2/255`; draw `x_tilde=x+η`, independently per scalar, with `η~Uniform(-1/255,1/255)`. This centered-cell convention allows a small dequantized tail outside `[-1,1]`. Use it for both tracks so their data targets agree. Store the convention and scale in each run configuration. Keep the existing dataset adapter unchanged.

For `D=3×64×64=12,288`, a continuous density `p_cont` with respect to normalized coordinates gives the uniform-dequantization bound

$$
\log P(\mathbf{u})\ge
\mathbb{E}_{\tilde{\mathbf{x}}\mid\mathbf{u}}
[\log p_{\mathrm{cont}}(\tilde{\mathbf{x}})] + D\log\Delta.
$$

The corresponding negative-log-likelihood upper bound in bits per dimension is

$$
\mathrm{bpd}=
-\frac{\mathbb{E}[\log p_{\mathrm{cont}}(\tilde{\mathbf{x}})]}{D\log 2}
-\log_2\Delta
=-\frac{\mathbb{E}[\log p_{\mathrm{cont}}(\tilde{\mathbf{x}})]}{D\log 2}
+\log_2(255/2).
$$

This correction applies only to the stated normalized-coordinate cell width. No likelihood or bpd may be reported until this convention, the continuous base density, and all coordinate Jacobians are implemented and tested. Stage 1 has selected the centered-cell convention; implementation must verify the correction against this normalized-coordinate definition before reporting likelihood or bpd.

## Track A: likelihood-trained CNF

Training maximizes the dequantized image likelihood induced by the ODE. Along a trajectory,

$$
\frac{d}{dt}\log p_t(\mathbf{Y}_t)
=-\nabla_{\mathbf{x}}\cdot\mathbf{v}_{\boldsymbol\theta}(\mathbf{Y}_t,t).
$$

For an observed dequantized image `Y_1=x`, integrate backward from `t=1` to `t=0` to obtain `z`. Then

$$
\log p_1^{\boldsymbol\theta}(\mathbf{x})
=\log p_0(\mathbf{z})
-\int_0^1\nabla_{\mathbf{x}}\cdot\mathbf{v}_{\boldsymbol\theta}(\mathbf{Y}_t,t)\,dt,
\qquad
\mathcal{L}_{\mathrm{CNF}}
=-\mathbb{E}_{\mathbf{X}\sim p_{\mathrm{dequant}}}
[\log p_1^{\boldsymbol\theta}(\mathbf{X})].
$$

The implementation must state how backward integration accumulates the density correction and test its sign on analytic fields. It must use a standard Gaussian base over all `D=12,288` pixel dimensions.

The full Jacobian is prohibited for image dimensions. Estimate the trace using Hutchinson probes satisfying `E[ξξᵀ]=I`:

$$
\nabla\cdot\mathbf{v}_{\boldsymbol\theta}
=\operatorname{tr}(D_{\mathbf{x}}\mathbf{v}_{\boldsymbol\theta})
\approx\boldsymbol\xi^{\mathsf T}D_{\mathbf{x}}\mathbf{v}_{\boldsymbol\theta}\boldsymbol\xi.
$$

Report the probe distribution and count. The probe used by one trajectory must remain fixed across all ODE right-hand-side evaluations. Training through the divergence requires higher-order autograd; record whether gradients use direct differentiation through solver steps or an adjoint method and validate the chosen strategy. Approximation sources include stochastic trace estimation, numerical integration, finite optimization, and dequantization.

Likelihood training is gated on analytical tests and a bounded image-space feasibility pilot. If the compact full-image model cannot meet the agreed memory/runtime budget, stop and present measured evidence and alternatives for review. Do not silently switch to latent likelihood or FM.

## Track B: conditional flow matching

For independent `Z~N(0,I_D)`, dequantized data `X~p_dequant`, and `U~Uniform(0,1)`, use the implemented straight interpolation and conditional target

$$
\mathbf{Y}_t=(1-t)\mathbf{Z}+t\mathbf{X},
\qquad \mathbf{V}=\mathbf{X}-\mathbf{Z},
\qquad \frac{d\mathbf{Y}_t}{dt}=\mathbf{V}.
$$

The objective is mean squared error averaged over batch and all scalar image entries:

$$
\mathcal{L}_{\mathrm{CFM}}(\boldsymbol\theta)
=\mathbb{E}_{\mathbf Z,\mathbf X,U}
\left[\frac{1}{D}\left\|
\mathbf{v}_{\boldsymbol\theta}(\mathbf{Y}_U,U)
-(\mathbf{X}-\mathbf{Z})\right\|_2^2\right].
$$

The population minimizer is the conditional expectation

$$
\mathbf{v}^{*}(\mathbf{x},t)
=\mathbb{E}[\mathbf{V}\mid\mathbf{Y}_t=\mathbf{x},t].
$$

The squared-error decomposition makes the conditional objective equal to regression onto this marginal field plus a parameter-independent conditional-variance term. For a smooth test function `f`, differentiating `E[f(Y_t)]` gives `E[grad f(Y_t)ᵀ V]`; conditioning on `(Y_t,t)` gives `E[grad f(Y_t)ᵀ v*(Y_t,t)]`. This is the weak form of the continuity equation for the marginal field. Per-pair interpolation paths are straight; learned marginal ODE trajectories are not guaranteed to be straight.

Ordinary FM training must not invoke the ODE solver or a divergence calculation. Generation starts from seeded Gaussian noise and integrates the learned velocity from `0` to `1`.

## ODE solver and diagnostics

- Provide explicit Euler and Heun fixed-step integration with chosen time direction, batch-shape preservation, selected trajectory snapshots, and actual velocity-network-evaluation counts.
- Evaluate an adaptive solver only after the fixed-step equations and diagnostics are correct. `torchdiffeq` is a candidate; adding it requires separate dependency review and a documented rationale.
- Count every vector-field call; Heun generally costs two evaluations per accepted step, while adaptive methods also evaluate rejected steps.
- Compare solver settings on identical trained weights and initial noise. Report NFE, wall time, peak memory, raw sample differences, and convergence against a finer numerical reference when practical.
- Keep intermediate snapshots sparse and selected by mathematical time; do not retain every internal solver state by default.
- Test forward/backward integration using analytic constant and linear fields before interpreting image reconstruction error.

## VP-SDE relationship

The existing VP-SDE uses diffusion time `s` (data at zero, noise at one). For its probability-flow ODE

$$
\mathbf{v}_{\mathrm{PF}}(\mathbf{x},s)
=-\frac{1}{2}\beta(s)\mathbf{x}
-\frac{1}{2}\beta(s)\nabla_{\mathbf{x}}\log p_s(\mathbf{x}),
$$

the new noise-to-data clock `t=1-s`, with `Y_t=X_{1-t}`, has velocity `-v_PF(Y_t,1-t)`. This only describes the reversed VP probability path. The straight-interpolation FM path is different in general, and fields inducing the same marginal path need not be identical.

The terms remain distinct: reverse SDE sampling is stochastic; probability-flow sampling is deterministic and shares the SDE's ideal marginals; either probability-flow ODE defines a CNF under regularity assumptions; CNF names the ODE transport, whereas FM names an objective for learning a velocity field. DSM regresses conditional scores; CFM regresses conditional velocities.

## Reproducibility and exclusions

Record git commit, dataset revision, image transform/dequantization, architecture and parameter count, objective, random seed, optimizer, batch/update budget, EMA, device/peak memory, elapsed time, solver/settings/NFE, checkpoint, validation diagnostics, artifacts, and anomalies. Use the same data and seed cohorts where meaningful. Do not claim convergence or fidelity from a training loss, one seed, a small sample cohort, numerical invertibility alone, or visual texture alone.

## VLM visual-review protocol

Use the existing structured patch-judgement/report workflow as a model-neutral evaluation path for DDPM, VP-SDE, flow-matching, and likelihood-CNF samples. The VLM prompt must not identify the generating model, run, or checkpoint. Use one fixed panel of 16 unlabeled EuroSAT validation patches for every model being compared, and record the reference selection so the panel can be reproduced. The prompt asks for visible aerial/satellite texture and spatial organization, not a named land-cover class or exact image match.

For each formal model comparison, judge exactly 100 generated patches per model. Generation chunk size and judgment request size are separate controls: generation remains local in its existing chunks of up to 8, while the evaluator sends 10 patches per request, for 10 judgment requests per model. Each patch receives an independent structured result keyed by a blinded patch ID: resemblance score from 1 to 5, visible gap categories, and concise visual evidence. Randomize patch IDs/order and omit model metadata from the judgment request. Save the model/run mapping locally with the report, along with the evaluator model identifier, prompt version, reference selection, generation seeds, raw structured results, token usage, and aggregate score/category summaries.

Where deterministic ODE samplers are compared, generate each model's patch from the same initial Gaussian tensor for a paired-seed comparison. For stochastic samplers, use the same seed only when the random-number streams and sampling procedures make that pairing meaningful; otherwise report the cohorts as unpaired. VLM scores are subjective visual assessments, not validated quality metrics or evidence of physical/spectral validity. Interpret them alongside raw-range, channel, diversity, trajectory, and solver diagnostics. They do not replace human review or establish convergence to the EuroSAT distribution.

No conditioning, latent flow, rectified-flow reflow, OT coupling, distillation, replacement dataset, cloud infrastructure, pretrained generator, repository rewrite, or test-split usage is in scope. The React Flows view supports the Flow Matching model after its CLI and artifact contracts; CNF UI support is deferred until its feasibility and math are established.

## Acceptance criteria

- All equations and tensor conventions above are represented in separate, inspectable code paths; existing DDPM and VP-SDE tests/checkpoint semantics remain unchanged.
- FM tests cover endpoint identities, interpolation derivative, seeded independent endpoint sampling, shapes, finite loss, gradients, and time direction.
- ODE tests cover constant/linear analytical solutions, forward/backward consistency, batch dimensions, NFE, trajectory order, determinism, and step-refinement convergence.
- CNF tests cover analytic divergence, Hutchinson estimates versus exact low-dimensional traces, backward density sign, Gaussian base log density, normalized dequantization correction, and finite likelihood gradients.
- A one-batch FM forward/backward preflight reports measured memory and a rough runtime estimate before approval of the 50-epoch run; the separate CNF feasibility pilot remains a later gate.
- Image experiments use train for optimization and validation for diagnostics/selection; test remains untouched.
- Generation and numerical diagnostics record seeds, solver, NFE, runtime, raw image ranges, channel summaries, diversity, and selected trajectories.
- VLM comparison uses 100 generated patches per model, independent blinded judgments in requests of 10, the same recorded 16-patch validation reference panel, and per-patch structured outputs; reports retain the subjective-assessment limitation.
- The Flows UI uses a separate Flow Matching identifier and rejects DDPM/VP-SDE checkpoints as velocity models; CNF UI support is deferred.
