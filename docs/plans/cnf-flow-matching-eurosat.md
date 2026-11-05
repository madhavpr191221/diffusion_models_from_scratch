# Milestone 0 Review: CNFs and Flow Matching on EuroSAT RGB

## Status and scope

This document records a read-only Milestone 0 review of the proposed continuous normalizing flow (CNF) and flow-matching work. No model code, dependencies, training runs, or existing diffusion behavior were changed. The repository remains on `main` at `c746ecf`; the pre-existing curriculum prompt remains untracked and out of scope.

## My view

This is a strong next research direction for this repository. It extends the current probability-transport story rather than adding an unrelated model family: the VP-SDE already exposes continuous time and a probability-flow ODE connection, while the new work would make the ODE itself the learned transport mechanism.

I recommend doing both tracks, but not treating them as equally feasible first experiments:

1. **Flow matching first.** It tests the new continuous velocity-field and ODE-sampling ideas with an uncomplicated training loop. There is no ODE solve or divergence calculation in its training loss.
2. **Likelihood CNF second, behind a compute gate.** It is the more distinctive and mathematically useful comparison, but its ODE-integrated divergence and gradients through that estimate make it substantially more expensive and numerically delicate.

The key scientific comparison should be: same image representation, same velocity-network capacity, same train/validation split, but different learning objective. It should not be framed as a contest over one scalar loss: FM regression loss and dequantized log-likelihood are different quantities and their numerical values are not comparable.

## Repository audit

The current checkout has a clear separation between discrete diffusion (`diffusion.py`, `sampling.py`, `train.py`) and continuous VP-SDE work (`vp_sde.py`, `score_sde_train.py`, `score_sde_sampling.py`). Both use the pinned EuroSAT RGB revision and the image-only dataset adapter. The adapter returns float32 `[3,64,64]` tensors normalized by `2u/255 - 1`; model workflows may load only `train` and `validation`, and cannot request `test`.

The existing model choices are materially different in size:

- The original-DDPM/VP-SDE U-Net has approximately **35.7 million parameters** and four image scales (64, 32, 16, 8), with attention at 16×16 and the bottleneck.
- The earlier compact U-Net has **2,309,539 parameters**, no attention, and widths 32/64/128.
- `ContinuousTimeOriginalDDPMUNet` already accepts floating normalized VP-SDE times and reuses the original U-Net weights/layout without adding parameters. Its output is still epsilon or score according to checkpoint metadata; it is not a velocity model.
- `data.py`, the run/checkpoint/JSONL conventions in the training modules, seed handling, EMA utilities, visualization helpers, and API run discovery can be reused. The existing API sampler dispatch and two-model UI should not be expanded until the new training and sampling contracts work independently.

The constitution favors explicit mathematics, small components, local artifacts, and review between milestones. That is a good fit here. The current spec-driven structure already provides a place for a separate specification, design, tasks, and learning-log entries.

## Important mathematical and data decisions

### Time orientation

Keep the new flow clock exactly as proposed: `t=0` is Gaussian noise and `t=1` is data. Existing VP-SDE time is the reverse: `s=0` is data and `s=1` is noise. With `t=1-s`, a VP probability-flow trajectory changes velocity sign:

$$
\frac{d\mathbf{Y}_t}{dt}
= -\mathbf{v}_{\mathrm{PF}}(\mathbf{Y}_t,1-t)
= \frac{1}{2}\beta(1-t)\mathbf{Y}_t
 + \frac{1}{2}\beta(1-t)\nabla_{\mathbf{x}}\log p_{1-t}(\mathbf{Y}_t).
$$

This reversed VP path is a useful reference, but it is generally not the same path as the linear Gaussian-to-image path used by the proposed FM objective. Nor is the FM marginal velocity required to equal the VP probability-flow velocity.

### Digital pixels and continuous densities

An exact likelihood for a continuous density evaluated at the repository's 8-bit pixel values is not, by itself, a probability for the observed discrete images. Both continuous-flow tracks should explicitly define a dequantized target distribution. I recommend leaving `image_to_tensor` unchanged and adding a separate, model-specific dequantization operation used by both FM and CNF.

One convention consistent with the existing normalization is to interpret each normalized level `x=2u/255-1` as the center of a quantization cell of width `Δ=2/255` and sample `x̃=x+η`, with `η ~ Uniform(-1/255,1/255)` independently per scalar pixel. This gives cell volume `Δ^D`, where `D=3×64×64=12,288`. It intentionally allows a small amount of dequantized mass outside `[-1,1]`; the source-image adapter and saved raw tensors remain unchanged.

For a continuous density `p_cont` with respect to these normalized coordinates, the uniform-dequantization lower bound on discrete log mass is:

$$
\log P(\mathbf{u}) \ge
\mathbb{E}_{\tilde{\mathbf{x}}\mid\mathbf{u}}
[\log p_{\mathrm{cont}}(\tilde{\mathbf{x}})]
+ D\log\Delta.
$$

Thus the reported negative-log-likelihood upper bound in bits per dimension is:

$$
\mathrm{bpd}
= -\frac{\mathbb{E}[\log p_{\mathrm{cont}}(\tilde{\mathbf{x}})]}{D\log 2}
-\log_2\Delta
= -\frac{\mathbb{E}[\log p_{\mathrm{cont}}(\tilde{\mathbf{x}})]}{D\log 2}
+\log_2(255/2).
$$

This correction is tied to the stated centered-cell convention. A standard `[0,1]` dequantization convention is also valid, but it would require a distinct transform and Jacobian correction. Do not mix the two. I recommend recording the convention, scale, and density units in every run's config and metadata. Dequantization is useful for FM too: its endpoint distribution should be continuous if we want an ordinary invertible ODE to transport to it, rather than trying to collapse a continuous Gaussian onto the finite empirical set of 8-bit images.

### Conditional versus marginal flow matching

For independently sampled noise and dequantized data endpoints, straight interpolation has conditional derivative `x_data-z`. Under squared error, conditioning on `(Y_t,t)` gives the population minimizer `v*(x,t)=E[V | Y_t=x,t]`. The standard conditional-expectation risk decomposition shows the conditional FM objective equals regression onto this marginal velocity plus a term independent of the model parameters. Using test functions and differentiating their expectation along the interpolant yields the continuity equation with this marginal field.

The endpoint-pair paths are straight; trajectories of the learned marginal ODE generally are not. The conditional target is not observable from `(Y_t,t)` alone, so the network learns its conditional expectation. In my view this is one of the most instructive visualizations in the proposed project.

### Likelihood integration direction and sign

Sampling integrates the ODE from `t=0` to `t=1`. Likelihood evaluation starts at a dequantized observation `Y_1=x` and integrates backward to `Y_0=z`. Along forward time, `d log p_t(Y_t)/dt = -div(v)`. Therefore:

$$
\log p_1(x) = \log p_0(z)
-\int_0^1 \nabla\cdot\mathbf{v}_{\theta}(Y_t,t)\,dt.
$$

If a numerical solver integrates the augmented state backward over `[1,0]`, its accumulated log-density change is `+∫_0^1 div(v) dt`; rearranging the backward result gives the boxed formula above. Tests must check both the integration direction and this sign using analytic fields.

At image dimension 12,288, do not form a Jacobian. Use a Hutchinson probe `ξ` with identity covariance and estimate per-example divergence by `ξᵀ Dv ξ`. For a scalar objective, the reverse-mode VJP `grad_x ((v * ξ).sum(), x, create_graph=True)` followed by an inner product with `ξ` is a transparent implementation. `create_graph=True` is needed to train parameters through the divergence estimate, so this introduces higher-order differentiation. Draw a probe once per sample and hold it fixed through all evaluations in that trajectory; resampling inside an ODE right-hand-side call would make the numerical solver see a stochastic, changing vector field. Start with one Rademacher probe for training and compare Gaussian/Rademacher variance and multiple probes during evaluation.

## Feasibility on the stated GPU

The repository's bug report records an original-U-Net pilot at about **0.517 effective updates/s** and **3.52 GiB peak memory** with microbatch 16 and effective batch 128. A 6,350-update FM run at that historical rate would take about **3.4 hours of update time**, before validation, checkpoints, data loading, and other overhead. Treat this only as a rough prior: it is an older measured run and not a current FM benchmark.

FM training is the plausible first full image experiment: it requires one field prediction, an MSE, and ordinary backprop per microbatch, much like current training. Generation cost is instead controlled by ODE network evaluations. Euler with `N` steps costs about `N` field evaluations; explicit Heun costs about `2N`; adaptive Dormand–Prince has variable NFE and can be slower or less stable for a poorly trained field. Always report NFE, elapsed time, solver settings, and output differences.

CNF likelihood training is in a different resource category. Every solver evaluation must compute the velocity and a divergence VJP; parameter gradients through that VJP require higher-order autograd. Direct differentiation through an ODE solve stores intermediate computation graphs and scales memory with solver work. Adjoint differentiation reduces stored-step memory by recomputing the solve, but its gradient is itself a numerical sensitivity solve and must be checked against direct gradients on small problems before it is trusted here. The 35.7M U-Net is therefore not my recommended starting architecture for CNF likelihood. An image-space compact velocity U-Net (about 2.31M parameters) preserves the full 12,288-dimensional EuroSAT problem while reducing activation and derivative cost. Even this option has no measured CNF throughput yet; it requires a short, explicitly approved feasibility pilot before committing to a full training plan.

I do not recommend latent-space CNF as a silent workaround: likelihood in a learned latent coordinate is not image-space likelihood unless the encoder/decoder density and Jacobians are accounted for. Nor do I recommend a 35.7M image CNF until a pilot shows that its higher-order divergence path fits and has useful throughput.

## Recommended implementation boundaries

Use the same new continuous-time vector-field architecture for FM and CNF so objective differences are interpretable. I recommend the compact 2.31M image U-Net topology with a continuous Fourier/time embedding and a 3-channel **velocity** output. Use no output activation, clipping, dropout, or stochastic layers in the ODE field. GroupNorm is compatible with per-example computation but should be checked for finite values and batch independence. Give the architecture and checkpoints a dedicated model ID; never infer semantics from tensor shape or load DDPM/VP-SDE checkpoints as velocity models.

Suggested module boundaries:

- `src/ddpm_eurosat/ode.py`: deterministic Euler and Heun solvers, optional adaptive-solver adapter, NFE accounting, integration direction, trajectory recording.
- `src/ddpm_eurosat/flow_matching.py`: endpoint sampling, interpolation, target velocity, and objective.
- `src/ddpm_eurosat/cnf.py`: augmented state, Gaussian base log density, divergence estimator, change-of-variables integration, likelihood calculation.
- Separate `flow_matching_train.py` and `cnf_train.py` entry points, reusing the data adapter, run-artifact conventions, and checkpoint utilities without changing diffusion trainers.
- New model identifiers and separate configs; later, API endpoints and a React **Flows** section after command-line generation and diagnostics are reliable.

For ODE numerics, my preference is transparent fixed-step Euler and Heun implementations in the repository, plus `torchdiffeq` only if the adaptive solver or adjoint strategy earns its dependency. `torchdiffeq` provides Euler, midpoint, RK4, adaptive Heun, and Dormand–Prince methods, and offers both direct `odeint` differentiation and the adjoint interface. A local Heun implementation is a small, useful equation-to-code teaching component; adaptive DOPRI5 is a reasonable later accuracy/runtime comparison. A solver wrapper should count actual field calls, including rejected adaptive steps, and return selected states without forcing storage of every full-resolution image at every internal evaluation.

## Proposed milestones and review gates

### Milestone 0 — this review

Repository audit, mathematical/data decisions, feasibility opinion, proposed boundaries, and unresolved risks. **Review before any code.**

### Milestone 1 — specification and approved conventions

Write separate requirements, design, and task documents for shared conventions, FM, and CNF. Confirm dequantization cells, time direction, architecture choice, likelihood units, solver set, and the limited meaning of the planned experiments. No model code yet.

### Milestone 2 — shared ODE math

Implement and test Euler/Heun, forward/backward direction, NFE, trajectory snapshots, constant and linear fields, step refinement, and seeded deterministic behavior. Compare any adaptive-library adapter against the fixed-step analytical cases.

### Milestone 3 — flow matching

Add continuous dequantization, straight-line endpoint coupling, conditional velocity target, same compact image-space vector field, and an isolated FM trainer/sampler. Validate shapes, finite loss, gradients, endpoint equations, and seeded ODE output. Run only a short, bounded GPU pilot; report measured VRAM, throughput, and estimated run duration before asking for approval for a full run.

### Milestone 4 — EuroSAT FM experiment

Train the approved run on `train`, select/diagnose with `validation`, never load `test`. Save trajectories, solver diagnostics, image-range/channel/diversity summaries, and qualitative cohorts. Compare Euler/Heun at matched seeds and include actual NFE.

### Milestone 5 — CNF mathematical feasibility pilot

First test analytic divergence, log-density sign, dequantization/Jacobian correction, and direct-versus-adjoint gradients in small-dimensional tests. Then measure one compact image-vector-field likelihood microbatch with a fixed Hutchinson probe and a small number of integration steps. Stop and report if memory or compute makes the proposed full-resolution likelihood experiment impractical; request approval before choosing any alternative.

### Milestone 6 — approved CNF image experiment

Only after the feasibility review, train the approved image-space likelihood model. Track dequantization likelihood bound, probe variance, solver tolerance/step sensitivity, NFE, runtime, peak memory, sample quality, and reverse/forward reconstruction error separately.

### Milestone 7 — synthesis and UI

Summarize what the ODE, continuity equation, and each objective actually estimate; distinguish guarantees, implementation checks, and empirical findings. Add the React Flows view only after the two command-line implementations and saved artifacts have stable contracts.

Each milestone ends with evidence and a review point. No long training run is implied by this plan.

## Focused test plan

- **FM:** endpoint identities, interpolation derivative equals the conditional target, independent seeded endpoint sampling, time range, tensor shapes, finite objective, parameter gradients, and deterministic ODE sampling.
- **ODE:** constant/linear analytic solutions, forward/backward consistency, batch shape, Euler/Heun convergence with step refinement, NFE accounting, and trajectory ordering.
- **CNF:** exact divergence for analytic fields; low-dimensional Hutchinson mean/variance versus exact trace; backward-integration sign; Gaussian base density; dequantization-cell correction; finite gradients through the likelihood objective.
- **Regression:** all existing DDPM, VP-SDE, direct-score, and Langevin tests remain unchanged and pass.

These tests validate equations and contracts, not that a trained model matches EuroSAT.

## Decisions to review before Milestone 1

My recommended defaults are: keep the existing source normalization untouched; explicitly dequantize both new model targets with centered cells of width `2/255`; use a shared compact 2.31M full-image velocity U-Net for FM/CNF; implement Euler and Heun directly; use `torchdiffeq` only for an adaptive comparison after fixed-step tests; try one fixed Rademacher Hutchinson probe per likelihood trajectory; do FM before CNF; and do not promise a full CNF training run until the image-space pilot establishes feasibility.

The open trade-off is capacity: matching the FM/CNF architecture to each other makes the objective comparison cleaner but differs from the 35.7M original-DDPM/VP-SDE model. A later controlled comparison can either train both new objectives at the large capacity if feasible, or present the cross-family comparison as a capacity-matched limitation rather than implying the architectures are identical.

## References consulted

- Chen et al., [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366).
- Grathwohl et al., [FFJORD](https://arxiv.org/abs/1810.01367).
- Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456).
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- Lipman et al., [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264).
- `torchdiffeq` solver and adjoint documentation: [project README](https://github.com/rtqichen/torchdiffeq#keyword-arguments).
