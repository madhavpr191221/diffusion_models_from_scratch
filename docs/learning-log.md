# DDPM Learning Log

This is the learner-facing record of what the project has implemented and what the evidence shows. It is intentionally separate from the normative specification and implementation plan. Last updated 2026-10-10.

## Current status

Stages 0, 1, 2, and 2.5 are implemented. The project has a dataset adapter and both mathematical forms of the forward diffusion process. **There is no neural network, trained checkpoint, reverse sampler, or generated model image yet.** The Stage 2.5 UI explores noise added to real images; it does not generate new images.

## Stage 0 — Project foundation

The repository contains a locked Python environment and a React/TypeScript/Vite frontend scaffold. PyTorch is used directly so the diffusion equations remain visible. Downloaded data and generated outputs are ignored by Git.

Set up the Python and frontend dependencies from the repository root:

```powershell
uv sync --locked
npm ci --prefix web
```

## Stage 1 — EuroSAT RGB data

The dataset is `blanchon/EuroSAT_RGB`, pinned to revision `655ef66ef2be07f89aec61407f24c772802eb87d`. Its 27,000 RGB images are 64 × 64 pixels and are split into 16,200 training, 5,400 validation, and 5,400 test examples. Training will use only the training split; the viewer uses validation examples; the test split is reserved and unused.

The model-facing adapter exposes images only. Class labels are intentionally omitted. A raw channel value $p$ is converted to float32 using

$$
x_0 = \frac{p}{127.5} - 1,
$$

so the tensor range is $[-1,1]$. Dataset cache files live in ignored `data/huggingface/`. The dataset card lists the license as unknown, so the downloaded data remains local and is not redistributed.

To inspect the pinned dataset and recreate its report and image check:

```powershell
uv run --locked python scripts/inspect_eurosat.py
```

The observed split counts, image properties, and normalization check are recorded in `outputs/eurosat_rgb_data_report.json`; the visual round-trip grid is `outputs/eurosat_rgb_data_check.png`.

## Stage 2 — Schedule and direct forward noising

The schedule uses $T=1000$ linearly spaced noise variances from $\beta_1=0.0001$ to $\beta_T=0.02$. Tensor arrays use zero-based indices $0,\ldots,999$, while the UI labels the corresponding mathematical timesteps $1,\ldots,1000$. Define

$$
\alpha_t = 1-\beta_t,
\qquad
\bar\alpha_t = \prod_{s=1}^{t}\alpha_s.
$$

The direct equation samples the marginal at a chosen timestep without simulating earlier steps:

$$
q(x_t\mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right),
\qquad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\quad \epsilon\sim\mathcal{N}(0,I).
$$

The implementation can receive a supplied $\epsilon$ and returns that exact tensor with $x_t$. The schedule check found finite values, $0<\alpha_t<1$, and a strictly decreasing $\bar\alpha_t$, from about 0.999900 at the first index to 0.00004036 at the last. The known-noise equation check had maximum absolute error 0.

Run the focused check and create the reproducible multi-timestep figure:

```powershell
uv run --locked python scripts/visualize_forward_process.py
```

Artifacts: `outputs/eurosat_forward_noising.png` and `outputs/eurosat_forward_noising_report.json`.

## Stage 2.5 — Interactive forward-noise viewer

The local viewer lets you choose an unlabeled validation image, set a seed, and scrub or play through 1,000 noise steps. It shows the clean image beside the current frame and reports the mathematical timestep and $\bar\alpha_t$. The same image, mode, and seed reproduce the same trace. The API and production frontend build have been checked locally; a visual browser click-through is still pending because the Windows browser helper could not initialize.

There are two views because the same marginal distributions can be connected across timesteps in different ways:

- **Direct marginal:** calculate $x_t$ from $x_0$ with one seeded $\epsilon$ reused across frames. Every selected timestep has the prescribed marginal distribution, but the frames are not one step-by-step random path.
- **Stepwise path:** start at $x_0$ and apply $x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}z_t$ with fresh Gaussian $z_t$ at each transition. The frames are one realized Markov trajectory.

Run the local API and frontend in separate PowerShell terminals from the repository root:

```powershell
uv run --locked uvicorn ddpm_eurosat.api:app --host 127.0.0.1 --port 8000
```

```powershell
npm run dev --prefix web
```

Open the local URL printed by Vite. The frontend proxies `/api` requests to the local service. For a production-style local run, build the frontend and then visit the API's local address:

```powershell
npm run build --prefix web
```

The API also serves this log at `/api/learning-log`. It caches only the most recently created trace in memory; creating a new trace replaces the old one. A 1,000-frame float32 trajectory for one RGB 64 × 64 image uses about 47 MiB after construction.

## Next stages

Stage 3 builds the timestep embedding and U-Net. Stage 4 trains the network to predict the noise used to form $x_t$. Only after that can Stage 5 use the learned reverse process to generate new images. The current viewer demonstrates the **forward** process on existing data, not learned generation.
