# Design: Unconditional EuroSAT RGB DDPM

## Repository layout

```text
docs/
  CONSTITUTION.md
  learning-log.md
  plans/eurosat-ddpm.md
  specs/001-eurosat-ddpm/{spec,design,tasks}.md
src/ddpm_eurosat/
  data.py
  diffusion.py
  model.py
  train.py
  sampling.py
  visualization.py
  api.py
scripts/
  inspect_eurosat.py
web/                         # React + TypeScript + Vite
configs/base.toml
data/                         # downloaded, ignored
runs/<run-id>/                # metrics, config, checkpoints, figures; ignored
```

Only files needed by the current milestone should be added. This layout is the target, not a request to create empty placeholders for every future module.

## Python boundaries

- `data.py`: explicit dataset download/loading, transforms, and batch contract.
- `diffusion.py`: schedule buffers, timestep coefficient extraction, independent timestep sampling, direct marginal noising, stepwise forward transitions/trajectory, and (in a later stage) reverse update math.
- `model.py`: timestep embedding, residual blocks, and U-Net only.
- `train.py`: batch-to-loss step, optimizer loop, metrics, and checkpoint writing.
- `sampling.py`: checkpoint loading and ancestral sampling, with optional intermediate states.
- `visualization.py`: image-grid creation and conversion from `[-1,1]` for display.
- `api.py`: local HTTP interface to the data, saved runs, forward traces, and (in a later stage) generation workflows; no diffusion math duplicated here.

## Initial model and run defaults

- Input/output: `[B,3,64,64]`; preserve native image resolution.
- Schedule: `T=1000`, linear beta endpoints `1e-4` and `0.02`; code uses zero-based array indices while explaining mathematical timesteps as `1..T`.
- Forward-process interface: `DiffusionSchedule` owns one-dimensional coefficient buffers; `sample_timesteps` draws one independent index per image; `q_sample(schedule, x0, t, noise=None)` returns `(x_t, noise)` for `[B,C,H,W]` input. The schedule, image, and timestep tensors share a device.
- Timestep embedding: cosine then sine ordering, dimension 128, learned MLP, separate learned projection in each conditioned residual block.
- U-Net: widths 32, 64, 128; downsample to an 8×8 bottleneck; GroupNorm and SiLU; matching-resolution skip concatenations; no attention.
- Data source: Hugging Face `blanchon/EuroSAT_RGB`, pinned to commit `655ef66ef2be07f89aec61407f24c772802eb87d`, cached under `data/huggingface/`.
- Data splits: preserve the published `train` (16,200), `validation` (5,400), and `test` (5,400) partitions; train only on `train`, use `validation` for diagnostics, and leave `test` unused in v1. The label and filename fields never enter the model-facing adapter.
- Pilot/default baseline: seed 42, batch size 8, AdamW at `2e-4`, 5 epochs. Confirm or revise batch size and epoch count only after measuring the tiny pilot's runtime and peak memory.
- Reverse variance: posterior variance `beta_tilde = beta_t * (1 - alpha_bar_(t-1)) / (1 - alpha_bar_t)`; at the final update, add no fresh noise.

## Run artifact contract

Each `runs/<run-id>/` contains a resolved configuration, seed and environment summary, epoch-level train/validation noise MSE, elapsed time, peak GPU memory when available, loadable checkpoint(s), and sample/trajectory images. A decreasing noise MSE is a training diagnostic; it is not by itself evidence of good generated imagery.

## Local UI boundary

React/TypeScript/Vite is a separate frontend. Stage 2.5 adds an early forward-noise viewer before the model exists; Stage 6 extends the same frontend and FastAPI service with dataset summaries, run artifacts, and checkpoint generation. The service binds to localhost. The UI does not start or control training in v1. During Stage 2.5, expose paginated validation image previews, create one active seeded trace at a time, and return frames with timestep and cumulative-alpha metadata. A new trace replaces the cached trace; a frame request for a replaced trace returns not found. Serve the built frontend from the local service; during development use the frontend dev server with an API proxy.
