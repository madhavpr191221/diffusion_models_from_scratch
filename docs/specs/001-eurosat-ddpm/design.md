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
- Model input: `x_t` is a floating tensor `[B,3,64,64]`; `t` is a `torch.long` tensor `[B]` with zero-based indices `0..999`. `TimeConditionedUNet(x_t, t)` returns a noise prediction with the same shape as `x_t`. Do not clamp `x_t`, because Gaussian noising can move values outside `[-1,1]`.
- Timestep embedding: use 64 geometric frequencies `exp(-log(10000) * k / 64)` for `k=0..63`; concatenate cosine features then sine features to form 128 values. A learned `128 -> 512 -> 128` MLP with SiLU after the first linear layer transforms the embedding.
- Conditioned residual block: GroupNorm with 8 groups and SiLU before each 3x3 convolution; add a learned linear projection of the timestep embedding after the first convolution; use an identity skip when channel counts match and a 1x1 convolution otherwise; apply SiLU after adding the residual.
- U-Net: stem 3->32 at 64x64; encoder blocks 32 at 64x64, 64 at 32x32, and 128 at 16x16; stride-2 3x3 convolutions reach a 128-channel 8x8 bottleneck. Use two 128-channel bottleneck blocks. Each decoder stage upsamples by nearest-neighbor 2x followed by a 3x3 convolution, concatenates its matching encoder feature, then applies one conditioned residual block. Decoder outputs are 128 at 16x16, 64 at 32x32, and 32 at 64x64. The output head is GroupNorm, SiLU, and a 3x3 convolution to 3 channels. Use no attention or class conditioning.
- Data source: Hugging Face `blanchon/EuroSAT_RGB`, pinned to commit `655ef66ef2be07f89aec61407f24c772802eb87d`, cached under `data/huggingface/`.
- Data splits: preserve the published `train` (16,200), `validation` (5,400), and `test` (5,400) partitions; train only on `train`, use `validation` for diagnostics, and leave `test` unused in v1. The label and filename fields never enter the model-facing adapter.
- Pilot/default baseline: seed 42, batch size 8, AdamW at `2e-4`, 5 epochs. Confirm or revise batch size and epoch count only after measuring the tiny pilot's runtime and peak memory.
- Reverse variance: posterior variance `beta_tilde = beta_t * (1 - alpha_bar_(t-1)) / (1 - alpha_bar_t)`; at the final update, add no fresh noise.

## Stage 3 feature-map shape map

For batch size `B`, the encoder skips are `[B,32,64,64]`, `[B,64,32,32]`, and `[B,128,16,16]`; the bottleneck is `[B,128,8,8]`. Decoder concatenations are `[B,256,16,16]`, `[B,128,32,32]`, and `[B,64,64,64]`, returning to `[B,32,64,64]` before the RGB output head.

## Run artifact contract

Each `runs/<run-id>/` contains a resolved configuration, seed and environment summary, epoch-level train/validation noise MSE, elapsed time, peak GPU memory when available, loadable checkpoint(s), and sample/trajectory images. A decreasing noise MSE is a training diagnostic; it is not by itself evidence of good generated imagery.

## Local UI boundary

React/TypeScript/Vite is a separate frontend. Stage 2.5 adds an early forward-noise viewer before the model exists; Stage 6 extends the same frontend and FastAPI service with dataset summaries, run artifacts, and checkpoint generation. The service binds to localhost. The UI does not start or control training in v1. During Stage 2.5, expose paginated validation image previews, create one active seeded trace at a time, and return frames with timestep and cumulative-alpha metadata. A new trace replaces the cached trace; a frame request for a replaced trace returns not found. Serve the built frontend from the local service; during development use the frontend dev server with an API proxy.
