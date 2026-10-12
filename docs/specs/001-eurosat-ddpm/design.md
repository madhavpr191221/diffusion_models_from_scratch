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
runs/<run-id>/                # status, config, JSONL metrics, checkpoints; ignored
```

Only files needed by the current milestone should be added. This layout is the target, not a request to create empty placeholders for every future module.

## Python boundaries

- `data.py`: explicit dataset download/loading, transforms, and batch contract.
- `diffusion.py`: schedule buffers, timestep coefficient extraction, independent timestep sampling, direct marginal noising, stepwise forward transitions/trajectory, and the ancestral posterior reverse update.
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

Each `runs/<run-id>/` contains `config.json`, atomically replaced `status.json`, append-only `metrics.jsonl`, `checkpoint_manifest.json`, and `checkpoints/`. Training records per-update noise MSE and epoch-level validation noise MSE with fixed validation noise/timesteps. The four progress snapshots, `best-val.pt`, and replaceable `latest.pt` contain model/optimizer state, epoch/batch/update position, resolved configuration, and Python/NumPy/PyTorch RNG states. Checkpoints are written through a temporary file followed by atomic replacement. A decreasing noise MSE is a training diagnostic; it is not by itself evidence of good generated imagery.

The training command loads the pinned `train` split directly and the `validation` split only for diagnostics. The `load_eurosat_split` training interface cannot request `test`, preventing the training workflow from loading test images.

## Local UI boundary

React/TypeScript/Vite is the only UI. The Stage 2.5 forward-noise view remains available and a Stage 4 Training Monitor polls the local FastAPI service for run summaries, status, incremental metrics, and checkpoint metadata. The CLI starts and controls training; UI endpoints are read-only. During Stage 2.5, expose paginated validation image previews, create one active seeded trace at a time, and return frames with timestep and cumulative-alpha metadata. A new trace replaces the cached trace; a frame request for a replaced trace returns not found. Serve the built frontend from the local service; during development use the frontend dev server with an API proxy.

## Stage 5 reverse sampling

`ddpm_reverse_step` implements the epsilon-prediction posterior update using zero-based schedule index `i`. For `i > 0`, it adds Gaussian noise scaled by `sqrt(beta_tilde_i)`, where `beta_tilde_i = beta_i * (1 - alpha_bar_(i-1)) / (1 - alpha_bar_i)`. At index 0, `alpha_bar_previous = 1` and no random noise is added. `sampling.py` loads a compatible checkpoint, uses the full configured schedule and a device-local seeded generator, and stores raw tensors without clipping. PNG conversion alone clips to `[-1,1]`.

The CLI writes `samples.pt`, `samples.png`, `trajectory.pt`, `trajectory.png`, and `metadata.json` under `runs/<run-id>/generation/seed-<seed>/`. The trajectory contains `x_T` followed by each reverse state through `x_0`, for `T+1` frames. One selected sample is retained to bound storage. Generation starts only from Gaussian noise and does not read any dataset split.

## Stage 6 live generation view

The Generate view lists only completed training runs with `checkpoints/best-val.pt`. It defaults to the newest eligible run, eight samples, and seed 42; users may select another eligible run, a count from 1 to 8, and a seed. Generation starts asynchronously through `POST /api/runs/{run_id}/generations`. The API reports state/progress through `GET /api/runs/{run_id}/generations/{generation_id}`, exposes each published PNG through `/frames/{frame_index}`, and serves the completed grid through `/samples.png`.

The API calls the shared `generate_from_checkpoint` and `sample_chain` functions. It publishes frame 0 (`x_T`) and then a frame every ten reverse updates, ending at `x_0`; the sampler still executes and saves all 1,000 reverse updates and the full trajectory. Only one generation job runs at a time in the local API process. A second start request returns conflict until the active job completes or fails. Generation artifacts are stored under the selected run's unique `generation/<generation-id>/` directory. No EuroSAT split is loaded for generation.


## Stage 7 longer training and checkpoint inference

Keep the baseline architecture, batch size, optimizer, learning rate, dataset revision, seed, and 1,000-step schedule fixed in a separate 50-epoch config. Save checkpoints at epochs 10, 20, 30, 40, and 50, in addition to best-validation and latest. The saved run config remains the source of truth for resume behavior.

The generation API accepts a checkpoint filename only when it is present in the run manifest and resolves inside that run's checkpoint directory. The UI can select checkpoints and accumulate same-seed result grids for visual comparison. Preserve generated tensors outside `[-1,1]`; record their range and out-of-range fraction, and clip only while creating preview PNGs. Sampling remains unconditional and does not load dataset images or labels.

## Stage 9 sampler diagnosis and EMA comparison

Keep `docs/bug-reports.md` as an append-only issue log. Record the current saturated-sample symptom as an open generation defect: include the run ID, matched-seed reproduction, out-of-range fractions, trajectory mean drift, diagnostic method, equation/index check results, and any remaining uncertainty about its root cause.

Before retraining, compare `ddpm_reverse_step` at the first, middle, and last schedule indices to the configured posterior mean/variance using supplied noise. Confirm the final index is deterministic and that the checkpoint's schedule equals the sampler schedule. Do not alter the sampler or clamp raw tensors if these checks pass.

Use a separate 50-epoch config matching Stage 7, with only `training.ema_decay = 0.999` added. Update the EMA shadow after every optimizer update. Evaluate raw and EMA weights against the same fixed validation timesteps/noise each epoch. Save `best-val-ema.pt` with EMA weights for inference while retaining raw training weights, optimizer state, RNG state, and EMA state in resumable checkpoints. `latest.pt` resumes raw optimization and restores the EMA shadow; the existing manifest/API/UI selection path exposes both best checkpoints.

Compare `best-val.pt` and `best-val-ema.pt` with the same seed, sample count, schedule, and ancestral sampler. Record raw ranges, out-of-range fraction, per-channel summaries, pairwise variation, and selected trajectory means, then inspect both previews in the existing React Generate view. EMA is an improvement only if quantitative diagnostics and visual structure support it; validation loss alone is not a generation-quality measure.

## Stage 10: original DDPM CIFAR-10 U-Net on EuroSAT

Port the authors' unconditional CIFAR-10 U-Net from [their released implementation](https://github.com/hojonathanho/diffusion) to PyTorch. Use 128 base channels, multipliers `(1, 2, 2, 2)`, two residual blocks at each level, attention at 16x16 and in the middle block, GroupNorm, Swish/SiLU, 0.1 dropout, and the original timestep embedding and zero-initialized residual/output projections. At EuroSAT resolution, feature scales are 64, 32, 16, and 8 pixels. The expected parameter count is about 35.7M; the input/output contract remains `[B,3,64,64]`.

New run configs declare `model.architecture = "ddpm_cifar_v1"`. Configs without an architecture identifier remain mapped to the compact U-Net so existing checkpoints continue to load. Training and sampling factories use the saved checkpoint config.

The dedicated 800,000-update run uses the paper's CIFAR training defaults where practical: Adam, effective batch 128, learning rate 2e-4 with 5,000-update warmup, global gradient clipping at 1.0, EMA 0.9999, train-only horizontal flips, epsilon MSE, and the 1,000-step linear beta schedule from 0.0001 to 0.02. Microbatching plus gradient accumulation maintains effective batch 128 on the 8 GB GPU. Use fixed-large reverse variance for this paper-recipe config; legacy configs default to posterior variance. The test split is never loaded.

Validate on a fixed 512-image subset every 10,000 updates using reproducible noise/timesteps, and run a full validation pass at completion. Save full `latest.pt` resume state every 5,000 updates, compact EMA-only milestones every 100,000 updates, and model-only best raw/EMA checkpoints. Keep raw per-update training metrics in `metrics.jsonl`; serve the existing React monitor a compact stream with one mean-loss point per 100 updates plus validation points. The existing Generate view uses architecture-aware loading for best EMA and milestone checkpoints.
