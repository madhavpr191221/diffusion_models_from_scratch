# DDPM Learning Log

This is the learner-facing record of what the project has implemented and what the evidence shows. It is intentionally separate from the normative specification and implementation plan. Last updated 2026-10-12.

## Current status

Stages 0 through 9 are implemented. The latest controlled EMA training run completed all 101,250 updates on CUDA in about 76 minutes. Its best raw validation MSE was 0.013333; its best EMA validation MSE was 0.013079. Reverse sampling remains the full unconditional 1,000-step ancestral process. The test split remains untouched.

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

## Stage 3 — Timestep embedding and U-Net

The model predicts the Gaussian noise used to form a noisy input, but it has not been trained. Its only conditioning input is the diffusion timestep; EuroSAT class labels are not used. The timestep embedding has 128 cosine-then-sine features followed by a learned `128 -> 512 -> 128` transform. A small U-Net uses GroupNorm, SiLU, and matching-resolution skip connections, with feature widths 32, 64, and 128 and an 8x8 bottleneck.

A CPU smoke check used a synthetic batch of two images and timesteps `[0,999]`. The embedding was finite with shape `[2,128]`; the model returned finite predictions shaped `[2,3,64,64]`. The encoder path was `[2,32,64,64]`, `[2,64,32,32]`, `[2,128,16,16]`, `[2,128,8,8]`; decoder concatenations were `[2,256,16,16]`, `[2,128,32,32]`, and `[2,64,64,64]`. The model contains 2,309,539 parameters. No loss, optimizer, or training step was run.

## Stage 4 — Noise prediction and the training monitor

The network learns to predict the exact Gaussian noise used to create a noisy training image. For each update, the trainer samples a training image $x_0$, a timestep $t$, and noise $\epsilon$, then constructs

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon.
$$

The optimizer minimizes the mean squared error

$$
L = \operatorname{mean}\left[(\epsilon-\epsilon_\theta(x_t,t))^2\right].
$$

This objective is unconditional: class labels are omitted. The CLI loads the `train` split for optimization and `validation` for diagnostics. The training workflow only accepts those split names and does not load the reserved test split. Validation uses the full 5,400-image split with fixed noise/timesteps across epochs, making epoch-to-epoch loss comparisons less affected by new validation randomness.

Start with the 64-image / 200-update pilot, then launch the five-epoch baseline:

```powershell
uv run --locked python -m ddpm_eurosat.train --pilot
uv run --locked python -m ddpm_eurosat.train
```

Runs are saved beneath `runs/<run-id>/`. The React app's **Training monitor** view polls run status and metrics while the CLI runs. It displays per-update loss, a 100-update moving average, validation loss by epoch, progress, device, runtime, throughput, GPU memory, and checkpoint metadata. Use the existing API and Vite commands in Stage 2.5 to open the app.

The raw training loss varies from batch to batch because each update sees new images, timesteps, and noise. Its moving average helps show the broad optimization trend. Validation loss is a diagnostic on held-out validation images. A lower loss says the model is better at predicting the sampled noise under this objective; it does not by itself establish that generated images look realistic.

Each run includes `config.json`, `status.json`, append-only `metrics.jsonl`, `checkpoint_manifest.json`, and a `checkpoints/` folder. Progress snapshots are saved around 25%, 50%, 75%, and 100% of planned updates, with `best-val.pt` and a replaceable resumable `latest.pt`. Resume from the latest checkpoint with:

```powershell
uv run --locked python -m ddpm_eurosat.train --resume runs/<run-id>/checkpoints/latest.pt
```

### Run evidence

The pilot completed 200 updates in about 13 seconds (about 15.4 updates/second) with roughly 257 MB peak allocated CUDA memory. Its per-update loss moved from 1.20 at the first update to 0.068 at the last update while repeatedly using the fixed 64-image subset. This checks that the optimization path can learn on a small repeated sample; it is not a measure of generalization.

The baseline used all 16,200 training images for five epochs, batch size 8, AdamW at `0.0002`, and seed 42. It completed 10,125 updates in about 31.9 minutes, with about 5.29 updates/second and 257 MB peak allocated CUDA memory. Each epoch evaluated all 5,400 validation images using the same validation noise/timesteps:

| Epoch | Train MSE | Validation MSE |
|---:|---:|---:|
| 1 | 0.06229 | 0.02512 |
| 2 | 0.02382 | 0.02077 |
| 3 | 0.01993 | 0.01845 |
| 4 | 0.01777 | 0.01714 |
| 5 | 0.01785 | 0.01663 |

The final validation MSE is the best observed value, saved in `best-val.pt`. Artifacts are under `runs/20261011-061727-train-seed-42/`; the run contains the 25%, 50%, 75%, and 100% snapshots, `best-val.pt`, and `latest.pt`. The training workflow loaded the train split and validation split by name and has no path that loads the test split. The React production build succeeded, and the user-provided browser screenshots confirm the completed-run monitor and checkpoint metadata display. An explicit resume smoke check remains as a Stage 4 follow-up. The loss trend is an optimization diagnostic, not evidence about generated-image realism.

## Stage 5 — Ancestral reverse sampling

At reverse index $i$, the model predicts the noise in $x_i$. The sampler computes

$$
\mu_\theta(x_i,i)=\frac{1}{\sqrt{\alpha_i}}\left(x_i-\frac{\beta_i}{\sqrt{1-\bar\alpha_i}}\epsilon_\theta(x_i,i)\right),
\qquad
\tilde\beta_i=\beta_i\frac{1-\bar\alpha_{i-1}}{1-\bar\alpha_i}.
$$

It draws $x_{i-1}=\mu_\theta+\sqrt{\tilde\beta_i}z$ for indices above zero. At index zero, it returns the mean without adding fresh noise. The implementation uses the full 1,000-step configured schedule. A supplied-noise equation check matched the posterior update, and two deliberately different supplied noises produced the same final update.

Generate eight unconditional images and keep the entire path for sample 0:

```powershell
uv run --locked python -m ddpm_eurosat.sampling --checkpoint runs/20261011-061727-train-seed-42/checkpoints/best-val.pt --num-samples 8 --seed 42 --trajectory-index 0
```

Artifacts are in `runs/20261011-061727-train-seed-42/generation/seed-42/`: `samples.pt` has shape `[8,3,64,64]`; `trajectory.pt` has shape `[1001,3,64,64]` and is ordered from $x_T$ to $x_0$; PNG previews show the eight samples and nine evenly spaced trajectory frames; `metadata.json` records the checkpoint, schedule, seed, device, and value range. Raw tensors remain unclipped; only PNG previews clip to `[-1,1]`.

The eight-image run completed in 4.3 seconds on CUDA. The saved tensors are finite and the raw sample range was approximately `[-3.67, 6.00]`. Visual inspection shows the current samples are highly saturated and noisy, with little recognizable land-cover structure. The implementation and training loss are evidence that the reverse calculation runs, not evidence that this five-epoch model has learned useful image generation. The test split was not accessed.

## Stage 6 — Live generation in the React app

The **Generate** view selects a completed training run and can select any checkpoint from the run manifest. Set a seed and request 1–8 patches. Generation runs in a FastAPI background task so the UI can poll progress while the model performs the full 1,000 reverse updates. The app animates the first sample using 101 preview frames (initial noise, every ten reverse updates, and the final patch), then displays the complete patch grid. Raw samples, all 1,001 trajectory states for sample 0, preview PNGs, and metadata are written under `runs/<run-id>/generation/<generation-id>/`.

For development, run the API and Vite frontend in separate terminals as in Stage 2.5, then open the Vite URL and select **Generate**. For a single-server local session, build the frontend with `npm run build --prefix web`, start the API, and open `http://127.0.0.1:8000`. The API exposes routes to start generation, poll progress, retrieve available trajectory frames, and retrieve the completed sample grid. Inference uses the shared CLI sampler and does not load EuroSAT images or labels.

## Stage 7 ? Longer retraining and checkpoint inference

The new `configs/eurosat_50_epoch.toml` keeps the model and optimizer settings fixed and increases the run to 50 epochs. It saves a selectable checkpoint every 10 epochs while retaining `best-val.pt` and resumable `latest.pt`. The run has 101,250 optimizer updates. Its first two epochs averaged about 34 updates/second, suggesting roughly 50 minutes before validation and checkpoint overhead. Training reads only the train split; validation supplies epoch diagnostics; the test split stays untouched.

An early checkpoint's samples were finite but many values fell outside `[-1,1]`; preview clipping therefore produced saturated patches. The Generate controls select a checkpoint so the same seed can be used to inspect outputs at different training stages. A lower validation MSE remains an optimization measure, not a guarantee of realistic patches.

The earlier epoch-10 sample from the 50-epoch run used seed 42 and eight patches. It completed all 1,000 reverse steps, but about 63.0% of its raw values were outside `[-1,1]`; its preview was visibly saturated. This is an intermediate checkpoint and is not a final quality assessment. Stage 8 compares checkpoints 10, 20, 30, 40, 50, and `best-val.pt` under the same seed/count, and records RGB channel means, standard deviations and quantiles, outside-range fraction, per-image pixel variation, and pairwise pixel RMSE. The reference summary uses the first 128 validation images by split index, without labels.

The controlled run is saved under `runs/20261012-045942-train-seed-42/generation/study-seed-42/`. Its combined visual is `checkpoint-comparison.png`. Across the six checkpoints, 62.3%–69.6% of raw generated values fell outside `[-1,1]`; pairwise pixel RMSE was 2.10–2.34. The fixed validation reference had no values outside the range and pairwise RMSE 0.399. The grid remains strongly saturated and does not show convincing land-cover structure. More epochs alone have not resolved the generation failure. Do not judge these samples by clipped PNG colors alone; use the saved raw tensors and diagnostics.

## Next stages

The broader dashboard dataset summary and Stage 4 checkpoint-resume smoke check remain follow-ups.

## Stage 9 — Diagnose drift and compare EMA

Bug report [BR-001](bug-reports.md) records the generation failure and evidence used to investigate it. The prior 50-epoch run's best-validation trajectory mean drifted from about `-0.003` at Gaussian initialization to `+2.05` at the final sample. Validation noise-prediction MSE was low on forward-noised validation images, but that evaluates inputs from the data distribution; it does not show how the model behaves when fed its own reverse-process outputs.

The posterior update matched its configured equation exactly at indices 0, 499, and 999. At index 0, supplying Gaussian noise or omitting it returned the same result. The checkpoint and sampler schedule matched, so no sampling formula change was justified by these checks.

The controlled run used `configs/eurosat_50_epoch_ema.toml`, identical to the previous 50-epoch configuration except for EMA decay `0.999`. It trained for 50 epochs, 101,250 updates, on the train split only; validation selected checkpoints; the test split was not loaded. Artifacts are under `runs/20261012-064627-train-seed-42/`. The training monitor now shows both raw and EMA validation loss, and the Generate view can select `best-val.pt` or `best-val-ema.pt`.

With eight samples and seed 42, raw best-val generation had 65.2% of values outside `[-1,1]`, pairwise pixel RMSE 2.161, and final trajectory mean `+2.082`. EMA best-val generation had 64.2% outside range, pairwise RMSE 2.028, and final trajectory mean `+1.050`. Both used the same full sampler. The first 128 validation images had no values outside `[-1,1]` and pairwise RMSE `0.399`. EMA reduced the trajectory mean drift and slightly reduced the out-of-range fraction, but previews remain strongly saturated and do not show consistently coherent EuroSAT patches. This is a partial improvement, not a resolution; BR-001 remains open and the underlying cause is not confirmed.


At matching timestep index 500, the raw sample path's implied clean estimate had mean `+2.031` with 87.5% outside `[-1,1]`; the forward-noised validation estimate had mean `-0.152` with 0.12% outside. For EMA, those values were `+1.042` / 54.4% versus `-0.164` / 0.09%. At index 300, 92.9% of raw and 42.6% of EMA generated estimates were outside range, compared with about 0.1% on validation. This localizes substantial drift to the model's own reverse states rather than ordinary forward-noised validation inputs.

A sensitivity probe added a known uniform `+0.5` input shift to validation `x_t`. If the clean image is held fixed, this corresponds to a noise-target shift of about `+0.50` to `+0.64` at indices 700 to 300. The model's predicted-noise mean changed by only about `+0.01` for both raw and EMA weights. Repeated GroupNorm layers could suppress global intensity information; this is a supported hypothesis, not a confirmed sole cause. The next controlled experiment should add a small branch that takes the three per-channel spatial means of `x_t` and the timestep embedding, predicts three channelwise noise offsets, and broadcasts them over the U-Net output. This gives global intensity information a path around GroupNorm. Keep the sampler, data, optimizer, seed, schedule, EMA, and 50-epoch budget fixed; change only this architecture path.

Verification: `pytest tests/test_ema.py` passed (2 tests), Python compilation passed, and the React production build passed. The EMA checkpoint also passed the existing API generation workflow: it appeared in the manifest, completed 1,000 reverse steps, and served a final grid and frame. A Windows browser surface was not available in this session, so visual review used the saved raw/EMA grids rather than a live browser click-through.

## Stage 10 � Original DDPM U-Net pilot and full run

The ported CIFAR-10 DDPM U-Net has 35,746,307 parameters and accepts `3�64�64` RGB patches. A 50-update GPU pilot used EuroSAT training examples, microbatch 16, effective batch 128, Adam, warmup, gradient clipping, and EMA. It completed in 96.8 seconds at 0.517 optimizer updates/second, with 3.52 GB peak allocated GPU memory. Linear extrapolation gives about 17.9 days for 800,000 updates; this excludes validation, checkpoint I/O, and any pauses, so it is a lower-bound planning estimate. This measures throughput and memory, not sample quality or convergence. The pilot did not load the test split.

The full run started as `20261012-103416-train-seed-42` with 800,000 planned updates. It is using only train data for gradient updates and a 512-image validation subset every 10,000 updates for checkpoint selection; a full validation pass is planned at completion. The run status metadata records `test_split_loaded: false`. At the latest check it was active at update 20; generation quality is not available until a checkpoint is saved.

### Runtime correction � practical original-U-Net run

The 800,000-update launch was impractical for this laptop: it stopped at update 125 before the first durable checkpoint, and therefore cannot be resumed. The 800,000-update config remains as a paper-scale reference only. The recommended EuroSAT experiment is now 50 epochs at effective batch 128: 6,350 updates, 40 warmup updates, EMA decay 0.999, validation every five epochs, latest checkpoint every epoch, and EMA snapshots every ten epochs. At the observed steady rate this should take about 3.5�4 hours plus validation and checkpoint overhead. This is a practical dataset-specific run, not a full reproduction of the paper's optimization budget.

The practical run `20261012-104411-train-seed-42` has started on CUDA. The test split remains unloaded. Assess generated samples and trajectory diagnostics before claiming the architecture resolves BR-001.
