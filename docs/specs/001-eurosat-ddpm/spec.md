# Specification: Unconditional EuroSAT RGB DDPM

## Status

Approved project direction; implementation is staged. Individual milestones receive their own review before work proceeds to the next one.

## User and need

The user is a statistics-oriented learner studying diffusion models. They want a first-principles implementation whose code makes the DDPM equations understandable, with an Earth-observation dataset rather than MNIST.

## Outcome

Train a small model to predict Gaussian noise for normalized EuroSAT RGB images and use ancestral DDPM sampling to generate image patches. Provide visual evidence of forward noising, generated images, and reverse denoising. Provide a local React/TypeScript interface for inspecting dataset and run artifacts and requesting generation from a saved checkpoint.

## Functional requirements

- Use EuroSAT RGB, preferably native 64×64; inspect and report observed data properties rather than assuming them.
- Load `blanchon/EuroSAT_RGB` from Hugging Face at revision `655ef66ef2be07f89aec61407f24c772802eb87d`, caching under ignored `data/huggingface/`.
- Preserve the source `train`, `validation`, and `test` splits. Use only `train` for optimization, `validation` for diagnostics, and leave `test` unused in v1.
- Ignore class labels. Do not add augmentation in the first version.
- Convert images to float32 `[B,3,H,W]` tensors normalized to `[-1,1]`.
- Implement a linear beta schedule and the direct forward equation
  `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`.
- Return the sampled noise with the noised tensor so the training target is exactly the noise used to construct it.
- Implement a cosine/sine sinusoidal timestep representation and a small learned transform.
- Implement a small timestep-conditioned U-Net mapping `[B,3,64,64]` to `[B,3,64,64]`.
- Train with mean squared error between sampled and predicted noise.
- Implement ancestral DDPM sampling with a documented fixed reverse variance and no fresh noise at the final update.
- Save training metrics, run configuration, checkpoints, sample grids, and a reverse trajectory.
- Provide a local React/TypeScript UI for dataset inspection, run/metric/checkpoint viewing, forward-noising visualization, and generation from a saved checkpoint. Training remains a CLI workflow in v1.
- Before training, provide an interactive forward-noise viewer where the user selects an unlabeled validation image and compares direct marginal samples with a seeded stepwise Markov path.
- Maintain a learner-facing log of completed stages, equations, commands, artifacts, and current model/training status.

## Acceptance criteria

- Dataset count, image shape/channels/dtype/value range, normalized range, and batch shape are recorded from an actual load.
- The observed split counts are recorded; labels and filenames are excluded from the model-facing dataset adapter.
- Normalization/de-normalization and a small image grid are visually checked.
- Schedule coefficients are finite, satisfy `0 < alpha_t < 1`, and `alpha_bar_t` decreases with timestep.
- Forward noising matches the equation, supports supplied noise, and handles an independent timestep per image.
- The U-Net returns finite predictions with the required shape.
- A tiny debug/overfit run completes before the baseline budget is chosen.
- A saved checkpoint can be loaded for a complete sampling loop; generated tensors and figures are saved.
- The UI loads real dataset/run artifacts and can request and display generation locally.
- Before any model is trained, the viewer can select validation images, create repeatable traces in both forward-process modes, scrub/play frames, and display the correct timestep and cumulative-alpha metadata.
- The learning log accurately describes completed work and clearly distinguishes implemented forward diffusion from the future trained model and reverse-generation work.

## Constraints and exclusions

- Use PyTorch directly for DDPM mechanics. No `diffusers`, Lightning, distributed training, attention, classifier guidance, text conditioning, latent diffusion, DDIM, or alternate sampler in v1.
- Keep code small and readable. Do not optimize for maximum sample quality.
- Keep downloaded data and generated run artifacts out of Git.
- The Hugging Face dataset card reports its license as unknown. Keep the downloaded copy local and do not redistribute it unless usage rights are verified.
- Do not add a large test suite or tracking system; verify only the mathematical contracts and user-visible workflows that matter.
