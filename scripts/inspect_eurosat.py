"""Download EuroSAT RGB and save empirical data checks and a preview grid."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ddpm_eurosat.data import (
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_SPLIT_COUNTS,
    UnconditionalEuroSATSplit,
    image_to_tensor,
    load_eurosat_rgb,
)


def inspect_split(split) -> dict:
    modes: set[str] = set()
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    pixel_min, pixel_max = 255, 0

    # Walk one row at a time so the full decoded image set is never held in RAM.
    for index in range(len(split)):
        image = split[index]["image"]
        pixels = np.asarray(image)
        modes.add(image.mode)
        shapes.add(tuple(int(dim) for dim in pixels.shape))
        dtypes.add(str(pixels.dtype))
        pixel_min = min(pixel_min, int(pixels.min()))
        pixel_max = max(pixel_max, int(pixels.max()))

    return {
        "examples": len(split),
        "columns": split.column_names,
        "observed_image_modes": sorted(modes),
        "observed_hwc_shapes": [list(shape) for shape in sorted(shapes)],
        "observed_dtypes": sorted(dtypes),
        "raw_pixel_min": pixel_min,
        "raw_pixel_max": pixel_max,
        "normalized_min_from_observed_extrema": pixel_min / 127.5 - 1.0,
        "normalized_max_from_observed_extrema": pixel_max / 127.5 - 1.0,
    }


def save_preview(train_split, output_path: Path) -> float:
    count = min(6, len(train_split))
    indices = random.Random(42).sample(range(len(train_split)), count)
    fig, axes = plt.subplots(2, count, figsize=(2.2 * count, 5.8), squeeze=False)
    max_roundtrip_error = 0

    for column, index in enumerate(indices):
        original = train_split[index]["image"]
        raw = np.asarray(original)
        normalized = image_to_tensor(original)
        display = normalized.add(1).div(2).clamp(0, 1)
        reconstructed = (
            display.mul(255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        max_roundtrip_error = max(
            max_roundtrip_error,
            int(np.abs(raw.astype(np.int16) - reconstructed.astype(np.int16)).max()),
        )

        axes[0, column].imshow(raw)
        axes[0, column].set_title("Original")
        axes[1, column].imshow(display.permute(1, 2, 0).cpu().numpy())
        axes[1, column].set_title("Round-trip")
        for row in range(2):
            axes[row, column].axis("off")

    fig.suptitle("EuroSAT RGB: original and normalized/de-normalized examples")
    fig.subplots_adjust(top=0.9, hspace=0.42, wspace=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(max_roundtrip_error)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / "data" / "huggingface"
    outputs_dir = project_root / "outputs"

    print(f"Loading {DATASET_ID} at {DATASET_REVISION} into {cache_dir}")
    source = load_eurosat_rgb(cache_dir)
    actual_counts = {name: len(split) for name, split in source.items()}
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Unexpected source splits: expected {EXPECTED_SPLIT_COUNTS}, got {actual_counts}"
        )

    split_reports = {}
    for name, split in source.items():
        print(f"Inspecting {name}: {len(split):,} images")
        split_reports[name] = inspect_split(split)

    train_images = UnconditionalEuroSATSplit(source["train"])
    batch = next(iter(DataLoader(train_images, batch_size=8, shuffle=False, num_workers=0)))
    batch_min, batch_max = float(batch.min()), float(batch.max())
    if batch.shape != (8, 3, 64, 64):
        raise ValueError(f"Unexpected batch shape: {tuple(batch.shape)}")
    if not torch.isfinite(batch).all() or batch_min < -1.0 or batch_max > 1.0:
        raise ValueError(f"Invalid normalized batch range: [{batch_min}, {batch_max}]")

    preview_path = outputs_dir / "eurosat_rgb_data_check.png"
    roundtrip_error = save_preview(source["train"], preview_path)
    report = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "cache_dir": str(cache_dir.relative_to(project_root)),
        "dataset_card_license": "unknown",
        "split_reports": split_reports,
        "observed_total_examples": sum(actual_counts.values()),
        "normalized_tensor_contract": {
            "shape": ["B", 3, 64, 64],
            "dtype": str(batch.dtype),
            "observed_batch_min": batch_min,
            "observed_batch_max": batch_max,
            "adapter_returns_tensor_only": isinstance(train_images[0], torch.Tensor),
        },
        "preview": {
            "path": str(preview_path.relative_to(project_root)),
            "max_roundtrip_pixel_error": roundtrip_error,
        },
    }
    report_path = outputs_dir / "eurosat_rgb_data_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")
    print(f"Saved preview: {preview_path}")


if __name__ == "__main__":
    main()
