"""EuroSAT RGB loading and the image-only tensor contract."""

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

DATASET_ID = "blanchon/EuroSAT_RGB"
DATASET_REVISION = "655ef66ef2be07f89aec61407f24c772802eb87d"
EXPECTED_SPLIT_COUNTS = {"train": 16_200, "validation": 5_400, "test": 5_400}
IMAGE_SIZE = 64


def load_eurosat_rgb(cache_dir: str | Path = "data/huggingface") -> DatasetDict:
    """Load the pinned HF dataset revision into the requested local cache."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        cache_dir=str(cache_path),
    )


def image_to_tensor(image: Image.Image) -> Tensor:
    """Map one RGB uint8 PIL image to float32 CHW in [-1, 1]."""
    if image.mode != "RGB":
        raise ValueError(f"Expected RGB image, received mode {image.mode!r}")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} image, received {image.size}"
        )

    # PIL is HWC; copy ensures the NumPy buffer is writable for torch.from_numpy.
    pixels_hwc = np.array(image, dtype=np.uint8, copy=True)
    pixels_chw = torch.from_numpy(pixels_hwc).permute(2, 0, 1).to(torch.float32)
    return pixels_chw.div(127.5).sub(1.0)


class UnconditionalEuroSATSplit(TorchDataset[Tensor]):
    """A source split exposing image tensors only; labels never reach training."""

    def __init__(self, split: Dataset) -> None:
        self.images = split.select_columns(["image"])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        return image_to_tensor(self.images[index]["image"])
