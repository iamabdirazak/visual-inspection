"""Reading the VisA split, its images and its defect masks."""
import csv
from dataclasses import dataclass

import cv2
import numpy as np

from src.paths import CALIBRATION_IMAGES, FIT_IMAGES, SEED, VISA

MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet statistics, which the backbone was trained on
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Sample:
    image: str
    mask: str | None  # None for a normal part
    defect: str       # "normal", or the annotated defect types

    @property
    def defective(self) -> bool:
        return self.mask is not None


def defect_types(product: str) -> dict:
    with open(VISA / product / "image_anno.csv") as f:
        return {row["image"]: row["label"] for row in csv.DictReader(f)}


def split(product: str) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """The official one-class split, with the normal training images divided into fit and calibration sets."""
    types = defect_types(product)
    train, test = [], []
    with open(VISA / "split_csv" / "1cls.csv") as f:
        for row in csv.DictReader(f):
            if row["object"] != product:
                continue
            sample = Sample(row["image"], row["mask"] or None, types.get(row["image"], row["label"]))
            (train if row["split"] == "train" else test).append(sample)
    order = np.random.default_rng(SEED).permutation(len(train))
    fit = [train[i] for i in order[:FIT_IMAGES]]
    calibration = [train[i] for i in order[FIT_IMAGES:FIT_IMAGES + CALIBRATION_IMAGES]]
    return fit, calibration, test


def size_for(shape: tuple[int, int], resolution: int) -> tuple[int, int]:
    """Width and height with the long side at the resolution and both sides multiples of 16, keeping the aspect."""
    h, w = shape
    scale = resolution / max(h, w)
    return max(16, round(w * scale / 16) * 16), max(16, round(h * scale / 16) * 16)


def read(path: str) -> np.ndarray:
    return cv2.imread(str(VISA / path), cv2.IMREAD_COLOR)


def prepare(image: np.ndarray, resolution: int) -> np.ndarray:
    """BGR image to a normalised (1, 3, H, W) float array at the working resolution."""
    image = cv2.resize(image, size_for(image.shape[:2], resolution), interpolation=cv2.INTER_AREA)
    image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
    return image.transpose(2, 0, 1)[None]


def mask(sample: Sample, size: tuple[int, int]) -> np.ndarray:
    """The defect mask as booleans at the given (width, height)."""
    m = cv2.imread(str(VISA / sample.mask), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 0
