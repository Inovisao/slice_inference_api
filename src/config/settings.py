from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AppPaths:
    source_dataset: str = "./dataset"
    generated_datasets: str = "./output"
    models: str = "./models"
    results: str = "./results"


@dataclass
class SlicingConfig:
    slicing_mode: str
    tile_size: Tuple[int, int]
    overlap_ratio: float

    def __post_init__(self):
        assert 0.0 <= self.overlap_ratio < 1.0, (
            f"overlap_ratio must be in [0, 1), got {self.overlap_ratio}"
        )


@dataclass
class DatasetConfig:
    input_path: str = "./dataset"
    output_path: str = "./output"


@dataclass
class CrossFoldsConfig:
    n_folds: int = 5
    seed: int = 42
    ioa_threshold: float = 0.20
    # Fraction carved from the non-test pool in each K-fold iteration.
    val_ratio: float = 0.15
    empty_tile_ratio: float = 0.08


@dataclass
class DataInferenceConfig:
    slicing_mode: str
    suppression: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    batch_size: int = 32
