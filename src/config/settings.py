from dataclasses import dataclass
from typing import Tuple


@dataclass
class SlicingConfig:
    slicing_mode: str
    tile_size: Tuple[int, int]
    overlap_ratio: float
    min_object_coverage: float

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
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    empty_tile_ratio: float = 0.08


@dataclass
class DataInferenceConfig:
    slicing_mode: str
    suppression: str
    dataset_path: str
    models_path: str
    output_results_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    batch_size: int = 32
    num_workers: int = 4
    save_original_annotations: bool = True
