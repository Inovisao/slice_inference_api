from dataclasses import dataclass
from typing import Tuple


@dataclass
class SlicingConfig:
    slicing_mode: str
    tile_size: Tuple[int, int]
    overlap_percentage: float
    min_object_coverage: float


@dataclass
class DatasetConfig:
    input_path: str = "./dataset"
    output_path: str = "./output"


@dataclass
class CrossFoldsConfig:
    n_folds: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


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
