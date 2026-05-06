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
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
@dataclass
class DataInferenceConfig:
    slicing_mode: str
    suppression: str
    dataset_path: str
    models_path: str
    output_results_path: str
    batch_size: int = 32
    num_workers: int = 4
    save_original_annotations: bool = True
