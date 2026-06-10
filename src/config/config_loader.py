from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

from config.settings import (
    CrossFoldsConfig,
    DataInferenceConfig,
    DatasetConfig,
    SlicingConfig,
)

_VALID_SLICING_MODES = ("sahi", "asahi")
_VALID_SUPPRESSIONS = ("nms", "bws", "nms_ioa", "wbf", "cluster_diou_nms")


@dataclass
class ProcessConfig:
    index: int
    dataset: DatasetConfig
    slicing: SlicingConfig
    crossfolds: CrossFoldsConfig
    inference: DataInferenceConfig


class ConfigLoader:
    """
    Loads config.yaml and exposes all processes as a list of ProcessConfig.

    config.yaml format:
        processes:
          - index: 1
            dataset: { ... }
            slicing: { ... }
            crossfolds: { ... }
            inference: { ... }
          - index: 2
            ...
    """

    def __init__(self, path: str = "config.yaml"):
        raw = Path(path).read_text()
        cfg = yaml.safe_load(raw)

        if "processes" not in cfg:
            raise ValueError(
                "config.yaml must have a top-level 'processes' list. "
                "See the format in the file header."
            )

        self._processes: List[ProcessConfig] = [
            self._parse_process(p) for p in cfg["processes"]
        ]

        for proc in self._processes:
            self._validate(proc)

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    def _parse_process(self, raw: dict) -> ProcessConfig:
        index = raw.get("index", 0)

        d = raw.get("dataset", {})
        dataset = DatasetConfig(
            input_path=d.get("input_path", "./dataset"),
            output_path=d.get("output_path", "./output"),
        )

        s = raw.get("slicing", {})
        slicing = SlicingConfig(
            slicing_mode=s.get("mode", "sahi"),
            tile_size=tuple(s.get("tile_size", [640, 640])),
            overlap_ratio=s.get("overlap_ratio", 0.2),
            min_object_coverage=s.get("min_object_coverage", 0.5),
        )

        cf = raw.get("crossfolds", {})
        crossfolds = CrossFoldsConfig(
            n_folds=cf.get("n_folds", 5),
            seed=cf.get("seed", 42),
            ioa_threshold=cf.get("ioa_threshold", 0.2),
            train_ratio=cf.get("train_ratio", 0.70),
            val_ratio=cf.get("val_ratio", 0.15),
            test_ratio=cf.get("test_ratio", 0.15),
        )

        inf = raw.get("inference", {})
        inference = DataInferenceConfig(
            slicing_mode=slicing.slicing_mode,
            suppression=inf.get("suppression", "nms"),
            dataset_path=dataset.input_path,
            models_path=inf.get("models_path", "./models"),
            output_results_path=inf.get("output_results_path", "./output"),
            conf_threshold=inf.get("conf_threshold", 0.25),
            iou_threshold=inf.get("iou_threshold", 0.5),
            batch_size=inf.get("batch_size", 32),
            num_workers=inf.get("num_workers", 4),
            save_original_annotations=inf.get("save_original_annotations", True),
        )

        return ProcessConfig(
            index=index,
            dataset=dataset,
            slicing=slicing,
            crossfolds=crossfolds,
            inference=inference,
        )

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def _validate(self, proc: ProcessConfig):
        self._validate_slicing(proc.slicing, proc.index)
        self._validate_crossfolds(proc.crossfolds, proc.index)
        self._validate_inference(proc.inference, proc.index)

    def _validate_slicing(self, s: SlicingConfig, idx: int):
        if s.slicing_mode not in _VALID_SLICING_MODES:
            raise ValueError(
                f"[process {idx}] slicing.mode must be one of {_VALID_SLICING_MODES}, "
                f"got '{s.slicing_mode}'"
            )
        if len(s.tile_size) != 2 or any(v <= 0 for v in s.tile_size):
            raise ValueError(
                f"[process {idx}] slicing.tile_size must be [w, h] with positive values, "
                f"got {s.tile_size}"
            )
        if not (0.0 < s.overlap_ratio < 1.0):
            raise ValueError(
                f"[process {idx}] slicing.overlap_ratio must be in (0, 1), "
                f"got {s.overlap_ratio}"
            )

    def _validate_crossfolds(self, cf: CrossFoldsConfig, idx: int):
        if cf.n_folds < 2:
            raise ValueError(
                f"[process {idx}] crossfolds.n_folds must be >= 2, got {cf.n_folds}"
            )
        total = cf.train_ratio + cf.val_ratio + cf.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"[process {idx}] crossfolds ratios must sum to 1.0, got {total:.6f}"
            )

    def _validate_inference(self, inf: DataInferenceConfig, idx: int):
        if inf.suppression not in _VALID_SUPPRESSIONS:
            raise ValueError(
                f"[process {idx}] inference.suppression must be one of "
                f"{_VALID_SUPPRESSIONS}, got '{inf.suppression}'"
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def processes(self) -> List[ProcessConfig]:
        return self._processes
