from pathlib import Path

import yaml

from config.settings import (
    CrossFoldsConfig,
    DataInferenceConfig,
    DatasetConfig,
    SlicingConfig,
)

_VALID_SLICING_MODES = ("sahi", "asahi")
_VALID_SUPPRESSIONS = ("nms", "bws", "nms_ioa", "wbf", "cluster_diou_nms")


class ConfigLoader:
    def __init__(self, path: str = "config.yaml"):
        raw = Path(path).read_text()
        self._cfg = yaml.safe_load(raw)
        self._validate()

    def _validate(self):
        self._validate_slicing()
        self._validate_crossfolds()

    def _validate_slicing(self):
        s = self._cfg.get("slicing", {})

        mode = s.get("mode", "")
        if mode not in _VALID_SLICING_MODES:
            raise ValueError(f"slicing.mode must be one of {_VALID_SLICING_MODES}, got '{mode}'")

        tile = s.get("tile_size", [])
        if len(tile) != 2 or any(v <= 0 for v in tile):
            raise ValueError(f"slicing.tile_size must be [width, height] with positive values, got {tile}")

        overlap = s.get("overlap_percentage", 0)
        if not (0.0 < overlap < 1.0):
            raise ValueError(f"slicing.overlap_percentage must be in (0, 1), got {overlap}")

        coverage = s.get("min_object_coverage", 0)
        if not (0.0 < coverage <= 1.0):
            raise ValueError(f"slicing.min_object_coverage must be in (0, 1], got {coverage}")

    def _validate_crossfolds(self):
        cf = self._cfg.get("crossfolds", {})

        n = cf.get("n_folds", 5)
        if n < 3:
            raise ValueError(f"crossfolds.n_folds must be >= 3, got {n}")

        total = sum([
            cf.get("train_ratio", 0),
            cf.get("val_ratio", 0),
            cf.get("test_ratio", 0),
        ])
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"crossfolds ratios must sum to 1.0, got {total:.6f}")

    @property
    def slicing(self) -> SlicingConfig:
        s = self._cfg["slicing"]
        return SlicingConfig(
            slicing_mode=s["mode"],
            tile_size=tuple(s["tile_size"]),
            overlap_percentage=s["overlap_percentage"],
            min_object_coverage=s.get("min_object_coverage", 0.5),
        )

    @property
    def dataset(self) -> DatasetConfig:
        d = self._cfg.get("dataset", {})
        return DatasetConfig(
            input_path=d.get("input_path", "./dataset"),
            output_path=d.get("output_path", "./output"),
        )

    @property
    def crossfolds(self) -> CrossFoldsConfig:
        cf = self._cfg.get("crossfolds", {})
        return CrossFoldsConfig(
            n_folds=cf.get("n_folds", 5),
            train_ratio=cf.get("train_ratio", 0.70),
            val_ratio=cf.get("val_ratio", 0.15),
            test_ratio=cf.get("test_ratio", 0.15),
        )

    @property
    def inference(self) -> DataInferenceConfig:
        inf = self._cfg.get("inference", {})
        suppression = inf.get("suppression", "nms")
        if suppression not in _VALID_SUPPRESSIONS:
            raise ValueError(f"inference.suppression must be one of {_VALID_SUPPRESSIONS}, got '{suppression}'")
        return DataInferenceConfig(
            slicing_mode=self._cfg["slicing"]["mode"],
            suppression=suppression,
            dataset_path=self._cfg.get("dataset", {}).get("input_path", "./dataset"),
            models_path=inf.get("models_path", "./models"),
            output_results_path=inf.get("output_results_path", "./output"),
            conf_threshold=inf.get("conf_threshold", 0.25),
            iou_threshold=inf.get("iou_threshold", 0.45),
            batch_size=inf.get("batch_size", 32),
            num_workers=inf.get("num_workers", 4),
            save_original_annotations=inf.get("save_original_annotations", True),
        )
