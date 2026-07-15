"""
Validates config.yaml before running the inference pipeline.
All assertions here are preconditions — if any fails, geraResultados.py will break.
"""

import pytest
import yaml

_VALID_MODES = ("sahi", "asahi", "asahi_rect")
_VALID_SUPPRESSIONS = ("nms", "bws", "nms_ioa", "wbf", "cluster_diou_nms")
_REQUIRED_PROCESS_KEYS = ("dataset", "slicing", "crossfolds", "inference")


@pytest.fixture(scope="module")
def config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def processes(config):
    return config["processes"]


class TestConfigStructure:
    def test_has_global_paths(self, config):
        paths = config.get("paths", {})
        for key in ("source_dataset", "generated_datasets", "models", "results"):
            assert paths.get(key), f"Global path is missing: '{key}'"

    def test_has_processes_key(self, config):
        assert "processes" in config

    def test_processes_is_non_empty_list(self, processes):
        assert isinstance(processes, list) and len(processes) > 0

    def test_each_process_has_required_keys(self, processes):
        for p in processes:
            for key in _REQUIRED_PROCESS_KEYS:
                assert key in p, f"Process {p.get('index')} missing key: '{key}'"


class TestSlicingConfig:
    def test_mode_is_valid(self, processes):
        for p in processes:
            mode = p["slicing"].get("mode")
            assert mode in _VALID_MODES, f"Invalid slicing mode: '{mode}'"

    def test_overlap_ratio_in_range(self, processes):
        for p in processes:
            overlap = p["slicing"].get("overlap_ratio")
            assert overlap is not None, "overlap_ratio is missing"
            assert 0.0 < overlap < 1.0, f"overlap_ratio must be in (0, 1), got {overlap}"

    def test_tile_size_is_positive(self, processes):
        for p in processes:
            tile = p["slicing"].get("tile_size", [])
            assert len(tile) == 2 and all(v > 0 for v in tile), \
                f"tile_size must be [w, h] with positive values, got {tile}"

class TestCrossFoldsConfig:
    def test_n_folds_at_least_two(self, processes):
        for p in processes:
            n = p["crossfolds"].get("n_folds", 0)
            assert n >= 2, f"n_folds must be >= 2, got {n}"

    def test_val_ratio_in_range(self, processes):
        for p in processes:
            cf = p["crossfolds"]
            val_ratio = cf.get("val_ratio")
            assert val_ratio is not None and 0.0 < val_ratio < 1.0

    def test_split_strategy_is_valid(self, processes):
        for p in processes:
            cf = p["crossfolds"]
            assert cf.get("split_strategy") in {"kfold_holdout", "fixed_ratios"}

    def test_fixed_ratio_strategy_has_valid_test_ratio(self, processes):
        for p in processes:
            cf = p["crossfolds"]
            if cf.get("split_strategy") != "fixed_ratios":
                continue
            test_ratio = cf.get("test_ratio")
            assert test_ratio is not None and 0.0 < test_ratio < 1.0
            assert cf["val_ratio"] + test_ratio < 1.0

    def test_current_config_documents_trained_split_protocol(self, processes):
        for p in processes:
            cf = p["crossfolds"]
            assert cf["split_strategy"] == "kfold_holdout"
            assert cf["n_folds"] == 5
            assert cf["val_ratio"] == 0.15


class TestInferenceConfig:
    def test_suppression_is_valid(self, processes):
        for p in processes:
            s = p["inference"].get("suppression")
            assert s in _VALID_SUPPRESSIONS, f"Invalid suppression: '{s}'"

    def test_conf_threshold_in_range(self, processes):
        for p in processes:
            conf = p["inference"].get("conf_threshold")
            assert conf is not None and 0.0 < conf <= 1.0, \
                f"conf_threshold must be in (0, 1], got {conf}"

    def test_iou_threshold_in_range(self, processes):
        for p in processes:
            iou = p["inference"].get("iou_threshold")
            assert iou is not None and 0.0 < iou <= 1.0, \
                f"iou_threshold must be in (0, 1], got {iou}"

    def test_batch_size_positive(self, processes):
        for p in processes:
            bs = p["inference"].get("batch_size", 32)
            assert bs > 0, f"batch_size must be positive, got {bs}"
