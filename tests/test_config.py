"""
Validates the setup configs before running the inference pipeline.
All assertions here are preconditions — if any fails, geraResultados.py will break.

The config.yaml is an index of setups (configs/<name>.yaml). These tests load
EVERY setup in the catalog through ConfigLoader (the pipeline's source of truth)
by temporarily selecting all of them, so any activatable setup is validated.
"""

import copy

import pytest
import yaml

from config.config_loader import ConfigLoader, engine_dir_for_model

_VALID_MODES = ("sahi", "asahi", "asahi_rect", "none")
_VALID_SUPPRESSIONS = ("nms", "bws", "nms_ioa", "wbf", "cluster_diou_nms")


@pytest.fixture(scope="module")
def raw_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def all_processes(tmp_path_factory, raw_config):
    """Load ALL cataloged setups through ConfigLoader (not just setups_to_run)."""
    cfg = copy.deepcopy(raw_config)
    cfg["setups_to_run"] = [s["name"] for s in cfg["setups"]]
    tmp = tmp_path_factory.mktemp("cfg")
    tmp_cfg = tmp / "config.yaml"
    # ConfigLoader resolves setup paths relative to the config file's dir, so
    # point the temp config's setup paths at the real absolute configs/.
    import os
    for s in cfg["setups"]:
        s["config"] = os.path.abspath(s["config"])
    tmp_cfg.write_text(yaml.safe_dump(cfg))
    return ConfigLoader(str(tmp_cfg)).processes


class TestConfigStructure:
    def test_has_global_paths(self, raw_config):
        paths = raw_config.get("paths", {})
        for key in ("source_dataset", "generated_datasets", "models", "results"):
            assert paths.get(key), f"Global path is missing: '{key}'"

    def test_has_setups_index(self, raw_config):
        assert "setups" in raw_config and isinstance(raw_config["setups"], list)
        assert raw_config["setups"], "setups catalog is empty"

    def test_setups_to_run_is_subset_of_catalog(self, raw_config):
        catalog = {s["name"] for s in raw_config["setups"]}
        for name in raw_config.get("setups_to_run", []):
            assert name in catalog, f"setups_to_run references unknown setup '{name}'"

    def test_all_setups_load(self, all_processes):
        assert len(all_processes) > 0


class TestSetupIdentity:
    def test_each_setup_has_output_name(self, all_processes):
        for p in all_processes:
            assert p.output_name, f"process {p.index} missing output_name"

    def test_each_setup_has_models(self, all_processes):
        for p in all_processes:
            assert p.models, f"setup {p.output_name} has empty models list"

    def test_model_names_map_to_engine(self, all_processes):
        for p in all_processes:
            for m in p.models:
                # raises if the prefix is unknown
                assert engine_dir_for_model(m) in {"yolo", "faster_rcnn", "detr"}


class TestSlicingConfig:
    def test_mode_is_valid(self, all_processes):
        for p in all_processes:
            assert p.slicing.slicing_mode in _VALID_MODES, \
                f"Invalid slicing mode: '{p.slicing.slicing_mode}'"

    def test_overlap_ratio_in_range(self, all_processes):
        for p in all_processes:
            overlap = p.slicing.overlap_ratio
            if p.slicing.slicing_mode == "none":
                assert overlap == 0.0
            else:
                assert 0.0 < overlap < 1.0, f"overlap_ratio must be in (0,1), got {overlap}"

    def test_tile_size_is_positive(self, all_processes):
        for p in all_processes:
            tile = p.slicing.tile_size
            assert len(tile) == 2 and all(v > 0 for v in tile)


class TestCrossFoldsConfig:
    def test_n_folds_at_least_two(self, all_processes):
        for p in all_processes:
            assert p.crossfolds.n_folds >= 2

    def test_val_ratio_in_range(self, all_processes):
        for p in all_processes:
            assert 0.0 < p.crossfolds.val_ratio < 1.0

    def test_split_strategy_is_valid(self, all_processes):
        for p in all_processes:
            assert p.crossfolds.split_strategy in {"kfold_holdout", "fixed_ratios"}

    def test_current_config_documents_trained_split_protocol(self, all_processes):
        for p in all_processes:
            assert p.crossfolds.split_strategy == "kfold_holdout"
            assert p.crossfolds.n_folds == 5
            assert p.crossfolds.val_ratio == 0.15


class TestInferenceConfig:
    def test_suppression_is_valid(self, all_processes):
        for p in all_processes:
            assert p.inference.suppression in _VALID_SUPPRESSIONS

    def test_conf_threshold_in_range(self, all_processes):
        for p in all_processes:
            assert 0.0 < p.inference.conf_threshold <= 1.0

    def test_iou_threshold_in_range(self, all_processes):
        for p in all_processes:
            assert 0.0 < p.inference.iou_threshold <= 1.0

    def test_batch_size_positive(self, all_processes):
        for p in all_processes:
            assert p.inference.batch_size > 0
