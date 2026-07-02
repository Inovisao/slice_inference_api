"""Checks requiring datasets and checkpoints produced outside the unit tests."""

import json
import os

import pytest

from config.config_loader import ConfigLoader

pytestmark = pytest.mark.integration

_ARCH_DIRS = ("yolo", "faster_rcnn", "detr")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _configured_folds():
    loader = ConfigLoader("config.yaml")
    return [
        (
            process.slicing.slicing_mode,
            fold,
            process.dataset.output_path,
            loader.paths.models,
        )
        for process in loader.processes
        for fold in range(1, process.crossfolds.n_folds + 1)
    ]


_FOLDS = _configured_folds()
_MODELS = [item + (arch,) for item in _FOLDS for arch in _ARCH_DIRS]


@pytest.mark.parametrize("mode,fold,output_root,models_root,arch", _MODELS)
def test_checkpoint_manifest_and_file_exist(mode, fold, output_root, models_root, arch):
    manifest_path = os.path.join(
        models_root, mode, f"fold_{fold}", arch, "manifest.json"
    )
    assert os.path.isfile(manifest_path), f"Missing manifest: {manifest_path}"
    with open(manifest_path, encoding="utf-8") as f:
        checkpoint = json.load(f).get("checkpoint")
    assert checkpoint and os.path.isfile(checkpoint), f"Missing checkpoint: {checkpoint}"


@pytest.mark.parametrize("mode,fold,output_root,models_root", _FOLDS)
def test_generated_test_split_is_consistent(mode, fold, output_root, models_root):
    test_dir = os.path.join(output_root, f"fold_{fold}", "test")
    images_dir = os.path.join(test_dir, "images")
    labels_dir = os.path.join(test_dir, "labels")
    assert os.path.isdir(images_dir)
    assert os.path.isdir(labels_dir)

    images = {
        os.path.splitext(name)[0]
        for name in os.listdir(images_dir)
        if name.lower().endswith(_IMAGE_EXTENSIONS)
    }
    labels = {
        os.path.splitext(name)[0]
        for name in os.listdir(labels_dir)
        if name.endswith(".txt")
    }
    assert images, f"No test images in {images_dir}"
    assert images == labels

    stats_path = os.path.join(output_root, f"fold_{fold}_stats.json")
    assert os.path.isfile(stats_path)
    with open(stats_path, encoding="utf-8") as f:
        stats_names = {item["image_name"] for item in json.load(f)["test_metrics"]}
    disk_names = {
        name for name in os.listdir(images_dir)
        if name.lower().endswith(_IMAGE_EXTENSIONS)
    }
    assert stats_names == disk_names
