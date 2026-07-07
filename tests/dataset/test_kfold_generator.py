import json

import cv2
import numpy as np
import yaml

from config.settings import SlicingConfig
from dataset.kfold_generator import AsahiKFoldValidator


def _write_coco_dataset(root, image_count=4):
    images, annotations = [], []
    for index in range(1, image_count + 1):
        name = f"image_{index}.jpg"
        cv2.imwrite(str(root / name), np.zeros((32, 32, 3), dtype=np.uint8))
        images.append({"id": index, "file_name": name, "width": 32, "height": 32})
        annotations.append({
            "id": index,
            "image_id": index,
            "category_id": 1,
            "bbox": [8, 8, 8, 8],
            "area": 64,
        })
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "insect"}],
    }
    (root / "_annotations.coco.json").write_text(json.dumps(coco))


def test_generator_writes_trainable_deterministic_folds(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_coco_dataset(source)

    generator = AsahiKFoldValidator(
        dataset_path=str(source),
        slicing_config=SlicingConfig("sahi", (16, 16), 0.25),
        n_splits=2,
        output_root=str(output),
        seed=7,
        val_ratio=0.5,
        ioa_threshold=0.2,
        empty_tile_ratio=0.1,
    )
    folds = generator.run()

    assert len(folds) == 2
    for fold in (1, 2):
        data = yaml.safe_load((output / f"fold_{fold}.yaml").read_text())
        assert data["names"] == {0: "insect"}
        assert list((output / f"fold_{fold}/train/images").glob("*.jpg"))
        assert list((output / f"fold_{fold}/train/labels").glob("*.txt"))
        assert list((output / f"fold_{fold}/test/images").glob("*.jpg"))
        assert not list((output / f"fold_{fold}/train").glob(".empty_tiles_*"))


def test_modes_use_the_same_source_image_splits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_coco_dataset(source, image_count=6)

    def split_names(mode):
        generator = AsahiKFoldValidator(
            dataset_path=str(source),
            slicing_config=SlicingConfig(mode, (16, 16), 0.15),
            n_splits=3,
            output_root=str(tmp_path / mode),
            seed=42,
            val_ratio=0.25,
        )
        return [
            tuple(tuple(img["file_name"] for img in split) for split in fold)
            for fold in generator._make_splits(generator._valid_images())
        ]

    assert split_names("sahi") == split_names("asahi")
    assert split_names("sahi") == split_names("asahi_rect")


def test_rectangular_tiles_are_letterboxed_with_valid_labels(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(source / "wide.jpg"), image)
    coco = {
        "images": [{"id": 1, "file_name": "wide.jpg", "width": 64, "height": 32}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [16, 8, 8, 8], "area": 64,
        }],
        "categories": [{"id": 1, "name": "insect"}],
    }
    (source / "_annotations.coco.json").write_text(json.dumps(coco))
    generator = AsahiKFoldValidator(
        dataset_path=str(source),
        slicing_config=SlicingConfig("asahi_rect", (640, 640), 0.15),
        n_splits=2,
        output_root=str(tmp_path / "output"),
        ioa_threshold=0.2,
    )

    annotated, _, _ = generator._collect_tiles(coco["images"][0])
    rectangular_tiles = [item for item in annotated if "_tile_" in item[1]]
    assert rectangular_tiles
    for tile, _, lines in rectangular_tiles:
        assert tile.shape[:2] == (640, 640)
        for line in lines:
            _, cx, cy, width, height = map(float, line.split())
            assert 0 <= cx <= 1 and 0 <= cy <= 1
            assert 0 < width <= 1 and 0 < height <= 1
