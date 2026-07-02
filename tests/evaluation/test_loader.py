import os
import tempfile

import numpy as np
import pytest

from evaluation.loader import FoldTestLoader


@pytest.fixture
def tmp_fold(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    return tmp_path, images_dir, labels_dir


class TestLoadGtBoxes:
    def test_single_box_converted_correctly(self, tmp_fold):
        _, images_dir, labels_dir = tmp_fold
        # cx=0.5 cy=0.5 w=0.4 h=0.4 → x1=0.3 y1=0.3 x2=0.7 y2=0.7
        (labels_dir / "img.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        loader = FoldTestLoader(str(tmp_fold[0]))
        boxes = loader.load_gt_boxes("img.jpg")
        np.testing.assert_allclose(boxes[0], [0.3, 0.3, 0.7, 0.7], atol=1e-6)

    def test_multiple_boxes_returns_correct_count(self, tmp_fold):
        _, images_dir, labels_dir = tmp_fold
        (labels_dir / "img.txt").write_text(
            "0 0.2 0.2 0.2 0.2\n"
            "0 0.8 0.8 0.2 0.2\n"
        )
        loader = FoldTestLoader(str(tmp_fold[0]))
        boxes = loader.load_gt_boxes("img.jpg")
        assert boxes.shape == (2, 4)

    def test_missing_label_returns_empty_array(self, tmp_fold):
        loader = FoldTestLoader(str(tmp_fold[0]))
        boxes = loader.load_gt_boxes("nonexistent.jpg")
        assert boxes.shape == (0, 4)

    def test_empty_label_file_returns_empty_array(self, tmp_fold):
        _, images_dir, labels_dir = tmp_fold
        (labels_dir / "img.txt").write_text("")
        loader = FoldTestLoader(str(tmp_fold[0]))
        boxes = loader.load_gt_boxes("img.jpg")
        assert boxes.shape == (0, 4)

    def test_output_dtype_is_float32(self, tmp_fold):
        _, images_dir, labels_dir = tmp_fold
        (labels_dir / "img.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        loader = FoldTestLoader(str(tmp_fold[0]))
        boxes = loader.load_gt_boxes("img.jpg")
        assert boxes.dtype == np.float32

    def test_x1_less_than_x2(self, tmp_fold):
        _, images_dir, labels_dir = tmp_fold
        (labels_dir / "img.txt").write_text("0 0.5 0.5 0.4 0.3\n")
        loader = FoldTestLoader(str(tmp_fold[0]))
        boxes = loader.load_gt_boxes("img.jpg")
        assert boxes[0, 0] < boxes[0, 2]  # x1 < x2

    def test_y1_less_than_y2(self, tmp_fold):
        _, images_dir, labels_dir = tmp_fold
        (labels_dir / "img.txt").write_text("0 0.5 0.5 0.4 0.3\n")
        loader = FoldTestLoader(str(tmp_fold[0]))
        boxes = loader.load_gt_boxes("img.jpg")
        assert boxes[0, 1] < boxes[0, 3]  # y1 < y2

    def test_boxes_stay_within_unit_square(self, tmp_fold):
        _, images_dir, labels_dir = tmp_fold
        (labels_dir / "img.txt").write_text("0 0.1 0.1 0.2 0.2\n")
        loader = FoldTestLoader(str(tmp_fold[0]))
        boxes = loader.load_gt_boxes("img.jpg")
        assert np.all(boxes >= 0.0) and np.all(boxes <= 1.0)


class TestListImages:
    def test_lists_only_image_files(self, tmp_fold):
        _, images_dir, labels_dir = tmp_fold
        (images_dir / "a.jpg").write_text("")
        (images_dir / "b.png").write_text("")
        (images_dir / "c.txt").write_text("")
        loader = FoldTestLoader(str(tmp_fold[0]))
        images = loader.list_images()
        assert sorted(images) == ["a.jpg", "b.png"]

    def test_returns_sorted_list(self, tmp_fold):
        _, images_dir, _ = tmp_fold
        for name in ["z.jpg", "a.jpg", "m.jpg"]:
            (images_dir / name).write_text("")
        loader = FoldTestLoader(str(tmp_fold[0]))
        assert loader.list_images() == sorted(loader.list_images())
