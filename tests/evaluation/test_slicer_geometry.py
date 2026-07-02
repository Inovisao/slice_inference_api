"""
Validates slicer geometry guarantees that the inference pipeline depends on:
  - tiles never exceed image bounds
  - uniform stride (ASAHI invariant — no border heuristics)
  - tile count matches grid formula
  - all coordinates are non-negative
"""

import math

import numpy as np
import pytest

from slicing.asahi import Asahi
from slicing.sahi import Sahi
from config.settings import SlicingConfig


def _asahi_config(overlap: float) -> SlicingConfig:
    return SlicingConfig(
        slicing_mode="asahi", tile_size=(640, 640),
        overlap_ratio=overlap,
    )


def _sahi_config(overlap: float) -> SlicingConfig:
    return SlicingConfig(
        slicing_mode="sahi", tile_size=(640, 640),
        overlap_ratio=overlap,
    )


def _blank_image(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


_IMAGE_SIZES_ASAHI = [(4032, 2268), (4000, 3000), (1920, 1080), (800, 600)]
# SAHI uses fixed 640×640 tiles — only test images larger than the tile in both dims
_IMAGE_SIZES_SAHI = [(4032, 2268), (4000, 3000), (1920, 1080)]
_OVERLAPS = [0.1, 0.15, 0.2, 0.3]


@pytest.mark.parametrize("img_w,img_h", _IMAGE_SIZES_ASAHI)
@pytest.mark.parametrize("overlap", _OVERLAPS)
class TestAsahiGeometry:
    def _tiles(self, img_w, img_h, overlap):
        slicer = Asahi(_asahi_config(overlap))
        image = _blank_image(img_w, img_h)
        return list(slicer.generate_tiles(image))

    def test_tile_size_does_not_exceed_image(self, img_w, img_h, overlap):
        slicer = Asahi(_asahi_config(overlap))
        p = slicer.compute_tile_size(img_w, img_h)
        assert p <= min(img_w, img_h), f"p={p} exceeds min({img_w},{img_h})"

    def test_no_tile_exceeds_image_bounds(self, img_w, img_h, overlap):
        for _, coords in self._tiles(img_w, img_h, overlap):
            assert coords["x"] + coords["width"] <= img_w + 1, \
                f"Tile exceeds image width: x={coords['x']} w={coords['width']} img_w={img_w}"
            assert coords["y"] + coords["height"] <= img_h + 1, \
                f"Tile exceeds image height: y={coords['y']} h={coords['height']} img_h={img_h}"

    def test_all_coordinates_non_negative(self, img_w, img_h, overlap):
        for _, coords in self._tiles(img_w, img_h, overlap):
            assert coords["x"] >= 0 and coords["y"] >= 0

    def test_tile_count_matches_grid_formula(self, img_w, img_h, overlap):
        slicer = Asahi(_asahi_config(overlap))
        p = slicer.compute_tile_size(img_w, img_h)
        a, b = slicer.compute_grid(img_w, img_h, p)
        tiles = self._tiles(img_w, img_h, overlap)
        assert len(tiles) == a * b

    def test_stride_is_uniform_per_axis(self, img_w, img_h, overlap):
        slicer = Asahi(_asahi_config(overlap))
        image = _blank_image(img_w, img_h)
        tiles = list(slicer.generate_tiles(image))

        xs = sorted(set(c["x"] for _, c in tiles))
        ys = sorted(set(c["y"] for _, c in tiles))

        if len(xs) > 2:
            strides_x = [xs[i+1] - xs[i] for i in range(len(xs) - 1)]
            # All strides must be within 1px of each other (float rounding)
            assert max(strides_x) - min(strides_x) <= 1, \
                f"Non-uniform x strides: {strides_x}"

        if len(ys) > 2:
            strides_y = [ys[i+1] - ys[i] for i in range(len(ys) - 1)]
            assert max(strides_y) - min(strides_y) <= 1, \
                f"Non-uniform y strides: {strides_y}"

    def test_first_tile_starts_at_origin(self, img_w, img_h, overlap):
        tiles = self._tiles(img_w, img_h, overlap)
        xs = [c["x"] for _, c in tiles]
        ys = [c["y"] for _, c in tiles]
        assert min(xs) == 0
        assert min(ys) == 0


@pytest.mark.parametrize("img_w,img_h", _IMAGE_SIZES_SAHI)
@pytest.mark.parametrize("overlap", _OVERLAPS)
class TestSahiGeometry:
    def _tiles(self, img_w, img_h, overlap):
        slicer = Sahi(_sahi_config(overlap))
        image = _blank_image(img_w, img_h)
        return list(slicer.generate_tiles(image))

    def test_all_tiles_are_exactly_tile_size(self, img_w, img_h, overlap):
        for tile, coords in self._tiles(img_w, img_h, overlap):
            assert coords["width"] == 640
            assert coords["height"] == 640

    def test_no_tile_exceeds_image_bounds(self, img_w, img_h, overlap):
        for _, coords in self._tiles(img_w, img_h, overlap):
            assert coords["x"] + coords["width"] <= img_w + 1
            assert coords["y"] + coords["height"] <= img_h + 1

    def test_all_coordinates_non_negative(self, img_w, img_h, overlap):
        for _, coords in self._tiles(img_w, img_h, overlap):
            assert coords["x"] >= 0 and coords["y"] >= 0

    def test_first_tile_starts_at_origin(self, img_w, img_h, overlap):
        tiles = self._tiles(img_w, img_h, overlap)
        assert tiles[0][1]["x"] == 0
        assert tiles[0][1]["y"] == 0
