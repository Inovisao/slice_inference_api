"""
Validates slicer geometry guarantees that the inference pipeline depends on:
  - tiles never exceed image bounds
  - uniform stride (ASAHI invariant — no border heuristics)
  - tile count matches grid formula
  - all coordinates are non-negative
"""

import math
from pathlib import Path

import numpy as np
import pytest

from config.config_loader import ConfigLoader
from slicing.asahi import Asahi
from slicing.asahi_rect import AsahiRect
from slicing.sahi import Sahi
from config.settings import SlicingConfig


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _selected_slicing_config(mode: str) -> SlicingConfig:
    loader = ConfigLoader(str(_PROJECT_ROOT / "config.yaml"))
    for process in loader.processes:
        if process.slicing.slicing_mode == mode:
            return process.slicing
    pytest.skip(f"Setup '{mode}' is not selected in config.yaml")


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


def _asahi_rect_config(overlap: float) -> SlicingConfig:
    return SlicingConfig(
        slicing_mode="asahi_rect", tile_size=(640, 640),
        overlap_ratio=overlap,
    )


def _blank_image(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


_IMAGE_SIZES_ASAHI = [(4032, 2268), (4000, 3000), (1920, 1080), (800, 600)]
# SAHI uses fixed 640×640 tiles — only test images larger than the tile in both dims
_IMAGE_SIZES_SAHI = [(4032, 2268), (4000, 3000), (1920, 1080)]
_OVERLAPS = sorted({0.1, _selected_slicing_config("asahi").overlap_ratio, 0.2, 0.3})


class TestAsahiRectGeometry:
    def test_reference_image_uses_four_by_two_grid(self):
        slicer = AsahiRect(_selected_slicing_config("asahi_rect"))
        assert slicer.compute_grid(4032, 2268) == (4, 2)
        assert slicer.compute_tile_size(4032, 2268, 4, 2) == (1136, 1226)

    def test_reference_image_redundancy_is_about_twenty_two_percent(self):
        slicer = AsahiRect(_selected_slicing_config("asahi_rect"))
        tiles = list(slicer.generate_tiles(_blank_image(4032, 2268)))
        total_area = sum(c["width"] * c["height"] for _, c in tiles)
        redundancy = (total_area - 4032 * 2268) / (4032 * 2268)
        assert len(tiles) == 8
        assert redundancy == pytest.approx(0.2184, abs=0.001)

    @pytest.mark.parametrize("img_w,img_h", _IMAGE_SIZES_ASAHI)
    @pytest.mark.parametrize("overlap", _OVERLAPS)
    def test_tiles_cover_bounds_with_uniform_stride(self, img_w, img_h, overlap):
        slicer = AsahiRect(_asahi_rect_config(overlap))
        tiles = list(slicer.generate_tiles(_blank_image(img_w, img_h)))
        xs = sorted({c["x"] for _, c in tiles})
        ys = sorted({c["y"] for _, c in tiles})
        assert min(xs) == min(ys) == 0
        for _, coords in tiles:
            assert coords["x"] + coords["width"] <= img_w
            assert coords["y"] + coords["height"] <= img_h
        for positions in (xs, ys):
            strides = [b - a for a, b in zip(positions, positions[1:])]
            if len(strides) > 1:
                assert max(strides) - min(strides) <= 1


@pytest.mark.parametrize("img_w,img_h", _IMAGE_SIZES_ASAHI)
@pytest.mark.parametrize("overlap", _OVERLAPS)
class TestAsahiGeometry:
    def test_reference_image_uses_canonical_four_by_three_grid(self, img_w, img_h, overlap):
        config = _selected_slicing_config("asahi")
        slicer = Asahi(config)
        l = config.overlap_ratio
        expected_tile = (
            math.ceil(4032 / (4 - 3 * l) + 1),
            math.ceil(2268 / (3 - 2 * l) + 1),
        )
        assert slicer.compute_grid(4032, 2268) == (4, 3)
        assert slicer.compute_tile_size(4032, 2268) == expected_tile

    def test_reference_image_redundancy_matches_canonical_asahi(self, img_w, img_h, overlap):
        config = _selected_slicing_config("asahi")
        slicer = Asahi(config)
        tiles = list(slicer.generate_tiles(_blank_image(4032, 2268)))
        total_area = sum(c["width"] * c["height"] for _, c in tiles)
        redundancy = (total_area - 4032 * 2268) / (4032 * 2268)
        tile_w, tile_h = slicer.compute_tile_size(4032, 2268)
        expected_redundancy = ((4 * 3 * tile_w * tile_h) - 4032 * 2268) / (4032 * 2268)
        assert len(tiles) == 12
        assert redundancy == pytest.approx(expected_redundancy, abs=0.001)

    def _tiles(self, img_w, img_h, overlap):
        slicer = Asahi(_asahi_config(overlap))
        image = _blank_image(img_w, img_h)
        return list(slicer.generate_tiles(image))

    def test_tile_size_does_not_exceed_image(self, img_w, img_h, overlap):
        slicer = Asahi(_asahi_config(overlap))
        tile_w, tile_h = slicer.compute_tile_size(img_w, img_h)
        assert tile_w <= img_w
        assert tile_h <= img_h

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
        a, b = slicer.compute_grid(img_w, img_h)
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
