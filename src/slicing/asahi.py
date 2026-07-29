import json
import math
import os
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import cv2
from cv2.typing import MatLike


class Asahi:
    """
    ASAHI fixes the number of slices (3x2 or 4x3, oriented along the image
    aspect ratio) and computes axis-dependent slice dimensions.

    The paper notation is kept close to the implementation:
      - mu: overlap ratio
      - r: limiting dimension, fixed at 640 in this experiment
      - T: resolution-dependent threshold, Eq. (1)
      - p: adaptive slice size, Eq. (2)
      - ncols/nrows: selected 3x2 or 4x3 slice grid, Algorithm 1
    """

    LIMITING_DIMENSION = 640

    def __init__(self, slicing_config):
        self.slicing_config = slicing_config
        self.overlap_ratio = slicing_config.overlap_ratio
        self.overlap = self.overlap_ratio

    def compute_resolution_dependent_threshold(self) -> float:
        """ASAHI Eq. (1): T = r * (4 - 3 * mu) + 1."""
        mu = self.overlap_ratio
        r = self.LIMITING_DIMENSION
        return r * (4 - 3 * mu) + 1

    def select_slice_grid(self, img_w: int, img_h: int) -> Tuple[int, int]:
        """ASAHI Algorithm 1: select 6 slices (3x2) or 12 slices (4x3)."""
        threshold_t = self.compute_resolution_dependent_threshold()
        uses_twelve_slices = max(img_w, img_h) > threshold_t
        long_axis_slices, short_axis_slices = (4, 3) if uses_twelve_slices else (3, 2)

        if img_w >= img_h:
            return long_axis_slices, short_axis_slices
        return short_axis_slices, long_axis_slices

    def compute_grid(self, img_w: int, img_h: int, _p: object = None) -> Tuple[int, int]:
        return self.select_slice_grid(img_w, img_h)

    def compute_long_short_slice_lengths(self, img_w: int, img_h: int) -> Tuple[int, int]:
        """ASAHI Eq. (3): derive llong and lshort from the selected slice count."""
        mu = self.overlap_ratio
        ncols, nrows = self.select_slice_grid(img_w, img_h)
        long_axis = max(img_w, img_h)
        short_axis = min(img_w, img_h)
        long_axis_slices = max(ncols, nrows)
        short_axis_slices = min(ncols, nrows)
        llong = math.ceil(long_axis / (long_axis_slices - (long_axis_slices - 1) * mu) + 1)
        lshort = math.ceil(short_axis / (short_axis_slices - (short_axis_slices - 1) * mu) + 1)
        return min(llong, long_axis), min(lshort, short_axis)

    def compute_tile_size(self, img_w: int, img_h: int) -> Tuple[int, int]:
        llong, lshort = self.compute_long_short_slice_lengths(img_w, img_h)
        if img_w >= img_h:
            slice_w, slice_h = llong, lshort
        else:
            slice_w, slice_h = lshort, llong
        return slice_w, slice_h

    def compute_adaptive_slice_size_p(self, img_w: int, img_h: int) -> int:
        """ASAHI Eq. (2): p is the larger of llong and lshort."""
        llong, lshort = self.compute_long_short_slice_lengths(img_w, img_h)
        return max(llong, lshort)

    def compute_reference_size(self, img_w: int, img_h: int) -> int:
        return self.compute_adaptive_slice_size_p(img_w, img_h)

    def _axis_positions(self, img_dim: int, tile_dim: int, n: int) -> List[int]:
        if n == 1:
            return [0]
        return [round(i * (img_dim - tile_dim) / (n - 1)) for i in range(n)]

    def generate_tiles(self, image: MatLike) -> Generator[Tuple[MatLike, Dict], None, None]:
        img_h, img_w = image.shape[:2]
        slice_w, slice_h = self.compute_tile_size(img_w, img_h)
        assert slice_w <= img_w and slice_h <= img_h, (
            f"Slice size {slice_w}x{slice_h} exceeds image dimensions ({img_w}x{img_h})"
        )
        ncols, nrows = self.select_slice_grid(img_w, img_h)

        for row, y in enumerate(self._axis_positions(img_h, slice_h, nrows)):
            for col, x in enumerate(self._axis_positions(img_w, slice_w, ncols)):
                tile = image[y: y + slice_h, x: x + slice_w]
                yield tile, {
                    "x": x,
                    "y": y,
                    "width": slice_w,
                    "height": slice_h,
                    "row_index": row,
                    "column_index": col,
                    "original_width": img_w,
                    "original_height": img_h,
                }


class AsahiPipeline:
    """Orchestrates I/O: reads images from disk, writes tiles, serializes metadata."""

    _IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(self, slicer: Asahi, dataset_config):
        self.slicer = slicer
        self.data_config = dataset_config

    def _list_images(self, path: str) -> List[str]:
        return [
            f for f in os.listdir(path)
            if f.lower().endswith(self._IMAGE_EXTENSIONS)
        ]

    def slice_image(self, image: MatLike, image_name: str, output_path: str) -> List[Dict]:
        os.makedirs(output_path, exist_ok=True)
        metadata = []
        for tile, coords in self.slicer.generate_tiles(image):
            tile_name = f"{Path(image_name).stem}_tile_{coords['x']}_{coords['y']}.jpg"
            cv2.imwrite(os.path.join(output_path, tile_name), tile)
            metadata.append({
                "source_image": image_name,
                "tile_file": tile_name,
                **coords,
                "overlap_ratio": self.slicer.overlap,
            })
        return metadata

    def apply_slicing(self):
        dataset_path = self.data_config.input_path
        output_path = self.data_config.output_path

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        image_files = self._list_images(dataset_path)
        all_metadata = []

        for i, img_name in enumerate(image_files, 1):
            img_path = os.path.join(dataset_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            metadata = self.slice_image(img, img_name, output_path)
            all_metadata.extend(metadata)
            print(f"[{i}/{len(image_files)}] {img_path} → {len(metadata)} tiles")

        metadata_path = os.path.join(output_path, "asahi_tiles_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(all_metadata, f, indent=2)

        print(f"{len(all_metadata)} tiles saved to {output_path}")
