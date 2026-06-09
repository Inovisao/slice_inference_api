import json
import math
import os
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import cv2
from cv2.typing import MatLike


class Asahi:
    """
    Adaptive tile size derived from image dimensions (Equations 1-4 from the ASAHI paper).
    Tile p floats freely (e.g. 1187px for 4032×2268). The network downscale p→640 is
    handled by the inference engine (imgsz=640), not here.
    RESTRICT_SIZE=640 is used only as the branch threshold in the tile-size formula (ls).
    """

    RESTRICT_SIZE = 640

    def __init__(self, slicing_config):
        self.slicing_config = slicing_config
        self.overlap = slicing_config.overlap_ratio

    def compute_tile_size(self, img_w: int, img_h: int) -> int:
        l = self.overlap
        ls = self.RESTRICT_SIZE * (4 - 3 * l) + 1
        if max(img_w, img_h) <= ls:
            p = max(img_w / (3 - 2 * l) + 1, img_h / (2 - l) + 1)
        else:
            p = max(img_w / (4 - 3 * l) + 1, img_h / (3 - 2 * l) + 1)
        return math.ceil(p)

    def compute_grid(self, img_w: int, img_h: int, p: int) -> Tuple[int, int]:
        l = self.overlap
        a = max(1, math.ceil((img_w - p * l) / (p * (1 - l))))
        b = max(1, math.ceil((img_h - p * l) / (p * (1 - l))))
        return a, b

    def _axis_positions(self, img_dim: int, p: int, n: int) -> List[int]:
        if n == 1:
            return [0]
        return [round(i * (img_dim - p) / (n - 1)) for i in range(n)]

    def generate_tiles(self, image: MatLike) -> Generator[Tuple[MatLike, Dict], None, None]:
        img_h, img_w = image.shape[:2]
        p = self.compute_tile_size(img_w, img_h)
        assert p <= min(img_w, img_h), (
            f"Slice size {p} exceeds image dimensions ({img_w}×{img_h})"
        )
        a, b = self.compute_grid(img_w, img_h, p)

        for row, y in enumerate(self._axis_positions(img_h, p, b)):
            for col, x in enumerate(self._axis_positions(img_w, p, a)):
                tile = image[y: y + p, x: x + p]
                yield tile, {
                    "x": x,
                    "y": y,
                    "width": p,
                    "height": p,
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
