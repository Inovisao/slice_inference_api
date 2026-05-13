from typing import Generator, Tuple, Dict, List
from cv2.typing import MatLike
from pathlib import Path
import cv2
import json
import os


class Sahi:
    """Pure geometric engine. Yields tiles on demand — never touches disk."""

    def __init__(self, slicing_config):
        self.slicing_config = slicing_config

    def compute_stride(self) -> Tuple[int, int]:
        tile_w, tile_h = self.slicing_config.tile_size
        overlap = self.slicing_config.overlap_percentage
        return (
            max(1, tile_w - int(tile_w * overlap)),
            max(1, tile_h - int(tile_h * overlap)),
        )

    def _axis_positions(self, img_dim: int, tile_dim: int, stride: int) -> List[int]:
        positions = list(range(0, max(img_dim - tile_dim + 1, 1), stride))
        last = max(img_dim - tile_dim, 0)
        if positions and positions[-1] != last:
            positions.append(last)
        return sorted(set(positions)) if positions else [0]

    def generate_tiles(
        self, image: MatLike
    ) -> Generator[Tuple[MatLike, Dict], None, None]:
        tile_w, tile_h = self.slicing_config.tile_size
        stride_x, stride_y = self.compute_stride()
        img_h, img_w = image.shape[:2]

        for row, y in enumerate(self._axis_positions(img_h, tile_h, stride_y)):
            for col, x in enumerate(self._axis_positions(img_w, tile_w, stride_x)):
                tile = image[y: y + tile_h, x: x + tile_w]
                yield tile, {
                    "x": x,
                    "y": y,
                    "width": tile.shape[1],
                    "height": tile.shape[0],
                    "row_index": row,
                    "column_index": col,
                    "original_width": img_w,
                    "original_height": img_h,
                }


class SahiPipeline:
    """Orchestrates I/O: reads images from disk, writes tiles, serializes metadata."""

    _IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(self, slicer: Sahi, dataset_config):
        self.slicer = slicer
        self.data_config = dataset_config

    def _list_images(self, path: str) -> List[str]:
        return [
            f for f in os.listdir(path)
            if f.lower().endswith(self._IMAGE_EXTENSIONS)
        ]

    def slice_image(self, image: MatLike, image_name: str, output_path: str) -> List[Dict]:
        os.makedirs(output_path, exist_ok=True)

        tile_w, tile_h = self.slicer.slicing_config.tile_size
        stride_x, stride_y = self.slicer.compute_stride()
        overlap_px = [tile_w - stride_x, tile_h - stride_y]

        metadata = []
        for tile, coords in self.slicer.generate_tiles(image):
            tile_name = f"{Path(image_name).stem}_tile_{coords['x']}_{coords['y']}.jpg"
            cv2.imwrite(os.path.join(output_path, tile_name), tile)
            metadata.append({
                "source_image": image_name,
                "tile_file": tile_name,
                **coords,
                "tile_size": [tile_w, tile_h],
                "overlap_px": overlap_px,
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

        metadata_path = os.path.join(output_path, "sahi_tiles_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(all_metadata, f, indent=2)

        print(f"{len(all_metadata)} tiles saved to {output_path}")
