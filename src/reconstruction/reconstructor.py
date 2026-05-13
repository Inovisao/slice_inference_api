from typing import Any, Dict

import cv2
import numpy as np


class ImageReconstructor:
    def __init__(self, config: Dict[str, Any], tiles_dir: str):
        self.config = config
        self.tiles_dir = tiles_dir

    def reconstruct(self, output_path: str) -> str:
        w = self.config["original_width"]
        h = self.config["original_height"]

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for tile_info in self.config["tiles"]:
            tile_path = f"{self.tiles_dir}/{tile_info['tile_file']}"
            tile = cv2.imread(tile_path)
            if tile is None:
                continue

            x, y = tile_info["x"], tile_info["y"]
            th, tw = tile.shape[:2]

            canvas[y:y + th, x:x + tw] += tile.astype(np.float32)
            weight_map[y:y + th, x:x + tw] += 1.0

        weight_map = np.maximum(weight_map, 1.0)
        reconstructed = (canvas / weight_map[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

        cv2.imwrite(output_path, reconstructed)
        return output_path
