from typing import Any, Dict, List, Tuple

import cv2


class SliceVisualizer:
    _BORDER_COLOR = (255, 255, 255)
    _OVERLAP_COLOR = (0, 215, 255)
    _OVERLAP_ALPHA = 0.35
    _BORDER_THICKNESS = 2

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _overlap_regions(self) -> List[Tuple[int, int, int, int]]:
        tiles = self.config["tiles"]
        regions = []
        for i, a in enumerate(tiles):
            ax1, ay1 = a["x"], a["y"]
            ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
            for b in tiles[i + 1:]:
                bx1, by1 = b["x"], b["y"]
                bx2, by2 = bx1 + b["width"], by1 + b["height"]
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                if ix1 < ix2 and iy1 < iy2:
                    regions.append((ix1, iy1, ix2, iy2))
        return regions

    def generate(self, reconstructed_path: str, output_path: str) -> str:
        img = cv2.imread(reconstructed_path)
        overlay = img.copy()

        for x1, y1, x2, y2 in self._overlap_regions():
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self._OVERLAP_COLOR, -1)

        cv2.addWeighted(overlay, self._OVERLAP_ALPHA, img, 1 - self._OVERLAP_ALPHA, 0, img)

        for tile_info in self.config["tiles"]:
            x, y = tile_info["x"], tile_info["y"]
            w, h = tile_info["width"], tile_info["height"]
            cv2.rectangle(img, (x, y), (x + w, y + h), self._BORDER_COLOR, self._BORDER_THICKNESS)

        cv2.imwrite(output_path, img)
        return output_path
