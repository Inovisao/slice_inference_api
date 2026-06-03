import math
from typing import Generator, Tuple, Dict, List

from cv2.typing import MatLike


class Asahi:
    """
    Adaptive tile size derived from image dimensions (Equations 1-4 from the ASAHI paper).
    Tile p is computed so that a uniform a×b grid covers the full image with overlap l.
    Positions are distributed uniformly — no border heuristics.
    """

    RESTRICT_SIZE = 640

    def __init__(self, slicing_config):
        self.slicing_config = slicing_config
        self.overlap = slicing_config.overlap_percentage

    def compute_tile_size(self, img_w: int, img_h: int) -> int:
        l = self.overlap
        ls = self.RESTRICT_SIZE * (4 - 3 * l) + 1
        if max(img_w, img_h) <= ls:
            p = max(img_w / (3 - 2 * l) + 1, img_h / (2 - l) + 1)
        else:
            p = max(img_w / (4 - 3 * l) + 1, img_h / (3 - 2 * l) + 1)
        return int(p)

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
