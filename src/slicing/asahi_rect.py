import math
from typing import Dict, Generator, List, Tuple

from cv2.typing import MatLike

from slicing.asahi import Asahi


class AsahiRect:
    """ASAHI variant with an independently sized tile on each image axis.

    The long-axis cell count comes from ASAHI. The short-axis count follows the
    image aspect ratio, then each tile dimension is solved so the requested
    overlap is achieved without forcing square tiles.
    """

    def __init__(self, slicing_config):
        self.slicing_config = slicing_config
        self.overlap = slicing_config.overlap_ratio
        self._square_reference = Asahi(slicing_config)

    def compute_grid(self, img_w: int, img_h: int) -> Tuple[int, int]:
        p = self._square_reference.compute_tile_size(img_w, img_h)
        square_cols, square_rows = self._square_reference.compute_grid(img_w, img_h, p)

        if img_w >= img_h:
            cols = square_cols
            rows = max(1, round(cols * img_h / img_w))
        else:
            rows = square_rows
            cols = max(1, round(rows * img_w / img_h))
        return cols, rows

    def compute_tile_size(
        self, img_w: int, img_h: int, cols: int, rows: int
    ) -> Tuple[int, int]:
        l = self.overlap
        tile_w = math.ceil(img_w / (cols - (cols - 1) * l))
        tile_h = math.ceil(img_h / (rows - (rows - 1) * l))
        return min(tile_w, img_w), min(tile_h, img_h)

    @staticmethod
    def _axis_positions(img_dim: int, tile_dim: int, count: int) -> List[int]:
        if count == 1:
            return [0]
        return [round(i * (img_dim - tile_dim) / (count - 1)) for i in range(count)]

    def generate_tiles(
        self, image: MatLike
    ) -> Generator[Tuple[MatLike, Dict], None, None]:
        img_h, img_w = image.shape[:2]
        cols, rows = self.compute_grid(img_w, img_h)
        tile_w, tile_h = self.compute_tile_size(img_w, img_h, cols, rows)

        xs = self._axis_positions(img_w, tile_w, cols)
        ys = self._axis_positions(img_h, tile_h, rows)
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                tile = image[y : y + tile_h, x : x + tile_w]
                yield tile, {
                    "x": x,
                    "y": y,
                    "width": tile_w,
                    "height": tile_h,
                    "row_index": row,
                    "column_index": col,
                    "original_width": img_w,
                    "original_height": img_h,
                }
