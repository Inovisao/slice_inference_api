from typing import Dict, Generator, Tuple

from cv2.typing import MatLike


class NoSlice:
    """Baseline mode: no tiling, the engine's full-image pass is the only inference."""

    def __init__(self, slicing_config):
        self.slicing_config = slicing_config

    def generate_tiles(self, image: MatLike) -> Generator[Tuple[MatLike, Dict], None, None]:
        return
        yield
