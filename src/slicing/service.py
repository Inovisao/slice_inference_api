import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from config.settings import SlicingConfig
from slicing.sahi import Sahi
from slicing.asahi import Asahi


def make_slicer(slicing_mode: str, overlap_ratio: float):
    slicing_config = SlicingConfig(
        slicing_mode=slicing_mode,
        tile_size=(640, 640),
        overlap_ratio=overlap_ratio,
        min_object_coverage=0.5,
    )
    return Asahi(slicing_config) if slicing_mode == "asahi" else Sahi(slicing_config)


def slice_image(slicer, image: cv2.typing.MatLike, image_name: str, output_path: str) -> List[Dict]:
    os.makedirs(output_path, exist_ok=True)
    metadata = []
    for tile, coords in slicer.generate_tiles(image):
        tile_name = f"{Path(image_name).stem}_tile_{coords['x']}_{coords['y']}.jpg"
        cv2.imwrite(os.path.join(output_path, tile_name), tile)
        metadata.append({"source_image": image_name, "tile_file": tile_name, **coords})
    return metadata


def _edge_coverage(metadata: List[Dict], tile_w: int, tile_h: int, stride_x: int, stride_y: int) -> Dict:
    xs = sorted(set(m["x"] for m in metadata))
    ys = sorted(set(m["y"] for m in metadata))

    def _axis(positions, tile_dim, nominal_overlap):
        if len(positions) < 2:
            return {}
        gap = positions[-1] - positions[-2]
        actual_overlap = tile_dim - gap
        return {
            "last_regular_position": positions[-2],
            "anchor_position": positions[-1],
            "gap_px": gap,
            "nominal_overlap_px": nominal_overlap,
            "actual_overlap_px": actual_overlap,
            "extra_overlap_px": actual_overlap - nominal_overlap,
            "has_extra": actual_overlap > nominal_overlap,
        }

    return {
        "x": _axis(xs, tile_w, tile_w - stride_x),
        "y": _axis(ys, tile_h, tile_h - stride_y),
    }


def save_slicing_config(
    base_path: str,
    source_image: str,
    metadata: List[Dict],
    slicer,
    id_image: Optional[int] = None,
):
    slicing_mode = slicer.slicing_config.slicing_mode
    config: Dict[str, Any] = {
        "slicing_method": slicing_mode,
        "source_image": source_image,
        "original_width": metadata[0]["original_width"] if metadata else 0,
        "original_height": metadata[0]["original_height"] if metadata else 0,
        "tile_count": len(metadata),
        "tiles": [
            {"tile_file": m["tile_file"], "x": m["x"], "y": m["y"], "width": m["width"], "height": m["height"]}
            for m in metadata
        ],
    }

    if slicing_mode == "asahi":
        p = metadata[0]["width"] if metadata else 0
        config.update({
            "tile_size": [p, p],
            "overlap_ratio": slicer.overlap,
            "grid_cols": metadata[-1]["column_index"] + 1 if metadata else 0,
            "grid_rows": metadata[-1]["row_index"] + 1 if metadata else 0,
        })
    else:
        tile_w, tile_h = slicer.slicing_config.tile_size
        stride_x, stride_y = slicer.compute_stride()
        config.update({
            "tile_size": [tile_w, tile_h],
            "stride_x": stride_x,
            "stride_y": stride_y,
            "overlap_x": tile_w - stride_x,
            "overlap_y": tile_h - stride_y,
            "edge_coverage": _edge_coverage(metadata, tile_w, tile_h, stride_x, stride_y),
        })

    if id_image is not None:
        config["id_image"] = id_image

    with open(os.path.join(base_path, "slicing_config.json"), "w") as f:
        json.dump(config, f, indent=2)
