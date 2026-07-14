import os
import random
from typing import List, Literal

import cv2
from fastapi import APIRouter
from pydantic import BaseModel, Field

from slicing.service import make_slicer, slice_image, save_slicing_config

from ..helpers import (
    DATASET_PATH,
    OUTPUT_PATH,
    next_id,
    read_image_or_raise,
    validate_dataset,
    validate_image_size,
)

router = APIRouter(prefix="/slicing", tags=["slicing"])


class SlicingRequest(BaseModel):
    slicing_mode: Literal["sahi", "asahi", "asahi_rect"] = "sahi"
    overlap_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)


@router.post("/single_image")
async def slicing_single_image(request: SlicingRequest = SlicingRequest()):
    images = validate_dataset(DATASET_PATH)
    img_name = random.choice(images)
    img_path = os.path.join(DATASET_PATH, img_name)

    img = read_image_or_raise(img_path, img_name)
    validate_image_size(img, (640, 640), img_name)

    id_image = next_id()
    base_path = os.path.join(OUTPUT_PATH, str(id_image))
    tiles_path = os.path.join(base_path, "tiles")

    slicer = make_slicer(request.slicing_mode, request.overlap_ratio)
    metadata = slice_image(slicer, img, img_name, tiles_path)
    save_slicing_config(base_path, img_name, metadata, slicer, id_image=id_image)

    return {
        "id_image": id_image,
        "slicing_mode": request.slicing_mode,
        "image_name": img_name,
        "tile_count": len(metadata),
        "output_path": base_path,
    }


@router.post("/dataset")
async def slicing_dataset(request: SlicingRequest = SlicingRequest()):
    images = validate_dataset(DATASET_PATH)
    skipped: List[str] = []
    results = []

    for img_name in images:
        img_path = os.path.join(DATASET_PATH, img_name)
        img = cv2.imread(img_path)
        if img is None:
            skipped.append(img_name)
            continue

        id_image = next_id()
        base_path = os.path.join(OUTPUT_PATH, str(id_image))
        tiles_path = os.path.join(base_path, "tiles")

        slicer = make_slicer(request.slicing_mode, request.overlap_ratio)
        metadata = slice_image(slicer, img, img_name, tiles_path)
        save_slicing_config(base_path, img_name, metadata, slicer, id_image=id_image)

        results.append({
            "id_image": id_image,
            "image_name": img_name,
            "tile_count": len(metadata),
            "output_path": base_path,
        })

    return {
        "slicing_mode": request.slicing_mode,
        "processed_images": len(results),
        "skipped_images": skipped,
        "total_tiles": sum(r["tile_count"] for r in results),
        "results": results,
    }
