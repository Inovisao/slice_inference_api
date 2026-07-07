import os
import random
import shutil
from pathlib import Path
from typing import List, Literal

import cv2
from fastapi import APIRouter, HTTPException
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


class CrossFoldsRequest(BaseModel):
    slicing_mode: Literal["sahi", "asahi", "asahi_rect"] = "sahi"
    n_folds: int = Field(default=5, ge=3)
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


@router.post(
    "/dataset/crossFolds",
    deprecated=True,
    description=(
        "Exploratory image-only slicing. No YOLO labels are generated; "
        "use `python main.py` to build trainable folds."
    ),
)
async def slicing_cross_folds(request: CrossFoldsRequest = CrossFoldsRequest()):
    images = validate_dataset(DATASET_PATH)

    if request.n_folds > len(images):
        raise HTTPException(
            status_code=422,
            detail={"error": f"n_folds cannot exceed image count ({len(images)})", "field": "n_folds"},
        )

    images_shuffled = images.copy()
    random.shuffle(images_shuffled)
    groups: List[List[str]] = [images_shuffled[i::request.n_folds] for i in range(request.n_folds)]
    folds_summary = []

    for k in range(request.n_folds):
        fold_name = f"fold_{k + 1}"
        fold_base = os.path.join(OUTPUT_PATH, "folds", fold_name)
        test_imgs = groups[k]
        val_imgs = groups[(k + 1) % request.n_folds]
        train_imgs = [
            img for i, group in enumerate(groups)
            if i != k and i != (k + 1) % request.n_folds
            for img in group
        ]

        splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
        fold_tiles = 0
        fold_status = "ok"

        for split_name, split_images in splits.items():
            split_dir = os.path.join(fold_base, split_name)
            originals_dir = os.path.join(split_dir, "originals")
            os.makedirs(originals_dir, exist_ok=True)

            for img_name in split_images:
                try:
                    shutil.copy2(os.path.join(DATASET_PATH, img_name), os.path.join(originals_dir, img_name))
                except OSError as e:
                    raise HTTPException(
                        status_code=500,
                        detail={"error": f"Failed to copy {img_name} to {fold_name}/{split_name}", "detail": str(e)},
                    )

                img = cv2.imread(os.path.join(DATASET_PATH, img_name))
                if img is None:
                    fold_status = f"warning: could not read {img_name}"
                    continue

                img_base = os.path.join(split_dir, Path(img_name).stem)
                tiles_path = os.path.join(img_base, "tiles")

                try:
                    slicer = make_slicer(request.slicing_mode, request.overlap_ratio)
                    metadata = slice_image(slicer, img, img_name, tiles_path)
                    save_slicing_config(img_base, img_name, metadata, slicer)
                    fold_tiles += len(metadata)
                except Exception as e:
                    fold_status = f"failed on {img_name}: {e}"

        folds_summary.append({
            "fold": fold_name,
            "slicing_mode": request.slicing_mode,
            "train_count": len(train_imgs),
            "val_count": len(val_imgs),
            "test_count": len(test_imgs),
            "total_tiles": fold_tiles,
            "status": fold_status,
        })

    return {
        "n_folds": request.n_folds,
        "slicing_mode": request.slicing_mode,
        "note": "Exploratory image-only split; no trainable labels were generated.",
        "folds": folds_summary,
    }
