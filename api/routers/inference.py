import os
import random
from pathlib import Path
from typing import Literal

import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from inference.service import make_inference_pipeline

from ..helpers import DATASET_PATH, OUTPUT_PATH, load_image_rgb, next_id, validate_dataset

router = APIRouter(prefix="/inference", tags=["inference"])


class InferenceRequest(BaseModel):
    model_path: str
    image_name: str | None = None
    slicing_mode: Literal["sahi", "asahi", "asahi_rect"] = "sahi"
    conf: float = Field(default=0.25, gt=0.0, le=1.0)
    iou_thr: float = Field(default=0.45, gt=0.0, le=1.0)
    suppression: Literal["nms", "bws", "nms_ioa", "wbf", "cluster_diou_nms"] = "wbf"
    overlap_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    device: Literal["cpu", "cuda"] = "cpu"
    include_full_image: bool | None = None
    batch_size: int = Field(default=32, ge=1)


def _resolve_pipeline(request: InferenceRequest):
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail={"error": "Model file not found", "path": request.model_path})
    return make_inference_pipeline(
        model_path=request.model_path,
        slicing_mode=request.slicing_mode,
        overlap_ratio=request.overlap_ratio,
        suppression=request.suppression,
        conf_thr=request.conf,
        iou_thr=request.iou_thr,
        device=request.device,
        include_full_inference=request.include_full_image,
        batch_size=request.batch_size,
    )


def _select_image(requested: str | None, images: list[str]) -> str:
    if requested is None:
        return random.choice(images)
    if Path(requested).name != requested or requested not in images:
        raise HTTPException(
            status_code=404,
            detail={"error": "Image not found in configured dataset", "image": requested},
        )
    return requested


@router.post("/single_image")
async def inference_single_image(request: InferenceRequest):
    images = validate_dataset(DATASET_PATH)
    img_name = _select_image(request.image_name, images)
    img_path = os.path.join(DATASET_PATH, img_name)

    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise HTTPException(status_code=500, detail={"error": f"Failed to read image: {img_name}"})

    id_image = next_id()
    base_path = os.path.join(OUTPUT_PATH, str(id_image))
    os.makedirs(base_path, exist_ok=True)

    out_path = os.path.join(base_path, f"{Path(img_name).stem}_resultado.jpg")
    stats = _resolve_pipeline(request).run(load_image_rgb(img_path), out_path)

    return {
        "id_image": id_image,
        "image_name": img_name,
        "slicing_mode": request.slicing_mode,
        "model": request.model_path,
        "suppression": request.suppression,
        "conf_threshold": request.conf,
        "include_full_image": (
            request.slicing_mode in ("asahi", "asahi_rect")
            if request.include_full_image is None
            else request.include_full_image
        ),
        "detections": stats["detections"],
        "raw_detections": stats["raw_detections"],
        "duplicates_removed": stats["duplicates_removed"],
        "scores": stats["scores"],
        "output_path": out_path,
    }


@router.post("/dataset")
async def inference_dataset(request: InferenceRequest):
    images = validate_dataset(DATASET_PATH)
    pipeline = _resolve_pipeline(request)
    results = []
    failures: list[dict] = []

    for img_name in images:
        img_path = os.path.join(DATASET_PATH, img_name)
        try:
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                failures.append({"image_name": img_name, "error": "Failed to read image"})
                continue

            id_image = next_id()
            base_path = os.path.join(OUTPUT_PATH, str(id_image))
            os.makedirs(base_path, exist_ok=True)

            out_path = os.path.join(base_path, f"{Path(img_name).stem}_resultado.jpg")
            stats = pipeline.run(load_image_rgb(img_path), out_path)

            results.append({
                "id_image": id_image,
                "image_name": img_name,
                "detections": stats["detections"],
                "raw_detections": stats["raw_detections"],
                "duplicates_removed": stats["duplicates_removed"],
                "output_path": out_path,
            })
        except Exception as exc:
            failures.append({
                "image_name": img_name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "slicing_mode": request.slicing_mode,
        "model": request.model_path,
        "suppression": request.suppression,
        "processed_images": len(results),
        "failed_images": failures,
        "total_detections": sum(r["detections"] for r in results),
        "results": results,
    }
