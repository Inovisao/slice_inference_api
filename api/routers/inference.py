import os
import random
from pathlib import Path
from typing import List, Literal

import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from inference.service import make_inference_pipeline

from ..helpers import DATASET_PATH, OUTPUT_PATH, load_image_rgb, next_id, validate_dataset

router = APIRouter(prefix="/inference", tags=["inference"])


class InferenceRequest(BaseModel):
    model_path: str
    slicing_mode: Literal["sahi", "asahi"] = "sahi"
    conf: float = Field(default=0.25, gt=0.0, le=1.0)
    iou_thr: float = Field(default=0.45, gt=0.0, le=1.0)
    suppression: Literal["nms", "bws", "nms_ioa", "wbf", "cluster_diou_nms"] = "wbf"
    overlap_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    device: Literal["cpu", "cuda"] = "cpu"


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
    )


@router.post("/single_image")
async def inference_single_image(request: InferenceRequest):
    images = validate_dataset(DATASET_PATH)
    img_name = random.choice(images)
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
        "detections": stats["detections"],
        "raw_detections": stats["raw_detections"],
        "duplicates_removed": stats["duplicates_removed"],
        "scores": stats["scores"],
        "output_path": out_path,
    }


@router.post("/dataset")
async def inference_dataset(request: InferenceRequest):
    images = validate_dataset(DATASET_PATH)
    results = []
    skipped: List[str] = []

    for img_name in images:
        img_path = os.path.join(DATASET_PATH, img_name)
        try:
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                skipped.append(img_name)
                continue
            if img_cv.shape[1] < 640 or img_cv.shape[0] < 640:
                skipped.append(img_name)
                continue

            id_image = next_id()
            base_path = os.path.join(OUTPUT_PATH, str(id_image))
            os.makedirs(base_path, exist_ok=True)

            out_path = os.path.join(base_path, f"{Path(img_name).stem}_resultado.jpg")
            stats = _resolve_pipeline(request).run(load_image_rgb(img_path), out_path)

            results.append({"id_image": id_image, "image_name": img_name, "detections": stats["detections"], "output_path": out_path})
        except Exception:
            skipped.append(img_name)

    return {
        "slicing_mode": request.slicing_mode,
        "model": request.model_path,
        "suppression": request.suppression,
        "processed_images": len(results),
        "skipped_images": skipped,
        "total_detections": sum(r["detections"] for r in results),
        "results": results,
    }
