import json
import os
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from inference.service import make_pipeline_from_config
from reconstruction.reconstructor import ImageReconstructor
from reconstruction.visualizer import SliceVisualizer

from ..helpers import DATASET_PATH, MODELS_PATH, OUTPUT_PATH, list_images, load_image_rgb

router = APIRouter(prefix="/reconstruct", tags=["reconstruct"])


class ReconstructRequest(BaseModel):
    id_image: int


class ReconstructValidateRequest(BaseModel):
    id_image: int
    model_path: str | None = None
    suppression: Literal["nms", "bws", "nms_ioa", "wbf", "cluster_diou_nms"] = "wbf"
    conf: float = Field(default=0.25, gt=0.0, le=1.0)
    iou_thr: float = Field(default=0.45, gt=0.0, le=1.0)
    device: Literal["cpu", "cuda"] = "cpu"


def _load_slicing_config(id_image: int) -> dict:
    base_path = os.path.join(OUTPUT_PATH, str(id_image))
    if not os.path.exists(base_path):
        raise HTTPException(status_code=404, detail={"error": f"id_image {id_image} not found", "path": base_path})
    config_path = os.path.join(base_path, "slicing_config.json")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail={"error": f"slicing_config.json not found for id_image {id_image}"})
    with open(config_path) as f:
        return json.load(f)


@router.post("/single_image")
async def reconstruct_single_image(request: ReconstructRequest):
    config = _load_slicing_config(request.id_image)
    base_path = os.path.join(OUTPUT_PATH, str(request.id_image))
    tiles_dir = os.path.join(base_path, "tiles")

    if not os.path.exists(tiles_dir) or not list_images(tiles_dir):
        raise HTTPException(status_code=422, detail={"error": f"No tiles found for id_image {request.id_image}"})

    reconstructed_path = os.path.join(base_path, "reconstructed.jpg")
    informative_path = os.path.join(base_path, "reconstructed_info.jpg")

    ImageReconstructor(config, tiles_dir).reconstruct(reconstructed_path)
    SliceVisualizer(config).generate(reconstructed_path, informative_path)

    return {
        "id_image": request.id_image,
        "source_image": config.get("source_image"),
        "reconstructed": reconstructed_path,
        "informative": informative_path,
    }


@router.post("/validate")
async def reconstruct_validate(request: ReconstructValidateRequest):
    config = _load_slicing_config(request.id_image)
    base_path = os.path.join(OUTPUT_PATH, str(request.id_image))

    source_image = config["source_image"]
    img_path = os.path.join(DATASET_PATH, source_image)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail={"error": f"Source image not found: {source_image}"})

    model_path = request.model_path or os.path.join(MODELS_PATH, "best.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail={"error": f"Model not found: {model_path}"})

    pipeline = make_pipeline_from_config(
        config=config,
        model_path=model_path,
        suppression=request.suppression,
        conf_thr=request.conf,
        iou_thr=request.iou_thr,
        device=request.device,
    )

    out_path = os.path.join(base_path, f"detections_{request.suppression}.jpg")
    stats = pipeline.run(load_image_rgb(img_path), out_path)

    return {
        "id_image": request.id_image,
        "source_image": source_image,
        "suppression": request.suppression,
        "tile_count": config["tile_count"],
        "raw_detections": stats["raw_detections"],
        "detections": stats["detections"],
        "duplicates_removed": stats["duplicates_removed"],
        "scores": stats["scores"],
        "output_path": out_path,
    }
