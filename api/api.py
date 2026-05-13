import json
import os
import random
import shutil
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import DatasetConfig, SlicingConfig
from slicing.sahi import Sahi, SahiPipeline
from reconstruction.reconstructor import ImageReconstructor
from reconstruction.visualizer import SliceVisualizer

app = FastAPI()

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DATASET_PATH = "./dataset"
OUTPUT_PATH = "./output"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# --- helpers ---

def _list_images(path: str) -> List[str]:
    return [
        f for f in os.listdir(path)
        if f.lower().endswith(_IMAGE_EXTENSIONS)
    ]


def _validate_dataset(path: str) -> List[str]:
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail={"error": "Dataset directory not found", "path": path},
        )
    images = _list_images(path)
    if not images:
        raise HTTPException(
            status_code=422,
            detail={"error": "No valid images found in dataset", "path": path},
        )
    return images


def _next_id() -> int:
    if not os.path.exists(OUTPUT_PATH):
        return 1
    existing = [
        int(d) for d in os.listdir(OUTPUT_PATH)
        if d.isdigit() and os.path.isdir(os.path.join(OUTPUT_PATH, d))
    ]
    return max(existing, default=0) + 1


class SahiRequest(BaseModel):
    overlap_percentage: float = Field(default=0.15, gt=0.0, lt=1.0)


class CrossFoldsRequest(BaseModel):
    n_folds: int = Field(default=5, ge=3)
    overlap_percentage: float = Field(default=0.15, gt=0.0, lt=1.0)


def _make_sahi_pipeline(input_path: str, output_path: str, overlap_percentage: float = 0.15) -> SahiPipeline:
    slicing_config = SlicingConfig(
        slicing_mode="sahi",
        tile_size=(640, 640),
        overlap_percentage=overlap_percentage,
        min_object_coverage=0.5,
    )
    dataset_config = DatasetConfig(
        input_path=input_path,
        output_path=output_path,
    )
    return SahiPipeline(Sahi(slicing_config), dataset_config)


def _read_image_or_raise(img_path: str, img_name: str) -> cv2.typing.MatLike:
    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to read image: {img_name}"},
        )
    return img


def _validate_image_size(img: cv2.typing.MatLike, tile_size: tuple, img_name: str):
    img_h, img_w = img.shape[:2]
    tile_w, tile_h = tile_size
    if img_w < tile_w or img_h < tile_h:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Image is smaller than tile_size",
                "image": img_name,
                "image_size": [img_w, img_h],
                "tile_size": [tile_w, tile_h],
            },
        )


def _edge_coverage(
    metadata: List[Dict[str, Any]],
    tile_w: int,
    tile_h: int,
    stride_x: int,
    stride_y: int,
) -> Dict[str, Any]:
    xs = sorted(set(m["x"] for m in metadata))
    ys = sorted(set(m["y"] for m in metadata))
    nominal_ox = tile_w - stride_x
    nominal_oy = tile_h - stride_y

    def _axis(positions: List[int], tile_dim: int, nominal_overlap: int) -> Dict[str, Any]:
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
        "x": _axis(xs, tile_w, nominal_ox),
        "y": _axis(ys, tile_h, nominal_oy),
    }


def _save_slicing_config(
    base_path: str,
    source_image: str,
    metadata: List[Dict[str, Any]],
    pipeline: SahiPipeline,
    id_image: Optional[int] = None,
):
    tile_w, tile_h = pipeline.slicer.slicing_config.tile_size
    stride_x, stride_y = pipeline.slicer.compute_stride()
    nominal_ox, nominal_oy = tile_w - stride_x, tile_h - stride_y

    config: Dict[str, Any] = {
        "slicing_method": pipeline.slicer.slicing_config.slicing_mode,
        "source_image": source_image,
        "original_width": metadata[0]["original_width"] if metadata else 0,
        "original_height": metadata[0]["original_height"] if metadata else 0,
        "tile_size": [tile_w, tile_h],
        "stride_x": stride_x,
        "stride_y": stride_y,
        "overlap_x": nominal_ox,
        "overlap_y": nominal_oy,
        "tile_count": len(metadata),
        "edge_coverage": _edge_coverage(metadata, tile_w, tile_h, stride_x, stride_y),
        "tiles": [
            {
                "tile_file": m["tile_file"],
                "x": m["x"],
                "y": m["y"],
                "width": m["width"],
                "height": m["height"],
            }
            for m in metadata
        ],
    }

    if id_image is not None:
        config["id_image"] = id_image

    with open(os.path.join(base_path, "slicing_config.json"), "w") as f:
        json.dump(config, f, indent=2)


# --- endpoints ---

@app.get("/")
async def health_check():
    return {"status": "API is running"}


@app.post("/sahi/single_image")
async def sahi_single_image(request: SahiRequest = SahiRequest()):
    images = _validate_dataset(DATASET_PATH)

    img_name = random.choice(images)
    img_path = os.path.join(DATASET_PATH, img_name)

    img = _read_image_or_raise(img_path, img_name)
    _validate_image_size(img, (640, 640), img_name)

    id_image = _next_id()
    base_path = os.path.join(OUTPUT_PATH, str(id_image))
    tiles_path = os.path.join(base_path, "tiles")

    pipeline = _make_sahi_pipeline(DATASET_PATH, tiles_path, request.overlap_percentage)
    metadata = pipeline.slice_image(img, img_name, tiles_path)
    _save_slicing_config(base_path, img_name, metadata, pipeline, id_image=id_image)

    tile_w, tile_h = pipeline.slicer.slicing_config.tile_size
    stride_x, stride_y = pipeline.slicer.compute_stride()

    return {
        "id_image": id_image,
        "image_name": img_name,
        "tile_count": len(metadata),
        "overlap_px": {"x": tile_w - stride_x, "y": tile_h - stride_y},
        "stride_px": {"x": stride_x, "y": stride_y},
        "output_path": base_path,
    }


@app.post("/sahi/dataset")
async def sahi_dataset(request: SahiRequest = SahiRequest()):
    images = _validate_dataset(DATASET_PATH)

    skipped: List[str] = []
    results = []

    for img_name in images:
        img_path = os.path.join(DATASET_PATH, img_name)
        img = cv2.imread(img_path)

        if img is None:
            skipped.append(img_name)
            continue

        id_image = _next_id()
        base_path = os.path.join(OUTPUT_PATH, str(id_image))
        tiles_path = os.path.join(base_path, "tiles")

        pipeline = _make_sahi_pipeline(DATASET_PATH, tiles_path, request.overlap_percentage)
        metadata = pipeline.slice_image(img, img_name, tiles_path)
        _save_slicing_config(base_path, img_name, metadata, pipeline, id_image=id_image)

        results.append({
            "id_image": id_image,
            "image_name": img_name,
            "tile_count": len(metadata),
            "output_path": base_path,
        })

    return {
        "processed_images": len(results),
        "skipped_images": skipped,
        "total_tiles": sum(r["tile_count"] for r in results),
        "results": results,
    }


@app.post("/sahi/dataset/crossFolds")
async def sahi_cross_folds(request: CrossFoldsRequest = CrossFoldsRequest()):
    n_folds = request.n_folds
    images = _validate_dataset(DATASET_PATH)

    if n_folds > len(images):
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"n_folds cannot exceed image count ({len(images)})",
                "field": "n_folds",
            },
        )

    if n_folds < 3:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "n_folds must be >= 3 to guarantee non-empty train/val/test splits",
                "field": "n_folds",
            },
        )

    images_shuffled = images.copy()
    random.shuffle(images_shuffled)

    groups: List[List[str]] = [images_shuffled[i::n_folds] for i in range(n_folds)]

    folds_summary = []

    for k in range(n_folds):
        fold_name = f"fold_{k + 1}"
        fold_base = os.path.join(OUTPUT_PATH, "folds", fold_name)

        test_imgs = groups[k]
        val_imgs = groups[(k + 1) % n_folds]
        train_imgs = [
            img
            for i, group in enumerate(groups)
            if i != k and i != (k + 1) % n_folds
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
                    shutil.copy2(
                        os.path.join(DATASET_PATH, img_name),
                        os.path.join(originals_dir, img_name),
                    )
                except OSError as e:
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": f"Failed to copy {img_name} to {fold_name}/{split_name}",
                            "detail": str(e),
                        },
                    )

                img_path = os.path.join(DATASET_PATH, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    fold_status = f"warning: could not read {img_name}"
                    continue

                img_stem = os.path.splitext(img_name)[0]
                img_base = os.path.join(split_dir, img_stem)
                tiles_path = os.path.join(img_base, "tiles")

                try:
                    pipeline = _make_sahi_pipeline(split_dir, tiles_path)
                    metadata = pipeline.slice_image(img, img_name, tiles_path)
                    _save_slicing_config(img_base, img_name, metadata, pipeline)
                    fold_tiles += len(metadata)
                except Exception as e:
                    fold_status = f"failed on {img_name}: {e}"

        folds_summary.append({
            "fold": fold_name,
            "train_count": len(train_imgs),
            "val_count": len(val_imgs),
            "test_count": len(test_imgs),
            "total_tiles": fold_tiles,
            "status": fold_status,
        })

    return {
        "n_folds": n_folds,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "folds": folds_summary,
    }


class ReconstructRequest(BaseModel):
    id_image: int


@app.post("/reconstruct/single_image")
async def reconstruct_single_image(request: ReconstructRequest):
    base_path = os.path.join(OUTPUT_PATH, str(request.id_image))

    if not os.path.exists(base_path):
        raise HTTPException(
            status_code=404,
            detail={"error": f"id_image {request.id_image} not found", "path": base_path},
        )

    config_path = os.path.join(base_path, "slicing_config.json")
    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=404,
            detail={"error": f"slicing_config.json not found for id_image {request.id_image}"},
        )

    tiles_dir = os.path.join(base_path, "tiles")
    if not os.path.exists(tiles_dir) or not _list_images(tiles_dir):
        raise HTTPException(
            status_code=422,
            detail={"error": f"No tiles found for id_image {request.id_image}"},
        )

    with open(config_path) as f:
        config = json.load(f)

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


# --- stubs ---

@app.post("/validate/single_image")
async def validate_single_image(_dataset_path: str):
    return {"validated": False}


@app.post("/imgshow/show_slice_frontiers")
async def show_slice_frontiers(_img_path: str):
    return {"exhibited": False}


@app.post("/imgshow/{index_slice}")
async def show_slice(_index_slice: int, _img_path: str):
    pass
