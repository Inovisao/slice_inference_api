import os
from typing import List

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

DATASET_PATH = "./dataset"
OUTPUT_PATH = "./output"
MODELS_PATH = "./models"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def list_images(path: str) -> List[str]:
    return [f for f in os.listdir(path) if f.lower().endswith(_IMAGE_EXTENSIONS)]


def validate_dataset(path: str) -> List[str]:
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail={"error": "Dataset directory not found", "path": path},
        )
    images = list_images(path)
    if not images:
        raise HTTPException(
            status_code=422,
            detail={"error": "No valid images found in dataset", "path": path},
        )
    return images


def next_id() -> int:
    if not os.path.exists(OUTPUT_PATH):
        return 1
    existing = [
        int(d)
        for d in os.listdir(OUTPUT_PATH)
        if d.isdigit() and os.path.isdir(os.path.join(OUTPUT_PATH, d))
    ]
    return max(existing, default=0) + 1


def read_image_or_raise(img_path: str, img_name: str) -> cv2.typing.MatLike:
    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to read image: {img_name}"},
        )
    return img


def validate_image_size(img: cv2.typing.MatLike, tile_size: tuple, img_name: str):
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


def load_image_rgb(img_path: str) -> np.ndarray:
    return np.array(Image.open(img_path).convert("RGB"))
