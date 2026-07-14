import os

from fastapi import APIRouter, HTTPException

from dataset.preprocessor import DatasetPreprocessor
from dataset.validator import DatasetValidator

from ..helpers import DATASET_PATH

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/validate")
async def validate_dataset_annotations():
    report = DatasetValidator(DATASET_PATH).validate()

    if report["status"] == "errors":
        raise HTTPException(status_code=422, detail=report)

    return report


@router.post("/clean")
async def clean_dataset_annotations():
    coco_path = os.path.join(DATASET_PATH, "_annotations.coco.json")
    if not os.path.isfile(coco_path):
        raise HTTPException(
            status_code=404,
            detail={"error": "Annotation file not found", "path": coco_path},
        )

    return DatasetPreprocessor(DATASET_PATH).run()
