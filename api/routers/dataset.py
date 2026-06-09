import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dataset.cleaner import DatasetCleaner
from dataset.crossfold import CrossFoldSplitter
from dataset.validator import DatasetValidator

from ..helpers import DATASET_PATH

router = APIRouter(prefix="/dataset", tags=["dataset"])


class CrossFoldRequest(BaseModel):
    n_folds: int = Field(default=5, ge=2)
    val_ratio: float = Field(default=0.3, gt=0.0, lt=1.0)
    seed: int = 42


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

    return DatasetCleaner(DATASET_PATH).clean()


@router.post("/crossfolds")
async def generate_crossfolds(request: CrossFoldRequest = CrossFoldRequest()):
    splitter = CrossFoldSplitter(DATASET_PATH)

    if not os.path.isfile(splitter.coco_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "No annotation file found. Run /dataset/clean first.",
                "expected": splitter.coco_path,
            },
        )

    return CrossFoldSplitter(DATASET_PATH).split(
        n_folds=request.n_folds,
        val_ratio=request.val_ratio,
        seed=request.seed,
    )
