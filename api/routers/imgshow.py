from fastapi import APIRouter

router = APIRouter(tags=["imgshow"])


@router.post("/validate/single_image")
async def validate_single_image(_dataset_path: str):
    return {"validated": False}


@router.post("/imgshow/show_slice_frontiers")
async def show_slice_frontiers(_img_path: str):
    return {"exhibited": False}


@router.post("/imgshow/{index_slice}")
async def show_slice(_index_slice: int, _img_path: str):
    pass
