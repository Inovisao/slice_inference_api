from inference.engine import TileInferenceEngine
from inference.pipeline import InferencePipeline
from slicing.service import make_slicer


def make_inference_pipeline(
    model_path: str,
    slicing_mode: str,
    overlap_ratio: float,
    suppression: str,
    conf_thr: float,
    iou_thr: float,
    device: str,
    include_full_inference: bool | None = None,
    batch_size: int = 32,
) -> InferencePipeline:
    engine = TileInferenceEngine(model_path, device=device)
    slicer = make_slicer(slicing_mode, overlap_ratio)
    return InferencePipeline(
        engine=engine,
        slicer=slicer,
        suppression=suppression,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        include_full_inference=(
            slicing_mode in ("asahi", "asahi_rect")
            if include_full_inference is None
            else include_full_inference
        ),
        batch_size=batch_size,
    )


def make_pipeline_from_config(
    config: dict,
    model_path: str,
    suppression: str,
    conf_thr: float,
    iou_thr: float,
    device: str,
    include_full_inference: bool | None = None,
    batch_size: int = 32,
) -> InferencePipeline:
    slicing_method = config["slicing_method"]
    tile_w = config["tile_size"][0]
    overlap = config.get("overlap_ratio") or config["overlap_x"] / tile_w
    return make_inference_pipeline(
        model_path=model_path,
        slicing_mode=slicing_method,
        overlap_ratio=overlap,
        suppression=suppression,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        device=device,
        include_full_inference=include_full_inference,
        batch_size=batch_size,
    )
