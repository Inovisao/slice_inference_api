"""Generate manifest.json shims under models/ pointing at pesos/ checkpoints.

pesos/ holds the raw training weights, laid out per crop mode
(sahi/asahi/asahi_rect/all_640) and per MODEL VARIANT folder — e.g. YOLOV8N,
YOLOV8L, Faster, Detr. The model-variant folder name is free-form; its prefix
decides the engine (YOLO* -> ultralytics, FASTER -> faster_rcnn, DETR -> detr),
which in turn decides where the actual weight file sits inside that folder.

geraResultados.py resolves checkpoints through
`models/<crop>/fold_N/<MODEL_NAME>/manifest.json`, so this script discovers every
model-variant folder present in pesos/ and writes that manifest at the canonical
location — without moving or duplicating the (large) weight files.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PESOS_ROOT = REPO_ROOT / "pesos"
MODELS_ROOT = REPO_ROOT / "models"

_MODES = ("sahi", "asahi", "asahi_rect", "all_640")
_N_FOLDS = 5

# engine (by model-name prefix) -> candidate weight paths inside the model folder
_WEIGHTS_BY_ENGINE = {
    "yolo": [Path("train/weights/best.pt")],
    "faster_rcnn": [Path("best.pth")],
    "detr": [Path("training/best_model.pth"), Path("best_model.pth")],
}
_ENGINE_BY_PREFIX = (("YOLO", "yolo"), ("FASTER", "faster_rcnn"), ("DETR", "detr"))


def _engine_for(model_name: str) -> str | None:
    up = model_name.upper()
    for prefix, engine in _ENGINE_BY_PREFIX:
        if up.startswith(prefix):
            return engine
    return None


def _pesos_fold_dir(mode: str, fold: int) -> Path:
    nested = PESOS_ROOT / mode / "model_checkpoints" / f"fold_{fold}"
    flat = PESOS_ROOT / mode / f"fold_{fold}"
    return nested if nested.is_dir() else flat


def _find_weight(model_dir: Path, engine: str) -> Path | None:
    for rel in _WEIGHTS_BY_ENGINE[engine]:
        cand = model_dir / rel
        if cand.is_file():
            return cand
    return None


def main() -> None:
    written, missing, skipped = 0, [], []

    for mode in _MODES:
        for fold in range(1, _N_FOLDS + 1):
            fold_dir = _pesos_fold_dir(mode, fold)
            if not fold_dir.is_dir():
                continue
            # Every subfolder is a model variant (YOLOV8N, YOLOV8L, Faster, Detr, ...)
            for model_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
                model_name = model_dir.name
                engine = _engine_for(model_name)
                if engine is None:
                    skipped.append(f"{fold_dir}/{model_name} (unknown engine)")
                    continue
                checkpoint = _find_weight(model_dir, engine)
                if checkpoint is None:
                    missing.append(str(model_dir))
                    continue

                manifest_dir = MODELS_ROOT / mode / f"fold_{fold}" / model_name
                manifest_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "mode": mode,
                    "fold": fold,
                    "model": model_name,
                    "engine": engine,
                    "checkpoint": str(checkpoint),
                    "source": "pesos",
                }
                (manifest_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                written += 1

    print(f"manifests written: {written}")
    if skipped:
        print(f"skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")
    if missing:
        print(f"weight file not found in ({len(missing)}):")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
