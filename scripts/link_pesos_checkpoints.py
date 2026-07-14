"""Generate manifest.json shims under paths.models pointing at pesos/ checkpoints.

pesos/ holds the raw weights from an external training run, laid out
differently per mode (sahi/asahi nest under `model_checkpoints/`, asahi_rect
does not) and per architecture (YOLOV8/Faster/Detr, each with its own weight
filename). geraResultados.py resolves checkpoints through
`<paths.models>/<mode>/fold_N/<yolo|faster_rcnn|detr>/manifest.json`, so this
script writes that manifest at the canonical location for every
mode x fold x architecture found in pesos/, without moving or duplicating
the (large) weight files themselves.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PESOS_ROOT = REPO_ROOT / "pesos"
MODELS_ROOT = REPO_ROOT / "models"

_MODES = ("sahi", "asahi", "asahi_rect")
_N_FOLDS = 5

# (manifest arch dir, pesos arch dir, checkpoint relative to pesos arch dir)
_ARCHS = (
    ("yolo", "YOLOV8", Path("train/weights/best.pt")),
    ("faster_rcnn", "Faster", Path("best.pth")),
    ("detr", "Detr", Path("training/best_model.pth")),
)


def _pesos_fold_dir(mode: str, fold: int) -> Path:
    nested = PESOS_ROOT / mode / "model_checkpoints" / f"fold_{fold}"
    flat = PESOS_ROOT / mode / f"fold_{fold}"
    return nested if nested.is_dir() else flat


def main() -> None:
    written, missing = 0, []

    for mode in _MODES:
        for fold in range(1, _N_FOLDS + 1):
            fold_dir = _pesos_fold_dir(mode, fold)
            for manifest_arch, pesos_arch, rel_checkpoint in _ARCHS:
                checkpoint = fold_dir / pesos_arch / rel_checkpoint
                if not checkpoint.is_file():
                    missing.append(str(checkpoint))
                    continue

                manifest_dir = MODELS_ROOT / mode / f"fold_{fold}" / manifest_arch
                manifest_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "mode": mode,
                    "fold": fold,
                    "architecture": manifest_arch,
                    "checkpoint": str(checkpoint),
                    "source": "pesos",
                }
                (manifest_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                written += 1

    print(f"manifests written: {written}")
    if missing:
        print(f"checkpoints not found ({len(missing)}):")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
