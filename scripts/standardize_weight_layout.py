#!/usr/bin/env python3
"""Move raw checkpoints to pesos/ and keep models/ for evaluator manifests only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ("all_640",)
ARCH_DIRS = ("YOLOV8", "Faster", "Detr")
MANIFEST_ARCH = {
    "YOLOV8": "yolo",
    "Faster": "faster_rcnn",
    "Detr": "detr",
}
CHECKPOINT_REL = {
    "YOLOV8": Path("train/weights/best.pt"),
    "Faster": Path("best.pth"),
    "Detr": Path("training/best_model.pth"),
}


def _move_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        if src.is_dir():
            shutil.rmtree(src)
        else:
            src.unlink()
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _move_arch_artifacts(models_fold: Path, pesos_fold: Path, arch: str) -> None:
    src_arch = models_fold / arch
    dst_arch = pesos_fold / arch
    if not src_arch.exists():
        return

    if arch == "YOLOV8":
        _move_path(src_arch / "train", dst_arch / "train")
    elif arch == "Faster":
        for file_name in ("best.pth", "last_checkpoint.pth", "training_params.json"):
            _move_path(src_arch / file_name, dst_arch / file_name)
    elif arch == "Detr":
        _move_path(src_arch / "training", dst_arch / "training")

    # Local manifests are intentionally not kept with raw weights.
    (src_arch / "manifest.json").unlink(missing_ok=True)
    try:
        src_arch.rmdir()
    except OSError:
        pass


def _checkpoint_path(pesos_fold: Path, arch: str) -> Path:
    return pesos_fold / arch / CHECKPOINT_REL[arch]


def _write_manifest(mode: str, fold: str, arch: str, checkpoint: Path, models_root: Path, dataset_root: Path) -> None:
    manifest_arch = MANIFEST_ARCH[arch]
    fold_num = int(fold.removeprefix("fold_"))
    manifest = {
        "mode": mode,
        "fold": fold_num,
        "fold_name": fold,
        "architecture": arch,
        "checkpoint": str(checkpoint.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "train_annotations": str((dataset_root / "filesJSON" / f"{fold}_train.json").resolve()),
        "val_annotations": str((dataset_root / "filesJSON" / f"{fold}_val.json").resolve()),
        "test_annotations": str((dataset_root / "filesJSON" / f"{fold}_test.json").resolve()),
        "source": "pesos",
    }
    manifest_path = models_root / mode / fold / manifest_arch / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def standardize_mode(mode: str, models_root: Path, pesos_root: Path, dataset_root: Path) -> tuple[int, int]:
    mode_models = models_root / mode
    mode_pesos = pesos_root / mode / "model_checkpoints"
    if not mode_models.exists():
        return 0, 0

    moved_or_seen = 0
    manifests = 0
    for models_fold in sorted(mode_models.glob("fold_*")):
        if not models_fold.is_dir():
            continue
        fold = models_fold.name
        pesos_fold = mode_pesos / fold
        for arch in ARCH_DIRS:
            _move_arch_artifacts(models_fold, pesos_fold, arch)
            checkpoint = _checkpoint_path(pesos_fold, arch)
            if checkpoint.is_file():
                moved_or_seen += 1
                _write_manifest(mode, fold, arch, checkpoint, models_root, dataset_root)
                manifests += 1

    return moved_or_seen, manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep raw weights in pesos/ and manifests in models/.")
    parser.add_argument("--models-root", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--pesos-root", type=Path, default=REPO_ROOT / "pesos")
    parser.add_argument("--dataset-root-base", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_weights = 0
    total_manifests = 0
    for mode in args.modes:
        weights, manifests = standardize_mode(
            mode=mode,
            models_root=args.models_root.resolve(),
            pesos_root=args.pesos_root.resolve(),
            dataset_root=(args.dataset_root_base / mode).resolve(),
        )
        print(f"{mode}: checkpoints available={weights}, manifests written={manifests}")
        total_weights += weights
        total_manifests += manifests
    print(f"total: checkpoints available={total_weights}, manifests written={total_manifests}")


if __name__ == "__main__":
    main()
