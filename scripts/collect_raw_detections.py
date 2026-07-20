"""Collect raw (pre-suppression) detections for every architecture x split,
so downstream threshold analysis can be done offline on CPU.

For each arch in {YOLO, Faster, Detr} and split in {val, test}, writes one
pickle under results/threshold_analysis/raw/. Existing caches are skipped
(the YOLO caches from the earlier sweep are reused as-is).

Keys are (dataset_name, fold) where dataset_name is the folder name
(sahi/asahi/asahi_rect/all_640) — all_640 is the no-tiling baseline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.threshold_sweep import collect, _cache_path, ARCHS

SPLITS = ["val", "test"]


def main():
    for arch in ARCHS:
        for split in SPLITS:
            path = _cache_path(split, arch)
            if os.path.exists(path):
                print(f"[exists] {arch} {split} -> {path} (skip)")
                continue
            print(f"\n{'='*70}\nCollecting {arch} / {split}\n{'='*70}")
            collect(split, arch)
    print("\nAll raw detection caches ready.")


if __name__ == "__main__":
    main()
