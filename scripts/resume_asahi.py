"""
Retoma a geração dos folds ASAHI a partir do fold 3.
Uso: python scripts/resume_asahi.py [resume_from]
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config.settings import SlicingConfig
from dataset.kfold_generator import AsahiKFoldValidator

resume_from = int(sys.argv[1]) if len(sys.argv) > 1 else 3

slicing_cfg = SlicingConfig(
    slicing_mode="asahi",
    tile_size=(640, 640),
    overlap_ratio=0.15,
)
validator = AsahiKFoldValidator(
    dataset_path=os.path.join(ROOT, "dataset", "all"),
    slicing_config=slicing_cfg,
    n_splits=5,
    output_root=os.path.join(ROOT, "dataset", "asahi"),
    seed=42,
    ioa_threshold=0.2,
    val_ratio=0.10,
    test_ratio=0.10,
)
validator.run(resume_from=resume_from)
print("Concluído.")
