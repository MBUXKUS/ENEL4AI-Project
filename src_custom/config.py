from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints_custom"
LOG_DIR = OUTPUT_DIR / "logs_custom"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for directory in (DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, REPORTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
