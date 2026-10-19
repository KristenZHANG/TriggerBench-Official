"""Path helpers for the Logical Adherence generation package."""

from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DIM_DIR = PIPELINE_DIR.parent
GEN_ROOT = DIM_DIR.parent
SEEDS_DIR = DIM_DIR / "seeds"
OUTPUTS_DIR = DIM_DIR / "outputs"
