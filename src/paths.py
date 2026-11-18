"""Where every file lives, and the settings every stage shares."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VISA = DATA / "visa"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
CACHE = REPORTS / "cache"

PRODUCTS = ["pcb1", "pcb2", "pcb3", "pcb4"]

# Only a subset of each product's normal training images is used. The fit set builds the memory of
# normal patches, and the calibration set, never seen by the memory, sets the decision threshold.
FIT_IMAGES = 200
CALIBRATION_IMAGES = 200
SEED = 0

FALSE_REJECT_TARGET = 0.01  # the share of good parts the station may reject
LATENCY_BUDGET_MS = 100     # per part, end to end, at the 95th percentile
THREADS = 4
