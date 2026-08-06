"""
config.py — path configuration for EPOCH (self-contained, reproducible layout).

Scripts are run from the repository root, e.g.  `python src/build_scale_cache.py`
or `uv run python src/fit_gdp_production.py`. Python puts src/ on the path,
so `from config import ...` resolves this file; all paths are anchored to the repo
root (the parent of src/), so the project is fully relocatable. Override the data
location with $EPOCH3_DATA if the CSVs live outside the tree.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIR_DATA = Path(os.environ["EPOCH3_DATA"]) if os.environ.get("EPOCH3_DATA") else REPO / "data"
DIR_RESULTS = REPO / "results"
DIR_FIGURES = REPO / "figures"
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

# --- processed inputs ---------------------------------------------------------
PATH_MERGED_AGE = DIR_DATA / "merged_age.csv"                 # country-year panel: real GDP, population, WAshare, region
PATH_AGE_SCEN   = DIR_DATA / "age_predictions_scenarios.csv"  # WAshare/OldDep 1950-2100 (Estimates + Medium variant)
PATH_POP_SCEN   = DIR_DATA / "pop_predictions_scenarios.csv"  # UN WPP Medium population to 2100
PATH_AGE_ANCHOR = DIR_DATA / "age_base_anchors.csv"           # base-year age-structure anchors

# --- production (B-spline) model artifacts ------------------------------------
DIR_RESULTS_IN = DIR_RESULTS                                  # posteriors regenerated in-place
PATH_SCALE_V2  = DIR_RESULTS / "cache" / "scale_rcs_v2.json"  # scaling constants + B-spline knots (input cache)
PATH_MODEL_V2  = DIR_RESULTS / "hierarchical_model_rcs_v2.nc" # production posterior (regenerate via fit_gdp_production.py)
# To use the adaptive posterior instead, set the env var:
#   EPOCH3_POSTERIOR=results/trace_adaptive.nc
_override = os.environ.get("EPOCH3_POSTERIOR")
if _override:
    PATH_MODEL_V2 = REPO / _override

RANDOM_SEED = 42
