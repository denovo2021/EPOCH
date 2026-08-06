"""
compare_models.py
=================
Leave-One-Out cross-validation (LOO) comparison of the two spline-regularization
choices for the hierarchical GDP model, the basis for the fragility/sensitivity result:

  - fixed_0.05         : strong FIXED first-difference (P-spline) penalty,
                         RW innovation SD = 0.05  -> smooth, SUB-unitary elasticity (~0.92)
  - adaptive_HalfNormal: data-driven ridge shrinkage, sigma_theta ~ HalfNormal(0.2)
                         -> volatile, SUPER-unitary decline elasticity (~1.22)

Both traces must be full InferenceData WITH a log_likelihood group, produced by:
  PRIOR=fixed005 SAVE_FULL=1 uv run python fit_gdp_production.py <data.csv> ../results/trace_fixed005.nc 42 2000 3000 4 0.95
  PRIOR=adaptive SAVE_FULL=1 uv run python fit_gdp_production.py <data.csv> ../results/trace_adaptive.nc 42 2000 3000 4 0.95

Per the user's environment (arviz==0.17.1), models are loaded with az.from_netcdf(PATH)
to obtain the full InferenceData (including log_likelihood) rather than just the posterior.

Usage:
  uv run python compare_models.py <fixed.nc> <adaptive.nc> [out_tag]
  uv run python compare_models.py results/trace_fixed005.nc results/trace_adaptive.nc
  uv run python compare_models.py results/trace_fixed005_real.nc results/trace_adaptive_real.nc _real
"""
import sys
import arviz as az
from config import DIR_RESULTS

FIXED = sys.argv[1] if len(sys.argv) > 1 else str(DIR_RESULTS / "trace_fixed005.nc")
ADAPT = sys.argv[2] if len(sys.argv) > 2 else str(DIR_RESULTS / "trace_adaptive.nc")
# Third argument is an output tag, so comparing the real pair does not overwrite the nominal
# table:  compare_models.py results/trace_fixed005_real.nc results/trace_adaptive_real.nc _real
OUT_TAG = sys.argv[3] if len(sys.argv) > 3 else ""

idata_fixed = az.from_netcdf(FIXED)
idata_adapt = az.from_netcdf(ADAPT)
for name, d in [("fixed_0.05", idata_fixed), ("adaptive_HalfNormal", idata_adapt)]:
    if "log_likelihood" not in d.groups():
        raise SystemExit("ERROR: %s has no log_likelihood group; refit with SAVE_FULL=1." % name)

cmp = az.compare({"fixed_0.05": idata_fixed, "adaptive_HalfNormal": idata_adapt}, ic="loo")
print(cmp.to_string())
out = DIR_RESULTS / ("model_comparison_loo%s.csv" % OUT_TAG)
cmp.to_csv(out)
print("\nSaved", out)
