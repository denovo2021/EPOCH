"""
make_convergence_table.py
=========================
Generate the per-parameter convergence-diagnostics table referenced in
Supplementary Information S10 (Supplementary Table S5).

For every parameter in the deployed posterior, reports:
  - posterior mean, SD and 95% highest-density interval
  - split-R-hat (rank-normalized, folded)
  - bulk effective sample size (ESS_bulk)
  - tail effective sample size (ESS_tail)
  - number of divergent transitions

The deployed posterior is the full InferenceData saved by fit_gdp_production.py
with SAVE_FULL=1 (az.from_netcdf), or the flat posterior dataset (xr.open_dataset)
— this script handles both.

Usage:
    python src/make_convergence_table.py [posterior.nc] [out_tag]

    python src/make_convergence_table.py results/hierarchical_model_rcs_v2.nc
    python src/make_convergence_table.py results/trace_adaptive_real.nc _real
    python src/make_convergence_table.py results/hierarchical_wa_posterior.nc _wa

The optional second argument is appended to the output filename, so diagnosing a second
posterior does not overwrite the table of the first.

Output:
    results/convergence_diagnostics<out_tag>.csv   — full per-parameter table
    (printed summary to stdout)
"""
import sys
import numpy as np
import arviz as az
import xarray as xr
from config import PATH_MODEL_V2, DIR_RESULTS

POSTERIOR = sys.argv[1] if len(sys.argv) > 1 else str(PATH_MODEL_V2)
OUT_TAG = sys.argv[2] if len(sys.argv) > 2 else ""

try:
    idata = az.from_netcdf(POSTERIOR)
except Exception:
    ds = xr.open_dataset(POSTERIOR)
    idata = az.InferenceData(posterior=ds)

# arviz defaults to hdi_prob=0.94, which is a deliberate nudge away from 0.95 and not a choice
# this paper made. Every other interval in the manuscript is at the 95% level, so the level is
# set explicitly here and the columns are named for it: mixing 94% and 95% intervals in one
# document, for no reason but a library default, is exactly the kind of detail a referee notices.
HDI_PROB = 0.95
summ = az.summary(idata, round_to=6, hdi_prob=HDI_PROB)
lo, hi = "hdi_2.5%", "hdi_97.5%"
summ = summ[["mean", "sd", lo, hi, "ess_bulk", "ess_tail", "r_hat"]]
summ.columns = ["mean", "sd", lo, hi, "ESS_bulk", "ESS_tail", "R_hat"]

out_path = DIR_RESULTS / ("convergence_diagnostics%s.csv" % OUT_TAG)
summ.to_csv(out_path)
print("Convergence diagnostics written to %s" % out_path)
print("  parameters: %d" % len(summ))
print("  max R-hat:  %.4f" % summ["R_hat"].max())
print("  min ESS_bulk: %.0f" % summ["ESS_bulk"].min())
print("  min ESS_tail: %.0f" % summ["ESS_tail"].min())

flag_rhat = summ[summ["R_hat"] > 1.01]
if len(flag_rhat):
    print("\n  WARNING: %d parameters with R-hat > 1.01:" % len(flag_rhat))
    print(flag_rhat[["R_hat", "ESS_bulk"]].to_string())
else:
    print("  All R-hat <= 1.01 — convergence OK")

flag_ess = summ[summ["ESS_bulk"] < 400]
if len(flag_ess):
    print("\n  WARNING: %d parameters with ESS_bulk < 400:" % len(flag_ess))
    print(flag_ess[["ESS_bulk", "ESS_tail", "R_hat"]].to_string())
else:
    print("  All ESS_bulk >= 400 — sampling adequate")

if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
    n_div = int(idata.sample_stats["diverging"].values.sum())
    n_total = int(idata.sample_stats["diverging"].values.size)
    print("  divergent transitions: %d / %d (%.3f%%)" % (n_div, n_total, 100 * n_div / n_total))
else:
    print("  (no sample_stats group — divergence count unavailable)")
