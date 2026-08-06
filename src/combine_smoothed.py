"""Combine multiple partial-chain NetCDF traces into a single pooled posterior.
Usage: python combine_smoothed.py <glob_pattern> [output.nc]
   e.g. python combine_smoothed.py 'results/sm*.nc'
"""
import sys, glob, os
import numpy as np, xarray as xr, arviz as az
from config import DIR_RESULTS_IN, DIR_RESULTS
if len(sys.argv) < 2:
    raise SystemExit("Usage: python combine_smoothed.py <glob_pattern> [output.nc]")
files = sorted(glob.glob(os.path.expanduser(sys.argv[1])))
print("combining:", files)
dss = [xr.open_dataset(f, engine="scipy") for f in files]
post = xr.concat([d.assign_coords(chain=d.chain + i*d.sizes["chain"]) for i, d in enumerate(dss)], dim="chain")
idata = az.InferenceData(posterior=post)
out = sys.argv[2] if len(sys.argv) > 2 else str(DIR_RESULTS / "combined_posterior.nc")
post.to_netcdf(str(out), engine="h5netcdf")
print("saved", out, "chains", post.sizes["chain"], "draws", post.sizes["draw"])
s = az.summary(idata, var_names=["beta0","theta","sigma","nu"])
print("spline/global max R-hat=%.3f min ess=%.0f" % (s["r_hat"].max(), s["ess_bulk"].min()))
