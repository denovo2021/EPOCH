"""
elasticity_inflection.py
========================
Local population-output elasticity, decline-phase vs growth-phase averages, and the
epsilon=1 tipping point (if one robustly exists) from the fitted hierarchical B-spline
posterior. GDP-only. Loads the posterior with raw xarray (xr.open_dataset).

  eps(x_s) = (1/v) * (beta0 + sum_j theta_j B_j'(x_s))
Output: results/elasticity_inflection.json
"""
import json
import numpy as np, pandas as pd
import xarray as xr
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter
from config import PATH_MODEL_V2, PATH_SCALE_V2, PATH_MERGED_AGE, DIR_RESULTS

scales = json.load(open(PATH_SCALE_V2))
MU, SD = scales["MU_GLOBAL"], scales["SD_GLOBAL"]
ka = np.array(scales["bspline_knots_aug"]); deg = scales["bspline_degree"]; nb = scales["bspline_n_basis"]

try:
    post = xr.open_dataset(PATH_MODEL_V2, group="posterior")
except Exception:
    post = xr.open_dataset(PATH_MODEL_V2)
theta = post["theta"].stack(z=("chain", "draw")).values
if theta.shape[0] != nb:
    theta = theta.T
theta = theta.T   # (S, nb)
beta0 = post["beta0"].stack(z=("chain", "draw")).values

def deriv(x, order):
    x = np.asarray(x, float); o = np.zeros((x.size, nb))
    for i in range(nb):
        c = np.zeros(nb); c[i] = 1.0
        o[:, i] = BSpline(ka, c, deg, extrapolate=False).derivative(order)(x)
    return np.nan_to_num(o)

df = pd.read_csv(PATH_MERGED_AGE).dropna(subset=["ISO3", "Population", "GDP"]).copy()
df["Year"] = df["Year"].astype(int); df["lp"] = np.log10(df["Population"])
df = df.sort_values(["ISO3", "Year"]); df["dlp"] = df.groupby("ISO3")["lp"].diff()
df = df.dropna(subset=["dlp"]); df["x_s"] = (df["lp"] - MU) / SD

def phase(sub):
    dbar = deriv(sub["x_s"].values, 1).mean(axis=0)
    e = (beta0 + theta @ dbar) / SD
    return float(e.mean()), float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5)), int(len(sub))

dec = phase(df[df["dlp"] < 0]); gro = phase(df[df["dlp"] > 0])
xs = np.linspace(*np.percentile(df["x_s"], [5, 95]), 400)
e_med = np.median((beta0[None, :] + deriv(xs, 1) @ theta.T) / SD, axis=1)
e_s = savgol_filter(e_med, 151, 2)
cr = np.where(np.diff(np.sign(e_s - 1.0)) != 0)[0]
if len(cr):
    xstar = min([xs[k] - (e_s[k]-1)*(xs[k+1]-xs[k])/((e_s[k+1]-1)-(e_s[k]-1)) for k in cr], key=abs)
    tip = float(10**(xstar*SD + MU)); tip_log = float(xstar*SD + MU)
else:
    tip = None; tip_log = None

res = {"prior": "hierarchical adaptive: sigma_theta ~ HalfNormal(0.2); theta ~ Normal(0, sigma_theta)",
       "decline_elasticity": dict(zip(["mean", "ci_lo", "ci_hi", "n"], dec)),
       "growth_elasticity": dict(zip(["mean", "ci_lo", "ci_hi", "n"], gro)),
       "core_mean_elasticity": float(e_s.mean()),
       "elasticity_core_min": float(e_s.min()), "elasticity_core_max": float(e_s.max()),
       "epsilon_subunitary": bool((e_s < 1).mean() > 0.5),
       "tipping_point_pop": tip, "tipping_point_log10pop": tip_log}
json.dump(res, open(DIR_RESULTS / "elasticity_inflection.json", "w"), indent=2)
print(json.dumps(res, indent=2))
