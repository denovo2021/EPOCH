"""
build_scale_cache.py
====================
Generate the deterministic scaling cache (results/cache/scale_rcs_v2.json)
required by fit_gdp_production.py, scenario_projection.py, elasticity_inflection.py,
make_figures.py, and all downstream B-spline posterior consumers.

The cache contains:
  - MU_GLOBAL, SD_GLOBAL   : mean/std of log10(Population) over the full panel
  - s_dWA, s_dOD, s_dt     : standardization scales for age-structure and time covariates
  - bspline_*              : cubic B-spline knot specification (8 interior at equal quantiles,
                              clamped boundaries, degree 3, 12 basis functions)
  - bspline_col_means      : column means for centering the B-spline design matrix
  - coef_dt_proj           : OLS coefficients for orthogonalizing the time covariate against
                              [1, x_s, B_1(x_s), ..., B_J(x_s)]
  - dt_clip                : clipping bound for the orthogonalized time covariate

All quantities are deterministic functions of the training data — no MCMC, no
fitted parameters, no leakage risk.  This must be run once before any sampling
script; the output is stable across runs given the same data.

Usage:
    python src/build_scale_cache.py [data/merged_age.csv] [results/cache/scale_rcs_v2.json]
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from config import PATH_MERGED_AGE, DIR_RESULTS

DATA = sys.argv[1] if len(sys.argv) > 1 else str(PATH_MERGED_AGE)
OUT  = sys.argv[2] if len(sys.argv) > 2 else str(DIR_RESULTS / "cache" / "scale_rcs_v2.json")

N_INTERIOR = 8
DEGREE = 3
DT_CLIP = 2.0

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO3", "Country Name", "Region", "Year",
                        "GDP", "Population", "WAshare", "OldDep"]).copy()
df["Year"] = df["Year"].astype(int)

lp = np.log10(df["Population"].values)
MU = float(np.mean(lp))
SD = float(np.std(lp))
x_s = (lp - MU) / SD

# --- cubic B-spline basis: 8 interior knots at equally-spaced quantiles, clamped ---
quantiles = np.linspace(0, 1, N_INTERIOR + 2)[1:-1]  # [0.111, 0.222, ..., 0.889]
knots_interior = np.quantile(x_s, quantiles).tolist()
x_min, x_max = float(x_s.min()), float(x_s.max())
knots_aug = [x_min] * (DEGREE + 1) + knots_interior + [x_max] * (DEGREE + 1)
n_basis = len(knots_interior) + DEGREE + 1   # 8 + 3 + 1 = 12

ka = np.array(knots_aug)

def bspline_matrix(x_vals):
    B = np.zeros((len(x_vals), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        B[:, i] = BSpline(ka, c, DEGREE, extrapolate=False)(np.asarray(x_vals, float))
    return np.nan_to_num(B)

B = bspline_matrix(x_s)
col_means = B.mean(axis=0).tolist()

# --- age-structure deltas vs country base year (last observed) ---
base = (df.sort_values(["ISO3", "Year"])
          .drop_duplicates("ISO3", keep="last")
          [["ISO3", "WAshare", "OldDep", "Year"]]
          .rename(columns={"WAshare": "WAb", "OldDep": "ODb", "Year": "Yb"}))
df2 = df.merge(base, on="ISO3", how="left")
dWA_raw = (df2["WAshare"] - df2["WAb"]).values
dOD_raw = (df2["OldDep"]  - df2["ODb"]).values
s_dWA = float(np.nanstd(dWA_raw)) or 0.1
s_dOD = float(np.nanstd(dOD_raw)) or 0.1

# --- time covariate: standardize, then orthogonalize against [1, x_s, B] ---
dt_raw = ((df2["Year"] - df2["Yb"]).values / 10.0)
s_dt = float(np.nanstd(dt_raw)) or 1.0
dt_s = dt_raw / s_dt

B_centered = B - np.array(col_means)
X_dt = np.column_stack([np.ones(len(df)), x_s, B_centered])
coef_dt = np.linalg.lstsq(X_dt, dt_s, rcond=None)[0].tolist()

# --- write ---
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cache = {
    "MU_GLOBAL": MU,
    "SD_GLOBAL": SD,
    "s_dWA": s_dWA,
    "s_dOD": s_dOD,
    "s_dt": s_dt,
    "bspline_knots_interior": knots_interior,
    "bspline_knots_aug": knots_aug,
    "bspline_col_means": col_means,
    "bspline_degree": DEGREE,
    "bspline_n_interior": N_INTERIOR,
    "bspline_n_basis": n_basis,
    "coef_dt_proj": coef_dt,
    "dt_clip": DT_CLIP,
}
json.dump(cache, open(OUT, "w"), indent=2)
print("scale cache written to %s" % OUT)
print("  MU=%.6f  SD=%.6f  n_basis=%d" % (MU, SD, n_basis))
print("  s_dWA=%.6f  s_dOD=%.6f  s_dt=%.6f" % (s_dWA, s_dOD, s_dt))
print("  interior knots: %s" % [round(k, 4) for k in knots_interior])
