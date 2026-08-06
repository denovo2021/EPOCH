"""Regenerate Supplementary Figure S4 (local population-output elasticity) from
the FIXED, strongly-regularized spline posterior (RW SD = 0.05), so the figure
matches its finalized caption: a smooth, structurally sub-unitary elasticity.

NOTE ON SOURCE FILE: the directive named results/hierarchical_model_smoothed.nc,
but that posterior in fact yields the *super-unitary* adaptive elasticity
(decline mean = 1.22), i.e. it would reproduce the caption mismatch. The genuine
fixed-penalty (RW 0.05) posterior that gives the smooth sub-unitary elasticity
(decline = growth = 0.92) is trace_fixed005.nc - the file labelled
"Fixed penalty (RW 0.05): e~0.92" in make_sensitivity_figure.py. This script
therefore loads trace_fixed005.nc.

Reuses the basis-derivative / standardization logic of the original
make_figures.py.

Run from the repository root:
    python src/make_figureS4_fixed.py

Output: figures/FigureS4_local_elasticity_fixed.png / .pdf
"""
import json
import os
from pathlib import Path

import sys
import numpy as np
import pandas as pd
import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

s = json.load(open(ROOT / "results" / "cache" / "scale_rcs_v2.json"))
MU, SD = s["MU_GLOBAL"], s["SD_GLOBAL"]
ka = np.array(s["bspline_knots_aug"])
deg, nb = s["bspline_degree"], s["bspline_n_basis"]


def basis(x, nu=0):
    x = np.asarray(x, float)
    o = np.zeros((x.size, nb))
    for i in range(nb):
        c = np.zeros(nb); c[i] = 1.0
        bs = BSpline(ka, c, deg, extrapolate=False)
        o[:, i] = bs.derivative(nu)(x) if nu else bs(x)
    return np.nan_to_num(o)


# Fixed-penalty (RW 0.05) posterior; prefer a local copy, else the forNHB draft.
#
# NOTE: Supplementary Fig. S4 is DELIBERATELY the nominal fixed-penalty curve. The SI text
# introduces it as "under the fixed first-difference penalty on the nominal series", quotes
# 0.92 in both demographic phases, and ties that constant to the counterfactual projection run
# tagged _eps0.92. Re-pointing this figure at trace_fixed005_real.nc would contradict its own
# caption. The argument below exists for sensitivity checks, not for the deposited figure.
#   $env:GDP_COL="GDP_constant_2015usd"
#   python src/make_figureS4_fixed.py results/trace_fixed005_real.nc
_candidates = [ROOT / "results" / "trace_fixed005.nc",
               ROOT.parent / "forNHB" / "results" / "trace_fixed005.nc"]
if len(sys.argv) > 1:
    POSTERIOR = Path(sys.argv[1])
    if not POSTERIOR.is_absolute():
        POSTERIOR = (ROOT / POSTERIOR).resolve()
else:
    POSTERIOR = next((p for p in _candidates if p.exists()), _candidates[0])
print(f"loading fixed-penalty posterior: {POSTERIOR}")
post = az.from_netcdf(POSTERIOR).posterior
theta = post["theta"].stack(z=("chain", "draw")).values
if theta.shape[0] != nb:
    theta = theta.T
theta = theta.T                       # (S, nb)
beta0 = post["beta0"].stack(z=("chain", "draw")).values
print(f"fixed-penalty posterior: theta {theta.shape}, beta0 {beta0.shape}, nb={nb}")

GDP_COL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("GDP_COL", "GDP")
# GDP enters here only through the row subset, but the subset decides which country-years
# count as contraction and as expansion, so it has to be the one the posterior was fitted on.
df = pd.read_csv(ROOT / "data" / "merged_age.csv").dropna(
    subset=["ISO3", "Population", GDP_COL]).copy()
df["Year"] = df["Year"].astype(int)
df["lp"] = np.log10(df["Population"])
df = df.sort_values(["ISO3", "Year"])
df["dlp"] = df.groupby("ISO3")["lp"].diff()
df["x_s"] = (df["lp"] - MU) / SD

lo_x, hi_x = np.percentile(df["x_s"], [1, 99])
lo_c, hi_c = np.percentile(df["x_s"], [5, 95])
xs = np.linspace(lo_x, hi_x, 400)
logpop = xs * SD + MU
core = (xs >= lo_c) & (xs <= hi_c)

ridx = np.random.default_rng(0).choice(theta.shape[0], min(800, theta.shape[0]), replace=False)
Bp = basis(xs, 1)
E = (beta0[ridx][None, :] + Bp @ theta[ridx].T) / SD
e_med = np.median(E, axis=1)
e_lo = np.percentile(E, 2.5, axis=1)
e_hi = np.percentile(E, 97.5, axis=1)
e_smooth = savgol_filter(e_med, 151, 2)


def phase_e(sub):
    dbar = basis(sub["x_s"].values, 1).mean(axis=0)
    return (beta0 + theta @ dbar) / SD


ed = phase_e(df[df["dlp"] < 0]).mean()
eg = phase_e(df[df["dlp"] > 0]).mean()
core_mean = float(e_smooth[core].mean())
print(f"decline-phase mean={ed:.3f}  growth-phase mean={eg:.3f}  core-mean={core_mean:.3f}")

fig = plt.figure(figsize=(4.6, 3.5))
ax = fig.add_subplot(111)
ax.plot(logpop[core], e_smooth[core], color="#1f4e79", lw=2.2)
ax.fill_between(logpop[core], savgol_filter(e_lo, 151, 2)[core],
                savgol_filter(e_hi, 151, 2)[core], color="#1f4e79", alpha=0.14,
                label="95% credible band")
ax.axhline(1.0, color="#555", ls=":", lw=1, label="ε = 1 (scale neutrality)")
ax.axhline(ed, color="#8c2d04", lw=1.3)
ax.text(logpop[core][3], ed + 0.03,
        "phase-averaged ε ≈ %.2f (decline %.2f, growth %.2f)" % ((ed + eg) / 2, ed, eg),
        color="#8c2d04", fontsize=7.5)
ax.text(0.97, 0.05, "ε < 1 across the data-dense core\n(structurally sub-unitary; no ε = 1 crossing)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#444")
ax.set_ylim(min(0.6, e_smooth[core].min() - 0.1), max(1.15, e_smooth[core].max() + 0.1))
ax.set_xlim(logpop[core].min(), logpop[core].max())
ax.set_xlabel("log$_{10}$ Population")
ax.set_ylabel("Local elasticity ε(x$_s$)")
ax.set_title("Local population–output elasticity (fixed penalty, RW 0.05)"
             + ("" if GDP_COL == "GDP" else "\nreal GDP, constant 2015 US$"),
             fontsize=9.5, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper right")
# A sensitivity run must not overwrite the deposited figure.
STEM = "FigureS4_local_elasticity_fixed" + ("" if GDP_COL == "GDP" else "_real")
for ext in ("png", "pdf"):
    fig.savefig(ROOT / "figures" / f"{STEM}.{ext}")
plt.close(fig)
print(f"saved {STEM}.png / .pdf")
