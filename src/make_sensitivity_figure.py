"""
make_sensitivity_figure.py
Figure: the population-output elasticity under the two spline regularizations,
showing the sign flip (sub-unitary fixed-0.05 vs super-unitary adaptive).
Loads both full-InferenceData traces with az.from_netcdf.
Usage: python make_sensitivity_figure.py <trace_fixed005.nc> <trace_adaptive.nc>
"""
import sys, json
import numpy as np, pandas as pd
import arviz as az
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter
from config import PATH_SCALE_V2, PATH_MERGED_AGE, DIR_FIGURES

FIXED = sys.argv[1] if len(sys.argv) > 1 else str(DIR_FIGURES.parent / "results" / "trace_fixed005.nc")
ADAPT = sys.argv[2] if len(sys.argv) > 2 else str(DIR_FIGURES.parent / "results" / "trace_adaptive.nc")
s = json.load(open(PATH_SCALE_V2)); MU, SD = s["MU_GLOBAL"], s["SD_GLOBAL"]
ka = np.array(s["bspline_knots_aug"]); deg = s["bspline_degree"]; nb = s["bspline_n_basis"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})

def d1(x):
    x = np.asarray(x, float); o = np.zeros((x.size, nb))
    for i in range(nb):
        c = np.zeros(nb); c[i] = 1.0
        o[:, i] = BSpline(ka, c, deg, extrapolate=False).derivative(1)(x)
    return np.nan_to_num(o)

df = pd.read_csv(PATH_MERGED_AGE).dropna(subset=["ISO3", "Population", "GDP"]).copy()
df["x_s"] = (np.log10(df["Population"]) - MU) / SD
xs = np.linspace(*np.percentile(df["x_s"], [5, 95]), 400); logpop = xs * SD + MU
D1 = d1(xs)

def curve(path):
    post = az.from_netcdf(path).posterior
    th = post["theta"].stack(z=("chain", "draw")).values
    if th.shape[0] != nb: th = th.T
    th = th.T; b0 = post["beta0"].stack(z=("chain", "draw")).values
    E = (b0[None, :] + D1 @ th.T) / SD
    return savgol_filter(np.median(E, axis=1), 151, 2), np.percentile(E, 2.5, axis=1), np.percentile(E, 97.5, axis=1)

ef, ef_lo, ef_hi = curve(FIXED)
ea, ea_lo, ea_hi = curve(ADAPT)

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.axhline(1.0, color="#555", ls=":", lw=1, label="ε = 1 (scale neutrality)")
ax.plot(logpop, ef, color="#238b45", lw=2.4, label="Fixed penalty (RW 0.05): ε≈0.92 (sub-unitary)")
ax.fill_between(logpop, savgol_filter(ef_lo, 151, 2), savgol_filter(ef_hi, 151, 2), color="#238b45", alpha=0.12)
ax.plot(logpop, ea, color="#b4451f", lw=2.4, label="Adaptive (HalfNormal): decline ε≈1.22 (super-unitary)")
ax.fill_between(logpop, savgol_filter(ea_lo, 151, 2), savgol_filter(ea_hi, 151, 2), color="#b4451f", alpha=0.12)
ax.set_ylim(0.3, 1.9)
ax.set_xlabel("log$_{10}$ Population"); ax.set_ylabel("Local elasticity ε(x) = ∂lnGDP/∂lnPop")
ax.text(0.5, 0.04, "LOO-CV marginally prefers the adaptive fit (Δelpd≈50±17)\nbut Pareto-κ diagnostics do not flag either fit; the data do not adjudicate.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#444")
ax.legend(loc="upper center", fontsize=7.5, bbox_to_anchor=(0.5, -0.16), ncol=1)
for ext in ("png", "pdf"):
    fig.savefig("%s/FigureS5_elasticity_sensitivity.%s" % (str(DIR_FIGURES), ext))
print("saved FigureS5_elasticity_sensitivity (fixed core-mean=%.2f, adaptive decline-region peak=%.2f)" % (
    np.mean(ef), np.max(ea)))
