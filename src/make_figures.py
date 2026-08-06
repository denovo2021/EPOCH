"""
make_figures.py — Figures 1, 2 and S4 from the hierarchical B-spline posterior.
Loads the posterior with xr.open_dataset. Handles either a sub-unitary elasticity
(no tipping point) or a non-linear elasticity that crosses 1 (tipping point shown).
"""
import json, os
import numpy as np, pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter
from config import PATH_MODEL_V2, PATH_SCALE_V2, PATH_MERGED_AGE, DIR_FIGURES, DIR_RESULTS

OUTDIR = str(DIR_FIGURES)
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelsize": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})

s = json.load(open(PATH_SCALE_V2))
MU = s["MU_GLOBAL"]; SD = s["SD_GLOBAL"]; ka = np.array(s["bspline_knots_aug"])
deg = s["bspline_degree"]; nb = s["bspline_n_basis"]; cm = np.array(s["bspline_col_means"])
try:
    post = xr.open_dataset(PATH_MODEL_V2, group="posterior")
except Exception:
    post = xr.open_dataset(PATH_MODEL_V2)
theta = post["theta"].stack(z=("chain", "draw")).values
if theta.shape[0] != nb:
    theta = theta.T
theta = theta.T   # (S, nb)
beta0 = post["beta0"].stack(z=("chain", "draw")).values
alpha0 = post["alpha0"].stack(z=("chain", "draw")).values
tm = theta.mean(0); b0 = beta0.mean()

def basis(x, nu=0):
    x = np.asarray(x, float); o = np.zeros((x.size, nb))
    for i in range(nb):
        c = np.zeros(nb); c[i] = 1.0
        bs = BSpline(ka, c, deg, extrapolate=False)
        o[:, i] = (bs.derivative(nu)(x) if nu else bs(x))
    return np.nan_to_num(o)

df = pd.read_csv(PATH_MERGED_AGE).dropna(subset=["ISO3", "Population", "GDP"]).copy()
df["Year"] = df["Year"].astype(int); df["lp"] = np.log10(df["Population"]); df["lg"] = np.log10(df["GDP"])
df = df.sort_values(["ISO3", "Year"]); df["dlp"] = df.groupby("ISO3")["lp"].diff()
df["x_s"] = (df["lp"] - MU) / SD
lo_x, hi_x = np.percentile(df["x_s"], [1, 99])
lo_c, hi_c = np.percentile(df["x_s"], [5, 95])
xs = np.linspace(lo_x, hi_x, 400); logpop = xs * SD + MU
core = (xs >= lo_c) & (xs <= hi_c)

# fitted association curve (deviation form), with credible band
Bx = basis(xs, 0) - cm
ridx = np.random.default_rng(0).choice(theta.shape[0], min(800, theta.shape[0]), replace=False)
G = (alpha0[ridx][None, :] + np.outer(xs, beta0[ridx]) + (Bx @ theta[ridx].T))
g_med = np.median(G, axis=1); g_lo = np.percentile(G, 2.5, axis=1); g_hi = np.percentile(G, 97.5, axis=1)

# elasticity curve
Bp = basis(xs, 1)
E = (beta0[ridx][None, :] + Bp @ theta[ridx].T) / SD
e_med = np.median(E, axis=1); e_lo = np.percentile(E, 2.5, axis=1); e_hi = np.percentile(E, 97.5, axis=1)
e_smooth = savgol_filter(e_med, 151, 2)
cr = np.where(np.diff(np.sign(e_smooth - 1.0)) != 0)[0]
def interp(k):
    return xs[k] - (e_smooth[k]-1)*(xs[k+1]-xs[k])/((e_smooth[k+1]-1)-(e_smooth[k]-1))
cross = [interp(k) for k in cr if core[k]]
xstar = min(cross, key=lambda v: abs(v)) if cross else None
pop_star = (10**(xstar*SD+MU)) if xstar is not None else None

# The crossing annotated on the figure must be the crossing the paper reports. This script
# smooths its own curve on a 1st-99th-percentile grid from an 800-draw subsample, while the
# reporting script (src/s13_levels_elasticity.py, and elasticity_inflection.py before it)
# uses a 5th-95th grid and every draw. Both are defensible; only one is cited, and a figure
# annotating the other is the same class of defect as a table built from two runs.
# So: if the reporting file describes this posterior, take its value.
def _basename(path):
    """Basename that works whichever platform wrote the path into the JSON."""
    return str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

# run_full_pipeline.ps1 creates the deployed posterior by copying the adaptive trace, so the
# alias and the trace are the same file under two names.
_ALIAS = {"hierarchical_model_rcs_v2.nc": "trace_adaptive.nc"}

_report = DIR_RESULTS / "levels_elasticity.json"
if _report.exists():
    _specs = json.load(open(_report)).get("specifications", {})
    _want = _basename(PATH_MODEL_V2)
    _want = _ALIAS.get(_want, _want)
    _key = next((k for k, v in _specs.items()
                 if _basename(v.get("posterior", "")) == _want), None)
    if _key is None:
        print("NOTE: %s describes no specification matching %s, so the tipping point below is "
              "this script's own and may differ from the one the manuscript reports."
              % (_report.name, _want))
    else:
        _tip = _specs[_key].get("unity_crossing_population")
        print("tipping point: figure routine %s, reporting script (%s) %s -- using the latter"
              % (("%.3gM" % (pop_star / 1e6)) if pop_star else "none",
                 _key, ("%.3gM" % (_tip / 1e6)) if _tip else "none"))
        pop_star = _tip
else:
    print("NOTE: %s not found; using this script's own tipping point." % _report.name)

# phase averages
def phase_e(sub):
    dbar = basis(sub["x_s"].values, 1).mean(axis=0)
    return (beta0 + theta @ dbar) / SD
ed = phase_e(df[df["dlp"] < 0]); eg = phase_e(df[df["dlp"] > 0])

res = {"decline_elasticity": {"mean": float(ed.mean())}, "growth_elasticity": {"mean": float(eg.mean())},
       "tipping_point_pop": (float(pop_star) if pop_star else None),
       "core_mean_elasticity": float(e_smooth[core].mean())}
json.dump(res, open(DIR_RESULTS / "elasticity_figure_summary.json", "w"), indent=2)
print("decline e=%.3f growth e=%.3f core-mean=%.3f tipping=%s" % (
    ed.mean(), eg.mean(), e_smooth[core].mean(), ("%.2gM" % (pop_star/1e6) if pop_star else "none (eps<1)")))

# ================= FIGURE 1 =================
fig = plt.figure(figsize=(7.1, 3.4)); gs = GridSpec(1, 2, width_ratios=[1.05, 1], wspace=0.36)
ax = fig.add_subplot(gs[0, 0])
samp = df.dropna(subset=["lg"]).sample(min(3000, df["lg"].notna().sum()), random_state=1)
ax.scatter(samp["lp"], samp["lg"], s=4, c="#c9d6e5", alpha=0.5, linewidths=0, label="Country-years")
gx_data = alpha0.mean() + beta0.mean()*df["x_s"].values + (basis(df["x_s"].values, 0)-cm) @ tm
shift = np.nanmean(df["lg"].values - gx_data)
ax.plot(logpop, g_med+shift, color="#1f4e79", lw=2, label="Posterior median")
ax.fill_between(logpop, g_lo+shift, g_hi+shift, color="#1f4e79", alpha=0.18, label="95% credible band")
if pop_star is not None:
    ax.axvline(np.log10(pop_star), color="#b4451f", ls="--", lw=1.2)
    ax.text(np.log10(pop_star)+0.05, ax.get_ylim()[0]+0.4, "ε=1 at ~%.2gM" % (pop_star/1e6), color="#b4451f", fontsize=7.5)
ax.set_xlabel("log$_{10}$ Population"); ax.set_ylabel("log$_{10}$ Total GDP (USD)")
ax.set_title("a  Population–GDP association", fontsize=10.5); ax.legend(loc="upper left", fontsize=7.5)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(logpop[core], e_smooth[core], color="#1f4e79", lw=2.2, label="Local elasticity ε(x)")
ax2.fill_between(logpop[core], savgol_filter(e_lo, 151, 2)[core], savgol_filter(e_hi, 151, 2)[core], color="#1f4e79", alpha=0.14)
ax2.axhline(1.0, color="#555", ls=":", lw=1)
ax2.axhline(ed.mean(), color="#8c2d04", lw=1.3); ax2.axhline(eg.mean(), color="#238b45", lw=1.3)
ax2.text(logpop[core][2], ed.mean()+0.04, "decline-phase mean %.2f" % ed.mean(), color="#8c2d04", fontsize=7.5)
ax2.text(logpop[core][2], eg.mean()-0.10, "growth-phase mean %.2f" % eg.mean(), color="#238b45", fontsize=7.5)
if pop_star is not None:
    ax2.axvline(np.log10(pop_star), color="#b4451f", ls="--", lw=1.2, label="ε=1 at ~%.2gM" % (pop_star/1e6))
else:
    ax2.text(0.97, 0.06, "ε<1 throughout ⇒ divergence:\nper-capita sustained as scale contracts",
             transform=ax2.transAxes, fontsize=7, ha="right", va="bottom", color="#444")
ax2.set_ylim(min(0.6, e_smooth[core].min()-0.1), max(1.35, e_smooth[core].max()+0.1))
ax2.set_xlim(logpop[core].min(), logpop[core].max())
ax2.set_xlabel("log$_{10}$ Population"); ax2.set_ylabel("Elasticity ∂lnGDP/∂lnPop")
ax2.set_title("b  Local elasticity", fontsize=10.5); ax2.legend(loc="upper right", fontsize=7.5)
for ext in ("png", "pdf"):
    fig.savefig("%s/Figure1_population_GDP_association.%s" % (OUTDIR, ext))
plt.close(fig); print("Figure 1 saved")

# ================= FIGURE 2 =================
d = np.load(str(DIR_RESULTS / "backtest_predictions.npz"), allow_pickle=True)
y = d["y"]; med = d["med"]; lo = d["lo"]; hi = d["hi"]
inside = (y >= lo) & (y <= hi); cov = inside.mean(); mae = np.mean(np.abs(med - y))
fig2 = plt.figure(figsize=(7.1, 3.4)); gs2 = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.34)
axa = fig2.add_subplot(gs2[0, 0])
lim = [min(y.min(), med.min())-0.3, max(y.max(), med.max())+0.3]
axa.plot(lim, lim, color="#888", lw=1, ls="--")
axa.scatter(y[~inside], med[~inside], s=7, c="#d1495b", alpha=0.6, linewidths=0, label="Outside 95% PI")
axa.scatter(y[inside], med[inside], s=7, c="#2e6e8e", alpha=0.5, linewidths=0, label="Inside 95% PI")
axa.set_xlim(lim); axa.set_ylim(lim)
axa.set_xlabel("Observed log$_{10}$ GDP (held-out)"); axa.set_ylabel("Predicted log$_{10}$ GDP (median)")
axa.set_title("a  Predicted vs observed", fontsize=10.5)
axa.text(0.05, 0.92, "MAE(log$_{10}$) = %.3f\n95%% PI coverage = %.1f%%\nn = %d" % (mae, cov*100, len(y)),
         transform=axa.transAxes, fontsize=8, va="top")
axa.legend(loc="lower right", fontsize=7.5)
axb = fig2.add_subplot(gs2[0, 1])
order = np.argsort(y); order = order[::max(1, len(order)//120)]
xx = np.arange(len(order))
axb.fill_between(xx, lo[order], hi[order], color="#bcd4e6", alpha=0.8, label="95% PI")
axb.plot(xx, med[order], color="#1f4e79", lw=0.8, label="Predicted median")
axb.scatter(xx, y[order], s=6, c="#222", label="Observed", zorder=3)
axb.set_xlabel("Held-out country-years (sorted)"); axb.set_ylabel("log$_{10}$ GDP")
axb.set_title("b  Interval calibration", fontsize=10.5); axb.legend(loc="upper left", fontsize=7.5)
for ext in ("png", "pdf"):
    fig2.savefig("%s/Figure2_backtest_accuracy.%s" % (OUTDIR, ext))
plt.close(fig2); print("Figure 2 saved")

# ================= FIGURE S4 =================
figs = plt.figure(figsize=(4.5, 3.4)); axs = figs.add_subplot(111)
axs.plot(logpop[core], e_smooth[core], color="#1f4e79", lw=2.2)
axs.fill_between(logpop[core], savgol_filter(e_lo, 151, 2)[core], savgol_filter(e_hi, 151, 2)[core], color="#1f4e79", alpha=0.14, label="95% credible band")
axs.axhline(1.0, color="#555", ls=":", lw=1, label="ε=1 (scale neutrality)")
axs.axhline(ed.mean(), color="#8c2d04", lw=1.2); axs.axhline(eg.mean(), color="#238b45", lw=1.2)
if pop_star is not None:
    axs.axvline(np.log10(pop_star), color="#b4451f", ls="--", lw=1.2, label="ε=1 at ~%.2gM" % (pop_star/1e6))
axs.set_ylim(min(0.6, e_smooth[core].min()-0.1), max(1.35, e_smooth[core].max()+0.1))
axs.set_xlim(logpop[core].min(), logpop[core].max())
axs.set_xlabel("log$_{10}$ Population"); axs.set_ylabel("Local elasticity ε(x$_s$)")
axs.set_title("Local population–output elasticity"); axs.legend(fontsize=7.5)
for ext in ("png", "pdf"):
    figs.savefig("%s/FigureS4_local_elasticity.%s" % (OUTDIR, ext))
plt.close(figs); print("Figure S4 saved")
