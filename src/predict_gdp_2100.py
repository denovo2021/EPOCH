"""
predict_gdp_2100.py  (EPOCH3 — working-age driver + dynamic growth/decline asymmetry)
=====================================================================================
Project real GDP (constant 2015 USD) to 2100 under the hierarchical WORKING-AGE
elasticity model (results/hierarchical_wa_posterior.nc), with two upgrades retained
from the hardened version plus two new structural ones:

  NEW (1) WORKING-AGE DRIVER. The demographic growth term is driven by the forecasted
      working-age population (15-64), WApop = Population * WAshare, built by applying the
      projected WAshare (age_predictions_scenarios.csv, Medium variant) to the UN WPP
      Medium population trajectory to 2100 — not total population.
  NEW (2) DYNAMIC ASYMMETRY. For each decade block the regime is set by the sign of the
      block's working-age growth:
          Delta ln(WApop) >= 0  ->  slope = beta_country               (expansion)
          Delta ln(WApop) <  0  ->  slope = beta_country + delta_region (contraction)
      and that slope is applied to every annual step within the block.

  (A) CONVERGENCE on the drift (unchanged). Decadal per-capita drift mean-reverts from
      alpha_c toward an advanced-economy steady state s_ss as per-capita GDP (GDP/total
      population) approaches the global frontier (US per-capita, growing at s_ss).
  (B) RESIDUAL SHOCKS (unchanged). Decadal residual SD sigma injected as annual
      innovations sigma/sqrt(10), accumulating into widening predictive bands.

GDP law (annual step):
   ln GDP_c(t+1) = ln GDP_c(t) + alpha_eff/10
                   + slope_block * [ln WApop(t+1) - ln WApop(t)] + N(0, sigma/sqrt(10))
   slope_block = beta_c                  if  Delta10 ln WApop(block) >= 0
               = beta_c + delta_region   if  Delta10 ln WApop(block) <  0

Run from the repository root in the project's virtual environment:
   python src/predict_gdp_2100.py [--draws 1000] [--s_ss 0.175]
                                                           [--no-converge] [--no-residual]
Outputs: results/gdp_rankings_2100.csv , gdp_projection_2100_topN.csv ; figures/Figure3_top10_gdp_2100.*
"""
import argparse
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PATH_POP_SCEN, PATH_AGE_SCEN, PATH_MERGED_AGE, DIR_RESULTS, DIR_FIGURES

ap = argparse.ArgumentParser()
ap.add_argument("--draws", type=int, default=1000)
ap.add_argument("--s_ss", type=float, default=0.175, help="frontier steady-state decadal drift (~1.75%/yr)")
ap.add_argument("--scenario", default="Medium variant")
ap.add_argument("--no-converge", dest="converge", action="store_false")
ap.add_argument("--no-residual", dest="residual", action="store_false")
args = ap.parse_args()
BASE = 2024; YEARS = np.arange(2024, 2101); TABLE_YEARS = [2024, 2050, 2075, 2100]
BLOCK_ENDS = list(range(2030, 2101, 10))   # decade-block grid for the projection horizon
S_SS = args.s_ss

# ---- posterior (full InferenceData -> open the posterior group) ----
post = xr.open_dataset(DIR_RESULTS / "hierarchical_wa_posterior.nc", group="posterior")
isos = [str(c) for c in post["country"].values]
regions = [str(r) for r in post["region"].values]
A = post["alpha_country"].stack(z=("chain", "draw")).values     # (C, S)
B = post["beta_country"].stack(z=("chain", "draw")).values      # (C, S) growth slope
D = post["delta_region"].stack(z=("chain", "draw")).values      # (R, S) decline increment
nS = A.shape[1]; idx = np.random.default_rng(42).choice(nS, min(args.draws, nS), replace=False)
A = A[:, idx]; B = B[:, idx]; D = D[:, idx]
sigma = float(post["sigma"].stack(z=("chain", "draw")).values.mean())
aidx = {iso: i for i, iso in enumerate(isos)}
r2i = {r: i for i, r in enumerate(regions)}

# ---- base GDP, names, region map ----
m = pd.read_csv(PATH_MERGED_AGE)
g = m[m["Year"] == 2023].dropna(subset=["GDP_constant_2015usd"]).set_index("ISO3")["GDP_constant_2015usd"]
cname = m.drop_duplicates("ISO3").set_index("ISO3")["Country Name"]
iso2region = m.drop_duplicates("ISO3").set_index("ISO3")["Region"].to_dict()

# ---- total population trajectory (frontier / per-capita) ----
pop = pd.read_csv(PATH_POP_SCEN); pop = pop[pop["Scenario"] == args.scenario]
popw = pop.pivot_table(index="ISO3", columns="Year", values="Population")
lp = np.log(popw)

# ---- working-age population trajectory: WApop = Population * WAshare (15-64) ----
age = pd.read_csv(PATH_AGE_SCEN); age = age[age["Scenario"] == args.scenario]
waw = age.pivot_table(index="ISO3", columns="Year", values="WAshare")
wapopw = popw * waw                          # pandas aligns on ISO3 x Year; NaN where either missing
lwap = np.log(wapopw)

yrs_in_pop = set(popw.columns); yrs_in_age = set(waw.columns)
def _full(iso, frame):
    return iso in frame.index and np.all(np.isfinite(frame.loc[iso, YEARS].values))
usable = [iso for iso in isos
          if iso in g.index and set(YEARS).issubset(yrs_in_pop) and set(YEARS).issubset(yrs_in_age)
          and _full(iso, lp) and _full(iso, lwap)
          and iso2region.get(iso) in r2i]

# global frontier: US per-capita 2024, growing at s_ss (on total population)
ln_yF0 = np.log(g["USA"] / popw.loc["USA", BASE])
ln_yF = ln_yF0 + S_SS * (YEARS - BASE) / 10.0          # (nyears,)
print("Projecting %d countries | working-age driver | convergence=%s residual=%s s_ss=%.3f(dec)=%.2f%%/yr"
      % (len(usable), args.converge, args.residual, S_SS, (np.exp(S_SS/10)-1)*100))


def project(iso):
    i = aidx[iso]; a = A[i]; b = B[i]; Sd = a.shape[0]
    delta = D[r2i[iso2region[iso]]]                    # region decline increment (S,)
    slope_growth = b
    slope_decline = b + delta
    lnP = lp.loc[iso, YEARS].values                    # total pop (frontier / per-capita)
    lnW = lwap.loc[iso, YEARS].values                  # working-age pop (elasticity driver)

    # ---- classify each decade block by working-age growth sign ----
    block_decline = {}
    for E in BLOCK_ENDS:
        Sy = max(E - 10, BASE)                          # first block is the partial 2024->2030
        block_decline[E] = (lnW[E - BASE] - lnW[Sy - BASE]) < 0.0

    lnG = np.full(Sd, np.log(g[iso]))                  # all draws start at observed GDP_2023
    ln_y0 = np.log(g[iso]) - lnP[0]; gap0 = ln_yF0 - ln_y0
    rng = np.random.default_rng(42 + sum(b for b in iso.encode("ascii")))
    med = np.empty(len(YEARS)); lo = np.empty(len(YEARS)); hi = np.empty(len(YEARS))
    G0 = np.exp(lnG); med[0] = np.median(G0); lo[0] = np.percentile(G0, 2.5); hi[0] = np.percentile(G0, 97.5)
    for j in range(1, len(YEARS)):
        # decade block this step falls in -> regime-dependent slope
        E = min(int(np.ceil(YEARS[j] / 10.0) * 10), 2100)
        slope = slope_decline if block_decline[E] else slope_growth
        # drift convergence (on total-population per-capita gap)
        if args.converge and gap0 > 1e-6:
            gap = ln_yF[j-1] - (lnG - lnP[j-1])         # per-capita gap (S,)
            w = np.clip(gap / gap0, 0.0, 1.0)
            alpha_eff = S_SS + (a - S_SS) * w
        elif args.converge:                              # at/above frontier -> steady state
            alpha_eff = np.full(Sd, S_SS)
        else:
            alpha_eff = a
        step = alpha_eff / 10.0 + slope * (lnW[j] - lnW[j-1])
        if args.residual:
            step = step + rng.normal(0.0, sigma / np.sqrt(10.0), Sd)
        lnG = lnG + step
        G = np.exp(lnG); med[j] = np.median(G); lo[j] = np.percentile(G, 2.5); hi[j] = np.percentile(G, 97.5)
    return med, lo, hi


med = {}; lo = {}; hi = {}
for iso in usable:
    med[iso], lo[iso], hi[iso] = project(iso)
MED = pd.DataFrame(med, index=YEARS)

rows = []
for iso in usable:
    r = {"ISO3": iso, "Country": cname.get(iso, iso)}
    for y in TABLE_YEARS: r["GDP_%d" % y] = MED.loc[y, iso]
    rows.append(r)
rk = pd.DataFrame(rows)
for y in TABLE_YEARS: rk["rank_%d" % y] = rk["GDP_%d" % y].rank(ascending=False).astype(int)
rk = rk.sort_values("GDP_2100", ascending=False).reset_index(drop=True)
rk.to_csv(DIR_RESULTS / "gdp_rankings_2100.csv", index=False)
top10 = rk.sort_values("GDP_2024", ascending=False).head(10)["ISO3"].tolist()

print("\nTop-10 by 2024 GDP:", top10)
disp = rk.head(12).copy()
for y in TABLE_YEARS: disp["GDP_%d" % y] = (disp["GDP_%d" % y] / 1e12).round(1)
print(disp[["Country"] + ["GDP_%d" % y for y in TABLE_YEARS] + ["rank_2024", "rank_2100"]].to_string(index=False))

band = []
for iso in top10:
    for j, y in enumerate(YEARS):
        band.append({"ISO3": iso, "Country": cname.get(iso, iso), "Year": int(y),
                     "GDP_median": med[iso][j], "GDP_lo": lo[iso][j], "GDP_hi": hi[iso][j]})
pd.DataFrame(band).to_csv(DIR_RESULTS / "gdp_projection_2100_topN.csv", index=False)

# ---- Figure 6 ----
DIR_FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
cmap = plt.get_cmap("tab10"); fig, ax = plt.subplots(figsize=(9.0, 6.0))
fig.subplots_adjust(right=0.84)
hist = m[(m["Year"] >= 2020) & (m["Year"] <= 2023)].dropna(subset=["GDP_constant_2015usd"])
colors = {}
for k, iso in enumerate(top10):
    c = cmap(k % 10); colors[iso] = c
    ax.fill_between(YEARS, lo[iso] / 1e12, hi[iso] / 1e12, color=c, alpha=0.12)
    ax.plot(YEARS, med[iso] / 1e12, color=c, lw=2, label=cname.get(iso, iso))
    h = hist[hist["ISO3"] == iso].sort_values("Year")
    if not h.empty: ax.plot(h["Year"], h["GDP_constant_2015usd"] / 1e12, color=c, lw=1.4, ls=":")

# dodge overlapping country labels on the right edge
label_info = sorted([(med[iso][-1] / 1e12, iso) for iso in top10], key=lambda x: x[0])
log_pos = [np.log10(y) for y, _ in label_info]
min_gap = 0.085
for _ in range(300):
    moved = False
    for i in range(1, len(log_pos)):
        if log_pos[i] - log_pos[i - 1] < min_gap:
            mid = (log_pos[i] + log_pos[i - 1]) / 2
            log_pos[i - 1] = mid - min_gap / 2
            log_pos[i] = mid + min_gap / 2
            moved = True
    if not moved:
        break
for (y_orig, iso), lp in zip(label_info, log_pos):
    y_adj = 10 ** lp
    if abs(lp - np.log10(y_orig)) > 0.008:
        ax.annotate("", xy=(2100.2, y_orig), xytext=(2101.5, y_adj),
                    arrowprops=dict(arrowstyle="-", color=colors[iso], lw=0.6, alpha=0.4),
                    clip_on=False)
    ax.text(2101.8, y_adj, iso, color=colors[iso], fontsize=7.5, va="center", clip_on=False,
            fontweight="bold")

ax.set_yscale("log"); ax.set_xlim(2020, 2108); ax.set_xlabel("Year")
ax.set_ylabel("Real GDP (trillions, constant 2015 USD) — log scale")
ax.legend(loc="upper left", fontsize=7.6, ncol=2)
for ext in ("png", "pdf"): fig.savefig("%s/Figure3_top10_gdp_2100.%s" % (str(DIR_FIGURES), ext))
print("\nSaved Figure3_top10_gdp_2100.png + rankings/projection CSVs")
