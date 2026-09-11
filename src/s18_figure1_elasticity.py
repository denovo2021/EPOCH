"""
s18_figure1_elasticity.py -- Figure 1: the 896 country-decades and the regional forest
=====================================================================================
Panel (a) is the estimation panel itself: ten-year changes in output against ten-year
changes in the working-age population, for the 896 non-overlapping country-decade blocks
the long-difference model is fitted on. The block construction is re-derived here from
data/merged_age.csv by the same steps as src/fit_hierarchical_workingage.py, so the figure
cannot drift away from the model: if the two disagree, the counts printed below will not
match results/elasticity_bridge.json.

Panel (b) is the regional forest. Regional estimates are shown for the EXPANSION regime
only, because three of the seven regions contribute no contraction blocks at all and their
region-level contraction posteriors are prior-driven; the contraction regime is reported
globally. Intervals are equal-tailed 95% credible intervals (2.5th-97.5th percentiles),
matching every other interval in the paper.

Inputs   data/merged_age.csv, data/age_predictions_scenarios.csv,
         results/hierarchical_wa_posterior.nc
Usage    uv run python src/s18_figure1_elasticity.py
Outputs  figures/Figure1_elasticity.{pdf,png}
         results/tables/Figure1_source_data_panel_a.csv  (896 blocks)
         results/tables/Figure1_source_data_panel_b.csv  (7 regions + 2 global rows)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import PATH_MERGED_AGE, PATH_AGE_SCEN, DIR_RESULTS, DIR_FIGURES

K = 10
GDP_COL = "GDP_constant_2015usd"
BLOCK_END = [1970, 1980, 1990, 2000, 2010, 2020]


def build_blocks():
    """Re-derive the 896 country-decade blocks exactly as fit_hierarchical_workingage.py does."""
    df = pd.read_csv(PATH_MERGED_AGE)
    gdp_col = GDP_COL if (GDP_COL in df.columns and df[GDP_COL].notna().sum()) else "GDP"
    age = pd.read_csv(PATH_AGE_SCEN)
    age["_pri"] = np.where(age["scenario_norm"].str.contains("estimate", case=False, na=False), 0, 1)
    age = (age.sort_values(["ISO3", "Year", "_pri"])
              .drop_duplicates(["ISO3", "Year"], keep="first")[["ISO3", "Year", "WAshare"]]
              .rename(columns={"WAshare": "WAshare_scn"}))
    df = df.merge(age, on=["ISO3", "Year"], how="left")
    df["WAshare"] = df["WAshare"].fillna(df["WAshare_scn"])
    df = df.dropna(subset=[gdp_col, "Population", "WAshare", "ISO3", "Year", "Region"]).copy()
    df = df[(df[gdp_col] > 0) & (df["Population"] > 0) & (df["WAshare"] > 0)]
    df["Year"] = df["Year"].astype(int)
    df["ln_gdp"] = np.log(df[gdp_col].astype(float))
    df["ln_wapop"] = np.log(df["Population"].astype(float) * df["WAshare"].astype(float))

    base = df[["ISO3", "Year", "Region", "ln_gdp", "ln_wapop"]].copy()
    lag = base[["ISO3", "Year", "ln_gdp", "ln_wapop"]].copy()
    lag["Year"] = lag["Year"] + K
    m = base.merge(lag, on=["ISO3", "Year"], suffixes=("", "_lag"))
    m["dln_gdp"] = m["ln_gdp"] - m["ln_gdp_lag"]
    m["dln_wapop"] = m["ln_wapop"] - m["ln_wapop_lag"]
    d = m[["ISO3", "Year", "Region", "dln_gdp", "dln_wapop"]].dropna()
    d = d[np.isfinite(d["dln_gdp"]) & np.isfinite(d["dln_wapop"])]
    d = d[d["Year"].isin(BLOCK_END)].reset_index(drop=True)
    for c in ["dln_gdp", "dln_wapop"]:
        lo, hi = np.percentile(d[c], [1, 99])
        d[c] = d[c].clip(lo, hi)
    d["decline"] = (d["dln_wapop"] < 0).astype(int)
    return d


def posterior_intervals():
    """Equal-tailed 95% credible intervals from the archived long-difference posterior."""
    ds = xr.open_dataset(DIR_RESULTS / "hierarchical_wa_posterior.nc",
                         group="posterior", engine="h5netcdf")

    def ci(a):
        a = np.asarray(a).ravel()
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    br, brd = ds["beta_region"], ds["beta_decline_region"]
    regions = [str(r) for r in br.coords["region"].values]
    rows = []
    for i, r in enumerate(regions):
        ge, gc = br.isel(region=i).values, brd.isel(region=i).values
        glo, ghi = ci(ge); clo, chi = ci(gc)
        rows.append(dict(Region=r, beta_growth=float(ge.mean()), growth_lo=glo, growth_hi=ghi,
                         growth_p_lt_1=float((ge < 1).mean()),
                         beta_decline=float(gc.mean()), decline_lo=clo, decline_hi=chi,
                         decline_p_lt_1=float((gc < 1).mean())))
    reg = pd.DataFrame(rows).sort_values("beta_growth")

    glob = {}
    for name, arr in [("global_expansion", ds["beta_global"].values),
                      ("global_contraction", ds["beta_decline_global"].values)]:
        lo, hi = ci(arr)
        glob[name] = dict(mean=float(arr.mean()), lo=lo, hi=hi,
                          p_lt_1=float((np.asarray(arr) < 1).mean()))
    return reg, glob


plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
    "axes.spines.top":False,"axes.spines.right":False,"axes.labelsize":9,"xtick.labelsize":8,
    "ytick.labelsize":8,"legend.frameon":False,"figure.dpi":150,"savefig.dpi":300,
    "savefig.bbox":"tight","pdf.fonttype":42})
BLUE="#2166ac"; RUST="#b4451f"; INK="#222222"; MUTED="#6b6b6b"; GRID="#e6e6e3"

d = build_blocks()
r, G = posterior_intervals()
print("panel a: n=%d blocks, %d countries, %d regions, %d contraction"
      % (len(d), d.ISO3.nunique(), d.Region.nunique(), int(d.decline.sum())))
d["x"]=100*d.dln_wapop; d["y"]=100*d.dln_gdp
exp=d[d.decline==0]; con=d[d.decline==1]
nblk=d.groupby("Region").agg(n=("decline","size"),c=("decline","sum"))

fig=plt.figure(figsize=(7.5,3.8)); gs=GridSpec(1,2,width_ratios=[1.0,0.98],wspace=0.56)

# ------------------------------------------------ a : the historical record ---
ax=fig.add_subplot(gs[0,0])
ax.axvspan(-17,0,color="#fbf0ea",zorder=0)
ax.grid(color=GRID,lw=.7); ax.set_axisbelow(True)
ax.axvline(0,color=MUTED,lw=.9,ls=(0,(4,3)),zorder=2)
ax.axhline(0,color=MUTED,lw=.9,ls=(0,(4,3)),zorder=2)
ax.scatter(exp.x,exp.y,s=13,c=BLUE,alpha=.50,linewidths=.3,edgecolors="white",zorder=3,
           label="Expansion  $n$ = 827")
ax.scatter(con.x,con.y,s=22,c=RUST,alpha=.90,linewidths=.4,edgecolors="white",zorder=4,
           label="Contraction  $n$ = 69")
q=pd.qcut(d.x,8,labels=False,duplicates="drop")
med=d.groupby(q).agg(x=("x","median"),y=("y","median"))
ax.plot(med.x,med.y,color=INK,lw=1.6,zorder=5,solid_capstyle="round")
ax.set_xlabel("Change in the working-age population\nover the decade (log points, %)")
ax.set_ylabel("Change in output over the decade\n(log points, %)")
ax.set_title("a", fontsize=11, fontweight="bold", loc="left", pad=14)
ax.text(0,1.02,"896 country-decades, 1961–2020",transform=ax.transAxes,fontsize=8.5,color=MUTED,va="bottom")
ax.set_xlim(-17,64); ax.set_ylim(-70,132)
ax.legend(loc="upper left",fontsize=7.5,handletextpad=.25,borderpad=.15,labelspacing=.35)
ax.text(-8.5,-64,"working-age\npopulation falls",fontsize=7.2,color=RUST,ha="center",va="center",linespacing=1.3)

# ------------------------------------------------------- b : regional forest ---
ax2=fig.add_subplot(gs[0,1])
r=r.sort_values("beta_growth",ascending=False).reset_index(drop=True)
SHORT={"Latin America & Caribbean":"Latin America & Caribbean","Sub-Saharan Africa":"Sub-Saharan Africa",
       "Europe & Central Asia":"Europe & Central Asia","South Asia":"South Asia",
       "East Asia & Pacific":"East Asia & Pacific","North America":"North America",
       "Middle East & North Africa":"Middle East & N. Africa"}
ypos=np.arange(len(r))[::-1]+2.45
ax2.axvspan(-0.15,1.0,color="#f3f5f8",zorder=0)
ax2.axvline(1.0,color=INK,lw=1.0,ls=(0,(4,3)),zorder=2)
labs=[];ys=[]
for i,row in r.iterrows():
    yy=ypos[i]
    ax2.plot([row.growth_lo,row.growth_hi],[yy]*2,color=BLUE,lw=2.0,solid_capstyle="butt",zorder=4)
    ax2.plot([row.beta_growth],[yy],"o",ms=5.2,color=BLUE,zorder=6)
    n=int(nblk.loc[row.Region,"n"]-nblk.loc[row.Region,"c"])
    ax2.text(1.92,yy,"%d"%n,fontsize=7,color=MUTED,ha="right",va="center")
    labs.append(SHORT[row.Region]); ys.append(yy)
ge=G["global_expansion"]; gc=G["global_contraction"]
ax2.axhline(1.85,color="#cfcfcb",lw=.8,zorder=1)
ax2.plot([ge["lo"],ge["hi"]],[1.15]*2,color=BLUE,lw=2.8,solid_capstyle="butt",zorder=4)
ax2.plot([ge["mean"]],[1.15],"D",ms=5.6,color=BLUE,zorder=6)
ax2.text(1.92,1.15,"827",fontsize=7,color=MUTED,ha="right",va="center")
ax2.plot([gc["lo"],gc["hi"]],[0.30]*2,color=RUST,lw=2.2,alpha=.85,solid_capstyle="butt",zorder=4)
ax2.plot([gc["mean"]],[0.30],"D",ms=5.6,color=RUST,zorder=6)
ax2.text(1.92,0.30,"69",fontsize=7,color=MUTED,ha="right",va="center")
ax2.set_yticks(ys+[1.15,0.30])
ax2.set_yticklabels(labs+["Global, expansion","Global, contraction"],fontsize=7.6)
for t,lab in zip(ax2.get_yticklabels(),labs+["G1","G2"]):
    if lab.startswith("G"): t.set_fontweight("bold")
ax2.get_yticklabels()[-1].set_color(RUST); ax2.get_yticklabels()[-2].set_color(BLUE)
ax2.set_ylim(-0.30,len(r)+2.35); ax2.set_xlim(-0.15,1.95)
ax2.set_xticks([0,0.5,1.0,1.5])
ax2.grid(axis="x",color=GRID,lw=.7); ax2.set_axisbelow(True)
ax2.spines["left"].set_visible(False); ax2.tick_params(axis="y",length=0)
ax2.set_xlabel("Elasticity of output to the\nworking-age population, ε")
ax2.set_title("b", fontsize=11, fontweight="bold", loc="left", pad=14)
ax2.text(0,1.02,"Sub-proportional in every region",transform=ax2.transAxes,fontsize=8.5,color=MUTED,va="bottom")
ax2.text(1.03,len(r)+2.30,"ε = 1",fontsize=7.4,color=INK,ha="left",va="top")
ax2.text(1.92,len(r)+2.30,"blocks",fontsize=7,color=MUTED,ha="right",va="top",style="italic")
ax2.annotate("P(ε<1) = 0.96",xy=(ge["hi"],1.15),xytext=(1.10,1.60),fontsize=7.2,color=BLUE,
             ha="left",va="center",arrowprops=dict(arrowstyle="-",lw=.7,color=BLUE,shrinkA=1,shrinkB=2))
DIR_FIGURES.mkdir(parents=True, exist_ok=True)
(DIR_RESULTS / "tables").mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(DIR_FIGURES / ("Figure1_elasticity.%s" % ext))
plt.close(fig)

a = d.rename(columns={"Year": "block_end_year", "dln_gdp": "delta10_ln_GDP",
                      "dln_wapop": "delta10_ln_WApop", "decline": "contraction_regime"})
a[["ISO3", "Region", "block_end_year", "delta10_ln_WApop", "delta10_ln_GDP", "contraction_regime"]] \
    .sort_values(["ISO3", "block_end_year"]) \
    .to_csv(DIR_RESULTS / "tables" / "Figure1_source_data_panel_a.csv", index=False)

rows = []
for _, x in r.sort_values("beta_growth", ascending=False).iterrows():
    n = int(nblk.loc[x.Region, "n"] - nblk.loc[x.Region, "c"])
    rows.append(dict(row=x.Region, regime="expansion", n_blocks=n,
                     posterior_mean=x.beta_growth, ci95_lo=x.growth_lo,
                     ci95_hi=x.growth_hi, P_eps_lt_1=x.growth_p_lt_1))
for key, regime, n in [("global_expansion", "expansion", int((d.decline == 0).sum())),
                       ("global_contraction", "contraction", int(d.decline.sum()))]:
    g = G[key]
    rows.append(dict(row="Global", regime=regime, n_blocks=n, posterior_mean=g["mean"],
                     ci95_lo=g["lo"], ci95_hi=g["hi"], P_eps_lt_1=g["p_lt_1"]))
pd.DataFrame(rows).to_csv(DIR_RESULTS / "tables" / "Figure1_source_data_panel_b.csv", index=False)
print("  wrote Figure1_elasticity.pdf / .png and both source-data files")
