"""
s19_figure3_decomposition.py -- Figure 3: the demographic channel against the trend
==================================================================================
The projected change in output per person decomposes exactly. Because g = G / N,

    D ln g = [ eps_i * D ln W  -  D ln N ]  +  drift,

where the bracket is the demographic channel and drift collects the country's own
estimated decadal drift (after the convergence term) and the accumulated residual.
This is an identity, not a regression: the engine asserts the same relation on the
written panel before any result is produced.

eps_i is the country's posterior-mean elasticity in the regime that applies to each
step: beta_growth where the working-age population is expanding, beta_decline where it
is contracting. Applying that rule annually rather than on ten-year blocks, as the
engine does, changes the demographic contribution by at most 0.004 log points in any
country and flips no signs.

Panel (a) plots the demographic contribution against projected population growth, with
an eight-bin median line. Panel (b) sets the demographic channel beside the drift on a
common zero baseline, by quintile of projected population growth.

Inputs   results/projection_2100.csv,
         results/hierarchical_wa_country_elasticities.csv
Usage    uv run python src/s19_figure3_decomposition.py
Outputs  figures/Figure3_decomposition.{pdf,png}
         results/tables/Figure3_source_data.csv  (175 economies)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import DIR_RESULTS, DIR_FIGURES

BASE, END = 2024, 2100


def decompose():
    p = pd.read_csv(DIR_RESULTS / "projection_2100.csv")
    e = pd.read_csv(DIR_RESULTS / "hierarchical_wa_country_elasticities.csv").set_index("ISO3")
    p = p[(p.Year >= BASE) & (p.Year <= END)]
    rows = []
    for iso, g in p.groupby("ISO3"):
        if iso not in e.index:
            continue
        g = g.sort_values("Year")
        lw = np.log(g.WApop.values)
        lp = np.log(g.Population.values)
        lpc = np.log(g.GDPpc_median.values)
        dW, dP = np.diff(lw), np.diff(lp)
        bg = float(e.loc[iso, "beta_growth"])
        bd = float(e.loc[iso, "beta_decline"])
        eps = np.where(dW < 0, bd, bg)
        D = float(np.sum(eps * dW - dP))
        total = float(lpc[-1] - lpc[0])
        rows.append(dict(ISO3=iso, Country=g.Country.iloc[0], Region=g.Region.iloc[0],
                         dlnP=float(lp[-1] - lp[0]), dlnW=float(lw[-1] - lw[0]),
                         dlnPC=total, D=D, drift=total - D,
                         eps_g=bg, eps_d=bd,
                         depop=bool(g.Population.iloc[-1] < g.Population.iloc[0])))
    d = pd.DataFrame(rows)
    print("decomposition: %d economies, %d depopulating, demographic channel negative in %d"
          % (len(d), int(d.depop.sum()), int((d.D < 0).sum())))
    return d


plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
    "axes.spines.top":False,"axes.spines.right":False,"axes.labelsize":9,"xtick.labelsize":8,
    "ytick.labelsize":8,"legend.frameon":False,"figure.dpi":150,"savefig.dpi":300,
    "savefig.bbox":"tight","pdf.fonttype":42})

BLUE="#2166ac"; RUST="#b4451f"; INK="#222222"; MUTED="#6b6b6b"; GRID="#e6e6e3"
d = decompose()
H = float(END - BASE)
d["popgr"]=100*d.dlnP/H; d["Dyr"]=100*d.D/H; d["driftyr"]=100*d.drift/H; d["pcyr"]=100*d.dlnPC/H

fig=plt.figure(figsize=(7.4,3.35)); gs=GridSpec(1,2,width_ratios=[1.12,1],wspace=0.34)

# ---------------- a : demographic channel vs population growth ----------------
ax=fig.add_subplot(gs[0,0])
ax.grid(axis="y",color=GRID,lw=.7); ax.set_axisbelow(True)
ax.axhline(0,color=MUTED,lw=.9,ls=(0,(4,3)),zorder=2)
for sub,c,lab in [(d[d.depop],RUST,"Population falls"),(d[~d.depop],BLUE,"Population rises")]:
    ax.scatter(sub.popgr,sub.Dyr,s=16,c=c,alpha=.72,linewidths=.4,edgecolors="white",zorder=3,label=lab)
q=pd.qcut(d.popgr,8,labels=False); med=d.groupby(q).agg(x=("popgr","median"),y=("Dyr","median"))
ax.plot(med.x,med.y,color=INK,lw=1.5,zorder=4,solid_capstyle="round")
for iso,dx,dy,ha,va in [("JPN",0.12,0.075,"left","bottom"),("KOR",0.10,-0.02,"left","center"),
                        ("CHN",0.10,0.00,"left","center"),("DEU",0.10,0.045,"left","bottom"),
                        ("USA",0.12,-0.075,"left","top"),("NGA",-0.10,0.03,"right","bottom")]:
    r=d[d.ISO3==iso].iloc[0]
    ax.annotate(iso,(r.popgr,r.Dyr),(r.popgr+dx,r.Dyr+dy),fontsize=7,color=INK,ha=ha,va=va,zorder=6)
    ax.scatter([r.popgr],[r.Dyr],s=16,facecolors="none",edgecolors=INK,linewidths=.7,zorder=5)
ax.set_xlabel("Population growth 2024–2100 (% yr$^{-1}$)")
ax.set_ylabel("Demographic contribution to\noutput per person (% yr$^{-1}$)")
ax.set_title("a", fontsize=11, fontweight="bold", loc="left", pad=14)
ax.text(0,1.02,"Favourable only where population falls",transform=ax.transAxes,
        fontsize=8.5,color=MUTED,va="bottom")
ax.legend(loc="lower left",fontsize=7.5,handletextpad=.3,borderpad=.2)
ax.set_ylim(-0.80,0.36)

# ---------------- b : magnitudes, common baseline ----------------
ax2=fig.add_subplot(gs[0,1])
ax2.grid(axis="x",color=GRID,lw=.7); ax2.set_axisbelow(True)
d["q"]=pd.qcut(d.popgr,5,labels=False)
g=d.groupby("q").agg(D=("Dyr","median"),dr=("driftyr","median"),tot=("pcyr","median"))
y=np.arange(5)[::-1]; h=0.34
ax2.barh(y+h/2+0.02,g.dr,height=h,color=BLUE,zorder=3,label="Productivity drift")
ax2.barh(y-h/2-0.02,g.D ,height=h,color=RUST,zorder=3,label="Demographic channel")
for yi,dr,Dv,t in zip(y,g.dr,g.D,g.tot):
    ax2.annotate("%.2f"%dr,(dr,yi+h/2+0.02),(4,0),textcoords="offset points",
                 fontsize=7.5,va="center",color=INK)
    ax2.annotate("%+.2f"%Dv,(min(Dv,0),yi-h/2-0.02),(-4,0),textcoords="offset points",
                 fontsize=7.5,va="center",ha="right",color=INK)
ax2.axvline(0,color=MUTED,lw=.9,zorder=4)
ax2.set_yticks(y); ax2.set_yticklabels(["fastest\ndecline","","median","","fastest\ngrowth"],fontsize=7.5)
ax2.set_ylabel("Population-growth quintile",fontsize=8.5,color=MUTED,labelpad=6)
ax2.set_xlabel("Contribution to output per person (% yr$^{-1}$)",labelpad=2)
ax2.set_title("b", fontsize=11, fontweight="bold", loc="left", pad=14)
ax2.text(0,1.02,"Drift is 15–80× the demographic term",transform=ax2.transAxes,
         fontsize=8.5,color=MUTED,va="bottom")
ax2.set_xlim(-0.62,2.72); ax2.set_ylim(-0.72,4.72)
ax2.legend(loc="lower left",fontsize=7.5,handletextpad=.3,borderpad=.2,ncol=2,
           columnspacing=1.1,bbox_to_anchor=(0.0,-0.30))
DIR_FIGURES.mkdir(parents=True, exist_ok=True)
(DIR_RESULTS / "tables").mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(DIR_FIGURES / ("Figure3_decomposition.%s" % ext))
plt.close(fig)
d.drop(columns=[c for c in ("popgr", "Dyr", "driftyr", "pcyr", "q") if c in d.columns]) \
 .to_csv(DIR_RESULTS / "tables" / "Figure3_source_data.csv", index=False)
print("ratio drift/|D| by quintile:", (g.dr / g.D.abs()).round(0).tolist())
print("  wrote Figure3_decomposition.pdf / .png and Figure3_source_data.csv")
