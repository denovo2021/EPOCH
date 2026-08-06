"""
s10_make_figures_2100.py -- every projection figure, from the single-source CSVs
==============================================================================
Reads only what s07_projection_2100.py / s09_decision_layer.py wrote, so no figure can
disagree with Table 2 or with another figure. Produces, at 300 dpi and 180 mm max width:

  Figure3_projection.{pdf,png}        (a) real GDP and (b) real GDP per capita to 2100 for
                                      the ten largest 2024 economies. Colour encodes the
                                      sign of population change, so the reordering between
                                      the panels is the finding rather than a caption claim.
  Figure4_divergence.{pdf,png}        aggregate against per-capita growth, 175 economies,
                                      with the 45-degree line. Distance from the line is
                                      the population growth rate by identity.
  Figure5_two_elasticities.{pdf,png}  the flagship. (a) 2100 per-capita output under the two
                                      elasticities the levels model cannot separate;
                                      (b) the wedge between them against population growth.
  Figure6_decision_layer.{pdf,png}    (a) demographic pressure on age-related outlays, which
                                      is the same under either elasticity; (b) what the
                                      elasticity decides -- the sustainable real benefit
                                      per retiree.

Palette: validated categorical slots (blue #2a78d6, orange #eb6834) and the blue-gray-red
diverging pair, checked with the design-system validator on a white surface for
protan/deutan/tritan separation. No series relies on hue alone: every mark carries a direct
label or a legend entry.

Usage:  python src/s10_make_figures_2100.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from config import DIR_RESULTS, DIR_FIGURES

# ------------------------------------------------------------------------------- palette
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("epoch_div", [RED, NEUTRAL, BLUE], N=256)

MM = 1 / 25.4
W_FULL = 180 * MM          # journal double-column maximum
W_HALF = 88 * MM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, stem):
    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(DIR_FIGURES / ("%s.%s" % (stem, ext)))
    plt.close(fig)
    print("  wrote %s.pdf / .png" % stem)


def panel_letter(ax, s, dx=-0.085, dy=1.045):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            color=INK, va="top", ha="left")


def label_gap(ax, fontsize, pad=1.35):
    """Minimum separation, in log10 data units, that two stacked labels need on this axis.
    Computed from the rendered axis height so the dodge cannot be silently too small."""
    lo, hi = ax.get_ylim()
    span = np.log10(hi) - np.log10(lo)
    h_in = ax.get_window_extent().height / ax.figure.dpi
    return (fontsize * pad / 72.0) / h_in * span


def dodge(values, min_gap):
    """Spread label positions apart on a log axis without changing their order."""
    pos = list(values)
    for _ in range(400):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < min_gap:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - min_gap / 2, mid + min_gap / 2
                moved = True
        if not moved:
            break
    return pos


# =========================================================================== FIGURE 3
def figure3(long_df, summ):
    top10 = summ.nsmallest(10, "rank_agg_2024").ISO3.tolist()
    shrink = set(summ[summ.g_pop < 0].ISO3)

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.7))
    fig.subplots_adjust(wspace=0.42)

    for ax, col, ttl, ylab, scale in [
        (axes[0], "GDP_median", "Aggregate output",
         "Real GDP (trillion, constant 2015 USD)", 1e12),
        (axes[1], "GDPpc_median", "Output per person",
         "Real GDP per capita (thousand, constant 2015 USD)", 1e3)]:
        ends = []
        for iso in top10:
            g = long_df[long_df.ISO3 == iso].sort_values("Year")
            c = ORANGE if iso in shrink else BLUE
            ax.plot(g.Year, g[col] / scale, color=c, lw=1.1, alpha=0.92,
                    solid_capstyle="round")
            ends.append((np.log10(g[col].iloc[-1] / scale), iso, c))
        ax.set_yscale("log")
        ax.set_xlim(2024, 2100)
        ax.autoscale(axis="y")
        fig.canvas.draw()
        ends.sort()
        gap = label_gap(ax, 6.0)
        for (y0, iso, c), y in zip(ends, dodge([e[0] for e in ends], gap)):
            if abs(y - y0) > 0.004:
                ax.plot([2101, 2107], [10 ** y0, 10 ** y], color=c, lw=0.4, alpha=0.45,
                        clip_on=False)
            ax.text(2109, 10 ** y, iso, color=c, fontsize=6, va="center", ha="left",
                    fontweight="bold", clip_on=False)
        ax.set_xticks([2024, 2040, 2060, 2080, 2100])
        ax.set_xlabel("Year")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, color=INK, pad=6, loc="left", fontsize=7.5)
        ax.grid(axis="y", which="major")
        ax.set_axisbelow(True)

    axes[0].legend(handles=[
        Line2D([], [], color=ORANGE, lw=1.4, label="Population falls 2024–2100"),
        Line2D([], [], color=BLUE, lw=1.4, label="Population rises 2024–2100")],
        loc="upper left", handlelength=1.4, borderpad=0.2)
    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    save(fig, "Figure3_projection")


# =========================================================================== FIGURE 4
def figure4(summ):
    fig, ax = plt.subplots(figsize=(W_FULL * 0.66, 4.15))
    x, y, p = summ.g_agg * 100, summ.g_pc * 100, summ.g_pop * 100
    lim = [min(x.min(), y.min()) - 0.25, max(x.max(), y.max()) + 0.25]

    ax.fill_between(lim, lim, lim[1], color=ORANGE, alpha=0.055, lw=0, zorder=0)
    ax.plot(lim, lim, color=BASELINE, lw=0.8, zorder=1)
    ax.axhline(0, color=GRID, lw=0.5, zorder=0)
    ax.axvline(0, color=GRID, lw=0.5, zorder=0)

    norm = TwoSlopeNorm(vmin=p.min(), vcenter=0.0, vmax=p.max())
    sc = ax.scatter(x, y, c=p, cmap=DIVERGING, norm=norm, s=15, linewidths=0.35,
                    edgecolors="white", zorder=3)

    ax.text(lim[0] + 0.15, lim[1] - 0.20,
            "per-capita growth exceeds aggregate growth\n(population falling)",
            fontsize=6, color=INK2, va="top", ha="left", linespacing=1.35)

    for iso, dx, dy, ha in [("JPN", -0.12, 0.10, "right"), ("ITA", -0.12, 0.08, "right"),
                            ("CHN", 0.10, 0.10, "left"), ("KOR", -0.12, 0.06, "right"),
                            ("USA", 0.10, -0.12, "left"), ("IND", 0.10, -0.10, "left"),
                            ("NGA", -0.10, -0.14, "right"), ("DEU", 0.10, 0.06, "left")]:
        r = summ[summ.ISO3 == iso]
        if r.empty:
            continue
        ax.annotate(iso, (r.g_agg.iloc[0] * 100, r.g_pc.iloc[0] * 100),
                    xytext=(r.g_agg.iloc[0] * 100 + dx, r.g_pc.iloc[0] * 100 + dy),
                    fontsize=6, color=INK, ha=ha, va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4,
                                    shrinkA=0, shrinkB=2))

    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("Population growth, 2024–2100 (% yr$^{-1}$)", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2, color=MUTED)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Aggregate real GDP growth, 2024–2100 (% yr$^{-1}$)")
    ax.set_ylabel("Real GDP per capita growth (% yr$^{-1}$)")
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    save(fig, "Figure4_divergence")


# =========================================================================== FIGURE 5
def figure5(a_summ, b_summ, panel):
    a = a_summ.set_index("ISO3")
    b = b_summ.set_index("ISO3").loc[a.index]
    sel = [i for i in panel if i in a.index]
    sel = sorted(sel, key=lambda i: a.loc[i, "g_pop"])

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.9),
                             gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": 0.34})

    # ---- (a) dumbbell of 2100 per-capita output under the two elasticities
    ax = axes[0]
    ypos = np.arange(len(sel))
    va, vb = (a.loc[sel, "GDPpc_2100"] / 1e3).to_numpy(), (b.loc[sel, "GDPpc_2100"] / 1e3).to_numpy()
    for i, (p, q) in enumerate(zip(va, vb)):
        ax.plot([q, p], [i, i], color=BASELINE, lw=1.0, zorder=1, solid_capstyle="round")
    ax.scatter(vb, ypos, s=20, color=RED, zorder=3, linewidths=0.4, edgecolors="white")
    ax.scatter(va, ypos, s=20, color=BLUE, zorder=3, linewidths=0.4, edgecolors="white")
    ax.set_yticks(ypos)
    ax.set_yticklabels(sel, fontsize=6)
    ax.tick_params(axis="y", length=0)
    for lbl, iso in zip(ax.get_yticklabels(), sel):
        lbl.set_color(INK)
        if a.loc[iso, "g_pop"] < 0:
            lbl.set_fontweight("bold")
    # two empty rows at the bottom reserve space for the legend, so it never sits on data
    ax.set_ylim(-2.9, len(sel) - 0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Real GDP per capita in 2100\n(thousand, constant 2015 USD)")
    ax.grid(axis="x", which="major")
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, ms=4,
               label="$\\varepsilon$ = 0.92  (fixed-penalty levels model)"),
        Line2D([], [], marker="o", ls="", color=RED, ms=4,
               label="$\\varepsilon$ = 0.98 / 1.22  (adaptive levels model)")],
        loc="lower left", handletextpad=0.35, borderpad=0.2, labelspacing=0.35)
    ax.set_title("Bold labels: population falls by 2100", color=INK2, loc="left",
                 fontsize=6.5, pad=5)
    panel_letter(ax, "a", dx=-0.13)

    # ---- (b) the wedge against population growth
    ax = axes[1]
    gp = a.g_pop * 100
    wedge = (b.GDPpc_2100 / a.GDPpc_2100 - 1) * 100
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
    ax.axvline(0, color=GRID, lw=0.5, zorder=0)
    ax.scatter(gp, wedge, s=13, color=BLUE, alpha=0.55, linewidths=0.3,
               edgecolors="white", zorder=3)
    m = gp < 0
    if m.sum() > 2:
        co = np.polyfit(gp[m], wedge[m], 1)
        xs = np.linspace(gp.min(), 0, 50)
        ax.plot(xs, np.polyval(co, xs), color=INK2, lw=1.0, ls=(0, (4, 2)), zorder=4)
    for iso in ["JPN", "KOR", "CHN", "ITA", "DEU", "USA", "IND", "UKR", "NGA", "POL"]:
        if iso not in a.index:
            continue
        ax.annotate(iso, (gp[iso], wedge[iso]), xytext=(3.5, 3.5),
                    textcoords="offset points", fontsize=6, color=INK, fontweight="bold")
    ax.set_xlabel("Population growth 2024–2100 (% per year)")
    ax.set_ylabel("Difference in 2100 output per capita,\n$\\varepsilon$=1.22 world vs "
                  "$\\varepsilon$=0.92 world (%)")
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.text(0.98, 0.04, "the two elasticities agree\nwhere population grows",
            transform=ax.transAxes, fontsize=6, color=INK2, ha="right", va="bottom",
            linespacing=1.35)
    panel_letter(ax, "b", dx=-0.14)
    save(fig, "Figure5_two_elasticities")


# =========================================================================== FIGURE 6
def figure6(dec, panel):
    sel = [i for i in panel if i in dec.index]
    sel = sorted(sel, key=lambda i: dec.loc[i, "g_pop_pct"])
    ypos = np.arange(len(sel))

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.9),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.30})

    # ---- (a) demographic pressure, identical under either elasticity
    ax = axes[0]
    p24 = np.full(len(sel), 10.0)
    p00 = dec.loc[sel, "pressure_2100_pct"].to_numpy()
    # drawn with plot(), not annotate(): annotate does not update the data limits, which
    # silently collapsed this panel to a single column in an earlier revision.
    for i, (s0, e0) in enumerate(zip(p24, p00)):
        ax.plot([s0, e0], [i, i], color=AQUA, lw=1.1, zorder=2, solid_capstyle="round")
    ax.scatter(p00, ypos, s=17, color=AQUA, zorder=3, linewidths=0.35, edgecolors="white")
    ax.scatter(p24, ypos, s=14, color=MUTED, zorder=4, linewidths=0.35, edgecolors="white")
    for i, e0 in enumerate(p00):
        ax.text(e0 + 0.7, i, "%.0f" % e0, fontsize=5.8, va="center", ha="left", color=INK2)
    ax.set_xlim(8.6, max(p00) * 1.14)
    ax.set_xlabel("Age-related outlay (% of GDP)")
    ax.set_title("Demographic pressure, 2024 → 2100:\nthe same under either "
                 "elasticity", color=INK, loc="left", fontsize=7, pad=6)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=MUTED, ms=3.6, label="2024"),
        Line2D([], [], marker="o", ls="", color=AQUA, ms=3.6, label="2100")],
        loc="lower right", handletextpad=0.3, borderpad=0.2, labelspacing=0.3)

    # ---- (b) what the elasticity decides
    ax2 = axes[1]
    sh = dec.loc[sel, "benefit_shortfall_2100_pct"].to_numpy()
    ax2.barh(ypos, sh, height=0.62, color=RED, alpha=0.9, linewidth=0)
    ax2.axvline(0, color=BASELINE, lw=0.8)
    for i, v in enumerate(sh):
        ax2.text(v - 0.6 if v < 0 else v + 0.6, i, "%+.0f" % v, fontsize=5.8,
                 va="center", ha="right" if v < 0 else "left", color=INK2)
    ax2.set_xlabel("Change in sustainable real benefit\nper retiree in 2100 (%)")
    ax2.set_title("What the elasticity decides:\n$\\varepsilon$=1.22 world vs "
                  "$\\varepsilon$=0.92 world", color=INK, loc="left", fontsize=7, pad=6)
    ax2.set_xlim(min(sh.min() * 1.28, -3), max(sh.max() * 1.5, 8))

    for ax_ in (ax, ax2):
        ax_.set_yticks(ypos)
        ax_.set_yticklabels(sel, fontsize=6)
        ax_.tick_params(axis="y", length=0)
        for lbl, iso in zip(ax_.get_yticklabels(), sel):
            lbl.set_color(INK)
            if dec.loc[iso, "g_pop_pct"] < 0:
                lbl.set_fontweight("bold")
        ax_.set_ylim(-0.75, len(sel) - 0.25)
        ax_.grid(axis="x", which="major")
        ax_.set_axisbelow(True)
    panel_letter(ax, "a")
    panel_letter(ax2, "b")
    save(fig, "Figure6_decision_layer")


# =============================================================================== main
def main():
    long_df = pd.read_csv(DIR_RESULTS / "projection_2100.csv")
    summ = pd.read_csv(DIR_RESULTS / "projection_summary.csv")
    a = pd.read_csv(DIR_RESULTS / "projection_summary_eps0.92.csv")
    b = pd.read_csv(DIR_RESULTS / "projection_summary_eps1.22.csv")
    dec = pd.read_csv(DIR_RESULTS / "decision_layer.csv").set_index("ISO3")
    panel = json.load(open(DIR_RESULTS / "decision_layer.json"))["panel"].keys()
    panel = list(panel)

    resid = np.abs((np.log(long_df.GDP_median) - np.log(long_df.GDPpc_median))
                   - np.log(long_df.Population)).max()
    if resid > 1e-9:
        raise ValueError("refusing to draw figures from a CSV that violates the identity "
                         "(max residual %.2e)" % resid)
    print("figures from projection_2100.csv (identity residual %.1e)" % resid)

    figure3(long_df, summ)
    figure4(summ)
    figure5(a, b, panel)
    figure6(dec, panel)


if __name__ == "__main__":
    main()
