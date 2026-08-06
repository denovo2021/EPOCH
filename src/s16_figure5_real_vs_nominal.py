"""
s16_figure5_real_vs_nominal.py -- Figure 5
==========================================
(a) The local population-output elasticity fitted to REAL GDP under both spline priors, with
    95% credible bands, against the two nominal phase averages the same model returns.
(b) 2100 output per person under the discredited nominal estimate, and under the other real-GDP
    prior, both relative to the real-GDP fixed-penalty projection, against population growth.

WHY THIS FILE EXISTS
--------------------
Figure 5 was assembled outside the pipeline while the paper was being reframed, so the figure
in the manuscript had no generating script in the archive -- the one defect class this project
exists to remove. This script restores it, and does so from the REPORTED artifacts rather than
from the posteriors: panel (a) is drawn from `results/levels_elasticity.json`, the same file the
manuscript quotes, so the curve, the annotated phase averages and the text cannot drift apart.
The 1.4 GB posteriors are not needed to redraw the figure.

Panel (b) reads three projection runs of `src/s07_projection_2100.py`. Each of those files
records the elasticities it was run with in `beta_contraction` and `beta_expansion`; this script
checks them against the reported values and refuses to draw a figure built from a projection
that does not correspond to the elasticity it is labelled with.

Usage
-----
    python src/s16_figure5_real_vs_nominal.py

    python src/s16_figure5_real_vs_nominal.py \\
        --report results/levels_elasticity.json \\
        --baseline results/projection_summary_real_fixed.csv \\
        --nominal  results/projection_summary_eps1.22.csv \\
        --realalt  results/projection_summary_real_adaptive.csv

Output: figures/Figure5_real_vs_nominal.png / .pdf
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DIR_RESULTS, DIR_FIGURES

# Specification labels in levels_elasticity.json.
REAL_FIXED, REAL_ADAPT = "real_fixed", "real_adaptive"
NOM_FIXED, NOM_ADAPT = "nominal_fixed", "nominal_adaptive"

C_FIXED, C_ADAPT = "#1f77b4", "#f4622a"      # real: fixed penalty, adaptive shrinkage
C_NOMADAPT, C_NOMFIXED = "#e8393c", "#7f7f7f"
# Panel (b) point colours: the nominal counterfactual and the real-prior disagreement.
C_PTS_NOM, C_PTS_REAL = "#f1808a", "#f79b62"
# Labelled economies and their text offsets in points. USA sits inside the dense cluster
# around zero population growth, so it is placed above the point rather than below it.
LABELS_B = {"NGA": (-6, -14), "USA": (-16, 7), "DEU": (-6, -14), "JPN": (-6, -14),
            "ITA": (-6, -14), "KOR": (-6, -14), "CHN": (-6, -14), "UKR": (-6, -14)}

# beta_contraction / beta_expansion recorded in a projection run may be the reported elasticity
# rounded for the run's command line (the nominal counterfactual was run at 1.220 / 0.980
# against reported 1.2199 / 0.9778). Anything beyond this is a real mismatch.
BETA_TOL = 0.01

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def load_report(path):
    with open(path) as fh:
        payload = json.load(fh)
    specs = payload.get("specifications", {})
    missing = [k for k in (REAL_FIXED, REAL_ADAPT, NOM_FIXED, NOM_ADAPT) if k not in specs]
    if missing:
        raise SystemExit(
            "%s is missing %s. Regenerate it with all four specifications:\n"
            "  python src/s13_levels_elasticity.py \\\n"
            "    --posterior real_fixed:results/trace_fixed005_real.nc:GDP_constant_2015usd \\\n"
            "    --posterior real_adaptive:results/trace_adaptive_real.nc:GDP_constant_2015usd \\\n"
            "    --posterior nominal_fixed:results/trace_fixed005.nc:GDP \\\n"
            "    --posterior nominal_adaptive:results/trace_adaptive.nc:GDP"
            % (path, ", ".join(missing)))
    return specs


def check_betas(name, df, spec, tol=BETA_TOL):
    """A projection run records the elasticities it used; they must be the reported ones."""
    got = {}
    for col, phase in (("beta_contraction", "contraction"), ("beta_expansion", "expansion")):
        vals = np.unique(np.round(df[col].values, 10))
        if vals.size != 1:
            raise SystemExit("%s: %s is not constant across economies (%d distinct values)"
                             % (name, col, vals.size))
        got[phase] = float(vals[0])
    want = {p: float(spec[p]["mean"]) for p in ("contraction", "expansion")}
    bad = [p for p in want if abs(got[p] - want[p]) > tol]
    print("    %-34s run at %.4f / %.4f, reported %.4f / %.4f%s"
          % (name, got["contraction"], got["expansion"],
             want["contraction"], want["expansion"], "   MISMATCH" if bad else ""))
    if bad:
        raise SystemExit(
            "ABORT: %s was projected with %s = %s but the reporting file gives %s. The figure\n"
            "       would label a projection with an elasticity it was not run at. Re-run\n"
            "       src/s07_projection_2100.py at the reported values, or point --%s elsewhere."
            % (name, ", ".join(bad), [round(got[p], 4) for p in bad],
               [round(want[p], 4) for p in bad], "baseline/nominal/realalt"))


def panel_a(ax, specs):
    rf, ra = specs[REAL_FIXED], specs[REAL_ADAPT]
    nf, na = specs[NOM_FIXED], specs[NOM_ADAPT]

    for spec, colour, label in ((rf, C_FIXED, "fixed penalty"),
                                (ra, C_ADAPT, "adaptive shrinkage")):
        pop = np.array(spec["curve"]["population"])
        med = np.array(spec["curve"]["elasticity_median"])
        lo = np.array(spec["curve"]["elasticity_lo"])
        hi = np.array(spec["curve"]["elasticity_hi"])
        ax.fill_between(pop, lo, hi, color=colour, alpha=0.16, lw=0)
        ax.plot(pop, med, color=colour, lw=2.4, zorder=3,
                label="%s:  %.3f / %.3f"
                      % (label, spec["contraction"]["mean"], spec["expansion"]["mean"]))
        ax.axhline(spec["contraction"]["mean"], color=colour, ls="--", lw=1.4, zorder=2)

    ax.axhline(1.0, color="#333333", ls="--", lw=1.8, zorder=2)
    ax.axhline(na["contraction"]["mean"], color=C_NOMADAPT, ls=":", lw=2.0, zorder=2)
    ax.axhline(nf["contraction"]["mean"], color=C_NOMFIXED, ls=":", lw=1.8, zorder=2)

    ax.set_xscale("log")
    pop_all = np.array(rf["curve"]["population"])
    ax.set_xlim(pop_all.min(), pop_all.max())
    ax.set_ylim(0.15, 1.45)

    # Annotate in axes fractions, after the scale and limits are fixed: placing text with data
    # coordinates taken before set_xscale is what once rendered a 220-megapixel figure.
    ax.text(0.02, 1.0 + 0.012, "ε = 1  scale neutrality",
            transform=ax.get_yaxis_transform(), color="#333333", fontsize=10.5,
            ha="left", va="bottom")
    ax.text(0.30, na["contraction"]["mean"] + 0.012,
            "nominal GDP, adaptive prior:  %.3f" % na["contraction"]["mean"],
            transform=ax.get_yaxis_transform(), color=C_NOMADAPT, fontsize=10.5, va="bottom")
    ax.text(0.42, nf["contraction"]["mean"] + 0.012,
            "nominal GDP, fixed prior:  %.3f" % nf["contraction"]["mean"],
            transform=ax.get_yaxis_transform(), color=C_NOMFIXED, fontsize=10.5, va="bottom")

    p_min = min(specs[k][p]["P_lt_1"] for k in (REAL_FIXED, REAL_ADAPT)
                for p in ("contraction", "expansion"))
    ax.set_title("Real GDP: both priors give sub-unitary phase averages\n"
                 "contraction / expansion, P(ε<1) = %.2f for all four" % p_min,
                 fontsize=12.5, loc="left")
    ax.set_xlabel("Population", fontsize=12)
    ax.set_ylabel(r"Local elasticity  ε = ∂ln GDP / ∂ln Population", fontsize=12)
    ax.grid(axis="y", color="#cccccc", lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=12, frameon=False)


def panel_b(ax, base, nom, alt):
    def rel(other):
        m = base.merge(other[["ISO3", "GDPpc_2100"]], on="ISO3", suffixes=("", "_o"))
        return m["g_pop"].values * 100.0, (m["GDPpc_2100_o"] / m["GDPpc_2100"] - 1.0) * 100.0, m

    xg, yn, mn = rel(nom)
    _, yr, _ = rel(alt)

    ax.axhline(0.0, color="#999999", lw=1.4, zorder=1)
    ax.scatter(xg, yn, s=38, color=C_PTS_NOM, alpha=0.85, lw=0, zorder=3,
               label="the discredited nominal estimate")
    ax.scatter(xg, yr, s=38, color=C_PTS_REAL, alpha=0.85, lw=0, zorder=2,
               label="the two real-GDP priors against each other")

    for iso, off in LABELS_B.items():
        hit = np.where(mn["ISO3"].values == iso)[0]
        if not hit.size:
            print("    NOTE: %s is not in the projection panel; label skipped" % iso)
            continue
        i = hit[0]
        ax.annotate(iso, (xg[i], yn[i]), textcoords="offset points", xytext=off,
                    fontsize=11, fontweight="bold", ha="center", zorder=4)

    ax.set_title("The artifact rotates the cross-section about zero\n"
                 "population growth rather than shifting its level",
                 fontsize=12.5, loc="left")
    ax.set_xlabel("Population growth 2024–2100 (% yr$^{-1}$)", fontsize=12)
    ax.set_ylabel("2100 output per capita relative to the\nreal-GDP fixed-penalty projection (%)",
                  fontsize=12)
    ax.grid(color="#cccccc", lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=11.5, frameon=False, scatterpoints=1)

    dep, grow = xg < 0, xg > 0
    return {
        "n_economies": int(len(xg)),
        "nominal_median_gap_depopulating_pct": float(np.median(yn[dep])),
        "nominal_min_gap_depopulating_pct": float(yn[dep].min()),
        "nominal_median_gap_growing_pct": float(np.median(yn[grow])),
        "real_prior_median_gap_depopulating_pct": float(np.median(yr[dep])),
        "n_depopulating": int(dep.sum()), "n_growing": int(grow.sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(DIR_RESULTS / "levels_elasticity.json"))
    ap.add_argument("--baseline", default=str(DIR_RESULTS / "projection_summary_real_fixed.csv"))
    ap.add_argument("--nominal", default=str(DIR_RESULTS / "projection_summary_eps1.22.csv"))
    ap.add_argument("--realalt", default=str(DIR_RESULTS / "projection_summary_real_adaptive.csv"))
    ap.add_argument("--stem", default="Figure5_real_vs_nominal")
    args = ap.parse_args(argv)

    specs = load_report(args.report)
    base = pd.read_csv(args.baseline)
    nom = pd.read_csv(args.nominal)
    alt = pd.read_csv(args.realalt)

    print("  provenance of the three projection runs:")
    check_betas(args.baseline.split("/")[-1].split("\\")[-1], base, specs[REAL_FIXED])
    check_betas(args.nominal.split("/")[-1].split("\\")[-1], nom, specs[NOM_ADAPT])
    check_betas(args.realalt.split("/")[-1].split("\\")[-1], alt, specs[REAL_ADAPT])

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.4))
    panel_a(axes[0], specs)
    stats = panel_b(axes[1], base, nom, alt)
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=20,
                fontweight="bold", va="top", ha="left")
    fig.tight_layout(w_pad=3.0)

    for ext in ("png", "pdf"):
        fig.savefig(DIR_FIGURES / ("%s.%s" % (args.stem, ext)))
    plt.close(fig)
    print("  saved %s.png / .pdf" % args.stem)
    print("  panel (b), %d economies (%d depopulating, %d growing):"
          % (stats["n_economies"], stats["n_depopulating"], stats["n_growing"]))
    print("    nominal estimate vs real fixed: median %+.1f%% where population falls "
          "(worst %+.1f%%), median %+.1f%% where it rises"
          % (stats["nominal_median_gap_depopulating_pct"],
             stats["nominal_min_gap_depopulating_pct"],
             stats["nominal_median_gap_growing_pct"]))
    print("    real adaptive vs real fixed:    median %+.1f%% where population falls"
          % stats["real_prior_median_gap_depopulating_pct"])
    with open(DIR_RESULTS / "figure5_summary.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print("  wrote %s" % (DIR_RESULTS / "figure5_summary.json"))
    return stats


if __name__ == "__main__":
    main()
