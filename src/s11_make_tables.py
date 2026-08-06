"""
s11_make_tables.py -- every table in the manuscript and SI, from the single-source outputs
=========================================================================================
No number in a table is typed by hand, and every table that reports projections reads the
same CSV the figures read. Emits both CSV (for the submission's source-data requirement)
and GitHub-flavoured markdown (for pasting into the manuscript).

  Table 1   out-of-sample calibration of the levels model (backtest)
  Table 2   projected real GDP and real GDP per capita, 2024 / 2050 / 2100, twelve largest
            2100 economies, with aggregate and per-capita ranks and the implied population
            growth rate -- so the identity is visible in the table itself
  Table S1  regional elasticities, expansion and contraction regimes, with P(eps < 1)
  Table S2  sensitivity of the 2100 projection to s_ss and to the convergence term
  Table S3  the decision layer for the twenty-economy panel
  Table S4  posterior probability of the four aggregate x per-capita outcome modes for the
            twenty fastest-depopulating economies

Usage:  python src/s11_make_tables.py
Output: results/tables/*.csv , results/tables/all_tables.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DIR_RESULTS

OUT = DIR_RESULTS / "tables"
N_TABLE2 = 12


def md(df, floatfmt=None, index=False):
    d = df.copy()
    if floatfmt:
        for c, f in floatfmt.items():
            if c in d:
                d[c] = d[c].map(lambda v: "" if pd.isna(v) else f % v)
    cols = ([d.index.name or ""] + list(d.columns)) if index else list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join([":---"] + [" ---:"] * (len(cols) - 1)) + "|"]
    for k, r in d.iterrows():
        vals = ([str(k)] if index else []) + [str(v) for v in r.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = []

    summ = pd.read_csv(DIR_RESULTS / "projection_summary.csv")
    man = json.load(open(DIR_RESULTS / "projection_manifest.json"))
    bt = json.load(open(DIR_RESULTS / "gdp_backtest_metrics.json"))
    bridge = json.load(open(DIR_RESULTS / "elasticity_bridge.json"))
    reg = pd.read_csv(DIR_RESULTS / "elasticity_bridge_regions.csv")
    dec = pd.read_csv(DIR_RESULTS / "decision_layer.csv")
    dmodes = pd.read_csv(DIR_RESULTS / "divergence_modes.csv")

    # ------------------------------------------------------------------------ Table 1
    cov = bt["COV95"] * 100
    t1 = pd.DataFrame([
        ["Training window", "1960–2010"],
        ["Evaluation window", "2011–%s" % bt["test_period"].split("-")[1]],
        ["Held-out country-years", "%d" % bt["test_n"]],
        ["Mean absolute error (log₁₀ real GDP)", "%.2f" % bt["MAE_log"]],
        ["Continuous ranked probability score", "%.3f" % bt["CRPS"]],
        ["Coverage of the nominal 95% predictive interval",
         "%.1f%% (intervals are slightly too narrow)" % cov],
        ["Maximum split-$\\hat{R}$", "%.2f" % bt["max_rhat"]],
        ["Minimum bulk effective sample size", "%d" % bt["min_ess_bulk"]],
        ["Divergent transitions", "%d of %d post-warmup draws"
         % (bt["divergences"], bt["draws"] * bt["chains"])],
    ], columns=["Diagnostic", "Value"])
    t1.to_csv(OUT / "Table1_backtest.csv", index=False)
    blocks.append(("Table 1. Out-of-sample calibration of the levels model.", md(t1)))

    # ------------------------------------------------------------------------ Table 2
    t2 = summ.nsmallest(N_TABLE2, "rank_agg_2100").copy()
    t2["Delta_agg"] = t2.rank_agg_2024 - t2.rank_agg_2100
    t2["Delta_pc"] = t2.rank_pc_2024 - t2.rank_pc_2100
    out2 = pd.DataFrame({
        "Economy": t2.Country,
        "GDP 2024 (T)": t2.GDP_2024 / 1e12,
        "GDP 2050 (T)": t2.GDP_2050 / 1e12,
        "GDP 2100 (T)": t2.GDP_2100 / 1e12,
        "GDP 2100 95% CrI (T)": ["%.1f–%.0f" % (lo / 1e12, hi / 1e12)
                                 for lo, hi in zip(t2.GDP_2100_lo, t2.GDP_2100_hi)],
        "GDPpc 2024 (k)": t2.GDPpc_2024 / 1e3,
        "GDPpc 2100 (k)": t2.GDPpc_2100 / 1e3,
        "Agg rank 2024→2100": ["%d→%d" % (a, b) for a, b in
                               zip(t2.rank_agg_2024, t2.rank_agg_2100)],
        "Per-capita rank 2024→2100": ["%d→%d" % (a, b) for a, b in
                                      zip(t2.rank_pc_2024, t2.rank_pc_2100)],
        "GDP growth (%/yr)": t2.g_agg * 100,
        "GDPpc growth (%/yr)": t2.g_pc * 100,
        "Population growth (%/yr)": t2.g_pop * 100,
    })
    out2.to_csv(OUT / "Table2_projection.csv", index=False)
    blocks.append((
        "Table 2. Projected real GDP and real GDP per capita, twelve largest 2100 economies.",
        md(out2, {"GDP 2024 (T)": "%.2f", "GDP 2050 (T)": "%.2f", "GDP 2100 (T)": "%.2f",
                  "GDPpc 2024 (k)": "%.1f", "GDPpc 2100 (k)": "%.1f",
                  "GDP growth (%/yr)": "%.2f", "GDPpc growth (%/yr)": "%.2f",
                  "Population growth (%/yr)": "%+.2f"})))

    # ----------------------------------------------------------------------- Table S1
    s1 = pd.DataFrame({
        "Region": reg.Region,
        "Expansion ε": reg.beta_expansion_mean,
        "Expansion 95% CrI": ["%.2f–%.2f" % (a, b) for a, b in
                              zip(reg.beta_expansion_lo, reg.beta_expansion_hi)],
        "P(ε<1) expansion": reg.P_expansion_lt_1,
        "Contraction ε": reg.beta_contraction_mean,
        "Contraction 95% CrI": ["%.2f–%.2f" % (a, b) for a, b in
                                zip(reg.beta_contraction_lo, reg.beta_contraction_hi)],
        "P(ε<1) contraction": reg.P_contraction_lt_1,
    })
    L = bridge["long_difference_model"]
    s1.loc[len(s1)] = ["**Global**", L["expansion_elasticity"]["mean"],
                       "%.2f–%.2f" % (L["expansion_elasticity"]["q2.5_50_97.5"][0],
                                      L["expansion_elasticity"]["q2.5_50_97.5"][2]),
                       L["expansion_elasticity"]["P_lt_1"],
                       L["contraction_elasticity"]["mean"],
                       "%.2f–%.2f" % (L["contraction_elasticity"]["q2.5_50_97.5"][0],
                                      L["contraction_elasticity"]["q2.5_50_97.5"][2]),
                       L["contraction_elasticity"]["P_lt_1"]]
    s1.to_csv(OUT / "TableS1_regional_elasticities.csv", index=False)
    blocks.append((
        "Table S1. Population–output elasticity by region and regime, long-difference model.",
        md(s1, {"Expansion ε": "%.3f", "Contraction ε": "%.3f",
                "P(ε<1) expansion": "%.3f", "P(ε<1) contraction": "%.3f"})))

    # ----------------------------------------------------------------------- Table S2
    rows = []
    base = summ.set_index("ISO3")
    for tag, label in [("", "s_ss = 1.77 %/yr (reported)"),
                       ("_sss1.51", "s_ss = 1.51 %/yr"),
                       ("_sss2.02", "s_ss = 2.02 %/yr"),
                       ("_noconv", "convergence term removed"),
                       ("_eps0.92", "ε = 0.92 in both regimes"),
                       ("_eps1.22", "ε = 0.98 expansion / 1.22 contraction")]:
        try:
            d = pd.read_csv(DIR_RESULTS / ("projection_summary%s.csv" % tag)).set_index("ISO3")
        except FileNotFoundError:
            continue
        d = d.loc[base.index]
        rho = float(base.GDP_2100.rank().corr(d.GDP_2100.rank(), method="spearman"))
        top = base.GDP_2100.nlargest(20).index
        shift = int(np.abs(base.loc[top].GDP_2100.rank() - d.loc[top].GDP_2100.rank()).max())
        shrink = d[d.g_pop < 0]
        rows.append({
            "Specification": label,
            "Median GDP growth (%/yr)": d.g_agg.median() * 100,
            "Max GDP growth (%/yr)": d.g_agg.max() * 100,
            "USA 2100 (T)": d.loc["USA", "GDP_2100"] / 1e12,
            "CHN 2100 (T)": d.loc["CHN", "GDP_2100"] / 1e12,
            "JPN 2100 GDPpc (k)": d.loc["JPN", "GDPpc_2100"] / 1e3,
            "Median GDPpc growth, depopulating (%/yr)": shrink.g_pc.median() * 100,
            "Spearman ρ of 2100 ranking vs reported": rho,
            "Largest rank shift in top 20": shift,
        })
    s2 = pd.DataFrame(rows)
    s2.to_csv(OUT / "TableS2_projection_sensitivity.csv", index=False)
    blocks.append(("Table S2. Sensitivity of the 2100 projection.",
                   md(s2, {"Median GDP growth (%/yr)": "%.2f", "Max GDP growth (%/yr)": "%.2f",
                           "USA 2100 (T)": "%.1f", "CHN 2100 (T)": "%.1f",
                           "JPN 2100 GDPpc (k)": "%.0f",
                           "Median GDPpc growth, depopulating (%/yr)": "%.2f",
                           "Spearman ρ of 2100 ranking vs reported": "%.4f"})))

    # ----------------------------------------------------------------------- Table S3
    panel = list(json.load(open(DIR_RESULTS / "decision_layer.json"))["panel"].keys())
    d3 = dec[dec.ISO3.isin(panel)].sort_values("g_pop_pct")
    s3 = pd.DataFrame({
        "Economy": d3.ISO3,
        "Population growth (%/yr)": d3.g_pop_pct,
        "Old-age dependency 2024": d3.oadr_2024,
        "Old-age dependency 2100": d3.oadr_2100,
        "Outlay 2100, ε-invariant (% GDP)": d3.pressure_2100_pct,
        "Benefit per retiree 2050, ε=1.22 vs 0.92 (%)": d3.benefit_shortfall_2050_pct,
        "Benefit per retiree 2100, ε=1.22 vs 0.92 (%)": d3.benefit_shortfall_2100_pct,
        "Extra primary surplus per 100% GDP debt (pp)": d3.debt_wedge_pp_per_100pct_debt,
        "Delay in doubling GDP per capita (yr)": d3.doubling_delay_years,
    })
    s3.to_csv(OUT / "TableS3_decision_layer.csv", index=False)
    blocks.append((
        "Table S3. Decision layer: what the two candidate elasticities imply for planning.",
        md(s3, {"Population growth (%/yr)": "%+.2f", "Old-age dependency 2024": "%.2f",
                "Old-age dependency 2100": "%.2f",
                "Outlay 2100, ε-invariant (% GDP)": "%.1f",
                "Benefit per retiree 2050, ε=1.22 vs 0.92 (%)": "%+.1f",
                "Benefit per retiree 2100, ε=1.22 vs 0.92 (%)": "%+.1f",
                "Extra primary surplus per 100% GDP debt (pp)": "%.2f",
                "Delay in doubling GDP per capita (yr)": "%.0f"})))

    # ----------------------------------------------------------------------- Table S4
    d4 = dmodes.nsmallest(20, "g_pop").copy()
    s4 = pd.DataFrame({
        "Economy": d4.Country,
        "Population growth (%/yr)": d4.g_pop * 100,
        "P(per-capita output rises)": d4.P_pc_gain,
        "P(aggregate output falls)": d4.P_agg_loss,
        "P(benign divergence)": d4.P_benign_divergence,
        "P(concordant expansion)": d4.P_concordant_expansion,
        "P(concordant contraction)": d4.P_concordant_contraction,
    })
    s4.to_csv(OUT / "TableS4_divergence_modes.csv", index=False)
    blocks.append((
        "Table S4. Posterior probability of the four outcome modes, twenty fastest-"
        "depopulating economies.",
        md(s4, {"Population growth (%/yr)": "%+.2f", "P(per-capita output rises)": "%.3f",
                "P(aggregate output falls)": "%.3f", "P(benign divergence)": "%.3f",
                "P(concordant expansion)": "%.3f", "P(concordant contraction)": "%.3f"})))

    with open(OUT / "all_tables.md", "w") as fh:
        fh.write("# EPOCH tables\n\nGenerated by `src/s11_make_tables.py` from "
                 "`results/projection_*.csv`, `results/elasticity_bridge*`, "
                 "`results/decision_layer.csv` and `results/gdp_backtest_metrics.json`. "
                 "Every projected figure traces to the run recorded in "
                 "`results/projection_manifest.json` (identity residual %.1e).\n\n"
                 % man["checks"]["identity_max_residual"])
        for title, body in blocks:
            fh.write("## %s\n\n%s\n\n" % (title, body))

    print("wrote %d tables to %s" % (len(blocks), OUT))
    for title, _ in blocks:
        print("  " + title)


if __name__ == "__main__":
    main()
