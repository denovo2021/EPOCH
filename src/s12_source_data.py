"""
s12_source_data.py -- per-figure source data files
=================================================
Most journals require the numerical values behind each plotted panel to be published
alongside it. This writes one CSV per figure containing exactly the columns that figure draws, so a
reader can reconstruct the panel without running the pipeline. Everything is read from
the same result files the figures read, so a source-data file cannot drift from its figure.

The decision-layer figure is drawn from whichever decision layer was asked for, so
--dec-tag selects the contrast: no flag reproduces the archived eps=0.92 vs eps=1.22
comparison bit for bit, and --dec-tag _epsband writes the fitted-versus-eps=1.00 contrast
the manuscript reports, to a separate filename. The panel rows follow the order
s10_make_figures_2100.figure6 draws them in, sorted by population growth, so the file
reconstructs the figure top to bottom.

Usage:  python src/s12_source_data.py [--dec-tag _epsband]
Output: results/source_data/Figure{3,4,5}_source_data.csv ,
        results/source_data/Figure6_source_data{dec_tag}.csv , plus a manifest
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from config import DIR_RESULTS

OUT = DIR_RESULTS / "source_data"


def main(dec_tag: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    long_df = pd.read_csv(DIR_RESULTS / "projection_2100.csv")
    summ = pd.read_csv(DIR_RESULTS / "projection_summary.csv")
    a = pd.read_csv(DIR_RESULTS / "projection_summary_eps0.92.csv").set_index("ISO3")
    b = pd.read_csv(DIR_RESULTS / "projection_summary_eps1.22.csv").set_index("ISO3")
    dec = pd.read_csv(DIR_RESULTS / ("decision_layer%s.csv" % dec_tag)).set_index("ISO3")
    dec_meta = json.load(open(DIR_RESULTS / ("decision_layer%s.json" % dec_tag)))
    panel = list(dec_meta["panel"].keys())

    written = {}

    # ---- Figure 3: trajectories for the ten largest 2024 economies, both panels
    top10 = summ.nsmallest(10, "rank_agg_2024").ISO3.tolist()
    f3 = long_df[long_df.ISO3.isin(top10)][
        ["ISO3", "Country", "Year", "Population", "GDP_median", "GDPpc_median"]].copy()
    f3["population_falls_2024_2100"] = f3.ISO3.map(
        summ.set_index("ISO3").g_pop.lt(0).to_dict())
    f3.to_csv(OUT / "Figure3_source_data.csv", index=False)
    written["Figure3_source_data.csv"] = ("panel a: GDP_median; panel b: GDPpc_median; "
                                          "colour: population_falls_2024_2100")

    # ---- Figure 4: the scatter, all economies
    f4 = summ[["ISO3", "Country", "Region"]].copy()
    f4["g_agg_pct_per_yr"] = summ.g_agg * 100
    f4["g_gdppc_pct_per_yr"] = summ.g_pc * 100
    f4["g_pop_pct_per_yr"] = summ.g_pop * 100
    f4.to_csv(OUT / "Figure4_source_data.csv", index=False)
    written["Figure4_source_data.csv"] = ("x: g_agg_pct_per_yr; y: g_gdppc_pct_per_yr; "
                                          "colour: g_pop_pct_per_yr")

    # ---- Figure 5: the two counterfactual worlds
    f5 = pd.DataFrame({
        "ISO3": a.index, "Country": a.Country.values,
        "g_pop_pct_per_yr": a.g_pop.values * 100,
        "GDPpc_2100_eps0.92": a.GDPpc_2100.values,
        "GDPpc_2100_eps1.22": b.loc[a.index, "GDPpc_2100"].values,
        "difference_pct": (b.loc[a.index, "GDPpc_2100"].values / a.GDPpc_2100.values - 1) * 100,
        "in_panel_a": [i in panel for i in a.index],
    })
    f5.to_csv(OUT / "Figure5_source_data.csv", index=False)
    written["Figure5_source_data.csv"] = ("panel a: the two GDPpc_2100 columns for rows with "
                                          "in_panel_a; panel b: difference_pct against "
                                          "g_pop_pct_per_yr for all rows")

    # ---- Figure 6: the decision layer
    sel = sorted([i for i in panel if i in dec.index], key=lambda i: dec.loc[i, "g_pop_pct"])
    f6 = dec.loc[sel, [
        "g_pop_pct", "oadr_2024", "oadr_2100", "pressure_2050_pct", "pressure_2100_pct",
        "benefit_shortfall_2050_pct", "benefit_shortfall_2100_pct",
        "debt_wedge_pp_per_100pct_debt", "doubling_delay_years"]].reset_index()
    f6.insert(2, "outlay_2024_pct",
              dec_meta["settings"]["s0_illustrative_share_of_gdp"] * 100)
    name6 = "Figure6_source_data%s.csv" % dec_tag
    f6.to_csv(OUT / name6, index=False)
    written[name6] = ("panel a: outlay_2024_pct -> pressure_2100_pct; "
                      "panel b: benefit_shortfall_2100_pct; "
                      "world A = %s, world B = %s; rows ordered by g_pop_pct as drawn"
                      % (dec_meta["settings"]["world_A"], dec_meta["settings"]["world_B"]))

    # ---- Figures 1, 2, S1-S5 derive from the two levels posteriors, which are archived
    #      separately because of their size. Record that rather than shipping a stub.
    note = {
        "Figure1_population_GDP_association": (
            "derived from results/trace_adaptive.nc via src/make_figures.py; the posterior "
            "is archived in the Zenodo deposit rather than the git repository (1.4 GB)"),
        "Figure2_backtest_accuracy": (
            "derived from results/backtest_predictions.npz via src/make_figures.py"),
        "FigureS1_trace / FigureS2_ppc / FigureS3_energy": (
            "derived from results/trace_adaptive.nc via src/make_si_diagnostics.py"),
        "FigureS4_local_elasticity_fixed": (
            "derived from results/trace_fixed005.nc via src/make_figureS4_fixed.py"),
        "FigureS5_elasticity_sensitivity": (
            "derived from both levels posteriors via src/make_sensitivity_figure.py"),
        "FigureS6_regional_elasticity_5yr_forest": (
            "derived from results/hierarchical_5yr_posterior.nc via src/fit_hierarchical_5yr.py; "
            "the per-region values are in results/hierarchical_region_elasticities_5yr.csv"),
    }
    with open(OUT / "MANIFEST.json", "w") as fh:
        json.dump({"generated_by": "src/s12_source_data.py",
                   "dec_tag": dec_tag or "(archived comparison)",
                   "files": written, "figures_without_a_csv": note}, fh, indent=2)

    print("wrote %d source-data files to %s" % (len(written), OUT))
    for k, v in written.items():
        print("  %-32s %s" % (k, v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec-tag", dest="dec_tag", default="",
                    help="suffix of the decision-layer run to draw Figure 6 from, e.g. _epsband")
    main(**vars(ap.parse_args()))
