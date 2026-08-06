"""
s09_decision_layer.py -- what the two candidate elasticities cost a finance ministry
===================================================================================
The levels specification cannot say whether the contraction-regime elasticity is 0.92 or
1.22. That is an inferential statement; on its own it does not tell a planner what to do.
This script converts it into the quantities a long-run fiscal projection actually needs,
computed from the two otherwise-identical projection runs produced by
s07_projection_2100.py with --eps-growth/--eps-decline.

A ratio of two GDP-denominated flows is invariant to the elasticity: if outlays are
indexed to output, both numerator and denominator move together. The elasticity therefore
bites on quantities measured in real levels or in time, and the decision layer has to be
built out of those. Four are reported.

  (0) DEMOGRAPHIC PRESSURE (the elasticity-invariant baseline).
      Under wage indexation -- benefits set as a fixed fraction of output per worker, the
      rule that Germany's sustainability factor and Japan's macroeconomic slide approximate
      -- age-related outlays are

          outlay(t)/GDP(t) = s0 * OADR(t) / OADR(2024),      OADR = (65+) / (15-64)

      which depends on demography alone. This is the pressure a planner already faces, and
      it is the baseline against which the elasticity wedge should be read. Reported at
      s0 = 10 % of GDP in 2024, stated as an illustrative benchmark, not a country estimate.

  (1) SUSTAINABLE REAL BENEFIT PER RETIREE.
      Under that same rule the real transfer per person aged 65+ is proportional to output
      per worker, so it *is* elasticity-sensitive. Holding the outlay ratio fixed, the
      sustainable real benefit per retiree in 2100 is lower under the super-unitary
      elasticity by exactly the output shortfall. This is the number a pension actuary
      would want, and it is where "we cannot tell 0.92 from 1.22" becomes expensive.

  (2) DEBT-STABILISING PRIMARY BALANCE.
      Holding the debt ratio constant requires a primary surplus of d0 * (r - g). Only the
      growth term differs between the two worlds, so the elasticity-attributable wedge is
      d0 * (g_0.92 - g_1.22) and does not require taking a stand on r. Reported per 100 %
      of GDP of initial debt.

  (3) TIME TO DOUBLE PER-CAPITA OUTPUT.
      The horizon on which living standards double, under each elasticity. The difference
      in years is the planning quantity that a non-economist reader understands immediately.

Also reported: cumulative discounted output 2024-2100 under each world (3 % real discount
rate), because a wedge that is small per year can be large in present value.

Inputs   results/projection_2100_eps0.92.csv, results/projection_2100_eps1.22.csv,
         results/projection_2100.csv (the fitted long-difference run, for reference),
         data/age_predictions_scenarios.csv, data/pop_predictions_scenarios.csv
Usage    python src/s09_decision_layer.py [--s0 0.10] [--discount 0.03]
Outputs  results/decision_layer.csv , results/decision_layer.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import PATH_AGE_SCEN, PATH_POP_SCEN, DIR_RESULTS

BASE, END = 2024, 2100
WORLD_A, WORLD_B = "_eps0.92", "_eps1.22"           # sub-unitary / super-unitary in contraction
LABEL_A, LABEL_B = "eps=0.92", "eps=0.98/1.22"

# Economies for the main-text panel: the largest economies plus the fastest-ageing ones.
PANEL = ["JPN", "KOR", "ITA", "DEU", "ESP", "CHN", "POL", "RUS", "PRT", "GRC", "THA", "UKR",
         "USA", "GBR", "FRA", "CAN", "IND", "BRA", "IDN", "NGA"]


def load(tag):
    d = pd.read_csv(DIR_RESULTS / ("projection_2100%s.csv" % tag))
    return d.pivot(index="Year", columns="ISO3", values="GDP_median")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=float, default=0.10,
                    help="illustrative 2024 age-related outlay as a share of GDP")
    ap.add_argument("--discount", type=float, default=0.03, help="real discount rate")
    args = ap.parse_args(argv)

    GA, GB = load(WORLD_A), load(WORLD_B)
    GF = load("")                                    # fitted long-difference run
    isos = [c for c in GA.columns if c in GB.columns and c in GF.columns]

    age = pd.read_csv(PATH_AGE_SCEN)
    age = age[age["Scenario"] == "Medium variant"]
    pop = pd.read_csv(PATH_POP_SCEN)
    pop = pop[pop["Scenario"] == "Medium variant"]
    POP = pop.pivot_table(index="Year", columns="ISO3", values="Population")
    WASH = age.pivot_table(index="Year", columns="ISO3", values="WAshare")
    ODEP = age.pivot_table(index="Year", columns="ISO3", values="OldDep")
    O65 = POP * WASH * ODEP                          # OldDep is (65+)/(15-64)

    years = np.arange(BASE, END + 1)
    disc = 1.0 / (1.0 + args.discount) ** (years - BASE)

    rows = []
    for iso in isos:
        if iso not in O65.columns or not np.isfinite(O65.loc[years, iso]).all():
            continue
        gA, gB, gF = GA[iso].loc[years], GB[iso].loc[years], GF[iso].loc[years]
        o = O65.loc[years, iso]
        p = POP.loc[years, iso]
        wa = (POP * WASH).loc[years, iso]
        oadr = (o / wa)

        # (0) demographic pressure: wage-indexed outlay ratio, elasticity-invariant
        press = args.s0 * oadr / oadr.iloc[0]

        # (1) sustainable real benefit per retiree = s0 * GDP / O, indexed to output per worker
        benA, benB, benF = (args.s0 * gA / o, args.s0 * gB / o, args.s0 * gF / o)

        # (2) debt-stabilising wedge per 100 % of GDP of initial debt
        g_a = float(np.log(gA.iloc[-1] / gA.iloc[0]) / (END - BASE))
        g_b = float(np.log(gB.iloc[-1] / gB.iloc[0]) / (END - BASE))
        g_f = float(np.log(gF.iloc[-1] / gF.iloc[0]) / (END - BASE))

        # (3) years for per-capita output to double
        def doubling(g):
            pc = (g / p).to_numpy()
            hit = np.where(pc >= 2.0 * pc[0])[0]
            return float(years[hit[0]] - BASE) if len(hit) else np.nan

        rows.append({
            "ISO3": iso,
            "g_pop_pct": float(np.log(p.iloc[-1] / p.iloc[0]) / (END - BASE) * 100),
            "g_agg_A_pct": g_a * 100, "g_agg_B_pct": g_b * 100, "g_agg_fitted_pct": g_f * 100,
            "oadr_2024": float(oadr.iloc[0]), "oadr_2050": float(oadr.loc[2050]),
            "oadr_2100": float(oadr.iloc[-1]),
            "pressure_2050_pct": float(press.loc[2050] * 100),
            "pressure_2100_pct": float(press.iloc[-1] * 100),
            "benefit_shortfall_2050_pct": float((benB.loc[2050] / benA.loc[2050] - 1) * 100),
            "benefit_shortfall_2100_pct": float((benB.iloc[-1] / benA.iloc[-1] - 1) * 100),
            "benefit_shortfall_2100_fitted_vs_B_pct":
                float((benB.iloc[-1] / benF.iloc[-1] - 1) * 100),
            "debt_wedge_pp_per_100pct_debt": (g_a - g_b) * 100,
            "doubling_years_A": doubling(gA), "doubling_years_B": doubling(gB),
            "doubling_delay_years": doubling(gB) - doubling(gA),
            "npv_ratio_B_over_A": float((gB * disc).sum() / (gA * disc).sum()),
            "gdp_2100_ratio_B_over_A": float(gB.iloc[-1] / gA.iloc[-1]),
        })

    df = pd.DataFrame(rows).set_index("ISO3")
    df.to_csv(DIR_RESULTS / "decision_layer.csv")

    shrink = df[df.g_pop_pct < 0]
    grow = df[df.g_pop_pct >= 0]
    summary = {
        "settings": {"s0_illustrative_share_of_gdp": args.s0,
                     "real_discount_rate": args.discount,
                     "world_A": LABEL_A, "world_B": LABEL_B,
                     "base_year": BASE, "end_year": END,
                     "outlay_rule": ("wage indexation: benefits set as a fixed fraction of output per "
                                    "worker, so the outlay ratio is s0 * OADR(t)/OADR(2024) and "
                                    "depends on demography alone; the real benefit per retiree "
                                    "is the elasticity-sensitive quantity"),
                     "n_countries": int(len(df)),
                     "n_depopulating": int(len(shrink))},
        "depopulating_countries": {
            "median_pressure_2050_pct": float(shrink.pressure_2050_pct.median()),
            "median_pressure_2100_pct": float(shrink.pressure_2100_pct.median()),
            "median_benefit_shortfall_2050_pct": float(shrink.benefit_shortfall_2050_pct.median()),
            "median_benefit_shortfall_2100_pct": float(shrink.benefit_shortfall_2100_pct.median()),
            "median_debt_wedge_pp": float(shrink.debt_wedge_pp_per_100pct_debt.median()),
            "median_doubling_delay_years": float(shrink.doubling_delay_years.median()),
            "median_npv_ratio": float(shrink.npv_ratio_B_over_A.median()),
        },
        "growing_countries": {
            "median_pressure_2100_pct": float(grow.pressure_2100_pct.median()),
            "median_benefit_shortfall_2100_pct": float(grow.benefit_shortfall_2100_pct.median()),
            "median_debt_wedge_pp": float(grow.debt_wedge_pp_per_100pct_debt.median()),
            "median_doubling_delay_years": float(grow.doubling_delay_years.median()),
            "median_npv_ratio": float(grow.npv_ratio_B_over_A.median()),
        },
        "panel": {i: {k: (None if pd.isna(v) else round(float(v), 3))
                      for k, v in df.loc[i].items()}
                  for i in PANEL if i in df.index},
        "headline": {},
    }
    worst = shrink.sort_values("benefit_shortfall_2100_pct").head(5)
    summary["headline"] = {
        "largest_benefit_shortfalls_2100": [
            {"ISO3": i, "shortfall_pct": round(float(r.benefit_shortfall_2100_pct), 1),
             "g_pop_pct": round(float(r.g_pop_pct), 2),
             "doubling_delay_years": None if pd.isna(r.doubling_delay_years)
             else int(r.doubling_delay_years)}
            for i, r in worst.iterrows()],
        "statement": ("Demography alone raises age-related outlays from 10 %% of GDP in "
                      "2024 to a median %.1f %% by 2100 in depopulating economies, and that "
                      "pressure is the same under either elasticity. What the elasticity "
                      "decides is what the transfer buys: the sustainable real benefit per "
                      "retiree in 2100 is a median %.1f %% lower under eps=0.98/1.22 than "
                      "under eps=0.92, per-capita output doubles a median %.0f years later, "
                      "and holding the debt ratio constant costs a further %.2f pp of GDP a "
                      "year for every 100 %% of GDP of initial debt."
                      % (shrink.pressure_2100_pct.median(),
                         -shrink.benefit_shortfall_2100_pct.median(),
                         shrink.doubling_delay_years.median(),
                         shrink.debt_wedge_pp_per_100pct_debt.median())),
    }
    with open(DIR_RESULTS / "decision_layer.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---------------------------------------------------------------------- report
    print("DECISION LAYER  |  %d countries (%d depopulating)  |  s0 = %.0f%% of GDP, "
          "discount %.1f%%" % (len(df), len(shrink), args.s0 * 100, args.discount * 100))
    print("\n  Columns 'pressure' are elasticity-INVARIANT (demography only). Columns")
    print("  'benefit shortfall' are the cost of not knowing whether eps is 0.92 or 1.22.\n")
    print("  %-5s %7s | %6s %6s | %9s %9s | %9s %9s %7s" %
          ("ISO3", "gPop%", "OADR24", "OADR00", "press50%", "press00%",
           "ben50%", "ben00%", "2xdelay"))
    for i in PANEL:
        if i not in df.index:
            continue
        r = df.loc[i]
        print("  %-5s %7.2f | %6.2f %6.2f | %9.1f %9.1f | %+9.1f %+9.1f %7.0f" %
              (i, r.g_pop_pct, r.oadr_2024, r.oadr_2100,
               r.pressure_2050_pct, r.pressure_2100_pct,
               r.benefit_shortfall_2050_pct, r.benefit_shortfall_2100_pct,
               r.doubling_delay_years))
    print("\n  Among depopulating economies (n=%d): demography alone takes a 10%%-of-GDP "
          "age-related\n  outlay to a median %.1f%% by 2100 -- identical under either "
          "elasticity. The elasticity\n  decides what it buys: the sustainable real benefit "
          "per retiree in 2100 is a median\n  %.1f%% lower under %s than under %s, "
          "per-capita output doubles %.0f years later,\n  and present-value output over "
          "2024-2100 is %.1f%% smaller." %
          (len(shrink), shrink.pressure_2100_pct.median(),
           -shrink.benefit_shortfall_2100_pct.median(), LABEL_B, LABEL_A,
           shrink.doubling_delay_years.median(),
           (1 - shrink.npv_ratio_B_over_A.median()) * 100))
    print("  Per 100%% of GDP of initial public debt, the two elasticities differ by a "
          "median of\n  %.2f pp of GDP in the annual primary surplus needed to hold the "
          "debt ratio constant." % shrink.debt_wedge_pp_per_100pct_debt.median())
    print("\n  wrote decision_layer.csv, decision_layer.json")


if __name__ == "__main__":
    main()
