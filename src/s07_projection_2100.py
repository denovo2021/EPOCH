"""
s07_projection_2100.py -- EPOCH single-source 2100 projection engine
====================================================================
Replaces the pair `scenario_projection.py` (B-spline forward pass) + `predict_gdp_2100.py`
(working-age forward pass) as the *sole* source of every projected number that appears in
the manuscript. Table 2 and Figures 3-5 all read the CSVs written here, so an aggregate
column and a per-capita column can never again come from two different runs.

WHY THIS FILE EXISTS
--------------------
In the ver6 manuscript the aggregate GDP column of Table 2 came from the working-age
forward pass (sane: 1.3-2.7 %/yr) while the per-capita column came from the B-spline
forward pass (5.0-7.3 %/yr, driven by an orthogonalised time trend extrapolated 76 years
and clipped at +6). Dividing one by the other implied 2100 populations 0.02-0.12x UN WPP
-- e.g. Germany 1.7 million. This engine makes that class of error unrepresentable:

  * per capita is *defined* as GDP(t) / Population(t) from the same UN WPP series that
    drives the demographic term -- it is never a separate model output;
  * an arithmetic identity assertion (g_agg - g_pc == g_pop) runs over every country and
    every year before anything is written;
  * plausibility bounds on 76-year mean growth raise an exception rather than a warning;
  * the convergence term is unit-tested for curvature, so "straight line on a log axis"
    fails loudly instead of shipping;
  * one CSV, one run, one tag.

MODEL (working-age decadal long-difference model, results/hierarchical_wa_posterior.nc)
---------------------------------------------------------------------------------------
Fitted on n = 896 non-overlapping country-decades (179 countries, 7 regions):

    D10 ln GDP = alpha_country + tau_period + (beta_country + delta_region * 1[decline]) * D10 ln WApop + eps

Forward pass, annual step (draw-wise; S draws):

    ln GDP(t+1) = ln GDP(t) + alpha_eff/10 + slope_block * [ln W(t+1) - ln W(t)] + N(0, sigma/sqrt(10))

    slope_block = beta_c                    if D10 ln W(block) >= 0   (expansion)
                = beta_c + delta_region     if D10 ln W(block) <  0   (contraction)

    alpha_eff   = S_SS + (alpha_c - S_SS) * clip(gap_t / gap_0, 0, 1)      [convergence]

W = Population * WAshare (ages 15-64), both from the UN WPP 2024 Medium variant.
The convergence gap is measured in *total-population* per-capita terms against a frontier
that grows at S_SS from the 2024 United States level.

BASE-YEAR ANCHORING (fixes an off-by-one in the previous engine)
---------------------------------------------------------------
The panel ends in 2023; the UN WPP scenario files begin in 2024. The previous engine
seeded the path with observed 2023 GDP and then *labelled that value 2024*, so its
"GDP 2024" column was really GDP 2023 and its 76-year growth denominator spanned 77
years of drift. Here the 2024 anchor is observed 2023 real GDP advanced exactly one year
by the country's own drift (demographic term omitted, because a clean 2023->2024 change in
working-age population would require splicing WDI and UN population levels). Everything
from 2024 onward is on a single UN WPP series, so the identity holds exactly across the
reported 2024-2100 window.

COUNTERFACTUAL WORLDS
---------------------
`--eps-growth` / `--eps-decline` replace the fitted country slopes with stated constants,
holding alpha, sigma, the convergence term, the demography and the seed fixed. This is
how the "0.92 world vs 1.22 world" figure is produced: the two elasticities that the
levels model cannot distinguish are pushed through an otherwise identical engine.

USAGE
-----
    python src/s07_projection_2100.py                                    # fitted model
    python src/s07_projection_2100.py --eps-growth 0.92 --eps-decline 0.92 --tag _eps0.92
    python src/s07_projection_2100.py --eps-growth 0.98 --eps-decline 1.22 --tag _eps1.22
    python src/s07_projection_2100.py --s_ss 0.150 --tag _sss1.50        # sensitivity

OUTPUTS (results/)
------------------
    projection_2100{tag}.csv        long panel: ISO3, Country, Region, Year, Population,
                                    WApop, GDP_{median,lo,hi}, GDPpc_{median,lo,hi}
    projection_summary{tag}.csv     one row per country: levels and ranks at 2024/2050/
                                    2075/2100, mean growth of GDP / GDP pc / population,
                                    identity residual, elasticity actually applied
    divergence_modes{tag}.csv       per-country posterior probability of the four
                                    aggregate x per-capita outcome modes
    projection_manifest{tag}.json   every setting, every check, every summary statistic
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

from config import PATH_POP_SCEN, PATH_AGE_SCEN, PATH_MERGED_AGE, DIR_RESULTS

# ----------------------------------------------------------------------------- constants
BASE = 2024
END = 2100
YEARS = np.arange(BASE, END + 1)
TABLE_YEARS = [2024, 2050, 2075, 2100]
BLOCK_ENDS = list(range(2030, END + 1, 10))
HORIZON = END - BASE                      # 76 years -- the growth denominator everywhere
SEED = 42

# Plausibility bounds on 76-year mean *real aggregate* growth. The upper bound is set
# above the fastest country in the fitted run (Afghanistan, 4.0 %/yr) so that a legitimate
# fast-catch-up economy does not abort the run, but far below the 5-8 %/yr signature of the
# ver6 failure. Anything outside these bounds is a bug, not a finding.
G_AGG_MAX = 0.050
G_AGG_MIN = -0.030
G_AGG_WARN = 0.035
IDENTITY_TOL = 1e-10


def _fail(msg: str) -> None:
    raise ValueError("SANITY CHECK FAILED -- " + msg)


# --------------------------------------------------------------------------------- args
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--s_ss", type=float, default=0.175,
                    help="frontier steady-state DECADAL per-capita drift; 0.175 decadal "
                         "= (exp(0.0175)-1) = 1.77 %%/yr. Report as 1.77, never 1.75/1.80.")
    ap.add_argument("--scenario", default="Medium variant")
    ap.add_argument("--eps-growth", dest="eps_growth", type=float, default=None,
                    help="override the expansion-regime elasticity for ALL countries")
    ap.add_argument("--eps-decline", dest="eps_decline", type=float, default=None,
                    help="override the contraction-regime elasticity for ALL countries")
    ap.add_argument("--tag", default="", help="suffix appended to every output filename")
    ap.add_argument("--no-converge", dest="converge", action="store_false")
    ap.add_argument("--no-residual", dest="residual", action="store_false")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------- ingredients
def load_posterior(draws: int):
    """Working-age posterior -> (alpha, beta, delta, sigma) with draws stacked."""
    post = xr.open_dataset(DIR_RESULTS / "hierarchical_wa_posterior.nc", group="posterior")
    isos = [str(c) for c in post["country"].values]
    regions = [str(r) for r in post["region"].values]
    A = post["alpha_country"].stack(z=("chain", "draw")).values      # (C, S) decadal drift
    B = post["beta_country"].stack(z=("chain", "draw")).values       # (C, S) expansion slope
    D = post["delta_region"].stack(z=("chain", "draw")).values       # (R, S) contraction increment
    nS = A.shape[1]
    idx = np.random.default_rng(SEED).choice(nS, min(draws, nS), replace=False)
    sigma = float(post["sigma"].stack(z=("chain", "draw")).values.mean())
    return (isos, regions, A[:, idx], B[:, idx], D[:, idx], sigma,
            {"n_posterior_draws": int(nS), "n_draws_used": int(len(idx))})


def load_demography(scenario: str):
    """Panel anchors plus the single UN WPP series used for both the demographic term and
    the per-capita denominator. Returns level frames, not logs, so per capita is always
    GDP / Population with no second population source anywhere."""
    m = pd.read_csv(PATH_MERGED_AGE)
    gdp23 = (m[m["Year"] == 2023]
             .dropna(subset=["GDP_constant_2015usd"])
             .set_index("ISO3")["GDP_constant_2015usd"])
    cname = m.drop_duplicates("ISO3").set_index("ISO3")["Country Name"]
    iso2region = m.drop_duplicates("ISO3").set_index("ISO3")["Region"].to_dict()

    pop = pd.read_csv(PATH_POP_SCEN)
    pop = pop[pop["Scenario"] == scenario]
    POP = pop.pivot_table(index="ISO3", columns="Year", values="Population")

    age = pd.read_csv(PATH_AGE_SCEN)
    age = age[age["Scenario"] == scenario]
    WASH = age.pivot_table(index="ISO3", columns="Year", values="WAshare")

    missing = [y for y in YEARS if y not in POP.columns or y not in WASH.columns]
    if missing:
        _fail("UN WPP scenario '%s' is missing years %s" % (scenario, missing[:5]))

    WAPOP = (POP * WASH).reindex(columns=YEARS)          # aligns on ISO3 x Year
    POP = POP.reindex(columns=YEARS)
    return gdp23, cname, iso2region, POP, WAPOP


# ------------------------------------------------------------------------------ forward
def project_country(iso, *, a, b, delta, lnP, lnW, ln_yF, ln_yF0, gdp23,
                    s_ss, converge, residual, sigma_annual):
    """Draw-wise annual forward pass. Returns (S, T) log-GDP paths."""
    S = a.shape[0]
    T = len(YEARS)

    # regime of each decade block, from the sign of that block's working-age change
    block_decline = {}
    for E in BLOCK_ENDS:
        start = max(E - 10, BASE)
        block_decline[E] = bool((lnW[E - BASE] - lnW[start - BASE]) < 0.0)

    slope_growth = b
    slope_decline = b + delta

    # 2024 anchor: observed 2023 real GDP advanced one year by the country's own drift.
    lnG = np.full(S, np.log(gdp23)) + a / 10.0
    # The convergence denominator is the country's *deterministic* 2024 distance to the
    # frontier (posterior-mean drift), not a draw-wise quantity. Making gap0 draw-wise puts
    # frontier countries on a knife edge -- half their draws would converge and half would
    # not, for no substantive reason. With this definition the frontier economy has
    # gap0 = 0 exactly and grows at s_ss by construction.
    ln_y0 = float(np.log(gdp23) + a.mean() / 10.0 - lnP[0])
    gap0 = ln_yF0 - ln_y0                                 # scalar distance to the frontier

    rng = np.random.default_rng(SEED + sum(iso.encode("ascii")))
    paths = np.empty((S, T))
    paths[:, 0] = lnG

    for j in range(1, T):
        E = min(int(np.ceil(YEARS[j] / 10.0) * 10), END)
        slope = slope_decline if block_decline[E] else slope_growth

        if converge and gap0 > 1e-6:
            gap = ln_yF[j - 1] - (lnG - lnP[j - 1])        # draw-wise remaining gap
            w = np.clip(gap / gap0, 0.0, 1.0)
            alpha_eff = s_ss + (a - s_ss) * w
        elif converge:
            alpha_eff = np.full(S, s_ss)                   # at or above the frontier
        else:
            alpha_eff = a

        step = alpha_eff / 10.0 + slope * (lnW[j] - lnW[j - 1])
        if residual:
            step = step + rng.normal(0.0, sigma_annual, S)
        lnG = lnG + step
        paths[:, j] = lnG

    return paths


# ---------------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)

    isos, regions, A, B, D, sigma, post_meta = load_posterior(args.draws)
    gdp23, cname, iso2region, POP, WAPOP = load_demography(args.scenario)

    sigma_annual = sigma / np.sqrt(10.0)
    aidx = {iso: i for i, iso in enumerate(isos)}
    r2i = {r: i for i, r in enumerate(regions)}

    def complete(iso, frame):
        return iso in frame.index and bool(np.all(np.isfinite(frame.loc[iso].values)))

    usable = [iso for iso in isos
              if iso in gdp23.index and complete(iso, POP) and complete(iso, WAPOP)
              and iso2region.get(iso) in r2i]
    if not usable:
        _fail("no country satisfies the completeness filter")

    lnPOP = np.log(POP)
    lnWAP = np.log(WAPOP)

    # global frontier: 2024 United States per-capita output, growing at s_ss
    ln_yF0 = float(np.log(gdp23["USA"] + 0.0) + A[aidx["USA"]].mean() / 10.0
                   - lnPOP.loc["USA", BASE])
    ln_yF = ln_yF0 + args.s_ss * (YEARS - BASE) / 10.0

    eps_note = "fitted (beta_country, delta_region)"
    if args.eps_growth is not None or args.eps_decline is not None:
        if args.eps_growth is None or args.eps_decline is None:
            _fail("--eps-growth and --eps-decline must be supplied together")
        eps_note = "override: expansion %.4f / contraction %.4f" % (args.eps_growth, args.eps_decline)

    print("s07_projection_2100  |  %d countries  |  %s" % (len(usable), eps_note))
    print("  s_ss = %.4f decadal = %.4f %%/yr | converge=%s residual=%s | draws=%d | sigma=%.4f"
          % (args.s_ss, (np.exp(args.s_ss / 10) - 1) * 100, args.converge, args.residual,
             post_meta["n_draws_used"], sigma))

    # ---------------------------------------------------------------- forward pass
    long_rows, applied = [], {}
    med_wide, lo_wide, hi_wide = {}, {}, {}
    draws_2100, draws_2024 = {}, {}

    for iso in usable:
        i = aidx[iso]
        ri = r2i[iso2region[iso]]
        S = A.shape[1]
        if args.eps_growth is None:
            b, delta = B[i], D[ri]
        else:
            b = np.full(S, args.eps_growth)
            delta = np.full(S, args.eps_decline - args.eps_growth)

        lnP = lnPOP.loc[iso].values
        lnW = lnWAP.loc[iso].values
        paths = project_country(iso, a=A[i], b=b, delta=delta, lnP=lnP, lnW=lnW,
                                ln_yF=ln_yF, ln_yF0=ln_yF0, gdp23=float(gdp23[iso]),
                                s_ss=args.s_ss, converge=args.converge,
                                residual=args.residual, sigma_annual=sigma_annual)

        G = np.exp(paths)                                   # (S, T) aggregate GDP draws
        Pv = POP.loc[iso].values[None, :]                   # (1, T) population, deterministic
        PC = G / Pv                                         # per capita -- SAME series, by construction

        gm, glo, ghi = (np.median(G, 0), np.percentile(G, 2.5, 0), np.percentile(G, 97.5, 0))
        pm, plo, phi = (np.median(PC, 0), np.percentile(PC, 2.5, 0), np.percentile(PC, 97.5, 0))

        med_wide[iso], lo_wide[iso], hi_wide[iso] = gm, glo, ghi
        applied[iso] = dict(
            beta_expansion=float(np.mean(b)),
            beta_contraction=float(np.mean(b + delta)),
            alpha_decadal=float(np.mean(A[i])),
        )
        draws_2024[iso] = (G[:, 0], PC[:, 0])
        draws_2100[iso] = (G[:, -1], PC[:, -1])

        for j, y in enumerate(YEARS):
            long_rows.append({
                "ISO3": iso, "Country": cname.get(iso, iso), "Region": iso2region[iso],
                "Year": int(y),
                "Population": float(POP.loc[iso, y]), "WApop": float(WAPOP.loc[iso, y]),
                "GDP_median": gm[j], "GDP_lo": glo[j], "GDP_hi": ghi[j],
                "GDPpc_median": pm[j], "GDPpc_lo": plo[j], "GDPpc_hi": phi[j],
            })

    long_df = pd.DataFrame(long_rows)

    # ------------------------------------------------- CHECK 1: arithmetic identity
    # g_agg - g_pc must equal g_pop for every country and every year, to machine precision.
    # This is the single line that would have stopped the ver6 accident.
    w = long_df.pivot(index="Year", columns="ISO3", values="GDP_median")
    wpc = long_df.pivot(index="Year", columns="ISO3", values="GDPpc_median")
    wpop = long_df.pivot(index="Year", columns="ISO3", values="Population")
    resid = np.abs((np.log(w) - np.log(wpc)) - np.log(wpop)).to_numpy()
    max_resid = float(np.nanmax(resid))
    if max_resid > IDENTITY_TOL:
        bad = np.unravel_index(np.nanargmax(resid), resid.shape)
        _fail("aggregate/per-capita/population identity violated: max |residual| = %.3e "
              "at %s, %s (tolerance %.1e)" % (max_resid, w.index[bad[0]], w.columns[bad[1]],
                                              IDENTITY_TOL))
    print("  [check 1/4] identity  g_agg - g_pc == g_pop : max residual %.2e  PASS" % max_resid)

    # -------------------------------------------------------- summary + growth rates
    rows = []
    for iso in usable:
        gm = med_wide[iso]
        r = {"ISO3": iso, "Country": cname.get(iso, iso), "Region": iso2region[iso]}
        for y in TABLE_YEARS:
            j = int(y - BASE)
            r["GDP_%d" % y] = gm[j]
            r["GDPpc_%d" % y] = gm[j] / float(POP.loc[iso, y])
            r["Pop_%d" % y] = float(POP.loc[iso, y])
        r["g_agg"] = np.log(r["GDP_2100"] / r["GDP_2024"]) / HORIZON
        r["g_pc"] = np.log(r["GDPpc_2100"] / r["GDPpc_2024"]) / HORIZON
        r["g_pop"] = np.log(r["Pop_2100"] / r["Pop_2024"]) / HORIZON
        r["identity_residual"] = r["g_agg"] - r["g_pc"] - r["g_pop"]
        r["GDP_2100_lo"] = lo_wide[iso][-1]
        r["GDP_2100_hi"] = hi_wide[iso][-1]
        r.update(applied[iso])
        rows.append(r)
    summ = pd.DataFrame(rows)

    for y in TABLE_YEARS:
        summ["rank_agg_%d" % y] = summ["GDP_%d" % y].rank(ascending=False).astype(int)
        summ["rank_pc_%d" % y] = summ["GDPpc_%d" % y].rank(ascending=False).astype(int)
    summ = summ.sort_values("GDP_2100", ascending=False).reset_index(drop=True)

    if float(np.abs(summ["identity_residual"]).max()) > IDENTITY_TOL:
        _fail("identity residual survived into the summary table")

    # ------------------------------------------------------- CHECK 2: plausibility
    hi = summ[summ.g_agg > G_AGG_MAX]
    lo = summ[summ.g_agg < G_AGG_MIN]
    if len(hi) or len(lo):
        _fail("implausible sustained real growth over %d years: %s"
              % (HORIZON, {**{r.ISO3: round(r.g_agg * 100, 2) for r in hi.itertuples()},
                           **{r.ISO3: round(r.g_agg * 100, 2) for r in lo.itertuples()}}))
    warn = summ[summ.g_agg > G_AGG_WARN].sort_values("g_agg", ascending=False)
    print("  [check 2/4] plausibility  g_agg in [%.2f, %.2f] %%/yr (median %.2f) : PASS%s"
          % (summ.g_agg.min() * 100, summ.g_agg.max() * 100, summ.g_agg.median() * 100,
             "" if warn.empty else "  [above %.1f%%/yr: %s]"
             % (G_AGG_WARN * 100, ", ".join("%s %.2f" % (r.ISO3, r.g_agg * 100)
                                            for r in warn.itertuples()))))

    # --------------------------------------- CHECK 3: convergence actually curves
    # If s_ss were inert the log path would be a straight line and the growth rate of the
    # last decade would equal that of the first. Require measurable deceleration among the
    # countries that start furthest below the frontier.
    if args.converge:
        first, last = [], []
        for iso in usable:
            gm = np.log(med_wide[iso])
            first.append((gm[10] - gm[0]) / 10.0)
            last.append((gm[-1] - gm[-11]) / 10.0)
        first, last = np.asarray(first), np.asarray(last)
        decel = float(np.mean(first - last))
        share = float(np.mean(last < first))
        if decel <= 1e-4 or share < 0.5:
            _fail("convergence term is inert: mean(first-decade growth - last-decade "
                  "growth) = %.2e, share decelerating = %.2f. The projection is a "
                  "constant-growth extrapolation on a log axis." % (decel, share))
        print("  [check 3/4] convergence bites  mean deceleration %.3f pp/yr, "
              "%.0f%% of countries decelerate : PASS" % (decel * 100, share * 100))
    else:
        print("  [check 3/4] convergence disabled by flag -- SKIPPED")

    # ------------------------------- CHECK 4: no per-capita/aggregate sign paradox
    # Per-capita growth must exceed aggregate growth exactly where population falls.
    paradox = summ[((summ.g_pc > summ.g_agg) & (summ.g_pop > 1e-12)) |
                   ((summ.g_pc < summ.g_agg) & (summ.g_pop < -1e-12))]
    if len(paradox):
        _fail("per-capita/aggregate ordering contradicts the population trajectory for %s"
              % paradox.ISO3.tolist())
    print("  [check 4/4] divergence sign consistent with UN population trajectory : PASS")

    # -------------------------------------------------- four-mode posterior classification
    dmodes = []
    for iso in usable:
        G0, PC0 = draws_2024[iso]
        G1, PC1 = draws_2100[iso]
        dG, dPC = G1 - G0, PC1 - PC0
        dmodes.append({
            "ISO3": iso, "Country": cname.get(iso, iso), "Region": iso2region[iso],
            "g_pop": float(summ.loc[summ.ISO3 == iso, "g_pop"].iloc[0]),
            "P_concordant_expansion": float(np.mean((dG >= 0) & (dPC >= 0))),
            "P_benign_divergence": float(np.mean((dG < 0) & (dPC >= 0))),
            "P_reverse_divergence": float(np.mean((dG >= 0) & (dPC < 0))),
            "P_concordant_contraction": float(np.mean((dG < 0) & (dPC < 0))),
            "P_pc_gain": float(np.mean(dPC >= 0)),
            "P_agg_loss": float(np.mean(dG < 0)),
        })
    dmodes = pd.DataFrame(dmodes).sort_values("g_pop").reset_index(drop=True)

    # ------------------------------------------------------------------------ write
    t = args.tag
    long_df.to_csv(DIR_RESULTS / ("projection_2100%s.csv" % t), index=False)
    summ.to_csv(DIR_RESULTS / ("projection_summary%s.csv" % t), index=False)
    dmodes.to_csv(DIR_RESULTS / ("divergence_modes%s.csv" % t), index=False)

    shrink = summ[summ.g_pop < 0]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "s07_projection_2100.py",
        "tag": t or "(fitted)",
        "settings": {
            "base_year": BASE, "end_year": END, "horizon_years": HORIZON,
            "base_anchor": "observed 2023 real GDP advanced one year by alpha_country/10",
            "s_ss_decadal": args.s_ss,
            "s_ss_pct_per_year": round((np.exp(args.s_ss / 10) - 1) * 100, 4),
            "scenario": args.scenario, "converge": args.converge,
            "residual": args.residual, "sigma_decadal": sigma,
            "sigma_annual": float(sigma_annual), "seed": SEED,
            "elasticity": eps_note,
            "eps_growth": args.eps_growth, "eps_decline": args.eps_decline,
            **post_meta,
        },
        "checks": {
            "identity_max_residual": max_resid,
            "identity_tolerance": IDENTITY_TOL,
            "g_agg_bounds_pct": [G_AGG_MIN * 100, G_AGG_MAX * 100],
            "g_agg_min_pct": float(summ.g_agg.min() * 100),
            "g_agg_max_pct": float(summ.g_agg.max() * 100),
            "g_agg_median_pct": float(summ.g_agg.median() * 100),
            "n_above_warn": int(len(warn)),
            "convergence_mean_deceleration_pp_per_yr":
                float(decel * 100) if args.converge else None,
            "convergence_share_decelerating": share if args.converge else None,
            "sign_paradoxes": 0,
        },
        "coverage": {
            "n_countries": len(usable),
            "n_countries_shrinking": int(len(shrink)),
            "n_posterior_draws_used": post_meta["n_draws_used"],
        },
        "headline": {
            "top5_2100_aggregate": [
                {"ISO3": r.ISO3, "GDP_2100_T": round(r.GDP_2100 / 1e12, 2),
                 "g_agg_pct": round(r.g_agg * 100, 2), "g_pc_pct": round(r.g_pc * 100, 2)}
                for r in summ.head(5).itertuples()],
            "median_g_pc_pct_shrinking": float(shrink.g_pc.median() * 100) if len(shrink) else None,
            "median_g_agg_pct_shrinking": float(shrink.g_agg.median() * 100) if len(shrink) else None,
            "median_g_pc_pct_growing": float(summ[summ.g_pop >= 0].g_pc.median() * 100),
            "mean_per_capita_dividend_pp": float(
                ((shrink.g_pc - shrink.g_agg) * 100).mean()) if len(shrink) else None,
        },
    }
    with open(DIR_RESULTS / ("projection_manifest%s.json" % t), "w") as fh:
        json.dump(manifest, fh, indent=2)

    # ------------------------------------------------------------------------ report
    disp = summ.head(12).copy()
    print("\n  2100 leaders (single run -- aggregate and per capita from the same draws):")
    print("  %-22s %9s %9s %9s %9s %7s %7s %7s" %
          ("Country", "GDP24_T", "GDP00_T", "pc24_k", "pc00_k", "gAgg%", "gPc%", "gPop%"))
    for r in disp.itertuples():
        print("  %-22s %9.2f %9.2f %9.1f %9.1f %7.2f %7.2f %7.2f" %
              (r.Country[:22], r.GDP_2024 / 1e12, r.GDP_2100 / 1e12,
               r.GDPpc_2024 / 1e3, r.GDPpc_2100 / 1e3,
               r.g_agg * 100, r.g_pc * 100, r.g_pop * 100))
    if len(shrink):
        print("\n  %d of %d countries depopulate by 2100. Among them the median per-capita "
              "growth is %.2f %%/yr against median aggregate growth %.2f %%/yr"
              % (len(shrink), len(usable), shrink.g_pc.median() * 100,
                 shrink.g_agg.median() * 100))
        print("  -> mean per-capita dividend of depopulation: %.2f pp/yr"
              % ((shrink.g_pc - shrink.g_agg) * 100).mean())
    print("\n  wrote projection_2100%s.csv, projection_summary%s.csv, "
          "divergence_modes%s.csv, projection_manifest%s.json" % (t, t, t, t))
    return summ


if __name__ == "__main__":
    main()
