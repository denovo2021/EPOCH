"""
scenario_projection.py
======================
Standalone demographic forward pass for the hierarchical GDP model (B-spline posterior).
Projects posterior-predictive Total GDP and GDP per capita to 2050 under the UN WPP
Medium-variant input, and classifies each draw into the four outcome modes of SI S3.4 to
estimate the regional divergence probability. GDP-only. Loads posterior with xr.open_dataset.

Output:
  results/gdp_scenario_projection.csv
  results/divergence_probabilities_by_region.csv

Usage: python scenario_projection.py [--scenario "Medium variant"] [--draws 1000]
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
import xarray as xr
from scipy.interpolate import BSpline
from config import (PATH_MODEL_V2, PATH_SCALE_V2, PATH_MERGED_AGE,
                    PATH_POP_SCEN, PATH_AGE_SCEN, DIR_RESULTS)

BASE_YEAR = 2023
REF_YEAR = 2024
HORIZON = 2050


def bspline_centered(x, scales):
    ka = np.array(scales["bspline_knots_aug"]); deg = scales["bspline_degree"]
    nb = scales["bspline_n_basis"]; cm = np.array(scales["bspline_col_means"])
    B = np.zeros((np.size(x), nb))
    for i in range(nb):
        c = np.zeros(nb); c[i] = 1.0
        B[:, i] = BSpline(ka, c, deg, extrapolate=False)(np.asarray(x, float))
    return np.nan_to_num(B) - cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="Medium variant")
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args()

    scales = json.load(open(PATH_SCALE_V2))
    MU, SD = scales["MU_GLOBAL"], scales["SD_GLOBAL"]
    s_dWA, s_dOD, s_dt = scales["s_dWA"], scales["s_dOD"], scales["s_dt"]
    coef_dt = np.array(scales["coef_dt_proj"])

    try:
        post = xr.open_dataset(PATH_MODEL_V2, group="posterior")
    except Exception:
        post = xr.open_dataset(PATH_MODEL_V2)
    flat = lambda v: post[v].stack(z=("chain", "draw")).values
    beta0 = flat("beta0"); theta = flat("theta")
    if theta.shape[0] != scales["bspline_n_basis"]:
        theta = theta.T
    alpha_c = flat("alpha_country"); tau_r = flat("tau_region")
    d_wa = flat("delta_washare"); d_od = flat("delta_olddep")
    S_all = beta0.shape[0]
    idx = np.random.default_rng(42).choice(S_all, min(args.draws, S_all), replace=False)
    beta0, theta = beta0[idx], theta[:, idx]
    alpha_c, tau_r = alpha_c[:, idx], tau_r[:, idx]
    d_wa, d_od = d_wa[idx], d_od[idx]
    countries = [str(c) for c in post["Country"].values]; regions = [str(r) for r in post["Region"].values]
    c2i = {c: i for i, c in enumerate(countries)}; r2i = {r: i for i, r in enumerate(regions)}

    hist = pd.read_csv(PATH_MERGED_AGE)
    pop = pd.read_csv(PATH_POP_SCEN); age = pd.read_csv(PATH_AGE_SCEN)
    pop = pop[pop["Scenario"] == args.scenario]; age = age[age["Scenario"] == args.scenario]
    scen = pd.merge(pop[["ISO3", "Year", "Population"]], age[["ISO3", "Year", "WAshare", "OldDep"]],
                    on=["ISO3", "Year"], how="inner").dropna()
    scen["Year"] = scen["Year"].astype(int)

    rows = []
    reg_G_ref = {r: np.zeros(len(idx)) for r in regions}; reg_G_hor = {r: np.zeros(len(idx)) for r in regions}
    reg_N_ref = {r: 0.0 for r in regions}; reg_N_hor = {r: 0.0 for r in regions}

    for iso, g in scen.groupby("ISO3"):
        h = hist[hist["ISO3"] == iso]
        if h.empty:
            continue
        anc = h[h["Year"] == BASE_YEAR]
        if anc.empty or pd.isna(anc["GDP"].values[0]):
            anc = h.dropna(subset=["GDP"]).sort_values("Year").tail(1)
        if anc.empty:
            continue
        cn = h["Country Name"].iloc[0] if "Country Name" in h else iso
        reg = anc["Region"].values[0]
        if reg not in r2i:
            continue
        ci = c2i.get(str(cn), c2i.get(iso))
        if ci is None:
            continue
        y_base = float(np.log10(anc["GDP"].values[0]))
        pop_base = float(np.log10(anc["Population"].values[0]))
        yb = int(anc["Year"].values[0])
        h_age = h.dropna(subset=["WAshare", "OldDep"]).sort_values("Year")
        if h_age.empty:
            continue
        wa_base = float(h_age["WAshare"].values[-1]); od_base = float(h_age["OldDep"].values[-1])
        x_base = (pop_base - MU) / SD
        sp_base = (bspline_centered([x_base], scales) @ theta).ravel()

        gg = g[g["Year"] > yb].sort_values("Year")
        if gg.empty:
            continue
        x_t = (np.log10(gg["Population"].values) - MU) / SD
        B_t = bspline_centered(x_t, scales)
        sp_t = B_t @ theta
        lin = np.outer(x_t - x_base, beta0)
        dWA = np.outer((gg["WAshare"].values - wa_base) / (s_dWA + 1e-8), d_wa)
        dOD = np.outer((gg["OldDep"].values - od_base) / (s_dOD + 1e-8), d_od)
        dt = ((gg["Year"].values - 2000) / 10.0 - (yb - 2000) / 10.0) / (s_dt + 1e-8)
        Xv = np.column_stack([np.ones_like(x_t), x_t, B_t])
        # Orthogonalized time covariate, anchored to zero at the base year. The forward pass is
        # anchored at observed GDP (y_base), so the time term must be the CHANGE from its
        # base-year value. The orthogonalization residual carries a large constant offset
        # (Xv_base @ coef_dt); without subtracting dt_orth_base it inflates GDP from the very
        # first step (~3.7x) and overpowers the demographic signal, collapsing the divergence
        # probability. Differencing removes that offset so dt_orth = 0 at the anchor year.
        Xv_base = np.concatenate([[1.0, x_base], bspline_centered([x_base], scales).ravel()])
        dt_orth_base = np.clip(0.0 - Xv_base @ coef_dt, -3.0, 6.0)
        dt_orth = np.clip(dt - Xv @ coef_dt, -3.0, 6.0) - dt_orth_base
        ri = r2i[reg]
        time_term = np.outer(dt_orth, tau_r[ri])
        log_gdp = y_base + lin + (sp_t - sp_base) + dWA + dOD + time_term
        G = 10 ** log_gdp
        Npop = gg["Population"].values[:, None]
        gpc = G / Npop
        yrs = list(gg["Year"].values)
        med = np.median(G, axis=1); lo = np.percentile(G, 2.5, axis=1); hi = np.percentile(G, 97.5, axis=1)
        pcm = np.median(gpc, axis=1)
        for k, yr in enumerate(yrs):
            rows.append({"ISO3": iso, "Region": reg, "Year": int(yr),
                         "TotalGDP_median": med[k], "TotalGDP_lo": lo[k], "TotalGDP_hi": hi[k],
                         "GDPpc_median": pcm[k]})
        if REF_YEAR in yrs and HORIZON in yrs:
            kr = yrs.index(REF_YEAR); kh = yrs.index(HORIZON)
            reg_G_ref[reg] += G[kr]; reg_G_hor[reg] += G[kh]
            reg_N_ref[reg] += float(Npop[kr, 0]); reg_N_hor[reg] += float(Npop[kh, 0])

    pd.DataFrame(rows).to_csv(DIR_RESULTS / "gdp_scenario_projection.csv", index=False)

    drows = []
    for r in regions:
        if reg_N_hor[r] == 0:
            continue
        G_ref, G_hor = reg_G_ref[r], reg_G_hor[r]
        g_ref = G_ref / reg_N_ref[r]; g_hor = G_hor / reg_N_hor[r]
        dG = G_hor - G_ref; dg = g_hor - g_ref
        drows.append({"Region": r,
                      "P_concordant_expansion": round(float(np.mean((dG >= 0) & (dg >= 0))), 3),
                      "P_divergence": round(float(np.mean((dG < 0) & (dg >= 0))), 3),
                      "P_reverse_divergence": round(float(np.mean((dG >= 0) & (dg < 0))), 3),
                      "P_concordant_contraction": round(float(np.mean((dG < 0) & (dg < 0))), 3)})
    dfd = pd.DataFrame(drows)
    dfd.to_csv(DIR_RESULTS / "divergence_probabilities_by_region.csv", index=False)
    print(dfd.to_string(index=False))
    print("\nScenario='%s', draws=%d. Saved to %s" % (args.scenario, len(idx), DIR_RESULTS))


if __name__ == "__main__":
    main()
