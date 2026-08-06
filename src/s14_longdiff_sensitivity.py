"""
s14_longdiff_sensitivity.py -- what the long-difference elasticity depends on
============================================================================
The published long-difference estimate is 0.805 (95% CrI 0.577-1.029) with P(eps < 1) = 0.961.
Two specification choices that were not reported as sensitivities carry part of that result, and
a referee will find them:

  (1) SHRINKAGE ON THE COUNTRY DRIFT. Differencing removes each country's *level*, not its drift.
      The drift survives in the differenced equation and carries a three-level shrinkage prior,
      so the estimator is a random-effects blend rather than a within estimator, and
      between-country variation re-enters through the pooled intercept.
  (2) WINSORISATION. Both differenced variables are clipped at the 1st and 99th percentiles. The
      clipped rows are not random: they are the post-Soviet and Baltic working-age collapses,
      which carry a disproportionate share of the contraction-regime predictor variance.

This script fits the same model on the same 896 blocks under every combination of the two, and
adds the three textbook panel estimators (within, between, pooled) plus a two-way fixed-effects
estimate with country-clustered standard errors, so the Bayesian result can be located against
them. Everything is written to one JSON that the manuscript and the Supplementary Information
read.

Usage:  python src/s14_longdiff_sensitivity.py [draws] [tune] [chains]
Output: results/longdiff_sensitivity.json , results/longdiff_sensitivity.csv
"""
from __future__ import annotations

import json
import sys

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from config import PATH_MERGED_AGE, PATH_AGE_SCEN, DIR_RESULTS

K = 10
GDP_COL = "GDP_constant_2015usd"
BLOCK_END = [1970, 1980, 1990, 2000, 2010, 2020]
PERIODS = ["1961-1970", "1971-1980", "1981-1990", "1991-2000", "2001-2010", "2011-2020"]
SEED = 42

DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
TUNE = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
CHAINS = int(sys.argv[3]) if len(sys.argv) > 3 else 4


# ------------------------------------------------------------------------------ panel
def build_panel(winsorise: bool):
    """Identical construction to fit_hierarchical_workingage.py, with winsorisation optional."""
    df = pd.read_csv(PATH_MERGED_AGE)
    age = pd.read_csv(PATH_AGE_SCEN)
    age["_pri"] = np.where(age["scenario_norm"].str.contains("estimate", case=False, na=False), 0, 1)
    age = (age.sort_values(["ISO3", "Year", "_pri"])
              .drop_duplicates(["ISO3", "Year"], keep="first")[["ISO3", "Year", "WAshare"]]
              .rename(columns={"WAshare": "WAshare_scn"}))
    df = df.merge(age, on=["ISO3", "Year"], how="left")
    df["WAshare"] = df["WAshare"].fillna(df["WAshare_scn"])
    df = df.dropna(subset=[GDP_COL, "Population", "WAshare", "ISO3", "Year", "Region"]).copy()
    df = df[(df[GDP_COL] > 0) & (df["Population"] > 0) & (df["WAshare"] > 0)]
    df["Year"] = df["Year"].astype(int)
    df["ln_gdp"] = np.log(df[GDP_COL].astype(float))
    df["ln_wapop"] = np.log(df["Population"].astype(float) * df["WAshare"].astype(float))

    base = df[["ISO3", "Year", "Region", "ln_gdp", "ln_wapop"]].copy()
    b = base[["ISO3", "Year", "ln_gdp", "ln_wapop"]].copy()
    b["Year"] = b["Year"] + K
    m = base.merge(b, on=["ISO3", "Year"], suffixes=("", "_lag"))
    m["dln_gdp"] = m["ln_gdp"] - m["ln_gdp_lag"]
    m["dln_wapop"] = m["ln_wapop"] - m["ln_wapop_lag"]
    d = m[["ISO3", "Year", "Region", "dln_gdp", "dln_wapop"]].dropna()
    d = d[np.isfinite(d.dln_gdp) & np.isfinite(d.dln_wapop)]
    d = d[d["Year"].isin(BLOCK_END)].reset_index(drop=True)

    clipped = 0
    if winsorise:
        for c in ["dln_gdp", "dln_wapop"]:
            lo, hi = np.percentile(d[c], [1, 99])
            clipped += int(((d[c] < lo) | (d[c] > hi)).sum())
            d[c] = d[c].clip(lo, hi)
    d["decline"] = (d.dln_wapop < 0).astype(float)
    return d, clipped


def index_panel(d):
    d = d.copy()
    d["rc"] = d.Region.astype("category").cat.codes
    d["cc"] = d.ISO3.astype("category").cat.codes
    regions = list(d.Region.astype("category").cat.categories)
    countries = list(d.ISO3.astype("category").cat.categories)
    c2r = d[["cc", "rc"]].drop_duplicates().sort_values("cc")["rc"].values.astype(int)
    pi = d.Year.map({y: i for i, y in enumerate(BLOCK_END)}).values.astype(int)
    return d, regions, countries, c2r, pi


# ------------------------------------------------------------------------------ model
def fit(d, free_intercepts: bool, label: str):
    d, regions, countries, c2r, pi = index_panel(d)
    ci = d.cc.values.astype(int)
    ri = d.rc.values.astype(int)
    x = d.dln_wapop.values
    dec = d.decline.values
    y = d.dln_gdp.values
    coords = {"region": regions, "country": countries, "period": PERIODS}

    with pm.Model(coords=coords):
        beta_global = pm.Normal("beta_global", 1.0, 0.5)
        s_beta_r = pm.HalfNormal("sigma_beta_region", 0.5)
        z_beta_r = pm.Normal("z_beta_region", 0, 1, dims="region")
        beta_region = pm.Deterministic("beta_region", beta_global + s_beta_r * z_beta_r,
                                       dims="region")
        s_beta_c = pm.HalfNormal("sigma_beta_country", 0.5)
        z_beta_c = pm.Normal("z_beta_country", 0, 1, dims="country")
        beta_country = pm.Deterministic("beta_country", beta_region[c2r] + s_beta_c * z_beta_c,
                                        dims="country")

        delta_global = pm.Normal("delta_global", 0.0, 0.5)
        s_delta_r = pm.HalfNormal("sigma_delta_region", 0.3)
        z_delta_r = pm.Normal("z_delta_region", 0, 1, dims="region")
        delta_region = pm.Deterministic("delta_region", delta_global + s_delta_r * z_delta_r,
                                        dims="region")

        if free_intercepts:
            # flat, unpooled country drift: the within-estimator analogue of the intercept
            alpha_country = pm.Normal("alpha_country", 0.0, 5.0, dims="country")
        else:
            alpha_global = pm.Normal("alpha_global", 0.0, 1.0)
            s_a_r = pm.HalfNormal("sigma_alpha_region", 0.5)
            z_a_r = pm.Normal("z_alpha_region", 0, 1, dims="region")
            alpha_region = pm.Deterministic("alpha_region", alpha_global + s_a_r * z_a_r,
                                            dims="region")
            s_a_c = pm.HalfNormal("sigma_alpha_country", 0.5)
            z_a_c = pm.Normal("z_alpha_country", 0, 1, dims="country")
            alpha_country = pm.Deterministic("alpha_country",
                                             alpha_region[c2r] + s_a_c * z_a_c, dims="country")

        tau_period = pm.ZeroSumNormal("tau_period", sigma=0.5, dims="period")
        pm.Deterministic("beta_decline_global", beta_global + delta_global)
        slope = beta_country[ci] + delta_region[ri] * dec
        mu = alpha_country[ci] + tau_period[pi] + slope * x
        sigma = pm.HalfNormal("sigma", 0.5)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=min(CHAINS, 4),
                          target_accept=0.92, nuts_sampler="nutpie", random_seed=SEED,
                          progressbar=False)

    f = lambda v: idata.posterior[v].stack(z=("chain", "draw")).values
    bg, dg, bd = f("beta_global"), f("delta_global"), f("beta_decline_global")
    summ = az.summary(idata, var_names=["beta_global", "delta_global", "sigma"])
    ndiv = int(idata.sample_stats["diverging"].sum())
    return {
        "label": label, "n_obs": int(len(d)), "n_countries": len(countries),
        "free_country_intercepts": free_intercepts,
        "elasticity_mean": float(bg.mean()),
        "elasticity_q2.5": float(np.percentile(bg, 2.5)),
        "elasticity_q97.5": float(np.percentile(bg, 97.5)),
        "P_lt_1": float((bg < 1).mean()),
        "delta_mean": float(dg.mean()),
        "delta_q2.5": float(np.percentile(dg, 2.5)),
        "delta_q97.5": float(np.percentile(dg, 97.5)),
        "P_delta_lt_0": float((dg < 0).mean()),
        "contraction_elasticity_mean": float(bd.mean()),
        "contraction_P_lt_1": float((bd < 1).mean()),
        "sigma_alpha_country": (None if free_intercepts
                                else float(f("sigma_alpha_country").mean())),
        "residual_sigma": float(f("sigma").mean()),
        "max_rhat": float(summ.r_hat.max()), "min_ess_bulk": float(summ.ess_bulk.min()),
        "divergences": ndiv,
    }


# ------------------------------------------------- textbook panel estimators, for location
def panel_estimators(d):
    d = d.copy()
    g = d.groupby("ISO3")
    d["x_bar"] = g.dln_wapop.transform("mean")
    d["y_bar"] = g.dln_gdp.transform("mean")
    within = float(np.polyfit(d.dln_wapop - d.x_bar, d.dln_gdp - d.y_bar, 1)[0])
    bm = g[["dln_wapop", "dln_gdp"]].mean()
    between = float(np.polyfit(bm.dln_wapop, bm.dln_gdp, 1)[0])
    pooled = float(np.polyfit(d.dln_wapop, d.dln_gdp, 1)[0])

    # two-way fixed effects (country + decade) by within transformation, cluster-robust SE
    X = pd.get_dummies(d.Year.astype(str), prefix="yr", drop_first=True).astype(float)
    X.insert(0, "x", d.dln_wapop.values)
    yv = d.dln_gdp.values.copy()
    cc = d.ISO3.values
    Xd = X.to_numpy(dtype=float)
    for col in range(Xd.shape[1]):
        Xd[:, col] -= pd.Series(Xd[:, col]).groupby(cc).transform("mean").to_numpy()
    yd = yv - pd.Series(yv).groupby(cc).transform("mean").to_numpy()
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ yd)
    resid = yd - Xd @ beta
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    for c in np.unique(cc):
        m = cc == c
        u = (Xd[m].T @ resid[m]).reshape(-1, 1)
        meat += u @ u.T
    G = len(np.unique(cc))
    n, k = Xd.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    V = XtX_inv @ meat @ XtX_inv * adj
    return {"within": within, "between": between, "pooled": pooled,
            "twoway_fe": float(beta[0]), "twoway_fe_se_clustered": float(np.sqrt(V[0, 0])),
            "n_clusters": int(G)}


def main():
    out = {"settings": {"draws": DRAWS, "tune": TUNE, "chains": CHAINS, "seed": SEED,
                        "gdp_column": GDP_COL, "blocks": BLOCK_END},
           "specifications": [], "panel_estimators": {}, "winsorisation": {}}

    dw, n_clipped = build_panel(winsorise=True)
    dr, _ = build_panel(winsorise=False)

    # what the winsorisation actually does, in the contraction regime specifically
    lo_x, hi_x = np.percentile(dr.dln_wapop, [1, 99])
    lo_y, hi_y = np.percentile(dr.dln_gdp, [1, 99])
    contr = dr[dr.dln_wapop < 0]
    clipped_contr = contr[(contr.dln_wapop < lo_x) | (contr.dln_gdp < lo_y) |
                          (contr.dln_gdp > hi_y)]
    var_all = float(contr.dln_wapop.var())
    var_kept = float(contr[~contr.index.isin(clipped_contr.index)].dln_wapop.var())
    out["winsorisation"] = {
        "n_rows_clipped_either_variable": int(n_clipped),
        "n_contraction_blocks": int(len(contr)),
        "n_contraction_blocks_clipped": int(len(clipped_contr)),
        "share_of_contraction_blocks_clipped": float(len(clipped_contr) / max(len(contr), 1)),
        "contraction_predictor_variance_removed":
            float(1 - var_kept * (len(contr) - len(clipped_contr)) / (var_all * len(contr)))
            if len(contr) else None,
        "countries_clipped_in_contraction":
            sorted(clipped_contr.ISO3.unique().tolist()),
        "decades_clipped_in_contraction":
            {str(int(k)): int(v) for k, v in clipped_contr.Year.value_counts().items()},
    }

    for d, wins in [(dw, True), (dr, False)]:
        for free in (False, True):
            lab = "%s intercepts, %s winsorisation" % (
                "free country" if free else "pooled", "with" if wins else "no")
            print("fitting: %s ..." % lab, flush=True)
            r = fit(d, free_intercepts=free, label=lab)
            r["winsorised"] = wins
            out["specifications"].append(r)

    out["panel_estimators"] = {"winsorised": panel_estimators(dw),
                               "unwinsorised": panel_estimators(dr)}

    with open(DIR_RESULTS / "longdiff_sensitivity.json", "w") as fh:
        json.dump(out, fh, indent=2)
    tab = pd.DataFrame(out["specifications"])
    tab.to_csv(DIR_RESULTS / "longdiff_sensitivity.csv", index=False)

    print("\n%-42s %6s %-26s %8s %10s" % ("specification", "n", "elasticity", "P(eps<1)", "max Rhat"))
    print("-" * 100)
    for r in out["specifications"]:
        print("%-42s %6d %.3f [%.3f, %.3f]      %8.3f %10.4f"
              % (r["label"], r["n_obs"], r["elasticity_mean"], r["elasticity_q2.5"],
                 r["elasticity_q97.5"], r["P_lt_1"], r["max_rhat"]))
    print()
    for k, v in out["panel_estimators"].items():
        print("%-14s within %.3f | between %.3f | pooled %.3f | two-way FE %.3f (SE %.3f, G=%d)"
              % (k, v["winsorised"] if False else v["within"], v["between"], v["pooled"],
                 v["twoway_fe"], v["twoway_fe_se_clustered"], v["n_clusters"]))
    w = out["winsorisation"]
    print("\nwinsorisation clips %d of %d contraction blocks (%.0f%%): %s"
          % (w["n_contraction_blocks_clipped"], w["n_contraction_blocks"],
             100 * w["share_of_contraction_blocks_clipped"],
             ", ".join(w["countries_clipped_in_contraction"])))
    print("\n  wrote longdiff_sensitivity.json / .csv")


if __name__ == "__main__":
    main()
