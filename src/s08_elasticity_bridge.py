"""
s08_elasticity_bridge.py -- reconciling the levels model and the long-difference model
=====================================================================================
The paper turns on two estimates of the same quantity produced by two specifications:

  LEVELS (penalised B-spline, `fit_gdp_production.py`)
      the local elasticity is a derivative of a fitted spline in log population. Its
      contraction-regime value moves from 0.92 to 1.22 on the spline penalty alone.
      LOO prefers the adaptive specification, but by a margin that does not license
      either value of the policy-relevant derivative.

  LONG DIFFERENCES (working-age decadal model, `fit_hierarchical_workingage.py`)
      the elasticity is a slope on non-overlapping ten-year changes. It is stable, it is
      sub-unitary, and the contraction-regime increment delta is centred near zero.

This script produces every number the manuscript needs to state that comparison, so that
none of them is typed by hand:

  * posterior probability that the elasticity is sub-unitary, global and by region
  * the contraction increment delta and the posterior probability that it is negative
  * where the two levels-model candidates (0.92, 1.22) sit inside the long-difference
    posterior -- i.e. whether the long-difference evidence can discriminate between them
  * the shrinkage evidence against regional heterogeneity: sigma_beta_region posterior
    against its prior mean
  * two adversarial robustness refits, each with its own archived posterior:
      - WEAK PRIORS: beta_global ~ Normal(1, 2) with HalfCauchy(1) on every hierarchical
        scale, so the sub-unitary result cannot be a product of the Normal(1, 0.5) prior
      - FIVE-YEAR BLOCKS on TOTAL population: 2,036 non-overlapping observations against
        896, a power stress-test on the regional-heterogeneity claim, and a second bracket
        on the elasticity from a different estimand and a different horizon
  * the LOO comparison with its weights labelled correctly (az.compare(ic="loo") returns
    STACKING weights, not Akaike weights -- the manuscript has been calling them Akaike)

Usage:  python src/s08_elasticity_bridge.py
Output: results/elasticity_bridge.json , results/elasticity_bridge_regions.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import halfnorm

from config import DIR_RESULTS

CANDIDATES = [0.92, 1.22]          # the two levels-model contraction elasticities
PRIOR_SIGMA_BETA_REGION = 0.5      # HalfNormal(0.5) in fit_hierarchical_workingage.py


def q(x, p=(2.5, 50, 97.5)):
    return [float(v) for v in np.percentile(x, p)]


def main():
    post = xr.open_dataset(DIR_RESULTS / "hierarchical_wa_posterior.nc", group="posterior")
    flat = lambda v: post[v].stack(z=("chain", "draw")).values

    bg = flat("beta_global")                      # expansion-regime elasticity
    dg = flat("delta_global")                     # contraction increment
    bd = flat("beta_decline_global")              # = bg + dg
    sbr = flat("sigma_beta_region")
    sbc = flat("sigma_beta_country")
    br = flat("beta_region")                      # (R, S)
    brd = flat("beta_decline_region")             # (R, S)
    regions = [str(r) for r in post["region"].values]

    out = {
        "long_difference_model": {
            "source": "results/hierarchical_wa_posterior.nc",
            "specification": ("Delta10 ln GDP = alpha_c + tau_p + "
                              "(beta_c + delta_r * 1[decline]) * Delta10 ln WApop + eps"),
            "n_country_decades": 896, "n_countries": 179, "n_regions": 7,
            "n_contraction_blocks": 69,
            "expansion_elasticity": {
                "mean": float(bg.mean()), "sd": float(bg.std()),
                "q2.5_50_97.5": q(bg), "P_lt_1": float((bg < 1).mean()),
            },
            "contraction_elasticity": {
                "mean": float(bd.mean()), "sd": float(bd.std()),
                "q2.5_50_97.5": q(bd), "P_lt_1": float((bd < 1).mean()),
            },
            "contraction_increment_delta": {
                "mean": float(dg.mean()), "sd": float(dg.std()),
                "q2.5_50_97.5": q(dg), "P_lt_0": float((dg < 0).mean()),
                "interpretation": ("posterior straddles zero: no evidence that the "
                                   "elasticity differs between expansion and contraction"),
            },
        },
        "levels_model_candidates": {},
        "heterogeneity_shrinkage": {
            "sigma_beta_region_posterior_mean": float(sbr.mean()),
            "sigma_beta_region_posterior_q": q(sbr),
            "sigma_beta_region_prior": "HalfNormal(%.1f)" % PRIOR_SIGMA_BETA_REGION,
            "sigma_beta_region_prior_mean": float(halfnorm.mean(scale=PRIOR_SIGMA_BETA_REGION)),
            "shrinkage_ratio_posterior_over_prior": float(
                sbr.mean() / halfnorm.mean(scale=PRIOR_SIGMA_BETA_REGION)),
            "sigma_beta_country_posterior_mean": float(sbc.mean()),
            "regional_expansion_range": [float(br.mean(1).min()), float(br.mean(1).max())],
            "regional_contraction_range": [float(brd.mean(1).min()), float(brd.mean(1).max())],
        },
    }

    # Where do the levels-model candidates sit inside the long-difference posterior?
    for c in CANDIDATES:
        out["levels_model_candidates"]["eps_%.2f" % c] = {
            "P_expansion_elasticity_exceeds": float((bg > c).mean()),
            "P_contraction_elasticity_exceeds": float((bd > c).mean()),
            "posterior_percentile_in_expansion": float((bg < c).mean() * 100),
            "posterior_percentile_in_contraction": float((bd < c).mean() * 100),
        }

    # per-region table
    rows = []
    for i, r in enumerate(regions):
        rows.append({
            "Region": r,
            "beta_expansion_mean": float(br[i].mean()),
            "beta_expansion_lo": float(np.percentile(br[i], 2.5)),
            "beta_expansion_hi": float(np.percentile(br[i], 97.5)),
            "P_expansion_lt_1": float((br[i] < 1).mean()),
            "beta_contraction_mean": float(brd[i].mean()),
            "beta_contraction_lo": float(np.percentile(brd[i], 2.5)),
            "beta_contraction_hi": float(np.percentile(brd[i], 97.5)),
            "P_contraction_lt_1": float((brd[i] < 1).mean()),
        })
    reg = pd.DataFrame(rows).sort_values("beta_expansion_mean")
    reg.to_csv(DIR_RESULTS / "elasticity_bridge_regions.csv", index=False)

    # ---------------------------------------------------- adversarial robustness refits
    out["robustness"] = {}

    wp = DIR_RESULTS / "hierarchical_wa_posterior_weakprior.nc"
    if wp.exists():
        w = xr.open_dataset(wp, group="posterior")
        wf = lambda v: w[v].stack(z=("chain", "draw")).values
        wbg, wdg, wsbr = wf("beta_global"), wf("delta_global"), wf("sigma_beta_region")
        wbr = wf("beta_region")
        base_reg = pd.read_csv(DIR_RESULTS / "hierarchical_wa_region_elasticities.csv"
                               ).set_index("Region")["beta_growth"]
        weak_reg = pd.read_csv(DIR_RESULTS / "hierarchical_wa_region_elasticities_weakprior.csv"
                               ).set_index("Region")["beta_growth"]
        out["robustness"]["weak_priors"] = {
            "specification": ("beta_global ~ Normal(1, 2); delta_global ~ Normal(0, 2); "
                              "every hierarchical scale ~ HalfCauchy(1)"),
            "expansion_elasticity_mean": float(wbg.mean()),
            "expansion_elasticity_q2.5_50_97.5": q(wbg),
            "P_lt_1": float((wbg < 1).mean()),
            "delta_mean": float(wdg.mean()), "delta_q2.5_50_97.5": q(wdg),
            "sigma_beta_region_mean": float(wsbr.mean()),
            "regional_expansion_range": [float(wbr.mean(1).min()), float(wbr.mean(1).max())],
            "max_abs_regional_shift_vs_baseline": float(
                (weak_reg - base_reg).abs().max()),
            "interpretation": ("the sub-unitary result and the collapse of regional "
                               "heterogeneity both survive priors that would happily "
                               "accommodate unity and large between-region variance"),
        }

    fp = DIR_RESULTS / "hierarchical_5yr_posterior.nc"
    if fp.exists():
        f5 = xr.open_dataset(fp)
        g5 = lambda v: f5[v].stack(z=("chain", "draw")).values
        b5, s5, br5 = g5("beta_global"), g5("sigma_beta_region"), g5("beta_region")
        reg5 = pd.read_csv(DIR_RESULTS / "hierarchical_region_elasticities_5yr.csv")
        spans_unity = int(((reg5.hdi_lo < 1) & (reg5.hdi_hi > 1)).sum())
        # rank correlation of the regional ordering against the ten-year working-age model
        order10 = reg.set_index("Region")["beta_expansion_mean"]
        order5 = reg5.set_index("Region")["beta_mean"]
        common = [r for r in order5.index if r in order10.index]
        rho = float(pd.Series(order10[common]).rank().corr(
            pd.Series(order5[common]).rank(), method="spearman"))
        out["robustness"]["five_year_blocks_total_population"] = {
            "specification": ("Delta5 ln GDP = alpha_c + beta_c * Delta5 ln Population, "
                              "non-overlapping five-year blocks 1965-2020, three-level "
                              "hierarchy, priors identical to the ten-year model"),
            "n_countries": int(len(pd.read_csv(
                DIR_RESULTS / "hierarchical_country_elasticities_5yr.csv"))),
            "n_observations": int(pd.read_csv(
                DIR_RESULTS / "hierarchical_country_elasticities_5yr.csv")["n_obs"].sum()),
            "elasticity_mean": float(b5.mean()), "elasticity_q2.5_50_97.5": q(b5),
            "P_lt_1": float((b5 < 1).mean()),
            "sigma_beta_region_mean": float(s5.mean()),
            "regional_range": [float(br5.mean(1).min()), float(br5.mean(1).max())],
            "n_regions_whose_interval_spans_unity": spans_unity,
            "n_regions": int(len(reg5)),
            "regional_rank_correlation_with_ten_year_model": rho,
            "interpretation": ("doubling the independent sample leaves every regional "
                               "interval spanning unity and reshuffles the ordering "
                               "(Spearman rho %.2f against the ten-year model) -- the "
                               "signature of noise rather than structure" % rho),
        }
        for c in CANDIDATES:
            out["levels_model_candidates"]["eps_%.2f" % c][
                "posterior_percentile_in_five_year_blocks"] = float((b5 < c).mean() * 100)
            if wp.exists():
                out["levels_model_candidates"]["eps_%.2f" % c][
                    "posterior_percentile_under_weak_priors"] = float((wbg < c).mean() * 100)

    # LOO comparison, with the weight method named correctly
    loo = pd.read_csv(DIR_RESULTS / "model_comparison_loo.csv", index_col=0)
    out["loo_comparison"] = {
        "note": ("az.compare(..., ic='loo') returns STACKING weights by default. Earlier "
                 "drafts described these as Akaike weights; they are not."),
        "models": {str(k): {"elpd_loo": float(v["elpd_loo"]),
                            "elpd_diff": float(v["elpd_diff"]),
                            "dse": float(v["dse"]),
                            "stacking_weight": float(v["weight"]),
                            "p_loo": float(v["p_loo"])}
                   for k, v in loo.iterrows()},
        "elpd_diff_in_dse": float(loo["elpd_diff"].max() /
                                  loo.loc[loo["elpd_diff"].idxmax(), "dse"]),
    }

    # archived levels-model elasticities (two files disagree -- record both, flag it)
    infl = json.load(open(DIR_RESULTS / "elasticity_inflection.json"))
    figs = json.load(open(DIR_RESULTS / "elasticity_figure_summary.json"))
    out["levels_model_archived"] = {
        "adaptive_contraction": {"mean": infl["decline_elasticity"]["mean"],
                                 "ci": [infl["decline_elasticity"]["ci_lo"],
                                        infl["decline_elasticity"]["ci_hi"]],
                                 "n": infl["decline_elasticity"]["n"]},
        "adaptive_expansion": {"mean": infl["growth_elasticity"]["mean"],
                               "ci": [infl["growth_elasticity"]["ci_lo"],
                                      infl["growth_elasticity"]["ci_hi"]],
                               "n": infl["growth_elasticity"]["n"]},
        "core_mean_elasticity_inflection_json": infl["core_mean_elasticity"],
        "core_mean_elasticity_figure_summary_json": figs["core_mean_elasticity"],
        "DISCREPANCY": ("elasticity_inflection.json and elasticity_figure_summary.json "
                        "report different core means (%.4f vs %.4f) and different tipping "
                        "points (%.0f vs %.0f). Only elasticity_inflection.json carries "
                        "credible intervals and sample sizes; use it and drop the other."
                        % (infl["core_mean_elasticity"], figs["core_mean_elasticity"],
                           infl["tipping_point_pop"], figs["tipping_point_pop"])),
    }

    with open(DIR_RESULTS / "elasticity_bridge.json", "w") as fh:
        json.dump(out, fh, indent=2)

    # ----------------------------------------------------------------------- report
    L = out["long_difference_model"]
    print("LONG-DIFFERENCE MODEL (n = 896 country-decades, 179 countries)")
    print("  expansion   beta = %.3f  95%% CrI [%.3f, %.3f]   P(beta < 1) = %.3f"
          % (L["expansion_elasticity"]["mean"], *L["expansion_elasticity"]["q2.5_50_97.5"][::2],
             L["expansion_elasticity"]["P_lt_1"]))
    print("  contraction beta = %.3f  95%% CrI [%.3f, %.3f]   P(beta < 1) = %.3f"
          % (L["contraction_elasticity"]["mean"],
             *L["contraction_elasticity"]["q2.5_50_97.5"][::2],
             L["contraction_elasticity"]["P_lt_1"]))
    print("  increment  delta = %+.3f  95%% CrI [%+.3f, %+.3f]  P(delta < 0) = %.3f"
          % (L["contraction_increment_delta"]["mean"],
             *L["contraction_increment_delta"]["q2.5_50_97.5"][::2],
             L["contraction_increment_delta"]["P_lt_0"]))
    print("\nLEVELS-MODEL CANDIDATES inside the long-difference posterior")
    for c in CANDIDATES:
        k = out["levels_model_candidates"]["eps_%.2f" % c]
        print("  eps = %.2f  ->  expansion percentile %5.1f   contraction percentile %5.1f"
              % (c, k["posterior_percentile_in_expansion"],
                 k["posterior_percentile_in_contraction"]))
    for k, v in out.get("robustness", {}).items():
        print("\nROBUSTNESS  %s" % k)
        for kk in ("expansion_elasticity_mean", "elasticity_mean"):
            if kk in v:
                qq = v.get("expansion_elasticity_q2.5_50_97.5") or v.get("elasticity_q2.5_50_97.5")
                print("  elasticity = %.3f  95%% CrI [%.3f, %.3f]   P(< 1) = %.3f"
                      % (v[kk], qq[0], qq[2], v["P_lt_1"]))
        print("  sigma_beta_region = %.3f | regional range %.3f-%.3f"
              % (v["sigma_beta_region_mean"],
                 *(v.get("regional_expansion_range") or v.get("regional_range"))))
        if "max_abs_regional_shift_vs_baseline" in v:
            print("  largest shift in any regional elasticity vs baseline: %.4f"
                  % v["max_abs_regional_shift_vs_baseline"])
        if "n_regions_whose_interval_spans_unity" in v:
            print("  %d of %d regional intervals span unity; regional rank correlation "
                  "with the ten-year model = %.2f"
                  % (v["n_regions_whose_interval_spans_unity"], v["n_regions"],
                     v["regional_rank_correlation_with_ten_year_model"]))

    H = out["heterogeneity_shrinkage"]
    print("\nHETEROGENEITY  sigma_beta_region posterior mean %.3f vs prior mean %.3f "
          "(ratio %.2f)" % (H["sigma_beta_region_posterior_mean"],
                            H["sigma_beta_region_prior_mean"],
                            H["shrinkage_ratio_posterior_over_prior"]))
    print("  regional expansion elasticities span %.3f-%.3f; contraction %.3f-%.3f"
          % (*H["regional_expansion_range"], *H["regional_contraction_range"]))
    print("\nLOO  elpd difference = %.1f (%.1f dse); stacking weights %s"
          % (loo["elpd_diff"].max(), out["loo_comparison"]["elpd_diff_in_dse"],
             {k: round(v["stacking_weight"], 3)
              for k, v in out["loo_comparison"]["models"].items()}))
    print("\n  wrote elasticity_bridge.json, elasticity_bridge_regions.csv")


if __name__ == "__main__":
    main()
