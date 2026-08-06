"""
s15_chainwise_check.py -- is the reported elasticity chain-invariant?
=====================================================================
`make_convergence_table.py` on `results/trace_adaptive_real.nc` reports 53 parameters above the
pre-specified split-R-hat threshold of 1.01, a minimum bulk effective sample size of 7, and a
maximum split-R-hat of 1.53. Every one of them is an intercept: alpha0, the region intercepts,
the country intercepts and their non-centred z's. None of them enters the estimand, which is

    eps(x_s) = (1/SD) * ( beta0 + sum_j theta_j B_j'(x_s) )

-- a slope, in which no alpha appears. That is an argument. This script turns it into a
measurement: it recomputes the phase-averaged elasticity separately within each chain, using
exactly the estimator of s13_levels_elasticity.py, and reports the spread across chains against
the pooled posterior standard deviation. If four chains that disagree about a country intercept
agree about the elasticity to within a small fraction of its own posterior width, the sampling
failure is localised and the reported number is not at risk.

It also localises the failure -- per-chain means of the worst-R-hat parameters, which
distinguishes mode-splitting (chains in two clusters) from slow mixing (chains drifting) -- and
reports how much posterior mass the degrees-of-freedom parameter puts against its nu >= 2 floor,
because a likelihood pinned at the floor is what makes a bimodal intercept cheap to switch.

Usage
-----
    python src/s15_chainwise_check.py \\
        --posterior real_adaptive:results/trace_adaptive_real.nc:GDP_constant_2015usd \\
        --posterior real_fixed:results/trace_fixed005_real.nc:GDP_constant_2015usd \\
        --posterior nominal_adaptive:results/trace_adaptive.nc:GDP

Each --posterior is `label:path:gdp_column`, the same form s13 takes.

Output: results/chainwise_check.json, and a printed report.
"""
from __future__ import annotations

import argparse
import json

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import BSpline

from config import PATH_SCALE_V2, PATH_MERGED_AGE, DIR_RESULTS, REPO

# The pre-specified thresholds of Supplementary Information S10.
RHAT_MAX = 1.01
ESS_MIN = 400.0
NU_FLOOR = 2.0
NU_FLOOR_TOL = 0.01


def load_basis():
    s = json.load(open(PATH_SCALE_V2))
    return (s["MU_GLOBAL"], s["SD_GLOBAL"], np.array(s["bspline_knots_aug"]),
            s["bspline_degree"], s["bspline_n_basis"])


def open_posterior(path):
    """SAVE_FULL=1 writes a full InferenceData; SAVE_FULL=0 writes the posterior group alone."""
    try:
        return xr.open_dataset(path, group="posterior")
    except Exception:
        return xr.open_dataset(path)


def deriv_matrix(x, ka, deg, nb):
    x = np.asarray(x, float)
    out = np.zeros((x.size, nb))
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1.0
        out[:, i] = BSpline(ka, c, deg, extrapolate=False).derivative(1)(x)
    return np.nan_to_num(out)


def phase_split(gdp_col, MU, SD):
    """Identical to s13_levels_elasticity.phase_split, so the numbers are comparable."""
    df = pd.read_csv(PATH_MERGED_AGE).dropna(subset=["ISO3", "Population", gdp_col]).copy()
    df = df[(df[gdp_col] > 0) & (df["Population"] > 0)]
    df["Year"] = df["Year"].astype(int)
    df["lp"] = np.log10(df["Population"])
    df = df.sort_values(["ISO3", "Year"])
    df["dlp"] = df.groupby("ISO3")["lp"].diff()
    df = df.dropna(subset=["dlp"])
    df["x_s"] = (df["lp"] - MU) / SD
    return df


def _stacked(da, nb=None):
    """(chain, draw, ...) -> keep the chain axis; return array with chain first."""
    v = da.values
    return v


def analyse(label, path, gdp_col, MU, SD, ka, deg, nb):
    post = open_posterior(path)
    beta0 = post["beta0"].values                      # (chain, draw)
    theta = post["theta"].values                      # (chain, draw, Spline)
    if theta.shape[-1] != nb:                         # tolerate (chain, Spline, draw)
        theta = np.moveaxis(theta, 1, -1)
    n_chain, n_draw = beta0.shape
    df = phase_split(gdp_col, MU, SD)

    out = {"posterior": str(path), "gdp_column": gdp_col, "n_chains": int(n_chain),
           "n_draws_per_chain": int(n_draw), "n_country_years": int(len(df))}

    # --- the estimand, chain by chain ---------------------------------------------------
    # The pre-specified thresholds of S10 are stated per parameter, but the paper does not
    # report a parameter: it reports a fixed linear functional of beta0 and theta. Strongly
    # correlated coefficients routinely mix badly one at a time while the functional of them
    # mixes well, so the functional is diagnosed here on its own terms -- split-R-hat, ESS and
    # the Monte Carlo standard error of the reported mean, computed on the derived quantity.
    phases = {}
    derived = {}
    for name, sub in [("contraction", df[df.dlp < 0]), ("expansion", df[df.dlp > 0])]:
        dbar = deriv_matrix(sub.x_s.values, ka, deg, nb).mean(axis=0)   # (nb,)
        e = (beta0 + theta @ dbar) / SD                                  # (chain, draw)
        derived["eps_" + name] = e
        per_chain = [float(e[c].mean()) for c in range(n_chain)]
        pooled_sd = float(e.reshape(-1).std())
        spread = float(max(per_chain) - min(per_chain))
        phases[name] = {
            "per_chain_mean": per_chain,
            "pooled_mean": float(e.mean()),
            "pooled_sd": pooled_sd,
            "chain_spread": spread,
            "chain_spread_in_posterior_sd": float(spread / pooled_sd) if pooled_sd else None,
            "per_chain_P_lt_1": [float((e[c] < 1).mean()) for c in range(n_chain)],
            "n_country_years": int(len(sub)),
        }

    ds = xr.Dataset({k: (("chain", "draw"), v) for k, v in derived.items()},
                    coords={"chain": np.arange(n_chain), "draw": np.arange(n_draw)})
    id_d = az.InferenceData(posterior=ds)
    rh_d, eb_d, et_d = az.rhat(id_d), az.ess(id_d, method="bulk"), az.ess(id_d, method="tail")
    mc_d = az.mcse(id_d, method="mean")
    for name in ("contraction", "expansion"):
        k = "eps_" + name
        sd_ = phases[name]["pooled_sd"]
        phases[name]["rhat"] = float(rh_d[k])
        phases[name]["ess_bulk"] = float(eb_d[k])
        phases[name]["ess_tail"] = float(et_d[k])
        phases[name]["mcse_mean"] = float(mc_d[k])
        phases[name]["mcse_in_posterior_sd"] = float(float(mc_d[k]) / sd_) if sd_ else None
        phases[name]["passes_prespecified"] = bool(
            float(rh_d[k]) < RHAT_MAX and float(eb_d[k]) > ESS_MIN and float(et_d[k]) > ESS_MIN)
    out["phases"] = phases
    out["estimand_passes_prespecified"] = bool(
        all(phases[n]["passes_prespecified"] for n in phases))

    # --- convergence of the estimand block against everything else ----------------------
    idata = az.InferenceData(posterior=post)
    blocks = {
        "elasticity (beta0, theta)": ["beta0", "theta"],
        "scales (sigma_theta, sigma, nu)": [v for v in ("sigma_theta", "sigma", "nu")
                                            if v in post],
        "time (tau0, tau_region, sigma_tau_region)":
            [v for v in ("tau0", "tau_region", "sigma_tau_region") if v in post],
        "age (delta_washare, delta_olddep)":
            [v for v in ("delta_washare", "delta_olddep") if v in post],
        "hierarchical scales (sigma_alpha*)":
            [v for v in post.data_vars if str(v).startswith(("sigma_alpha", "sigma_beta"))],
        "intercepts (alpha*, z_alpha*)":
            [v for v in post.data_vars if str(v).startswith(("alpha", "z_alpha"))],
        "ALL parameters": list(map(str, post.data_vars)),
    }
    block_stats = {}
    for name, vars_ in blocks.items():
        vars_ = [v for v in vars_ if v in post]
        if not vars_:
            continue
        rh = az.rhat(idata, var_names=vars_)
        eb = az.ess(idata, var_names=vars_, method="bulk")
        et = az.ess(idata, var_names=vars_, method="tail")
        mx = float(max(float(rh[v].max()) for v in vars_))
        mb = float(min(float(eb[v].min()) for v in vars_))
        mt = float(min(float(et[v].min()) for v in vars_))
        block_stats[name] = {
            "n_variables": len(vars_), "max_rhat": mx,
            "min_ess_bulk": mb, "min_ess_tail": mt,
            "passes_prespecified": bool(mx < RHAT_MAX and mb > ESS_MIN and mt > ESS_MIN),
        }
    out["blocks"] = block_stats

    # --- localise the failure: per-chain means of the worst parameters -------------------
    worst = []
    rh_all = az.rhat(idata)
    flat = []
    for v in rh_all.data_vars:
        arr = rh_all[v].values
        if arr.ndim == 0:
            flat.append((str(v), None, float(arr)))
        else:
            coord = rh_all[v].dims[0]
            labels = rh_all[v][coord].values
            for i, lab in enumerate(labels):
                flat.append((str(v), str(lab), float(arr[i])))
    flat.sort(key=lambda t: -t[2])
    for v, lab, r in flat[:5]:
        da = post[v] if lab is None else post[v].sel({post[v].dims[-1]: lab})
        vals = da.values
        means = [float(vals[c].mean()) for c in range(n_chain)]
        sds = [float(vals[c].std()) for c in range(n_chain)]
        gap = (max(means) - min(means)) / max(np.mean(sds), 1e-12)
        worst.append({
            "parameter": v if lab is None else "%s[%s]" % (v, lab),
            "rhat": r, "per_chain_mean": means, "per_chain_sd": sds,
            "between_chain_gap_in_within_chain_sd": float(gap),
            "reads_as": "mode-splitting" if gap > 2 else "slow mixing",
        })
    out["worst_parameters"] = worst

    # --- the degrees-of-freedom floor ---------------------------------------------------
    if "nu" in post:
        nu = post["nu"].values.reshape(-1)
        out["nu"] = {
            "mean": float(nu.mean()), "sd": float(nu.std()),
            "q2.5": float(np.percentile(nu, 2.5)), "q97.5": float(np.percentile(nu, 97.5)),
            "min": float(nu.min()),
            "floor_tolerance": NU_FLOOR_TOL,
            "fraction_at_floor": float((nu < NU_FLOOR + NU_FLOOR_TOL).mean()),
            "implied_residual_sd_multiplier":
                (float(np.sqrt(nu.mean() / (nu.mean() - 2.0)))
                 if nu.mean() > 2.0 else None),
        }
    if "sigma" in post:
        s_ = post["sigma"].values.reshape(-1)
        out["sigma"] = {"mean": float(s_.mean()), "sd": float(s_.std())}
    return out


def report(label, r):
    print("=" * 100)
    print("%s   (%s, %s)" % (label, r["gdp_column"], r["posterior"]))
    print("-" * 100)
    nc = r["n_chains"]
    print("  phase-averaged elasticity, chain by chain")
    print("    %-13s %s   %9s %9s %9s" % ("phase",
          " ".join("%9s" % ("chain %d" % c) for c in range(nc)),
          "pooled", "sd", "spread/sd"))
    for name, p in r["phases"].items():
        print("    %-13s %s   %9.4f %9.4f %8.3f" % (
            name, " ".join("%9.4f" % v for v in p["per_chain_mean"]),
            p["pooled_mean"], p["pooled_sd"],
            p["chain_spread_in_posterior_sd"] or float("nan")))
    print()
    print("  the REPORTED QUANTITY diagnosed on its own terms (R-hat < %.2f, ESS > %d)"
          % (RHAT_MAX, ESS_MIN))
    print("    %-13s %9s %10s %10s %11s %11s  %s"
          % ("phase", "R-hat", "ESS_bulk", "ESS_tail", "MCSE", "MCSE/sd", "verdict"))
    for name, p in r["phases"].items():
        print("    %-13s %9.4f %10.0f %10.0f %11.5f %10.1f%%  %s"
              % (name, p["rhat"], p["ess_bulk"], p["ess_tail"], p["mcse_mean"],
                 100 * p["mcse_in_posterior_sd"],
                 "PASS" if p["passes_prespecified"] else "FAIL"))
    print()
    print("  convergence by parameter block (pre-specified: R-hat < %.2f, ESS > %d)"
          % (RHAT_MAX, ESS_MIN))
    print("    %-38s %5s %9s %10s %10s  %s"
          % ("block", "vars", "max R-hat", "min ESS_b", "min ESS_t", "verdict"))
    for name, b in r["blocks"].items():
        print("    %-38s %5d %9.4f %10.0f %10.0f  %s"
              % (name, b["n_variables"], b["max_rhat"], b["min_ess_bulk"],
                 b["min_ess_tail"], "PASS" if b["passes_prespecified"] else "FAIL"))
    print()
    print("  worst parameters, per-chain means")
    for w in r["worst_parameters"]:
        print("    %-42s R-hat %6.4f  [%s]  gap=%.1f within-chain SD -> %s"
              % (w["parameter"], w["rhat"],
                 " ".join("%.3f" % m for m in w["per_chain_mean"]),
                 w["between_chain_gap_in_within_chain_sd"], w["reads_as"]))
    if "nu" in r:
        n = r["nu"]
        print()
        print("  Student-t degrees of freedom: mean %.4f [%.4f, %.4f], min %.4f, "
              "%.1f%% of draws within %.2f of the nu>=2 floor"
              % (n["mean"], n["q2.5"], n["q97.5"], n["min"],
                 100 * n["fraction_at_floor"], NU_FLOOR_TOL))
    print()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", action="append", required=True,
                    metavar="LABEL:PATH:GDP_COLUMN")
    ap.add_argument("--out", default=str(DIR_RESULTS / "chainwise_check.json"))
    args = ap.parse_args(argv)

    MU, SD, ka, deg, nb = load_basis()
    res, missing = {}, []
    for spec in args.posterior:
        label, path, col = spec.split(":", 2)
        p = path if path.startswith("/") else str(REPO / path)
        try:
            res[label] = analyse(label, p, col, MU, SD, ka, deg, nb)
        except FileNotFoundError:
            missing.append((label, p))
            continue
        report(label, res[label])

    payload = {"thresholds": {"rhat_max": RHAT_MAX, "ess_min": ESS_MIN},
               "specifications": res,
               "not_found": [{"label": l, "path": p} for l, p in missing]}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    for l, p in missing:
        print("  (not found, skipped: %s -> %s)" % (l, p))
    print("  wrote %s" % args.out)
    return payload


if __name__ == "__main__":
    main()
