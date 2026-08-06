"""
s13_levels_elasticity.py -- phase-averaged local elasticity from any levels posterior
====================================================================================
The levels model's elasticity is the local slope of the fitted spline,

    eps(x_s) = (1/SD) * ( beta0 + sum_j theta_j B_j'(x_s) )

evaluated at the observed population domain and averaged separately over the country-years in
which population is falling and rising. This script computes it for an arbitrary list of
posteriors and writes one JSON holding all of them, so the manuscript can quote a single file
and a re-fit is a one-command swap.

WHY THIS FILE EXISTS
--------------------
The deployed levels model was fitted on `merged_age.csv`'s `GDP` column, which is NOMINAL
current US dollars, while the manuscript described the outcome as real GDP in constant 2015
US dollars. Refitted on the real series, the contraction-phase elasticity falls from 1.220 to
0.806 under adaptive shrinkage and from 0.920 to 0.657 under the fixed penalty, and the
posterior probability of being sub-unitary goes to 1.00 under both. `elasticity_inflection.py`
could only ever describe one posterior at a time and hard-coded the nominal column; this
replaces it for reporting purposes.

Usage
-----
    python src/s13_levels_elasticity.py \\
        --posterior real_fixed:results/trace_fixed005_real.nc:GDP_constant_2015usd \\
        --posterior real_adaptive:results/trace_adaptive_real.nc:GDP_constant_2015usd \\
        --posterior nominal_adaptive:results/trace_adaptive.nc:GDP

Each --posterior is `label:path:gdp_column`. The GDP column only selects which country-years
enter the phase split and the domain; it does not change the posterior.

Output: results/levels_elasticity.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter

from config import PATH_SCALE_V2, PATH_MERGED_AGE, DIR_RESULTS, REPO

CORE_PCTL = (5, 95)


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
    df = pd.read_csv(PATH_MERGED_AGE).dropna(subset=["ISO3", "Population", gdp_col]).copy()
    df = df[(df[gdp_col] > 0) & (df["Population"] > 0)]
    df["Year"] = df["Year"].astype(int)
    df["lp"] = np.log10(df["Population"])
    df = df.sort_values(["ISO3", "Year"])
    df["dlp"] = df.groupby("ISO3")["lp"].diff()
    df = df.dropna(subset=["dlp"])
    df["x_s"] = (df["lp"] - MU) / SD
    return df


def analyse(path, gdp_col, MU, SD, ka, deg, nb):
    post = open_posterior(path)
    theta = post["theta"].stack(z=("chain", "draw")).values
    if theta.shape[0] != nb:
        theta = theta.T
    theta = theta.T                                       # (S, nb)
    beta0 = post["beta0"].stack(z=("chain", "draw")).values
    df = phase_split(gdp_col, MU, SD)

    out = {"posterior": str(path), "gdp_column": gdp_col,
           "n_country_years": int(len(df)), "n_countries": int(df.ISO3.nunique()),
           "n_draws": int(beta0.size)}

    for name, sub in [("contraction", df[df.dlp < 0]), ("expansion", df[df.dlp > 0])]:
        dbar = deriv_matrix(sub.x_s.values, ka, deg, nb).mean(axis=0)
        e = (beta0 + theta @ dbar) / SD
        out[name] = {"mean": float(e.mean()), "sd": float(e.std()),
                     "q2.5": float(np.percentile(e, 2.5)),
                     "q97.5": float(np.percentile(e, 97.5)),
                     "P_lt_1": float((e < 1).mean()), "n_country_years": int(len(sub))}

    xs = np.linspace(*np.percentile(df.x_s, CORE_PCTL), 400)
    curve = (beta0[None, :] + deriv_matrix(xs, ka, deg, nb) @ theta.T) / SD
    med = np.median(curve, axis=1)
    out["core"] = {"mean": float(med.mean()), "min": float(med.min()), "max": float(med.max()),
                   "crosses_unity": bool(med.min() < 1 < med.max())}
    out["curve"] = {"population": (10 ** (xs * SD + MU)).tolist(),
                    "elasticity_median": med.tolist(),
                    "elasticity_lo": np.percentile(curve, 2.5, axis=1).tolist(),
                    "elasticity_hi": np.percentile(curve, 97.5, axis=1).tolist()}

    sm = savgol_filter(med, 151, 2)
    cross = np.where(np.diff(np.sign(sm - 1.0)) != 0)[0]
    if len(cross):
        xstar = min([xs[k] - (sm[k] - 1) * (xs[k + 1] - xs[k]) / ((sm[k + 1] - 1) - (sm[k] - 1))
                     for k in cross], key=abs)
        out["unity_crossing_population"] = float(10 ** (xstar * SD + MU))
    else:
        out["unity_crossing_population"] = None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", action="append", required=True,
                    metavar="LABEL:PATH:GDP_COLUMN")
    ap.add_argument("--out", default=str(DIR_RESULTS / "levels_elasticity.json"))
    args = ap.parse_args(argv)

    MU, SD, ka, deg, nb = load_basis()
    res, missing = {}, []
    for spec in args.posterior:
        label, path, col = spec.split(":", 2)
        p = path if path.startswith("/") else str(REPO / path)
        try:
            res[label] = analyse(p, col, MU, SD, ka, deg, nb)
        except FileNotFoundError:
            missing.append((label, p))
            continue

    payload = {"basis": {"MU_GLOBAL": MU, "SD_GLOBAL": SD, "n_basis": nb, "degree": deg},
               "specifications": res,
               "not_found": [{"label": l, "path": p} for l, p in missing]}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("%-22s %-11s %8s  %-26s %-26s %s"
          % ("specification", "GDP series", "n", "contraction", "expansion", "core (min-max)"))
    print("-" * 122)
    for label, r in res.items():
        c, e = r["contraction"], r["expansion"]
        print("%-22s %-11s %8d  %.3f [%.3f,%.3f] P<1=%.2f  %.3f [%.3f,%.3f] P<1=%.2f  %.2f (%.2f-%.2f)"
              % (label, "real" if "constant" in r["gdp_column"] else "nominal",
                 r["n_country_years"], c["mean"], c["q2.5"], c["q97.5"], c["P_lt_1"],
                 e["mean"], e["q2.5"], e["q97.5"], e["P_lt_1"],
                 r["core"]["mean"], r["core"]["min"], r["core"]["max"]))
    for l, p in missing:
        print("  (not found, skipped: %s -> %s)" % (l, p))
    print("\n  wrote %s" % args.out)
    return payload


if __name__ == "__main__":
    main()
