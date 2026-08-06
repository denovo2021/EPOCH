"""Render Supplementary diagnostic figures (S1 trace, S2 posterior-predictive
check, S3 energy) from the deployed Stage-3 hierarchical GDP posterior.

Run from the repository root:
    python src/make_si_diagnostics.py

Outputs (PNG + PDF) are written to figures/:
    FigureS1_trace.png    - trace + marginal posteriors for key parameters
    FigureS2_ppc.png      - posterior-predictive check vs observed log10 GDP
    FigureS3_energy.png    - NUTS energy-transition diagnostic
"""
from pathlib import Path
import json, os, warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import sys
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats
from scipy.interpolate import BSpline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
# Which posterior to diagnose. Defaults to the deployed alias, as before, so existing
# invocations are unchanged; pass a path to diagnose a specific fit without disturbing the
# alias, which elasticity_inflection.py and scenario_projection.py also read.
#
#   $env:GDP_COL="GDP_constant_2015usd"
#   python src/make_si_diagnostics.py results/trace_adaptive_real.nc
#
# GDP_COL must name the same column fit_gdp_production.py was run with. Figure S2 is a
# posterior-predictive check: it compares draws from this posterior against the observed
# outcome, so a real-GDP posterior checked against the nominal column compares two different
# series and the check is meaningless. The dropna subset and the >0 filter below are copied
# from fit_gdp_production.py for the same reason -- the country and region codes have to be
# the ones the sampler saw.
POSTERIOR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "hierarchical_model_rcs_v2.nc"
if not POSTERIOR.is_absolute():
    POSTERIOR = (ROOT / POSTERIOR).resolve()
GDP_COL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("GDP_COL", "GDP")
SCALE_CACHE = ROOT / "results" / "cache" / "scale_rcs_v2.json"
DATA_CSV = ROOT / "data" / "merged_age.csv"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)
SEED = 42


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


def main():
    print(f"Loading {POSTERIOR} ...")
    print(f"  outcome column for the posterior-predictive check: {GDP_COL}")
    idata = az.from_netcdf(POSTERIOR)
    post = idata.posterior

    # --- S1: trace plots for key parameters ---
    print("Rendering FigureS1_trace ...")
    az.plot_trace(
        idata,
        var_names=[
            "beta0", "theta", "sigma_theta",
            "sigma_alpha_region", "sigma_tau_region", "sigma_alpha_country",
            "sigma", "nu",
        ],
        coords={"Spline": ["0", "5", "11"]},
        compact=False,
    )
    fig = plt.gcf()
    fig.suptitle("Supplementary Figure S1 — Trace plots and marginal posteriors",
                 fontsize=12, y=1.0)
    fig.subplots_adjust(top=0.975, hspace=0.55)
    save(fig, "FigureS1_trace")

    # --- S2: posterior-predictive check (reconstructed from raw data) ---
    print("Rendering FigureS2_ppc ...")
    rng = np.random.default_rng(SEED)
    beta0 = post["beta0"].values.reshape(-1)
    theta = post["theta"].values.reshape(-1, post.sizes["Spline"])
    dWA = post["delta_washare"].values.reshape(-1)
    dOD = post["delta_olddep"].values.reshape(-1)
    alpha_c = post["alpha_country"].values.reshape(-1, post.sizes["Country"])
    tau_r = post["tau_region"].values.reshape(-1, post.sizes["Region"])
    sigma = post["sigma"].values.reshape(-1)
    nu = post["nu"].values.reshape(-1)

    sc = json.load(open(SCALE_CACHE))
    MU, SD = sc["MU_GLOBAL"], sc["SD_GLOBAL"]
    ka = np.array(sc["bspline_knots_aug"]); deg = sc["bspline_degree"]
    nb = sc["bspline_n_basis"]; cm = np.array(sc["bspline_col_means"])
    s_dWA, s_dOD, s_dt = sc["s_dWA"], sc["s_dOD"], sc["s_dt"]
    coef_dt = np.array(sc["coef_dt_proj"]); dt_clip = sc["dt_clip"]

    def bspline_centered(xv):
        B = np.zeros((len(xv), nb))
        for i in range(nb):
            c = np.zeros(nb); c[i] = 1.0
            B[:, i] = BSpline(ka, c, deg, extrapolate=False)(np.asarray(xv, float))
        return np.nan_to_num(B) - cm

    df = pd.read_csv(DATA_CSV)
    if GDP_COL not in df.columns:
        raise SystemExit("%s has no column %r; available GDP columns: %s"
                         % (DATA_CSV.name, GDP_COL,
                            [c for c in df.columns if "GDP" in c]))
    df = df.dropna(subset=["ISO3", "Country Name", "Region", "Year",
                            GDP_COL, "Population", "WAshare", "OldDep"]).copy()
    df = df[df[GDP_COL] > 0]
    # A country the sampler never saw has no alpha_country to index, and pandas .map would
    # hand a NaN to .astype(int) further down, which numpy turns into a silent garbage index
    # rather than an error. Drop such rows loudly instead. With the right GDP_COL there are
    # none, so a non-zero count here means GDP_COL does not match how the posterior was fitted.
    countries_post = [str(c) for c in post["Country"].values]
    regions_post = [str(r) for r in post["Region"].values]
    known = df["Country Name"].astype(str).isin(countries_post) & \
        df["Region"].astype(str).isin(regions_post)
    if not known.all():
        missing = sorted(set(df.loc[~known, "Country Name"].astype(str)))
        print("  WARNING: %d of %d rows are from %d unit(s) absent from this posterior "
              "(%s%s); dropping them. Check that GDP_COL matches the fit."
              % ((~known).sum(), len(df), len(missing), ", ".join(missing[:6]),
                 " ..." if len(missing) > 6 else ""))
        df = df[known].copy()
    print("  posterior-predictive panel: n=%d, countries=%d, regions=%d"
          % (len(df), df["Country Name"].nunique(), df["Region"].nunique()))
    df["Year"] = df["Year"].astype(int)
    df["x_s"] = (np.log10(df["Population"]) - MU) / SD
    obs = np.log10(df[GDP_COL].values)
    B_mat = bspline_centered(df["x_s"].values)
    base = (df.sort_values(["ISO3", "Year"]).drop_duplicates("ISO3", keep="last")
            [["ISO3", "WAshare", "OldDep", "Year"]].rename(
                columns={"WAshare": "WAb", "OldDep": "ODb", "Year": "Yb"}))
    df = df.merge(base, on="ISO3", how="left")
    dwa = ((df["WAshare"] - df["WAb"]) / (s_dWA + 1e-8)).values
    dod = ((df["OldDep"] - df["ODb"]) / (s_dOD + 1e-8)).values
    dt_raw = ((df["Year"] - df["Yb"]) / 10.0) / (s_dt + 1e-8)
    Xdt = np.column_stack([np.ones(len(df)), df["x_s"].values, B_mat])
    dt = np.clip(dt_raw.values - Xdt @ coef_dt, -dt_clip, dt_clip)

    df["rc"] = df["Region"].astype("category").cat.codes
    df["cc"] = df["Country Name"].astype("category").cat.codes
    cat_regions = list(df["Region"].astype("category").cat.categories)
    cat_countries = list(df["Country Name"].astype("category").cat.categories)
    ri_map = {r: regions_post.index(r) for r in cat_regions if r in regions_post}
    ci_map = {c: countries_post.index(c) for c in cat_countries if c in countries_post}
    ri = df["Region"].map(ri_map).values.astype(int)
    ci = df["Country Name"].map(ci_map).values.astype(int)
    x = df["x_s"].values

    n_draws = beta0.shape[0]
    n_pp = 150
    sel = rng.choice(n_draws, size=n_pp, replace=False)

    mu = (
        alpha_c[sel][:, ci]
        + beta0[sel][:, None] * x[None, :]
        + theta[sel] @ B_mat.T
        + dWA[sel][:, None] * dwa[None, :]
        + dOD[sel][:, None] * dod[None, :]
        + tau_r[sel][:, ri] * dt[None, :]
    )
    # A posterior-predictive check is only a check if the draws and the observations are on the
    # same series. Fitted on real GDP and compared against the nominal column (or the reverse),
    # the two densities are offset by the deflator and the figure is meaningless -- but it still
    # renders, and it renders plausibly enough to be believed. Refuse instead.
    # Threshold: on a correct pairing the country intercepts absorb the level, so the gap is a
    # few thousandths of a dex. Swapping the two columns of merged_age.csv moves the median by
    # 0.322 dex (10.241 real against 9.919 nominal). 0.15 sits between the two with room on
    # both sides, and a real fit whose median prediction misses the observed median by more
    # than 0.15 dex is misspecified badly enough that stopping is the right response anyway.
    # The gap is printed either way, so a borderline value is visible rather than silent.
    MAX_GAP = 0.15
    gap = float(np.median(mu) - np.median(obs))
    print("  median predicted %.3f vs median observed %.3f (log10, gap %+.3f)"
          % (np.median(mu), np.median(obs), gap))
    if abs(gap) > MAX_GAP:
        raise SystemExit(
            "ABORT: the posterior predicts a median log10 outcome of %.3f while the observed\n"
            "       median of %r is %.3f, a gap of %+.3f. This posterior was not fitted on this\n"
            "       column. Pass the matching series, e.g.\n"
            "         $env:GDP_COL=\"GDP_constant_2015usd\"; python %s <real posterior>\n"
            "       (merged_age.csv GDP columns: %s)"
            % (np.median(mu), GDP_COL, np.median(obs), gap, Path(__file__).name,
               ", ".join(c for c in df.columns if "GDP" in c)))

    yrep = stats.t.rvs(df=nu[sel][:, None], loc=mu, scale=sigma[sel][:, None],
                       random_state=SEED)

    grid = np.linspace(obs.min() - 0.5, obs.max() + 0.5, 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(n_pp):
        ax.plot(grid, stats.gaussian_kde(yrep[i])(grid), color="#1f77b4",
                alpha=0.08, lw=0.7)
    ax.plot([], [], color="#1f77b4", alpha=0.6, lw=1.2,
            label=f"Posterior predictive ({n_pp} draws)")
    ax.plot(grid, stats.gaussian_kde(obs)(grid), color="black", lw=2.5,
            label="Observed")
    ax.set_xlabel(r"$\log_{10}$ GDP")
    ax.set_ylabel("Density")
    ax.set_title("Supplementary Figure S2 — Posterior-predictive check")
    ax.legend(frameon=False)
    save(fig, "FigureS2_ppc")

    # --- S3: NUTS energy diagnostic ---
    print("Rendering FigureS3_energy ...")
    ax = az.plot_energy(idata)
    fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
    fig.suptitle("Supplementary Figure S3 — NUTS energy-transition diagnostic",
                 fontsize=12, y=1.02)
    save(fig, "FigureS3_energy")

    # Record what these three figures were drawn from. Fig. S1-S3 once shipped diagnosing the
    # superseded nominal fit and nothing in the package could tell: a PNG carries no provenance.
    manifest = {
        "posterior": str(POSTERIOR),
        "posterior_name": POSTERIOR.name,
        "gdp_column": GDP_COL,
        "n_country_years": int(len(df)),
        "n_countries": int(df["Country Name"].nunique()),
        "median_predicted_minus_observed": gap,
        "figures": ["FigureS1_trace", "FigureS2_ppc", "FigureS3_energy"],
    }
    out = ROOT / "results" / "si_diagnostics_manifest.json"
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  wrote {out}")
    print("Done.")


if __name__ == "__main__":
    main()
