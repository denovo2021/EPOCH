"""
fit_gdp_production.py
=====================
Refit the deployed hierarchical B-spline GDP model with STRONGER spline
regularization, to remove the artifactual wiggle in the first/second derivatives
of the population-output association (Figure 1, elasticity, inflection x*).

Difference from the original Stage-3 model (m03_hierarchical_model_rcs_v2.py):
the P-spline smoothing is tightened — the GaussianRandomWalk innovation scale prior
is changed from Exponential(5) (mean 0.2) to Exponential(50) (mean 0.02) and the
random-walk init scale from Normal(0,0.2) to Normal(0,0.05). This is the B-spline
analogue of setting the spline coefficient SD to 0.05.

The deterministic B-spline basis and all transforms are loaded from the persisted
scale_rcs_v2.json so that theta is directly comparable to elasticity_inflection.py.

Production (run on a machine with the project's environment):
    uv run python fit_gdp_production.py <merged_age.csv> <out.nc> <seed> <tune> <draws> <chains> <target_accept>
    e.g.  PRIOR=fixed005 uv run python fit_gdp_production.py data/merged_age.csv \
              results/trace_fixed005.nc 42 2000 3000 4 0.95
          PRIOR=adaptive SAVE_FULL=1 uv run python fit_gdp_production.py data/merged_age.csv \
              results/trace_adaptive.nc 42 2000 3000 4 0.95

Pooling-friendly — call several times with different seeds and small
tune/draws, then combine the traces (see combine_smoothed.py).
"""
import sys, os, json
import numpy as np, pandas as pd, pymc as pm
from scipy.interpolate import BSpline
from config import PATH_MERGED_AGE, PATH_SCALE_V2

DATA = sys.argv[1] if len(sys.argv) > 1 else str(PATH_MERGED_AGE)
OUT = sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
TUNE = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
DRAWS = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
CHAINS = int(sys.argv[6]) if len(sys.argv) > 6 else 4
TA = float(sys.argv[7]) if len(sys.argv) > 7 else 0.95
MTD = int(os.environ.get("MTD", "10"))

# strong-smoothing hyperparameters (override via env if desired)

s = json.load(open(PATH_SCALE_V2))
MU, SD = s["MU_GLOBAL"], s["SD_GLOBAL"]
ka = np.array(s["bspline_knots_aug"]); deg = s["bspline_degree"]; nb = s["bspline_n_basis"]
cm = np.array(s["bspline_col_means"]); s_dWA = s["s_dWA"]; s_dOD = s["s_dOD"]; s_dt = s["s_dt"]
coef_dt = np.array(s["coef_dt_proj"]); dt_clip = s["dt_clip"]

def bspline_centered(x):
    B = np.zeros((np.size(x), nb))
    for i in range(nb):
        c = np.zeros(nb); c[i] = 1.0
        B[:, i] = BSpline(ka, c, deg, extrapolate=False)(np.asarray(x, float))
    return np.nan_to_num(B) - cm

# Which GDP series the levels model is fitted on. The deployed run used "GDP", which in
# merged_age.csv is NOMINAL current US dollars, while the manuscript described the outcome as
# real GDP in constant 2015 US dollars. Set GDP_COL=GDP_constant_2015usd for the real-GDP fit.
GDP_COL = os.environ.get("GDP_COL", "GDP")
df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO3", "Country Name", "Region", "Year", GDP_COL, "Population",
                       "WAshare", "OldDep"]).copy()
df = df[df[GDP_COL] > 0]
# Optional country exclusions, for the leave-one-country-out sensitivity refits. The evaluation
# domain used by s13_levels_elasticity.py is deliberately NOT changed to match: the comparison
# asks whether one country's data drive the estimate, so the estimand has to stay the same and
# only the fitted posterior may move.
#   $env:EXCLUDE_ISO3 = "GNQ"
EXCLUDE = [c.strip().upper() for c in os.environ.get("EXCLUDE_ISO3", "").split(",") if c.strip()]
if EXCLUDE:
    n0, c0 = len(df), df["Country Name"].nunique()
    hit = sorted(df.loc[df["ISO3"].str.upper().isin(EXCLUDE), "Country Name"].unique())
    missed = [c for c in EXCLUDE if c not in set(df["ISO3"].str.upper())]
    if missed:
        raise SystemExit("EXCLUDE_ISO3 names %s, absent from the fitted panel" % ", ".join(missed))
    df = df[~df["ISO3"].str.upper().isin(EXCLUDE)].copy()
    print("excluded %s (%s): %d country-years and %d countries dropped; n %d -> %d"
          % (",".join(EXCLUDE), "; ".join(hit), n0 - len(df),
             c0 - df["Country Name"].nunique(), n0, len(df)))
df["Year"] = df["Year"].astype(int)
df["x_s"] = (np.log10(df["Population"]) - MU) / SD
df["y"] = np.log10(df[GDP_COL])
B = bspline_centered(df["x_s"].values)
# age deltas vs country base year (last observed)
base = (df.sort_values(["ISO3", "Year"]).drop_duplicates("ISO3", keep="last")
        [["ISO3", "WAshare", "OldDep", "Year"]].rename(columns={"WAshare": "WAb", "OldDep": "ODb", "Year": "Yb"}))
df = df.merge(base, on="ISO3", how="left")
dWA = ((df["WAshare"] - df["WAb"]) / (s_dWA + 1e-8)).values
dOD = ((df["OldDep"] - df["ODb"]) / (s_dOD + 1e-8)).values
dt = (((df["Year"] - 2000) / 10.0 - (df["Yb"] - 2000) / 10.0) / (s_dt + 1e-8)).values
Xdt = np.column_stack([np.ones(len(df)), df["x_s"].values, B])
dt_orth = np.clip(dt - Xdt @ coef_dt, -dt_clip, dt_clip)

df["rc"] = df["Region"].astype("category").cat.codes
df["cc"] = df["Country Name"].astype("category").cat.codes
regions = df["Region"].astype("category").cat.categories
countries = df["Country Name"].astype("category").cat.categories
ri = df["rc"].values.astype(int); ci = df["cc"].values.astype(int); y = df["y"].values
c2r = df[["cc", "rc"]].drop_duplicates().sort_values("cc").set_index("cc")["rc"].values.astype(int)
coords = {"Region": regions.tolist(), "Country": countries.tolist(), "Spline": [str(i) for i in range(nb)]}
print("fit: n=%d countries=%d regions=%d basis=%d | PRIOR=%s | GDP_COL=%s" % (
    len(df), len(countries), len(regions), nb, os.environ.get("PRIOR","adaptive"), GDP_COL))

with pm.Model(coords=coords) as m:
    # Intercept centred on the data mean; slope centred on the approximate
    # population-output elasticity.  These are weakly informative fixed priors,
    # not empirical hyperpriors from a Stage-2 regional fit (SI S6.1 should
    # document them as such).
    alpha0 = pm.Normal("alpha0", float(np.mean(y)), 1.0)
    beta0 = pm.Normal("beta0", 0.9, 0.5)
    # Spline prior selectable via env PRIOR:
    #   "adaptive"  -> data-driven ridge shrinkage (HalfNormal hyperprior on coef scale)
    #   "fixed005"  -> strong fixed first-difference (P-spline) penalty, RW innovation SD=0.05
    PRIOR = os.environ.get("PRIOR", "adaptive")
    if PRIOR == "fixed005":
        theta = pm.GaussianRandomWalk("theta", sigma=0.05,
                                      init_dist=pm.Normal.dist(0.0, 0.05), dims="Spline")
    else:
        sigma_theta = pm.HalfNormal("sigma_theta", sigma=0.2)
        theta = pm.Normal("theta", mu=0.0, sigma=sigma_theta, dims="Spline")
    delta_washare = pm.Normal("delta_washare", 0.0, 0.3)
    delta_olddep = pm.Normal("delta_olddep", 0.0, 0.3)
    sar = pm.Exponential("sigma_alpha_region", 2.0); zar = pm.Normal("z_alpha_region", 0, 1, dims="Region")
    alpha_r = pm.Deterministic("alpha_region", alpha0 + sar * zar, dims="Region")
    tau0 = pm.Normal("tau0", 0, 0.12); stra = pm.Exponential("sigma_tau_region", 10.0); ztr = pm.Normal("z_tau_region", 0, 1, dims="Region")
    tau_r = pm.Deterministic("tau_region", tau0 + stra * ztr, dims="Region")
    sac = pm.Exponential("sigma_alpha_country", 10.0); zac = pm.Normal("z_alpha_country", 0, 1, dims="Country")
    alpha_c = pm.Deterministic("alpha_country", alpha_r[c2r] + sac * zac, dims="Country")
    sigma = pm.HalfNormal("sigma", 0.15); nu = pm.Deterministic("nu", pm.Gamma("nu_minus2", 2.0, 0.1) + 2.0)
    mu = (alpha_c[ci] + beta0 * df["x_s"].values + pm.math.dot(B, theta)
          + delta_washare * dWA + delta_olddep * dOD + tau_r[ri] * dt_orth)
    pm.StudentT("Log_GDP_obs", nu=nu, mu=mu, sigma=sigma, observed=y)
    SAVE_FULL = os.environ.get("SAVE_FULL", "0") == "1"
    idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS, target_accept=TA,
                      nuts_sampler="nutpie", nuts_sampler_kwargs={"maxdepth": MTD},
                      random_seed=SEED, progressbar=False,
                      idata_kwargs={"log_likelihood": SAVE_FULL})
    if SAVE_FULL:
        # nutpie does not auto-populate log_likelihood; compute it explicitly for LOO.
        idata = pm.compute_log_likelihood(idata, model=m)
def _sanitize_attrs(attrs):
    for key, val in list(attrs.items()):
        if isinstance(val, (dict, bool)) or val is None:
            attrs[key] = str(val)
for group in idata.groups():
    _sanitize_attrs(getattr(idata, group).attrs)
if hasattr(idata, "_attrs"):
    _sanitize_attrs(idata._attrs)
if SAVE_FULL:
    # Full InferenceData incl. log_likelihood (for LOO/az.compare); load with az.from_netcdf.
    idata.to_netcdf(OUT)
else:
    # Flat posterior dataset for the downstream pipeline (load with xr.open_dataset).
    idata.posterior.to_netcdf(OUT)
print("saved", OUT, "PRIOR=%s SAVE_FULL=%s chains=%d draws=%d tune=%d" % (
    os.environ.get("PRIOR","adaptive"), SAVE_FULL, CHAINS, DRAWS, TUNE))
