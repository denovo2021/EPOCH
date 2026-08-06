"""
fit_hierarchical_workingage.py
==============================
EPOCH3 core model — Bayesian hierarchical LINEAR model for country-specific long-run
WORKING-AGE-population -> GDP elasticities on 10-year non-overlapping decade blocks.
Refactor of fit_hierarchical_10yr.py with three NHB upgrades:

  (1) Working-age predictor   : x = Delta10 ln(Population * WAshare)   [15-64 yrs]
  (2) Growth/decline asymmetry: decline = 1[ Delta10 ln(WApop) < 0 ]
                                slope_growth  = beta_country
                                slope_decline = beta_country + delta_region
  (3) Decade fixed effects    : sum-to-zero period intercepts tau_period[block]
                                (1970s oil shocks ... 2020 COVID) absorb global shocks.

Likelihood (per country i, decade block t):
  Delta10 ln_gdp = alpha_country[i] + tau_period[t]
                   + ( beta_country[i] + delta_region[r(i)] * decline ) * x  + eps

Hierarchies (non-centered, global -> region -> country):
  beta  : 3-level   (country-specific growth elasticity)
  alpha : 3-level   (country intercept)
  delta : 2-level   (global -> region; decline-regime increment, pooled above country)

Usage:
    python fit_hierarchical_workingage.py [draws] [tune] [chains] [target_accept]

Outputs (results/):
    hierarchical_wa_country_elasticities.csv   ISO3,Region,n,beta_growth,beta_decline,HDIs
    hierarchical_wa_region_elasticities.csv    region-level growth & decline elasticities
    hierarchical_wa_period_effects.csv         decade fixed effects (posterior mean + HDI)
    hierarchical_wa_posterior.nc               flat posterior for downstream projection
"""
import os, sys
import numpy as np, pandas as pd, pymc as pm, arviz as az, xarray as xr
from config import PATH_MERGED_AGE, PATH_AGE_SCEN, DIR_RESULTS

K = 10                                          # 10-year non-overlapping decade blocks
GDP_COL = "GDP_constant_2015usd"
BLOCK_END = [1970, 1980, 1990, 2000, 2010, 2020]
PERIOD_LABELS = ["1961-1970", "1971-1980", "1981-1990",
                 "1991-2000", "2001-2010", "2011-2020"]

DRAWS  = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
TUNE   = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
CHAINS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
TA     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
SEED   = 42

BETA_GLOBAL_SD  = float(os.environ.get("BETA_GLOBAL_SD", "0.5"))
DELTA_GLOBAL_SD = float(os.environ.get("DELTA_GLOBAL_SD", "0.5"))
VAR_PRIOR = os.environ.get("VAR_PRIOR", "halfnormal")   # halfnormal | halfcauchy | exponential
OUT_TAG   = os.environ.get("OUT_TAG", "")


def _scale(name):
    if VAR_PRIOR == "halfcauchy":
        return pm.HalfCauchy(name, beta=1.0)
    if VAR_PRIOR == "exponential":
        return pm.Exponential(name, 1.0)
    return pm.HalfNormal(name, 0.5)


# ---------------------------------------------------------------- data prep ----
df = pd.read_csv(PATH_MERGED_AGE)
gdp_col = GDP_COL if (GDP_COL in df.columns and df[GDP_COL].notna().sum()) else "GDP"

# (1) Working-age share: coalesce panel WAshare (1960-2017) with the age-scenario
#     file (Estimates -> Medium variant) so the 2010->2020 COVID block is covered.
age = pd.read_csv(PATH_AGE_SCEN)
age["_pri"] = np.where(age["scenario_norm"].str.contains("estimate", case=False, na=False), 0, 1)
age = (age.sort_values(["ISO3", "Year", "_pri"])
          .drop_duplicates(["ISO3", "Year"], keep="first")[["ISO3", "Year", "WAshare"]]
          .rename(columns={"WAshare": "WAshare_scn"}))
df = df.merge(age, on=["ISO3", "Year"], how="left")
df["WAshare"] = df["WAshare"].fillna(df["WAshare_scn"])

df = df.dropna(subset=[gdp_col, "Population", "WAshare", "ISO3", "Year", "Region"]).copy()
df = df[(df[gdp_col] > 0) & (df["Population"] > 0) & (df["WAshare"] > 0)]
df["Year"] = df["Year"].astype(int)
df["wapop"]  = df["Population"].astype(float) * df["WAshare"].astype(float)   # 15-64 pop
df["ln_gdp"] = np.log(df[gdp_col].astype(float))
df["ln_wapop"] = np.log(df["wapop"])

# 10-year long differences (non-overlapping decade blocks)
base = df[["ISO3", "Year", "Region", "ln_gdp", "ln_wapop"]].copy()
b = base[["ISO3", "Year", "ln_gdp", "ln_wapop"]].copy(); b["Year"] = b["Year"] + K
m = base.merge(b, on=["ISO3", "Year"], suffixes=("", "_lag"))
m["dln_gdp"]   = m["ln_gdp"]   - m["ln_gdp_lag"]
m["dln_wapop"] = m["ln_wapop"] - m["ln_wapop_lag"]
d = m[["ISO3", "Year", "Region", "dln_gdp", "dln_wapop"]].dropna()
d = d[np.isfinite(d["dln_gdp"]) & np.isfinite(d["dln_wapop"])]
d = d[d["Year"].isin(BLOCK_END)].reset_index(drop=True)
for c in ["dln_gdp", "dln_wapop"]:
    lo, hi = np.percentile(d[c], [1, 99]); d[c] = d[c].clip(lo, hi)

# (2) decline regime dummy on working-age-population growth
d["decline"] = (d["dln_wapop"] < 0).astype(float)

# indices
d["rc"] = d["Region"].astype("category").cat.codes
d["cc"] = d["ISO3"].astype("category").cat.codes
regions   = list(d["Region"].astype("category").cat.categories)
countries = list(d["ISO3"].astype("category").cat.categories)
ci = d["cc"].values.astype(int)
ri = d["rc"].values.astype(int)
c2r = d[["cc", "rc"]].drop_duplicates().sort_values("cc")["rc"].values.astype(int)
pi = d["Year"].map({y: i for i, y in enumerate(BLOCK_END)}).values.astype(int)  # (3) period idx
x   = d["dln_wapop"].values
dec = d["decline"].values
y   = d["dln_gdp"].values
n_obs_country = d.groupby("cc").size().reindex(range(len(countries))).fillna(0).astype(int).values
print("WA panel: n=%d, countries=%d, regions=%d, decline blocks=%d (%.1f%%)"
      % (len(d), len(countries), len(regions), int(dec.sum()), 100 * dec.mean()))

# ----------------------------------------------------------------- model ----
coords = {"region": [str(r) for r in regions], "country": [str(c) for c in countries], "period": PERIOD_LABELS}
with pm.Model(coords=coords) as model:
    # ---- slope hierarchy: growth-regime elasticity (global -> region -> country) ----
    beta_global = pm.Normal("beta_global", 1.0, BETA_GLOBAL_SD)
    s_beta_r = _scale("sigma_beta_region")
    z_beta_r = pm.Normal("z_beta_region", 0, 1, dims="region")
    beta_region = pm.Deterministic("beta_region", beta_global + s_beta_r * z_beta_r, dims="region")
    s_beta_c = _scale("sigma_beta_country")
    z_beta_c = pm.Normal("z_beta_country", 0, 1, dims="country")
    beta_country = pm.Deterministic("beta_country", beta_region[c2r] + s_beta_c * z_beta_c, dims="country")

    # ---- (2) asymmetry increment: decline regime, pooled global -> region only ----
    delta_global = pm.Normal("delta_global", 0.0, DELTA_GLOBAL_SD)
    s_delta_r = pm.HalfNormal("sigma_delta_region", 0.3)
    z_delta_r = pm.Normal("z_delta_region", 0, 1, dims="region")
    delta_region = pm.Deterministic("delta_region", delta_global + s_delta_r * z_delta_r, dims="region")

    # ---- intercept hierarchy ----
    alpha_global = pm.Normal("alpha_global", 0.0, 1.0)
    s_a_r = _scale("sigma_alpha_region")
    z_a_r = pm.Normal("z_alpha_region", 0, 1, dims="region")
    alpha_region = pm.Deterministic("alpha_region", alpha_global + s_a_r * z_a_r, dims="region")
    s_a_c = _scale("sigma_alpha_country")
    z_a_c = pm.Normal("z_alpha_country", 0, 1, dims="country")
    alpha_country = pm.Deterministic("alpha_country", alpha_region[c2r] + s_a_c * z_a_c, dims="country")

    # ---- (3) decade fixed effects: sum-to-zero period intercepts ----
    tau_period = pm.ZeroSumNormal("tau_period", sigma=0.5, dims="period")

    # ---- reporting deterministics: separate growth vs decline elasticities ----
    pm.Deterministic("beta_decline_global", beta_global + delta_global)
    pm.Deterministic("beta_decline_region", beta_region + delta_region, dims="region")

    # ---- likelihood ----
    slope = beta_country[ci] + delta_region[ri] * dec      # = beta (growth) or beta+delta (decline)
    mu = alpha_country[ci] + tau_period[pi] + slope * x
    sigma = pm.HalfNormal("sigma", 0.5)
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS, target_accept=TA,
                      nuts_sampler="nutpie", random_seed=SEED, progressbar=False)

# ------------------------------------------------------------ diagnostics ----
summ = az.summary(idata, var_names=["beta_global", "delta_global", "beta_region",
                                    "tau_period", "sigma_beta_country", "sigma"])
print("PRIOR cfg: beta_global~N(1,%.1f), delta_global~N(0,%.1f), var~%s | max R-hat=%.3f min ess=%.0f"
      % (BETA_GLOBAL_SD, DELTA_GLOBAL_SD, VAR_PRIOR, summ["r_hat"].max(), summ["ess_bulk"].min()))
n_div = int(idata.sample_stats["diverging"].sum())
n_draws = int(idata.sample_stats["diverging"].size)
print("Divergences: %d of %d post-warmup draws (%.2f%%)" % (n_div, n_draws, 100 * n_div / n_draws))

# ----- per-country growth & decline elasticities (posterior mean + 95% HDI) -----
bc = idata.posterior["beta_country"]                                  # growth slope
dr = idata.posterior["delta_region"]
# map each country's region-level decline increment onto the country dim
dr_by_country = dr.isel(region=xr.DataArray(c2r, dims="country"))
bc_dec = bc + dr_by_country                                          # decline slope per country
beta_g = bc.mean(("chain", "draw")).values
beta_d = bc_dec.mean(("chain", "draw")).values
hdi_g = az.hdi(idata, var_names=["beta_country"], hdi_prob=0.95)["beta_country"].values
hdi_d = az.hdi(bc_dec.to_dataset(name="bc_dec"), hdi_prob=0.95)["bc_dec"].values
cdf = pd.DataFrame({"ISO3": countries,
                    "Region": [regions[c2r[i]] for i in range(len(countries))],
                    "n_obs": n_obs_country,
                    "beta_growth": beta_g, "growth_hdi_lo": hdi_g[:, 0], "growth_hdi_hi": hdi_g[:, 1],
                    "beta_decline": beta_d, "decline_hdi_lo": hdi_d[:, 0], "decline_hdi_hi": hdi_d[:, 1]}
                   ).sort_values("beta_growth")
cdf.to_csv(DIR_RESULTS / ("hierarchical_wa_country_elasticities%s.csv" % OUT_TAG), index=False)

# ----- region-level growth vs decline -----
br = idata.posterior["beta_region"]; brd = idata.posterior["beta_decline_region"]
rdf = pd.DataFrame({
    "Region": regions,
    "beta_growth":  br.mean(("chain", "draw")).values,
    "growth_hdi_lo": az.hdi(idata, var_names=["beta_region"], hdi_prob=0.95)["beta_region"].values[:, 0],
    "growth_hdi_hi": az.hdi(idata, var_names=["beta_region"], hdi_prob=0.95)["beta_region"].values[:, 1],
    "beta_decline": brd.mean(("chain", "draw")).values,
    "decline_hdi_lo": az.hdi(idata, var_names=["beta_decline_region"], hdi_prob=0.95)["beta_decline_region"].values[:, 0],
    "decline_hdi_hi": az.hdi(idata, var_names=["beta_decline_region"], hdi_prob=0.95)["beta_decline_region"].values[:, 1],
}).sort_values("beta_growth")
rdf.to_csv(DIR_RESULTS / ("hierarchical_wa_region_elasticities%s.csv" % OUT_TAG), index=False)

# ----- decade fixed effects -----
tp = idata.posterior["tau_period"]
thdi = az.hdi(idata, var_names=["tau_period"], hdi_prob=0.95)["tau_period"].values
pdf = pd.DataFrame({"period": PERIOD_LABELS, "tau_mean": tp.mean(("chain", "draw")).values,
                    "hdi_lo": thdi[:, 0], "hdi_hi": thdi[:, 1]})
pdf.to_csv(DIR_RESULTS / ("hierarchical_wa_period_effects%s.csv" % OUT_TAG), index=False)

def _sanitize_attrs(attrs):
    for key, val in list(attrs.items()):
        if isinstance(val, (dict, bool)) or val is None:
            attrs[key] = str(val)
for group in idata.groups():
    _sanitize_attrs(getattr(idata, group).attrs)
if hasattr(idata, "_attrs"):
    _sanitize_attrs(idata._attrs)
idata.to_netcdf(DIR_RESULTS / ("hierarchical_wa_posterior%s.nc" % OUT_TAG))  # full InferenceData (incl. sample_stats)

# ----------------------------------------------------------------- report ----
print("\nGlobal beta_growth  = %.3f" % float(idata.posterior["beta_global"].mean()))
print("Global beta_decline = %.3f  (delta_global = %.3f)"
      % (float(idata.posterior["beta_decline_global"].mean()),
         float(idata.posterior["delta_global"].mean())))
print("\nRegion-level growth vs decline elasticities:")
print(rdf.to_string(index=False, float_format=lambda v: "%.3f" % v))
print("\nDecade fixed effects (deviation from global intercept):")
print(pdf.to_string(index=False, float_format=lambda v: "%.3f" % v))
print("\nSaved hierarchical_wa_{country,region}_elasticities.csv, _period_effects.csv, _posterior.nc")
