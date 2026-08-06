"""
fit_hierarchical_5yr.py
=======================
Step 3 power stress-test: identical Bayesian hierarchical LINEAR model as
fit_hierarchical_10yr.py, but on NON-OVERLAPPING 5-YEAR blocks (~2x the
independent observations, more within-country population-growth variation).

    Delta5 ln_gdp_{it} = alpha_c[i] + beta_c[i] * Delta5 ln_pop_{it} + eps
    3-level non-centered hierarchy (global -> region -> country) on slope AND intercept.
    Priors IDENTICAL to the 10-yr baseline: beta_global ~ Normal(1.0, 0.5);
    all hierarchical SDs ~ HalfNormal(0.5); residual sigma ~ HalfNormal(0.5).

Non-overlapping 5-yr blocks: end years {1965,1970,...,2020} (start = end-5), disjoint,
so the likelihood-independence assumption holds.

Usage: python fit_hierarchical_5yr.py [draws] [tune] [chains] [target_accept]
Outputs: results/hierarchical_country_elasticities_5yr.csv / region csv / posterior .nc
         figures/Figure5_regional_elasticity_5yr_forest.{png,pdf}
"""
import sys
import numpy as np, pandas as pd, pymc as pm, arviz as az
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PATH_MERGED_AGE, DIR_RESULTS, DIR_FIGURES

K = 5
GDP_COL = "GDP_constant_2015usd"
BLOCK_END = list(range(1965, 2021, 5))   # 1965,1970,...,2020  (non-overlapping 5-yr)
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
TUNE  = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
CHAINS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.92
SEED = 42

df = pd.read_csv(PATH_MERGED_AGE)
gc = GDP_COL if (GDP_COL in df.columns and df[GDP_COL].notna().sum()) else "GDP"
df = df.dropna(subset=[gc, "Population", "ISO3", "Year", "Region"]).copy()
df = df[(df[gc] > 0) & (df["Population"] > 0)]; df["Year"] = df["Year"].astype(int)
df["ln_gdp"] = np.log(df[gc].astype(float)); df["ln_pop"] = np.log(df["Population"].astype(float))
base = df[["ISO3", "Year", "Region", "ln_gdp", "ln_pop"]]
a = base.copy(); b = base[["ISO3", "Year", "ln_gdp", "ln_pop"]].copy(); b["Year"] += K
m = a.merge(b, on=["ISO3", "Year"], suffixes=("", "_lag"))
m["dln_gdp"] = m["ln_gdp"] - m["ln_gdp_lag"]; m["dln_pop"] = m["ln_pop"] - m["ln_pop_lag"]
d = m[m["Year"].isin(BLOCK_END)].dropna(subset=["dln_gdp", "dln_pop"]).reset_index(drop=True)
for c in ["dln_gdp", "dln_pop"]:
    lo, hi = np.percentile(d[c], [1, 99]); d[c] = d[c].clip(lo, hi)

# ---- power diagnostics ----
npc = d.groupby("ISO3").size()
wsd = d.groupby("ISO3")["dln_pop"].std().median()
bsd = d.groupby("ISO3")["dln_pop"].mean().std()
print("5yr non-overlapping: N=%d  countries=%d  regions=%d  (obs/country median=%.0f mean=%.2f)" % (
    len(d), d["ISO3"].nunique(), d["Region"].nunique(), npc.median(), npc.mean()))
print("within-country SD dln_pop median=%.4f | between-country SD=%.4f | within/between=%.2f  (10yr was 0.42)" % (
    wsd, bsd, wsd / bsd))

d["rc"] = d["Region"].astype("category").cat.codes
d["cc"] = d["ISO3"].astype("category").cat.codes
regions = list(d["Region"].astype("category").cat.categories)
countries = list(d["ISO3"].astype("category").cat.categories)
ci = d["cc"].values.astype(int); x = d["dln_pop"].values; y = d["dln_gdp"].values
c2r = d[["cc", "rc"]].drop_duplicates().sort_values("cc")["rc"].values.astype(int)
n_obs_country = d.groupby("cc").size().reindex(range(len(countries))).fillna(0).astype(int).values
coords = {"region": regions, "country": countries}

with pm.Model(coords=coords) as model:
    beta_global = pm.Normal("beta_global", 1.0, 0.5)
    s_beta_r = pm.HalfNormal("sigma_beta_region", 0.5)
    z_beta_r = pm.Normal("z_beta_region", 0, 1, dims="region")
    beta_region = pm.Deterministic("beta_region", beta_global + s_beta_r * z_beta_r, dims="region")
    s_beta_c = pm.HalfNormal("sigma_beta_country", 0.5)
    z_beta_c = pm.Normal("z_beta_country", 0, 1, dims="country")
    beta_country = pm.Deterministic("beta_country", beta_region[c2r] + s_beta_c * z_beta_c, dims="country")
    alpha_global = pm.Normal("alpha_global", 0.0, 1.0)
    s_a_r = pm.HalfNormal("sigma_alpha_region", 0.5)
    z_a_r = pm.Normal("z_alpha_region", 0, 1, dims="region")
    alpha_region = pm.Deterministic("alpha_region", alpha_global + s_a_r * z_a_r, dims="region")
    s_a_c = pm.HalfNormal("sigma_alpha_country", 0.5)
    z_a_c = pm.Normal("z_alpha_country", 0, 1, dims="country")
    alpha_country = pm.Deterministic("alpha_country", alpha_region[c2r] + s_a_c * z_a_c, dims="country")
    sigma = pm.HalfNormal("sigma", 0.5)
    mu = alpha_country[ci] + beta_country[ci] * x
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS, target_accept=TA,
                      nuts_sampler="nutpie", random_seed=SEED, progressbar=False)

summ = az.summary(idata, var_names=["beta_global", "sigma_beta_region", "sigma_beta_country", "sigma"])
print("max R-hat=%.3f min ess=%.0f" % (summ["r_hat"].max(), summ["ess_bulk"].min()))
def ms(v):
    s = idata.posterior[v].stack(z=("chain", "draw")).values; return s.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)
for v in ["beta_global", "sigma_beta_region", "sigma_beta_country"]:
    mu, lo, hi = ms(v); print("  %-20s mean=%.4f [%.4f, %.4f]" % (v, mu, lo, hi))

# region + country betas
br = idata.posterior["beta_region"]; rmean = br.mean(("chain", "draw")).values
rhdi = az.hdi(idata, var_names=["beta_region"], hdi_prob=0.95)["beta_region"].values
rdf = pd.DataFrame({"Region": regions, "beta_mean": rmean, "hdi_lo": rhdi[:, 0], "hdi_hi": rhdi[:, 1]}).sort_values("beta_mean")
bc = idata.posterior["beta_country"]; cmean = bc.mean(("chain", "draw")).values
chdi = az.hdi(idata, var_names=["beta_country"], hdi_prob=0.95)["beta_country"].values
cdf = pd.DataFrame({"ISO3": countries, "Region": [regions[c2r[i]] for i in range(len(countries))],
                    "n_obs": n_obs_country, "beta_mean": cmean, "hdi_lo": chdi[:, 0], "hdi_hi": chdi[:, 1]}).sort_values("beta_mean")
rdf.to_csv(DIR_RESULTS / "hierarchical_region_elasticities_5yr.csv", index=False)
cdf.to_csv(DIR_RESULTS / "hierarchical_country_elasticities_5yr.csv", index=False)
idata.posterior.to_netcdf(DIR_RESULTS / "hierarchical_5yr_posterior.nc")
GLOBAL = float(idata.posterior["beta_global"].mean())
print("\nGlobal beta = %.3f | region-beta spread SD = %.4f | range [%.3f, %.3f]" % (
    GLOBAL, rdf["beta_mean"].std(), rdf["beta_mean"].min(), rdf["beta_mean"].max()))
print(rdf.to_string(index=False, float_format=lambda v: "%.3f" % v))
for iso in ["USA", "IND", "JPN", "NGA"]:
    r = cdf[cdf["ISO3"] == iso]
    if not r.empty:
        r = r.iloc[0]; print("  %s (n=%d): beta=%.3f [%.3f, %.3f]" % (iso, r["n_obs"], r["beta_mean"], r["hdi_lo"], r["hdi_hi"]))

# ---- forest plot ----
DIR_FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.4, 4.0))
yy = np.arange(len(rdf))[::-1]
ax.axvline(1.0, color="#444", ls="--", lw=1.2, label="β = 1 (proportional)")
ax.axvline(GLOBAL, color="#2e6e8e", ls="-.", lw=1.2, label="global mean (β = %.2f)" % GLOBAL)
for i, (_, row) in enumerate(rdf.iterrows()):
    sub = row["hdi_hi"] < 1.0; sup = row["hdi_lo"] > 1.0
    col = "#238b45" if sub else ("#b4451f" if sup else "#7a7a7a")
    ax.plot([row["hdi_lo"], row["hdi_hi"]], [yy[i], yy[i]], color=col, lw=2.6)
    ax.plot(row["beta_mean"], yy[i], "o", color=col, ms=8)
    ax.text(row["hdi_hi"] + 0.02, yy[i], "%.2f [%.2f, %.2f]" % (row["beta_mean"], row["hdi_lo"], row["hdi_hi"]), va="center", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels(rdf["Region"], fontsize=9)
ax.set_xlabel("5-year hierarchical elasticity  β (pooled)")
ax.set_title("Regional elasticities, 5-year non-overlapping (max power)", fontsize=11, fontweight="bold")
ax.set_xlim(0.4, max(1.6, rdf["hdi_hi"].max() + 0.35)); ax.legend(loc="lower right", fontsize=8.5)
for ext in ("png", "pdf"):
    fig.savefig("%s/Figure5_regional_elasticity_5yr_forest.%s" % (str(DIR_FIGURES), ext))
print("\nSaved figures/Figure5_regional_elasticity_5yr_forest.png and results/hierarchical_*_5yr.csv")
