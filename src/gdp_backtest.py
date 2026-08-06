"""Standalone temporal backtest of the hierarchical GDP model (m03-faithful).
Train <=2010, test 2011+ (limited by age-covariate coverage to 2017).
Model: log10(GDP) = alpha_c + beta0*x_s + theta.B(x_s) + dWA + dOD + tau_r*dt + StudentT noise
with three-level non-centered hierarchy (region, country) as in m03_hierarchical_model_rcs_v2.
Reports MAE(log10), 95% PI coverage (COV95), CRPS, and test n.
Usage: python gdp_backtest.py DATA OUT [draws] [tune] [target_accept] [chains]
"""
import sys, os, json
import numpy as np, pandas as pd, pymc as pm, arviz as az
from scipy import stats as sp_stats
from config import DIR_RESULTS

DATA = sys.argv[1]; OUT = sys.argv[2]
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
TUNE = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
TARGET_ACCEPT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
CHAINS = int(sys.argv[6]) if len(sys.argv) > 6 else 4
TRAIN_END, TEST_START, TEST_END = 2010, 2011, 2023
KNOT_QUANTS = [0.05, 0.35, 0.65, 0.95]
SEED = 42
MTD = int(os.environ.get("MTD", "10"))

def rcs_design(x, knots):
    k = np.asarray(knots, float); K = k.size
    if K < 3: return np.zeros((x.size, 0))
    d = lambda u, j: np.maximum(u - k[j], 0.0) ** 3
    cols = [d(x, j) - d(x, K-1)*(k[K-1]-k[j])/(k[K-1]-k[0]) + d(x, 0)*(k[j]-k[0])/(k[K-1]-k[0])
            for j in range(1, K-1)]
    return np.column_stack(cols)

def crps_ensemble(obs, samples):
    n_obs, n_draws = samples.shape; out = np.empty(n_obs)
    for i in range(n_obs):
        s = np.sort(samples[i]); y = obs[i]
        w = 2.0*np.arange(1, n_draws+1) - n_draws - 1.0
        out[i] = np.mean(np.abs(s - y)) - np.dot(w, s)/(n_draws**2)
    return out

def coverage_95(obs, samples):
    lo = np.percentile(samples, 2.5, axis=1); hi = np.percentile(samples, 97.5, axis=1)
    return float(np.mean((obs >= lo) & (obs <= hi)))

df = pd.read_csv(DATA)
if "Country Code" in df.columns and "ISO3" not in df.columns:
    df = df.rename(columns={"Country Code": "ISO3"})
df["Year"] = df["Year"].astype(int)
gdp_col = "Log_GDP_constant" if ("Log_GDP_constant" in df.columns and df["Log_GDP_constant"].notna().sum()) else "Log_GDP"
if "Log_Population" not in df.columns:
    df["Log_Population"] = np.log10(df["Population"].clip(lower=1))
df = df.dropna(subset=[gdp_col, "Population", "Region", "ISO3"]).copy()
df = df[df["Population"] > 0]; df["y_target"] = df[gdp_col]
for c in ["y_target", "Log_Population", "WAshare", "OldDep"]:
    if c in df.columns: df = df[np.isfinite(df[c].astype(float))]
df_tr = df[df["Year"] <= TRAIN_END].copy()
df_te = df[(df["Year"] >= TEST_START) & (df["Year"] <= TEST_END)].copy()

# ---- training-set standardization (no leakage) ----
mu_g = float(df_tr["Log_Population"].mean())
x_tr = df_tr["Log_Population"].values - mu_g
knots = np.quantile(x_tr, KNOT_QUANTS)
Z_tr = rcs_design(x_tr, knots); m = Z_tr.shape[1]
anc = (df_tr.loc[df_tr.groupby("ISO3")["Year"].idxmax()][["ISO3","WAshare","OldDep","Year"]]
       .rename(columns={"WAshare":"WAb","OldDep":"ODb","Year":"Yb"}))
df_tr = df_tr.merge(anc, on="ISO3", how="left")
dWA_raw = (df_tr["WAshare"]-df_tr["WAb"]).values; dOD_raw = (df_tr["OldDep"]-df_tr["ODb"]).values
s_dWA = np.nanstd(dWA_raw) or 0.1; s_dOD = np.nanstd(dOD_raw) or 0.1
dWA_tr = dWA_raw/s_dWA; dOD_tr = dOD_raw/s_dOD
dt_raw = ((df_tr["Year"]-df_tr["Yb"])/10.0).values
s_dt = np.nanstd(dt_raw) or 1.0; dt_s = dt_raw/s_dt
Xdt = np.column_stack([np.ones_like(x_tr), x_tr, Z_tr])
coef_dt = np.linalg.lstsq(Xdt, dt_s, rcond=None)[0]
dt_tr = np.clip(dt_s - Xdt@coef_dt, -2.0, 2.0)
print("GDP col=%s | train n=%d (%d-%d) | test n=%d (%d-%d)" % (
    gdp_col, len(df_tr), df_tr.Year.min(), df_tr.Year.max(), len(df_te), df_te.Year.min(), df_te.Year.max()))

df_tr["rc"] = df_tr["Region"].astype("category").cat.codes
df_tr["cc"] = df_tr["ISO3"].astype("category").cat.codes
regions = df_tr["Region"].astype("category").cat.categories
countries = df_tr["ISO3"].astype("category").cat.categories
ri = df_tr["rc"].values.astype(int); ci = df_tr["cc"].values.astype(int)
y = df_tr["y_target"].values
c2r = df_tr[["cc","rc"]].drop_duplicates().sort_values("cc").set_index("cc")["rc"].values.astype(int)
coords = {"Region": regions.tolist(), "Country": countries.tolist(), "Spline": [str(i) for i in range(m)]}
y_mean = float(np.mean(y))

with pm.Model(coords=coords) as mdl:
    alpha0 = pm.Normal("alpha0", y_mean, 1.0)
    beta0 = pm.Normal("beta0", 0.9, 0.5)
    theta = pm.Normal("theta", 0.0, 0.2, dims="Spline")
    delta_washare = pm.Normal("delta_washare", 0.0, 0.3)
    delta_olddep = pm.Normal("delta_olddep", 0.0, 0.3)
    sig_ar = pm.Exponential("sigma_alpha_region", 2.0)
    z_ar = pm.Normal("z_alpha_region", 0.0, 1.0, dims="Region")
    alpha_r = pm.Deterministic("alpha_region", alpha0 + sig_ar*z_ar, dims="Region")
    tau0 = pm.Normal("tau0", 0.0, 0.12)
    sig_tr = pm.Exponential("sigma_tau_region", 10.0)
    z_tr_ = pm.Normal("z_tau_region", 0.0, 1.0, dims="Region")
    tau_r = pm.Deterministic("tau_region", tau0 + sig_tr*z_tr_, dims="Region")
    sig_ac = pm.Exponential("sigma_alpha_country", 10.0)
    z_ac = pm.Normal("z_alpha_country", 0.0, 1.0, dims="Country")
    alpha_c = pm.Deterministic("alpha_country", alpha_r[c2r] + sig_ac*z_ac, dims="Country")
    sigma = pm.HalfNormal("sigma", 0.15)
    nu = pm.Deterministic("nu", pm.Gamma("nu_minus2", 2.0, 0.1) + 2.0)
    mu = (alpha_c[ci] + beta0*x_tr + pm.math.dot(Z_tr, theta)
          + delta_washare*dWA_tr + delta_olddep*dOD_tr + tau_r[ri]*dt_tr)
    pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y)
    if os.environ.get("SAMPLER", "nutpie") == "pymc":
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                          target_accept=TARGET_ACCEPT, max_treedepth=MTD,
                          random_seed=SEED, progressbar=False)
    else:
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                          target_accept=TARGET_ACCEPT, nuts_sampler="nutpie",
                          nuts_sampler_kwargs={"maxdepth": MTD},
                          random_seed=SEED, progressbar=False)

# R-hat on PREDICTION-relevant quantities (country intercepts are data-pinned;
# the alpha0/offset additive split is non-identified and irrelevant to predictions)
summ_pred = az.summary(idata, var_names=["alpha_country","beta0","theta","sigma","nu","delta_washare","delta_olddep","tau_region"])
max_rhat = float(summ_pred["r_hat"].max()); min_ess = float(summ_pred["ess_bulk"].min())
summ_glob = az.summary(idata, var_names=["alpha0","tau0","sigma_alpha_region"])
print("global-param max R-hat (non-identified split, informational)=%.2f" % float(summ_glob["r_hat"].max()))
ndiv = int(idata.sample_stats["diverging"].values.sum())
print("max R-hat=%.3f  min ess_bulk=%.0f  divergences=%d" % (max_rhat, min_ess, ndiv))

# ---- project test (no refit) ----
post = idata.posterior
cidx = df_te["ISO3"].map({c:i for i,c in enumerate(countries)})
df_te = df_te[cidx.notna()].copy(); ci_te = cidx[cidx.notna()].astype(int).values
x_te = df_te["Log_Population"].values - mu_g
Z_te = rcs_design(x_te, knots)
df_te = df_te.merge(anc, on="ISO3", how="left")
dWA_te = (df_te["WAshare"]-df_te["WAb"]).values/s_dWA
dOD_te = (df_te["OldDep"]-df_te["ODb"]).values/s_dOD
dt_te_raw = ((df_te["Year"]-df_te["Yb"])/10.0).values/s_dt
Xdt_te = np.column_stack([np.ones_like(x_te), x_te, Z_te])
dt_te = np.clip(dt_te_raw - Xdt_te@coef_dt, -2.0, 2.0)
ri_te = c2r[ci_te]
flat = lambda a: a.values.reshape(-1, *a.values.shape[2:])
a_c = flat(post["alpha_country"]); b0 = flat(post["beta0"]).ravel(); th = flat(post["theta"])
tau = flat(post["tau_region"]); dwa = flat(post["delta_washare"]).ravel(); dod = flat(post["delta_olddep"]).ravel()
sig = flat(post["sigma"]).ravel(); nuv = flat(post["nu"]).ravel()
n_te = len(df_te); n_s = len(sig)
pred = np.empty((n_te, n_s))
for s in range(n_s):
    mu_s = a_c[s, ci_te] + b0[s]*x_te + Z_te@th[s] + dwa[s]*dWA_te + dod[s]*dOD_te + tau[s, ri_te]*dt_te
    pred[:, s] = mu_s + sp_stats.t.rvs(df=nuv[s], loc=0, scale=sig[s], size=n_te, random_state=np.random.RandomState(s))

y_te = df_te["y_target"].values
mae = float(np.mean(np.abs(np.median(pred, axis=1)-y_te)))
crps = float(np.mean(crps_ensemble(y_te, pred)))
cov = coverage_95(y_te, pred)
np.savez(str(DIR_RESULTS / "backtest_predictions.npz"),
         y=y_te, med=np.median(pred, axis=1),
         lo=np.percentile(pred, 2.5, axis=1), hi=np.percentile(pred, 97.5, axis=1),
         iso=df_te["ISO3"].values, region=df_te["Region"].values, year=df_te["Year"].values)
res = {"gdp_col": gdp_col, "train_period": "%d-%d" % (int(df_tr.Year.min()), TRAIN_END),
       "test_period": "%d-%d" % (int(df_te.Year.min()), int(df_te.Year.max())), "test_n": int(n_te),
       "MAE_log": round(mae,4), "COV95": round(cov,4), "CRPS": round(crps,4),
       "max_rhat": round(max_rhat,3), "min_ess_bulk": round(min_ess,0), "divergences": ndiv,
       "draws": DRAWS, "tune": TUNE, "chains": CHAINS, "target_accept": TARGET_ACCEPT, "seed": SEED}
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2))
