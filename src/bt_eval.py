"""Pool trace_*.nc chains, compute convergence diagnostics, project test set,
and report GDP backtest metrics (MAE log10, COV95, CRPS, test n)."""
import sys, os, glob, json
import numpy as np, xarray as xr, arviz as az
from scipy import stats as sp_stats
from bt_common import prep
from config import DIR_RESULTS
DATA=sys.argv[1]; OUT=sys.argv[2]; TRACE_GLOB=sys.argv[3]
d=prep(DATA)
files=sorted(glob.glob(os.path.expanduser(TRACE_GLOB)))
print("pooling traces:",files)
dss=[xr.open_dataset(f,engine="scipy") for f in files]
# renumber chains so concat is unique
post=xr.concat([ds.assign_coords(chain=ds.chain+ i*ds.sizes["chain"]) for i,ds in enumerate(dss)],dim="chain")
idata=az.InferenceData(posterior=post)
summ=az.summary(idata,var_names=["alpha_country","beta0","theta","sigma","nu","delta_washare","delta_olddep","tau_region"])
max_rhat=float(summ["r_hat"].max()); min_ess=float(summ["ess_bulk"].min()); med_ess=float(summ["ess_bulk"].median())
print("POOLED chains=%d draws=%d | predictive max R-hat=%.3f min ess=%.0f median ess=%.0f"%(
    post.sizes["chain"],post.sizes["draw"],max_rhat,min_ess,med_ess))

te=d["te"]; countries=list(d["countries"]); c2r=d["c2r"]; knots=d["knots"]; mu_g=d["mu_g"]
cidx=te["ISO3"].map({c:i for i,c in enumerate(countries)})
te=te[cidx.notna()].copy(); ci_te=cidx[cidx.notna()].astype(int).values
from bt_common import rcs_design
x_te=te["Log_Population"].values-mu_g; Z_te=rcs_design(x_te,knots)
te=te.merge(d["anc"],on="ISO3",how="left")
dWA_te=(te["WAshare"]-te["WAb"]).values/d["s_dWA"]; dOD_te=(te["OldDep"]-te["ODb"]).values/d["s_dOD"]
dt_raw=((te["Year"]-te["Yb"])/10.0).values/d["s_dt"]
Xdt=np.column_stack([np.ones_like(x_te),x_te,Z_te]); dt_te=np.clip(dt_raw-Xdt@d["coef"],-2.0,2.0)
ri_te=c2r[ci_te]
flat=lambda a: a.values.reshape(-1,*a.values.shape[2:])
a_c=flat(post["alpha_country"]); b0=flat(post["beta0"]).ravel(); th=flat(post["theta"])
tau=flat(post["tau_region"]); dwa=flat(post["delta_washare"]).ravel(); dod=flat(post["delta_olddep"]).ravel()
sig=flat(post["sigma"]).ravel(); nuv=flat(post["nu"]).ravel()
n_te=len(te); n_s=len(sig); pred=np.empty((n_te,n_s))
for s in range(n_s):
    mu_s=a_c[s,ci_te]+b0[s]*x_te+Z_te@th[s]+dwa[s]*dWA_te+dod[s]*dOD_te+tau[s,ri_te]*dt_te
    pred[:,s]=mu_s+sp_stats.t.rvs(df=nuv[s],loc=0,scale=sig[s],size=n_te,random_state=np.random.RandomState(s))
y_te=te["y_target"].values
def crps_ens(obs,sa):
    n,nd=sa.shape; o=np.empty(n)
    for i in range(n):
        ss=np.sort(sa[i]); w=2.0*np.arange(1,nd+1)-nd-1.0
        o[i]=np.mean(np.abs(ss-obs[i]))-np.dot(w,ss)/(nd**2)
    return o
lo=np.percentile(pred,2.5,axis=1); hi=np.percentile(pred,97.5,axis=1)
mae=float(np.mean(np.abs(np.median(pred,axis=1)-y_te)))
cov=float(np.mean((y_te>=lo)&(y_te<=hi))); crps=float(np.mean(crps_ens(y_te,pred)))
np.savez(str(DIR_RESULTS / "backtest_predictions.npz"),y=y_te,med=np.median(pred,axis=1),lo=lo,hi=hi,
         iso=te["ISO3"].values,region=te["Region"].values,year=te["Year"].values)
res={"gdp_col":d["gdp_col"],"train_period":"%d-%d"%(int(d["tr"].Year.min()),2010),
     "test_period":"%d-%d"%(int(te.Year.min()),int(te.Year.max())),"test_n":int(n_te),
     "MAE_log":round(mae,4),"COV95":round(cov,4),"CRPS":round(crps,4),
     "pooled_chains":int(post.sizes["chain"]),"pooled_draws_per_chain":int(post.sizes["draw"]),
     "predictive_max_rhat":round(max_rhat,3),"predictive_min_ess":round(min_ess,0),"predictive_median_ess":round(med_ess,0),
     "maxdepth":int(os.environ.get("MTD","8")),"target_accept":0.9,"seeds":[1,7,13]}
json.dump(res,open(OUT,"w"),indent=2)
print(json.dumps(res,indent=2))
