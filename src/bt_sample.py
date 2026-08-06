"""Sample N chains for the GDP backtest model and save the trace.
Usage: python bt_sample.py DATA TRACE_OUT SEED [tune] [draws] [chains] [target_accept]
"""
import sys, os, numpy as np, pymc as pm
from bt_common import prep
DATA=sys.argv[1]; TRACE=sys.argv[2]; SEED=int(sys.argv[3])
TUNE=int(sys.argv[4]) if len(sys.argv)>4 else 400
DRAWS=int(sys.argv[5]) if len(sys.argv)>5 else 400
CHAINS=int(sys.argv[6]) if len(sys.argv)>6 else 2
TA=float(sys.argv[7]) if len(sys.argv)>7 else 0.9
MTD=int(os.environ.get("MTD","8"))
d=prep(DATA)
coords={"Region":d["regions"],"Country":d["countries"],"Spline":np.arange(d["m"])}
y_mean=float(np.mean(d["y"]))
with pm.Model(coords=coords) as mdl:
    alpha0=pm.Normal("alpha0",y_mean,1.0); beta0=pm.Normal("beta0",0.9,0.5)
    sigma_theta=pm.HalfNormal("sigma_theta",0.2)
    theta=pm.Normal("theta",0.0,sigma_theta,dims="Spline")
    dwa=pm.Normal("delta_washare",0.0,0.3); dod=pm.Normal("delta_olddep",0.0,0.3)
    sar=pm.Exponential("sigma_alpha_region",2.0); zar=pm.Normal("z_alpha_region",0,1,dims="Region")
    alpha_r=pm.Deterministic("alpha_region",alpha0+sar*zar,dims="Region")
    tau0=pm.Normal("tau0",0,0.12); stra=pm.Exponential("sigma_tau_region",10.0); ztr=pm.Normal("z_tau_region",0,1,dims="Region")
    tau_r=pm.Deterministic("tau_region",tau0+stra*ztr,dims="Region")
    sac=pm.Exponential("sigma_alpha_country",10.0); zac=pm.Normal("z_alpha_country",0,1,dims="Country")
    alpha_c=pm.Deterministic("alpha_country",alpha_r[d["c2r"]]+sac*zac,dims="Country")
    sigma=pm.HalfNormal("sigma",0.15); nu=pm.Deterministic("nu",pm.Gamma("nu_minus2",2.0,0.1)+2.0)
    mu=(alpha_c[d["ci"]]+beta0*d["x_tr"]+pm.math.dot(d["Z_tr"],theta)
        +dwa*d["dWA_tr"]+dod*d["dOD_tr"]+tau_r[d["ri"]]*d["dt_tr"])
    pm.StudentT("y_obs",nu=nu,mu=mu,sigma=sigma,observed=d["y"])
    idata=pm.sample(draws=DRAWS,tune=TUNE,chains=CHAINS,cores=CHAINS,target_accept=TA,
                    nuts_sampler="nutpie",nuts_sampler_kwargs={"maxdepth":MTD},
                    random_seed=SEED,progressbar=False)
idata.posterior.to_netcdf(TRACE)
print("saved",TRACE,"chains",CHAINS,"draws",DRAWS,"tune",TUNE)
