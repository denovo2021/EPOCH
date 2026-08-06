"""Shared deterministic data-prep for the GDP backtest (no MCMC)."""
import numpy as np, pandas as pd
TRAIN_END = 2010; TEST_START, TEST_END = 2011, 2023
KNOT_QUANTS = [0.05, 0.35, 0.65, 0.95]
def rcs_design(x, knots):
    k = np.asarray(knots, float); K = k.size
    d = lambda u, j: np.maximum(u - k[j], 0.0) ** 3
    cols = [d(x, j) - d(x, K-1)*(k[K-1]-k[j])/(k[K-1]-k[0]) + d(x, 0)*(k[j]-k[0])/(k[K-1]-k[0])
            for j in range(1, K-1)]
    return np.column_stack(cols)
def prep(DATA):
    df = pd.read_csv(DATA)
    if "Country Code" in df.columns and "ISO3" not in df.columns:
        df = df.rename(columns={"Country Code": "ISO3"})
    df["Year"] = df["Year"].astype(int)
    gdp_col = "Log_GDP_constant" if ("Log_GDP_constant" in df.columns and df["Log_GDP_constant"].notna().sum()) else "Log_GDP"
    if "Log_Population" not in df.columns:
        df["Log_Population"] = np.log10(df["Population"].clip(lower=1))
    df = df.dropna(subset=[gdp_col, "Population", "Region", "ISO3"]).copy()
    df = df[df["Population"] > 0]; df["y_target"] = df[gdp_col]
    for c in ["y_target","Log_Population","WAshare","OldDep"]:
        if c in df.columns: df = df[np.isfinite(df[c].astype(float))]
    tr = df[df["Year"] <= TRAIN_END].copy()
    te = df[(df["Year"] >= TEST_START) & (df["Year"] <= TEST_END)].copy()
    mu_g = float(tr["Log_Population"].mean())
    x_tr = tr["Log_Population"].values - mu_g
    knots = np.quantile(x_tr, KNOT_QUANTS); Z_tr = rcs_design(x_tr, knots); m = Z_tr.shape[1]
    anc = (tr.loc[tr.groupby("ISO3")["Year"].idxmax()][["ISO3","WAshare","OldDep","Year"]]
           .rename(columns={"WAshare":"WAb","OldDep":"ODb","Year":"Yb"}))
    tr = tr.merge(anc, on="ISO3", how="left")
    dWA_raw=(tr["WAshare"]-tr["WAb"]).values; dOD_raw=(tr["OldDep"]-tr["ODb"]).values
    s_dWA=np.nanstd(dWA_raw) or 0.1; s_dOD=np.nanstd(dOD_raw) or 0.1
    dWA_tr=dWA_raw/s_dWA; dOD_tr=dOD_raw/s_dOD
    dt_raw=((tr["Year"]-tr["Yb"])/10.0).values; s_dt=np.nanstd(dt_raw) or 1.0; dt_s=dt_raw/s_dt
    Xdt=np.column_stack([np.ones_like(x_tr),x_tr,Z_tr]); coef=np.linalg.lstsq(Xdt,dt_s,rcond=None)[0]
    dt_tr=np.clip(dt_s-Xdt@coef,-2.0,2.0)
    tr["rc"]=tr["Region"].astype("category").cat.codes; tr["cc"]=tr["ISO3"].astype("category").cat.codes
    regions=tr["Region"].astype("category").cat.categories; countries=tr["ISO3"].astype("category").cat.categories
    ri=tr["rc"].values.astype(int); ci=tr["cc"].values.astype(int); y=tr["y_target"].values
    c2r=tr[["cc","rc"]].drop_duplicates().sort_values("cc").set_index("cc")["rc"].values.astype(int)
    return dict(gdp_col=gdp_col, tr=tr, te=te, mu_g=mu_g, knots=knots, Z_tr=Z_tr, m=m, x_tr=x_tr,
                dWA_tr=dWA_tr, dOD_tr=dOD_tr, dt_tr=dt_tr, s_dWA=s_dWA, s_dOD=s_dOD, s_dt=s_dt, coef=coef,
                regions=regions, countries=countries, ri=ri, ci=ci, y=y, c2r=c2r, anc=anc)
