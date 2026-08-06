# Posterior sample files

The `.nc` posterior sample files are not in this repository. Together they are about 8 GB, well
past what a Git repository should carry, so they are deposited on Zenodo instead (DOI in the
top-level `README.md`). Everything in `results/` here — every JSON, CSV and table the manuscript
cites — was produced *from* them and is sufficient to reproduce every figure and every reported
number without downloading them. You need the `.nc` files only to re-derive a quantity that is not
already summarised here, or to inspect the draws directly.

Each is written to `results/` by the command shown, with `SEED=42` throughout.

| File | Approx. size | Produced by |
|---|---:|---|
| `trace_fixed005_real_d10k.nc` | 2.8 GB | `python src/fit_gdp_production.py` with `PRIOR=fixed005 GDP_COL=GDP_constant_2015usd DRAWS=10000` — **the deployed levels posterior under the fixed penalty** |
| `trace_adaptive_real.nc` | 1.4 GB | `python src/fit_gdp_production.py` with `PRIOR=adaptive GDP_COL=GDP_constant_2015usd` — **the deployed levels posterior under adaptive shrinkage** |
| `trace_fixed005_real.nc` | 1.4 GB | as the first row at the default 5,000 draws; superseded by the 10,000-draw refit, retained because S10 reports the comparison |
| `trace_adaptive_real_noGNQ.nc` | 1.4 GB | as the second row with `EXCLUDE_ISO3=GNQ` — the Equatorial Guinea sensitivity refit of S10.2 |
| `trace_fixed005.nc` | 1.4 GB | `PRIOR=fixed005` on the **nominal** series (`GDP_COL=GDP`) — the superseded fit, kept because S3 reports it |
| `trace_adaptive.nc` | 1.4 GB | `PRIOR=adaptive` on the nominal series — likewise |
| `hierarchical_wa_posterior.nc` | 58 MB | `python src/fit_hierarchical_workingage.py` — the long-difference model |
| `hierarchical_wa_posterior_weakprior.nc` | 59 MB | the same script with `OUT_TAG=_weakprior` and weak priors |
| `hierarchical_5yr_posterior.nc` | 41 MB | `python src/fit_hierarchical_5yr.py` — the five-year-block robustness fit |

## What each stage needs

- The 2100 projection, the decision layer, every table and Figures 3–6 read
  `hierarchical_wa_posterior.nc` only, and run in about a minute once it is present.
- `src/s13_levels_elasticity.py`, `src/s15_chainwise_check.py` and `src/make_convergence_table.py`
  read the large levels traces. Each one is loaded and released in turn, so peak memory is one
  trace, not all of them, but budget roughly 16 GB of RAM.
- `src/s16_figure5_real_vs_nominal.py` deliberately reads `results/levels_elasticity.json`, not the
  traces, so Figure 5 cannot drift from the numbers in the text.

A full re-fit of the two deployed levels posteriors is the expensive step: roughly 6–10 hours each
on 4 chains. Nothing else in the pipeline takes more than a few minutes.
