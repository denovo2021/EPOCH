# Posterior sample files

The `.nc` posterior sample files are not in this repository. Together the deposited set is
9,991,913,033 bytes — about 10.0 GB — well past what a Git repository should carry, so they are
deposited on Zenodo instead (DOI in the top-level `README.md`). Everything in `results/` here —
every JSON, CSV and table the manuscript cites — was produced *from* them and is sufficient to
reproduce every figure and every reported number without downloading them. You need the `.nc`
files only to re-derive a quantity that is not already summarised here, or to inspect the draws
directly.

Each is written to `results/` by the command shown, with `SEED=42` throughout. Sizes are decimal GB.

## Deposited on Zenodo — seven files

| File | Size | Produced by |
|---|---:|---|
| `trace_fixed005_real.nc` | 2.82 GB | `src/fit_gdp_production.py` with `PRIOR=fixed005 GDP_COL=GDP_constant_2015usd` at **10,000 draws** — the deployed levels posterior under the fixed penalty. The doubled draw count is the reason given in SI S10.2, and is why this file is twice the size of the others |
| `trace_adaptive_real.nc` | 1.43 GB | the same script with `PRIOR=adaptive GDP_COL=GDP_constant_2015usd` — the deployed levels posterior under adaptive shrinkage |
| `trace_fixed005_real_5k_superseded.nc` | 1.43 GB | the 5,000-draw first pass at the row above it. Superseded, but deposited: the `real_fixed` block of `results/levels_elasticity.json` was computed from it, and SI S10.2 is the comparison between the two. The reported elasticities moved by 0.0001 |
| `trace_adaptive_real_noGNQ.nc` | 1.42 GB | as the adaptive row with `EXCLUDE_ISO3=GNQ` — the Equatorial Guinea sensitivity refit of SI S10.2 |
| `trace_adaptive.nc` | 1.42 GB | `PRIOR=adaptive` on the **nominal** series (`GDP_COL=GDP`) — the superseded fit, deposited because SI S3 reports it. Figure 1 is drawn from this one |
| `trace_fixed005.nc` | 1.42 GB | `PRIOR=fixed005` on the nominal series — likewise |
| `hierarchical_wa_posterior.nc` | 61 MB | `src/fit_hierarchical_workingage.py` — the long-difference model, and the only posterior the 2100 projection, the decision layer and Figures 3–6 need |

## Filenames changed after the runs — read this before re-running

The provenance strings recorded inside `results/*.json` are the paths **as they were at run time**.
Two files were renamed afterwards, so the same path string means different things depending on which
JSON you are reading. The JSONs are machine-written records of what actually ran and have not been
edited; use this table to map them onto the deposited files.

| Recorded in | Path as recorded | Deposited as | Which run |
|:---|:---|:---|:---|
| `levels_elasticity.json`, `chainwise_check.json` (`real_fixed`) | `results/trace_fixed005_real.nc` | `trace_fixed005_real_5k_superseded.nc` | fixed penalty, real GDP, **5,000 draws** |
| `levels_elasticity_refits.json` (`real_fixed_d10k`) | `results/trace_fixed005_real_d10k.nc` | `trace_fixed005_real.nc` | fixed penalty, real GDP, **10,000 draws** — the deployed fit |

## Not deposited, deliberately

| File | Size | Why not |
|---|---:|---|
| `hierarchical_model_rcs_v2.nc` | 1.42 GB | a byte-identical copy of `trace_adaptive.nc`, kept under the filename `src/make_figures.py` expects. Recreate it with `cp results/trace_adaptive.nc results/hierarchical_model_rcs_v2.nc` rather than downloading 1.4 GB twice |
| `hierarchical_wa_posterior_weakprior.nc` | 61 MB | four chains sampled in parallel are not bit-deterministic, so re-running these two refits produces different draws rather than recovering the archived ones. An independent re-run moves no country's elasticity by more than 0.010 and leaves every reported conclusion standing, but its draws did not generate the archived CSVs. Every quantity these two support is carried by `hierarchical_wa_*_weakprior.csv` and `hierarchical_*_5yr.csv` in `results/` |
| `hierarchical_5yr_posterior.nc` | 42 MB | as above |

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
