# EPOCH — the population–output elasticity

Analysis code, input data and result artifacts for

> **Real output scales sub-proportionally with population, and super-unitary estimates are an
> artifact of nominal GDP**
> Tomoki Kawahara, Shuhei Terada, Nobutoshi Nawa, Takeo Fujiwara

Let ε be the elasticity of aggregate real output with respect to population,
ε = ∂ ln GDP / ∂ ln Population. Because output per person is output divided by population, a
proportional fall in population changes output per person by (ε − 1) times that fall. The **sign
of ε − 1** is therefore the whole of the question of whether depopulation erodes living standards,
and this repository is the machinery for estimating it.

**Headline result.** Fitted to real GDP in constant 2015 US dollars, the contraction-phase
elasticity is **0.652 (95% credible interval 0.594–0.710)** under a fixed spline penalty and
**0.806 (0.748–0.863)** under adaptive shrinkage — sub-unitary with posterior probability 1.00
under both. Fitted to *nominal* GDP, the same model on the same panel returns 0.924 and 1.220,
straddling unity. The widely reported super-unitary estimates, including this project's own
earlier drafts, are an artifact of measuring output in current prices. Carried through United
Nations medium-variant demography to 2100 for 175 economies, output per person rises in every one,
including all 75 whose population falls.

The manuscript and Supplementary Information are added to this repository on publication.

---

## Layout

| | |
|:---|:---|
| `src/` | 29 analysis scripts — panel construction, both levels fits, the long-difference model, the projection engine, the decision layer, and every figure and table script |
| `tests/` | `test_projection_engine.py` — eight guards on the projection engine, including a case that must abort |
| `data/` | the analysis-ready panel and the United Nations projection inputs, 4 CSVs |
| `results/` | every JSON, CSV and table the paper cites. `POSTERIORS.md` lists the `.nc` sample files, which are on Zenodo rather than here |
| `figures/` | the published figures, PNG and PDF |

## Environment

```bash
pip install uv
uv venv
uv pip install pymc nutpie arviz scipy pandas matplotlib h5netcdf
```

Python 3.12. `arviz` 0.17.1 is the version `src/compare_models.py` was written against, and the
one whose `az.compare(ic="loo")` returns stacking weights.

Every script is run **from the repository root** and imports `src/config.py`, which anchors all
paths to the root, so the tree is relocatable. The global seed is **42** throughout — passed to
every sampler call, to the posterior-draw subsampling in the projection engine, and to the
per-country residual generators, which are seeded as 42 plus the sum of the ASCII codes of each
country's ISO3 code.

## Reproducing the paper

The commands below regenerate every reported number from the raw inputs, in order. Environment
overrides are shown in POSIX-shell form; in PowerShell use `$env:PRIOR="fixed005"` on its own line
and `Remove-Item Env:\PRIOR` to clear it.

**Nothing here has to be re-run to check the paper.** Every number the manuscript states is
already in `results/`, derived from the posteriors listed in `results/POSTERIORS.md`. Re-run only
what you want to re-derive.

```bash
# Stage 1 -- deterministic constants (~5 s)
python src/build_scale_cache.py

# Stage 2 -- levels posteriors  (LONG: about 2-4 h each, 4-8 h for the 10,000-draw fit)
# The nominal pair. These are the comparison fits; Figure 1 is drawn from the adaptive one.
SAVE_FULL=1 PRIOR=fixed005 \
  python src/fit_gdp_production.py data/merged_age.csv results/trace_fixed005.nc 42 2000 5000 4 0.95
SAVE_FULL=1 PRIOR=adaptive \
  python src/fit_gdp_production.py data/merged_age.csv results/trace_adaptive.nc  42 2000 5000 4 0.95
cp results/trace_adaptive.nc results/hierarchical_model_rcs_v2.nc   # deployed alias, Figure 1

# The real pair. Same panel, basis, priors, likelihood and seed; only the outcome column changes.
# The fixed-penalty fit draws 10,000 rather than 5,000, for the reason given in SI S10.2.
SAVE_FULL=1 GDP_COL=GDP_constant_2015usd PRIOR=fixed005 \
  python src/fit_gdp_production.py data/merged_age.csv results/trace_fixed005_real.nc 42 2000 10000 4 0.95
SAVE_FULL=1 GDP_COL=GDP_constant_2015usd PRIOR=adaptive \
  python src/fit_gdp_production.py data/merged_age.csv results/trace_adaptive_real.nc  42 2000 5000 4 0.95

# The leave-one-country-out sensitivity refit.
SAVE_FULL=1 GDP_COL=GDP_constant_2015usd PRIOR=adaptive EXCLUDE_ISO3=GNQ \
  python src/fit_gdp_production.py data/merged_age.csv results/trace_adaptive_real_noGNQ.nc 42 2000 5000 4 0.95

# Stage 3 -- levels model comparison, elasticity, chain-wise diagnostics (~3 min)
# The third argument tags the output so the two comparisons do not overwrite each other.
python src/compare_models.py results/trace_fixed005.nc      results/trace_adaptive.nc
python src/compare_models.py results/trace_fixed005_real.nc results/trace_adaptive_real.nc _real
python src/elasticity_inflection.py
python src/s13_levels_elasticity.py \
  --posterior real_fixed:results/trace_fixed005_real.nc:GDP_constant_2015usd \
  --posterior real_adaptive:results/trace_adaptive_real.nc:GDP_constant_2015usd \
  --posterior nominal_fixed:results/trace_fixed005.nc:GDP \
  --posterior nominal_adaptive:results/trace_adaptive.nc:GDP
python src/s15_chainwise_check.py \
  --posterior real_fixed:results/trace_fixed005_real.nc:GDP_constant_2015usd \
  --posterior real_adaptive:results/trace_adaptive_real.nc:GDP_constant_2015usd \
  --posterior real_adaptive_noGNQ:results/trace_adaptive_real_noGNQ.nc:GDP_constant_2015usd \
  --posterior nominal_fixed:results/trace_fixed005.nc:GDP \
  --posterior nominal_adaptive:results/trace_adaptive.nc:GDP

# Stage 4 -- out-of-sample backtest  (LONG: about 1-3 h)
python src/gdp_backtest.py data/merged_age.csv results/gdp_backtest_metrics.json 3000 4000 0.98 4

# Stage 5 -- long-difference posteriors (~15-45 min for the baseline)
python src/fit_hierarchical_workingage.py 1500 1000 4 0.9
BETA_GLOBAL_SD=2 DELTA_GLOBAL_SD=2 VAR_PRIOR=halfcauchy OUT_TAG=_weakprior \
  python src/fit_hierarchical_workingage.py 1500 1000 4 0.9
python src/fit_hierarchical_5yr.py 1500 1000 4 0.92

# Stage 6 -- projection engine, its guard suite, the bridge and the decision layer (~5 min)
python tests/test_projection_engine.py
python src/s07_projection_2100.py
python src/s07_projection_2100.py --eps-growth 0.92 --eps-decline 0.92 --tag _eps0.92
python src/s07_projection_2100.py --eps-growth 0.98 --eps-decline 1.22 --tag _eps1.22
python src/s07_projection_2100.py --s_ss 0.150 --tag _sss1.51
python src/s07_projection_2100.py --s_ss 0.200 --tag _sss2.02
python src/s07_projection_2100.py --no-converge --tag _noconv
python src/s08_elasticity_bridge.py
python src/s09_decision_layer.py --s0 0.10 --discount 0.03

# Stage 7 -- figures and tables (~5 min)
python src/s10_make_figures_2100.py
python src/s11_make_tables.py
python src/make_figures.py                 # Figure 1, deliberately the nominal adaptive fit
python src/s16_figure5_real_vs_nominal.py  # Figure 5, from levels_elasticity.json
python src/make_figureS4_fixed.py          # Fig. S4, deliberately the nominal fixed fit
python src/make_sensitivity_figure.py results/trace_fixed005.nc results/trace_adaptive.nc

# Figs. S1-S3 diagnose the deployed REAL adaptive posterior, so the outcome column must match it.
GDP_COL=GDP_constant_2015usd python src/make_si_diagnostics.py results/trace_adaptive_real.nc

# One per-parameter convergence table per posterior; the second argument tags the output file.
python src/make_convergence_table.py results/trace_fixed005_real.nc        _real_fixed_d10k
python src/make_convergence_table.py results/trace_adaptive_real.nc        _real
python src/make_convergence_table.py results/trace_adaptive_real_noGNQ.nc  _real_adaptive_noGNQ
python src/make_convergence_table.py results/trace_fixed005.nc             _nominal_fixed
python src/make_convergence_table.py results/trace_adaptive.nc             _nominal_adaptive
python src/make_convergence_table.py results/hierarchical_wa_posterior.nc  _wa
```

Four notes on the order.

1. The projection guard suite is listed **before** the projection runs, which is where it belongs:
   it exercises the engine, including the case that must abort, so a failure stops the pipeline
   before figures and tables are regenerated. It writes and then deletes its own temporary outputs.
2. `src/s09_decision_layer.py` requires the two counterfactual runs and the fitted run to exist,
   and `src/s11_make_tables.py` requires all six projection runs, so Stage 6 must run in the order
   shown.
3. The frontier-rate sensitivity runs are tagged by the annual rate they imply — 0.150 per decade
   is 1.51% a year, 0.200 per decade is 2.02% a year — while the command-line argument is the
   decadal value.
4. The elasticity comparison figure of the five-year model is written by
   `src/fit_hierarchical_5yr.py` under the filename `Figure5_regional_elasticity_5yr_forest`, and
   the same figure is archived and cited in the paper as Supplementary Fig. S6. The script's output
   filename has not been changed, so expect that name on disk.

Two quantities are recorded rather than recomputed, and a reader checking reproduction should
compare against them rather than re-deriving them: the sampler settings and diagnostics of the
backtest are in `results/gdp_backtest_metrics.json` (4 chains, 4,000 warmup, 3,000 post-warmup
draws, target acceptance 0.98, seed 42), and the settings and assertion results of every projection
run are in the corresponding `results/projection_manifest*.json`.

**If you only want the projection, the decision layer, the tables and Figures 3–6**, you need
`hierarchical_wa_posterior.nc` alone. Download it from the Zenodo deposit into `results/`, then run
Stages 6 and 7 — about five minutes in total.

## What is deliberately *not* fitted on the real series

Two figures diagnose the superseded nominal fit on purpose, because the paper's argument is that
the nominal fit is wrong and it shows you the thing it is arguing about:

- **Figure 1** is the nominal adaptive fit (`src/make_figures.py`), and its legend says so.
- **Supplementary Fig. S4** is the nominal fixed-penalty fit (`src/make_figureS4_fixed.py`).

Supplementary Figs. S1–S3 diagnose the deployed **real** adaptive posterior.
`src/make_si_diagnostics.py` refuses to draw the posterior-predictive check at all if the posterior
and the outcome column disagree by more than 0.15 in log₁₀ — swapping the two columns moves it by
0.322 — and records what it used in `results/si_diagnostics_manifest.json`.

## Data sources

| Series | Source | Accessed |
|:---|:---|:---|
| Real GDP (constant 2015 US$), nominal GDP (current US$), total population | World Bank, [World Development Indicators](https://databank.worldbank.org/source/world-development-indicators) | 18 May 2026 |
| Population and age-structure projections, medium variant | United Nations, [World Population Prospects 2024](https://population.un.org/wpp/) | 19 May 2026 |

Both are openly redistributable; no access restrictions apply. `data/` holds the processed,
analysis-ready panel derived from them.

## Archive

The posterior sample files (seven files, about 10.0 GB of NetCDF) and the complete result set are permanently
archived on Zenodo. The DOI is added here on release of this version; see `results/POSTERIORS.md`
for what each file is and which stage needs it.

## Citing

See `CITATION.cff`. The code is released under the MIT licence (`LICENSE`); the data are
redistributed under the terms of their original sources.
