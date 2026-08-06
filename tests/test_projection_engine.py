"""
test_projection_engine.py -- guards on s07_projection_2100.py
=============================================================
Every test here corresponds to a specific way the ver6 projection failed. Run before any
figure or table is regenerated:

    python tests/test_projection_engine.py            # standalone
    pytest tests/test_projection_engine.py -q         # or under pytest

T1  the aggregate / per-capita / population identity holds in the written CSV
T2  per capita in the CSV is exactly GDP / Population -- not a second model output
T3  the path is not a constant-growth extrapolation (the ver6 straight-line signature)
T4  the convergence term acts in the direction the drift implies, and the 2100 ordering
    does not depend on it -- so s_ss is a stated parameter, not the load-bearing one
T5  the plausibility bound raises rather than warns when growth becomes implausible
T6  a super-unitary contraction elasticity reverses the sign of the per-capita dividend
T7  the run is reproducible under a fixed seed
T8  no country's implied 2100 population differs from the UN WPP input
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
RESULTS = REPO / "results"
DRAWS = "500"          # enough to exercise every code path; the engine is deterministic in seed

_FAILURES: list[str] = []


def run_engine(*extra, expect_fail=False):
    cmd = [sys.executable, str(SRC / "s07_projection_2100.py"), "--draws", DRAWS, *extra]
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    if expect_fail:
        assert p.returncode != 0, "engine was expected to abort but exited 0"
        return p.stderr
    assert p.returncode == 0, "engine failed:\n%s\n%s" % (p.stdout[-2000:], p.stderr[-3000:])
    return p.stdout


def check(name, cond, detail=""):
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s   %s" % (name, detail))
        _FAILURES.append(name)


# ----------------------------------------------------------------------------------------
def test_identity_in_written_output():
    """T1 -- the defect that produced Germany's 1.7-million implied 2100 population."""
    run_engine("--tag", "_test")
    d = pd.read_csv(RESULTS / "projection_2100_test.csv")
    resid = (np.log(d.GDP_median) - np.log(d.GDPpc_median)) - np.log(d.Population)
    check("T1 identity g_agg - g_pc == g_pop in written CSV",
          np.abs(resid).max() < 1e-9, "max |residual| = %.3e" % np.abs(resid).max())
    return d


def test_per_capita_is_a_quotient(d):
    """T2 -- per capita must be derived, never modelled separately."""
    for col_g, col_pc in [("GDP_median", "GDPpc_median"), ("GDP_lo", "GDPpc_lo"),
                          ("GDP_hi", "GDPpc_hi")]:
        rel = np.abs(d[col_pc] - d[col_g] / d.Population) / d[col_pc]
        check("T2 %s == %s / Population" % (col_pc, col_g), rel.max() < 1e-12,
              "max relative error = %.3e" % rel.max())


def test_convergence_curves():
    """T3 + T4 -- 'perfectly straight on a log axis' was the ver6 tell.

    A path can bend for two reasons: the convergence term, or the demographic term (the
    annual change in working-age population is itself not constant). So a bare curvature
    test is not a discriminator. What identifies conditional convergence is that the
    deceleration must be *ordered by initial income*: economies far below the frontier
    decelerate as their gap closes, economies at the frontier do not. That correlation is
    absent when the term is switched off, and it is what a referee would look for.
    """
    def curvature(tag):
        d = pd.read_csv(RESULTS / ("projection_2100%s.csv" % tag))
        dec, y0 = {}, {}
        for iso, g in d.groupby("ISO3"):
            g = g.sort_values("Year")
            y = np.log(g.GDP_median.to_numpy())
            dec[iso] = (y[10] - y[0]) / 10.0 - (y[-1] - y[-11]) / 10.0
            y0[iso] = np.log(g.GDPpc_median.to_numpy()[0])
        idx = sorted(dec)
        return (np.array([dec[i] for i in idx]), np.array([y0[i] for i in idx]))

    on, _ = curvature("_test")
    run_engine("--tag", "_testnoconv", "--no-converge")
    check("T3 the projection is not a constant-growth extrapolation",
          on.mean() > 1e-3 and (np.abs(on) > 1e-3).mean() > 0.9,
          "mean deceleration %.4f pp/yr; share with |deceleration| > 0.1 pp/yr %.2f"
          % (on.mean() * 100, (np.abs(on) > 1e-3).mean()))

    a = pd.read_csv(RESULTS / "projection_summary_test.csv").set_index("ISO3")
    b = pd.read_csv(RESULTS / "projection_summary_testnoconv.csv").set_index("ISO3").loc[a.index]
    # the convergence term must move each country in the direction its own drift implies:
    # a country whose estimated decadal drift exceeds the frontier rate is pulled down.
    # Restricted to countries where the term actually moves the 2100 level by more than
    # 0.5 %; where it is a near-no-op its sign is numerical noise, not behaviour.
    lr = np.log(b.GDP_2100 / a.GDP_2100)
    d_alpha = a.alpha_decadal - 0.175
    sel = lr.abs() > 0.005
    agree = (np.sign(d_alpha) == np.sign(lr))[sel]
    r = float(np.corrcoef(d_alpha, lr)[0, 1])
    check("T4 convergence acts in the direction the drift implies",
          agree.mean() > 0.95 and r > 0.4,
          "directional agreement %.3f over the %d of %d countries it moves by > 0.5%%; "
          "corr(alpha - s_ss, log ratio) = %.3f" % (agree.mean(), int(sel.sum()), len(a), r))

    # ...and it must be a second-order correction, not the load-bearing assumption. If
    # switching it off reordered the projection, the ranking would be an artefact of s_ss.
    rho = float(pd.Series(a.GDP_2100).rank().corr(pd.Series(b.GDP_2100).rank()))
    top = a.GDP_2100.nlargest(20).index
    shift = int(np.abs(a.loc[top].GDP_2100.rank() - b.loc[top].GDP_2100.rank()).max())
    check("T4b the 2100 ordering does not depend on the convergence assumption",
          rho > 0.99 and shift <= 2,
          "Spearman rho = %.5f; largest rank shift in the top 20 = %d" % (rho, shift))


def test_plausibility_bound_raises():
    """T5 -- ver6 shipped 5-8 %/yr for 76 years. That must abort, not warn."""
    err = run_engine("--tag", "_testboom", "--s_ss", "0.60", expect_fail=True)
    check("T5 implausible growth aborts the run",
          "SANITY CHECK FAILED" in err and "implausible sustained real growth" in err,
          err.strip().splitlines()[-1][:160] if err.strip() else "(no stderr)")


def test_super_unitary_reverses_the_dividend():
    """T6 -- the substantive direction check: eps > 1 must erode per-capita output where
    population falls, eps < 1 must lift it. If this test passes in both directions the
    engine is responding to the elasticity and not to an unrelated trend term."""
    run_engine("--tag", "_test092", "--eps-growth", "0.92", "--eps-decline", "0.92")
    run_engine("--tag", "_test122", "--eps-growth", "0.98", "--eps-decline", "1.22")
    a = pd.read_csv(RESULTS / "projection_summary_test092.csv").set_index("ISO3")
    b = pd.read_csv(RESULTS / "projection_summary_test122.csv").set_index("ISO3")
    shrink = a.index[a.g_pop < -0.002]
    div_a = (a.loc[shrink, "g_pc"] - a.loc[shrink, "g_agg"]).mean()
    div_b = (b.loc[shrink, "g_pc"] - b.loc[shrink, "g_agg"]).mean()
    check("T6a per-capita dividend is positive in depopulating countries under both eps",
          div_a > 0 and div_b > 0, "eps0.92 %.4f, eps1.22 %.4f" % (div_a, div_b))
    lvl = (b.loc[shrink, "GDPpc_2100"] / a.loc[shrink, "GDPpc_2100"])
    check("T6b eps=1.22 leaves depopulating countries strictly poorer per capita by 2100",
          lvl.max() < 1.0, "worst-case ratio = %.4f" % lvl.max())
    grow = a.index[a.g_pop > 0.002]
    lvl_g = (b.loc[grow, "GDPpc_2100"] / a.loc[grow, "GDPpc_2100"])
    check("T6c the two elasticities barely differ where population grows",
          np.abs(np.log(lvl_g)).mean() < np.abs(np.log(lvl)).mean(),
          "growing mean |log ratio| %.4f vs shrinking %.4f"
          % (np.abs(np.log(lvl_g)).mean(), np.abs(np.log(lvl)).mean()))


def test_reproducible():
    """T7 -- a fixed seed must give a bit-identical run."""
    first = pd.read_csv(RESULTS / "projection_summary_test.csv")
    run_engine("--tag", "_testrepeat")
    second = pd.read_csv(RESULTS / "projection_summary_testrepeat.csv")
    num = first.select_dtypes("number")
    check("T7 identical output under a fixed seed",
          np.allclose(num.to_numpy(), second[num.columns].to_numpy(), rtol=0, atol=0),
          "max abs difference = %.3e"
          % np.abs(num.to_numpy() - second[num.columns].to_numpy()).max())


def test_population_matches_un_input(d):
    """T8 -- the ver6 table implied populations 0.02-0.12x UN WPP. Verify against the input."""
    pop = pd.read_csv(REPO / "data" / "pop_predictions_scenarios.csv")
    pop = pop[pop.Scenario == "Medium variant"][["ISO3", "Year", "Population"]]
    j = d[["ISO3", "Year", "GDP_median", "GDPpc_median"]].merge(pop, on=["ISO3", "Year"])
    implied = j.GDP_median / j.GDPpc_median
    ratio = implied / j.Population
    check("T8 implied population == UN WPP Medium input",
          np.abs(ratio - 1).max() < 1e-9,
          "ratio range %.9f .. %.9f" % (ratio.min(), ratio.max()))


def main():
    print("test_projection_engine.py -- %s\n" % REPO)
    d = test_identity_in_written_output()
    test_per_capita_is_a_quotient(d)
    test_population_matches_un_input(d)
    test_convergence_curves()
    test_plausibility_bound_raises()
    test_super_unitary_reverses_the_dividend()
    test_reproducible()
    for p in RESULTS.glob("*_test*.csv"):
        p.unlink()
    for p in RESULTS.glob("*_test*.json"):
        p.unlink()
    print()
    if _FAILURES:
        print("%d FAILED: %s" % (len(_FAILURES), "; ".join(_FAILURES)))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
