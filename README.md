# Complex explicit Runge–Kutta methods of orders five and six and complex Cash–Karp pairs

Code and data of the paper

> V. Rybin, I. Babkin, A. Karimov, D. Butusov. *Complex Explicit Runge–Kutta Methods of Orders Five
> and Six with as Many Stages as the Order, and Complex Cash–Karp Pairs.* Mathematics (MDPI),
> [??: year, volume, DOI].

The methods are explicit Runge–Kutta methods with complex coefficients applied to a real system
y' = f(t, y); after every step only the real part of the result is kept. The repository contains all
tableaux with 45 digits, the verification of the order conditions, the structure of the solution sets
(ranks, D(1), the interval Krawczyk test, the fifth-order formulas b + θd of the CCK6 pairs), the
minimization of the error constants, all fixed- and variable-step experiments, the C++ timing
benchmark, the latency model, and the scripts that generate every table and figure of the paper.

## Layout

| Path | Contents |
|---|---|
| `code/base/code/` | pipeline of the original computations (`crk.py`: order conditions and integrator in multiple precision; `tableaux.py`; `problems.py`; `measure.py`; `adaptive.py`; `optimize_E.py`; `tune_cck6.py`; ...) |
| `code/base/data/` | its data; `tableaux.json` holds the coefficients of all tableaux with 45 digits |
| `code/r1/code/` | scripts of the revision (`*_r1.py`); `r1common.py` sets all paths |
| `code/r1/data/` | their data; `merged/` holds the original data together with the new real pairs |
| `paper/tables/`, `paper/figures/` | the tables (LaTeX rows) and figures of the paper |
| `code/r1/code/bench/` | C++ timing benchmark (`bench.cpp`, `coeffs.h`, `build.bat`, `run_bench_r1.sh`) |

## Contents

The paper cites the scripts by name. A script named `*_r1.py` is in `code/r1/code/`, every other script
in `code/base/code/`; a data file is in the `data/` folder next to the script that writes it.
Section, proposition and appendix numbers refer to the MDPI version of the paper.

| Topic | What | Files |
|---|---|---|
| Coefficients | coefficients of all tableaux with 45 digits | `code/base/data/tableaux.json`, `code/r1/data/tableaux_r1.json` |
| Order conditions | computer-algebra check of the proof of Butcher's barrier over C (Proposition 2, Appendix C) | `code/base/code/barrier_proof_check.py` |
| Order conditions | refinement of the tableaux; residuals, error constants, ranks, stability polynomials | `code/base/code/check_tableaux.py` |
| Order conditions | coefficients rounded to double precision | `code/r1/code/roundoff_r1.py` |
| Order conditions | checks with the full-precision coefficients | `code/r1/code/checks45_r1.py` |
| Structure (Sect. 3) | singular-value spectra in 90 digits, continuation along the tangent directions | `code/r1/code/spectra_r1.py`, `code/r1/code/structure_r1.py` |
| Structure (Sect. 3) | projected D(1) and the Krawczyk test (Propositions 5, 6) | `code/r1/code/d1_r1.py`, `code/r1/code/krawczyk_r1.py` |
| Structure (Sect. 3) | Jacobian ranks and embedded kernels at all visited tableaux | `code/base/code/rank_survey.py` |
| Structure (Sect. 3) | fifth-order formulas b + θd and the reality of c6 (Proposition 7) | `code/r1/code/cck6_identity_r1.py` |
| Structure (Sect. 3) | rank certificates of the fifth-order conditions (Proposition 7(c)) | `code/r1/code/kernel_proof_r1.py` |
| Structure (Sect. 3) | existence of the pair CCK6(5)-v (Proposition 8) | `code/r1/code/krawczyk_pairs_r1.py`, `code/r1/data/cck6v_r1.json` |
| Structure (Sect. 3) | assumption (W), random walk and proof of Theorem 1 | `code/r1/code/conjecture_r1.py`, `code/r1/code/conjecture_proof_r1.py` |
| Structure (Sect. 3) | assumption (W0) and existence of the cascade (Proposition 10) | `code/r1/code/cascade_proof_r1.py` |
| Structure (Sect. 3) | repetition of all Krawczyk tests in Arb | `code/r1/code/krawczyk_arb_r1.py` |
| Error constants (Sect. 4) | minimization under a coefficient bound, fronts, degenerate branches | `code/base/code/optimize_E.py`, `code/base/code/front_E.py`, `code/base/code/polish_opt.py`, `code/base/code/degenerate_check.py` |
| Embedded formulas (Sect. 5) | embedded formulas and the pairs CRK5(4), CRK6(4) | `code/base/code/embedded.py` |
| Embedded formulas (Sect. 5) | fourth-order formulas of Proposition 11 and the cascades of the pairs | `code/r1/code/emb4_r1.py` |
| Embedded formulas (Sect. 5) | embedded formulas of the real seven-stage tableaux, pair RK6-opt(4) | `code/r1/code/real64_r1.py` |
| Stability (Sect. 6) | stability and preference regions | `code/base/code/stability.py`, `code/base/code/prefregions.py`, `code/r1/code/stability_r1.py` |
| Stability (Sect. 6) | harmonic oscillator and Kepler problem over long times | `code/r1/code/longtime_r1.py` |
| Experiments (Sect. 7) | reference solutions | `code/base/data/references.json` |
| Experiments (Sect. 7) | fixed-step measurements of the error constants in multiple precision | `code/base/code/measure.py`, `code/r1/code/measure_r1.py` |
| Experiments (Sect. 7) | variable-step runs, CRK6(4) against the real pairs, performance profiles | `code/base/code/adaptive.py`, `code/r1/code/adaptive_r1.py`, `code/r1/code/adaptive_tol15_r1.py`, `code/r1/code/profiles_r1.py` |
| Experiments (Sect. 7) | newer real pairs Tsit5(4), BS5(4), Vern6(5) | `code/r1/code/real_pairs_r1.py`, `code/r1/code/adaptive_new_r1.py`, `code/r1/code/r1patch.py` |
| Experiments (Sect. 7) | Vern6(5) with coefficients split into two doubles | `code/r1/code/vern6_split_r1.py` |
| Experiments (Sect. 7) | shifted tolerance grid; error along the trajectory and perturbed initial data | `code/r1/code/robust_r1.py`, `code/r1/code/traj_r1.py` |
| Experiments (Sect. 7) | Pleiades and outer solar system | `code/r1/code/nbody_r1.py` |
| Experiments (Sect. 7) | BRUSS-2D | `code/r1/code/bruss2d_r1.py` |
| Experiments (Sect. 7) | C++ timing benchmark and its tables | `code/r1/code/bench/`, `code/base/code/bench/`, `code/r1/code/tables_cpu_r1.py` |
| Discussion (Sect. 8) | latency model and break-even ratios ν* | `code/r1/code/latency_r1.py` |
| Discussion (Sect. 8) | GBS(2,4,6) and EX5: tableau, error constant, variable-step runs at equal latency | `code/r1/code/gbs246_r1.py`, `code/r1/code/gbs_r1.py` |
| Discussion (Sect. 8) | real methods with two chains of five stages | `code/r1/code/two_chain_r1.py` |
| Discussion (Sect. 8) | error constants with a free imaginary error estimate | `code/r1/code/imag_estimate_r1.py` |
| Discussion (Sect. 8) | distances of the stage values to the singularities of f | `code/r1/code/singular_r1.py` |
| Tools | tuning a family to a given problem by moving along its solution set | `code/base/code/tune_cck6.py` |
| Tools | tables, figures and appendix of the paper | `code/r1/code/tables_r1.py`, `code/r1/code/figures_r1.py`, `code/r1/code/make_appendix_r1.py` |

## Running

Python 3.12 with the packages of `requirements.txt`:

```
pip install -r requirements.txt
cd code/base/code
python check_tableaux.py
```

Selected entry points (run from the folder of the script):

| Script | What it does |
|---|---|
| `code/base/code/check_tableaux.py` | refines all tableaux, recomputes residuals, error constants, ranks, stability polynomials |
| `code/base/code/barrier_proof_check.py` | computer-algebra check of the proof of Butcher's barrier over C |
| `code/r1/code/spectra_r1.py`, `d1_r1.py`, `krawczyk_r1.py` | singular values in 90 digits, projected D(1), interval Krawczyk test |
| `code/r1/code/cck6_identity_r1.py` | fifth-order formulas b + θd and the reality of c6 on the CCK6 pairs |
| `code/r1/code/roundoff_r1.py` | order conditions with coefficients rounded to double precision |
| `code/r1/code/tables_r1.py`, `figures_r1.py`, ... | regenerate the tables and figures of the paper into `paper/` |
| `code/base/code/tune_cck6.py` | tunes a family to a user's problem |

The multiple-precision computations (90-digit spectra, Krawczyk test, 34-digit fixed-step runs) take
from minutes to hours on a laptop; the CPU benchmark needs MSVC and an otherwise idle machine.

## License

See `LICENSE`.
