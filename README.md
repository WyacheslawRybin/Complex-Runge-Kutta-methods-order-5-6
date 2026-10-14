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

In the paper, a script named `*_r1.py` is in `code/r1/code/`, every other script in
`code/base/code/`; a data file is in the `data/` folder next to the script that writes it.

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
| `code/base/code/tune_cck6.py` | tunes a family to a user's problem (Appendix D of the paper) |

The multiple-precision computations (90-digit spectra, Krawczyk test, 34-digit fixed-step runs) take
from minutes to hours on a laptop; the CPU benchmark needs MSVC and an otherwise idle machine.

## License

See `LICENSE`.
