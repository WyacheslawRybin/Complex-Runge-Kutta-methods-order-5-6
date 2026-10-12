# tune_cck6.py: tuning a complex Runge–Kutta tableau to one problem

This repository contains the tuning script from Appendix C of the paper

> V. Rybin, I. Babkin, A. Karimov, D. Butusov. *The Butcher barrier is only real:
> complex Runge–Kutta methods of orders five and six and a complex Cash–Karp pair.*
> [??: journal, year, DOI]

The script implements the procedure of Section 7.7 of the paper. It takes one of the
explicit Runge–Kutta families with complex coefficients and real projection studied
there, and moves the tableau along the solution set of the order conditions so that
the global error on **your** problem decreases. The order of every level stays exact
(residuals of about 1e-16) during the whole descent.

## Background

The methods are explicit Runge–Kutta methods with complex coefficients applied to a
real system $y' = f(t, y)$. After every step only the real part of the result is
kept:

$$y_{n+1} = \mathrm{Re}\Big(y_n + h \sum_i b_i k_i\Big).$$

The order conditions of such a method do not determine its coefficients uniquely.
Every family has a manifold of solutions, and the local dimension of this manifold is
the number of free directions the script can use:

| `LEVELS` | method | stages, order of the levels | real unknowns | conditions | free directions |
|---|---|---|---|---|---|
| `"CCK65"` | complex Cash–Karp pair CCK6(5) | 6/6 and 5/5 | 50 | 54 | 4 |
| `"CCK6-cascade"` | cascade CCK6 6(5)(4)(3)(2)(1) | 6/6, 5/5, …, 1/1 | 66 | 69 | 6 |
| `"CRK5"` | CRK5, single level | 5/5 | 29 | 17 | 12 |
| `"CRK6"` | CRK6, single level | 6/6 | 41 | 37 | 10 |

The number of conditions can exceed the number of unknowns because some conditions
are dependent (see Sections 3.2 and 3.3 of the paper).

The leading error constant $E_{p+1}$, the norm of the error coefficients over the trees
of order $p+1$, does not tell which member of a family is the best on a given problem.
The actual leading error is a projection of the error coefficients onto the problem.
The script therefore minimizes the error measured on the problem itself.

## Requirements

Python 3 and NumPy, nothing else. Tested with Python 3.12 and NumPy 2.5.

## Quick start

```bash
python tune_cck6.py
```

This tunes the pair CCK6(5) for the Rössler system on $[0, 5]$ with $N = 100$ steps,
which is the example of the paper. The family and the step count can also be given on
the command line:

```bash
python tune_cck6.py CRK6
python tune_cck6.py CCK6-cascade 200
```

## Setting up your problem

Everything problem-specific is in the block `USER INPUT` at the top of the script:

| variable | meaning | default |
|---|---|---|
| `f(t, y)` | right-hand side; returns a NumPy array | Rössler system, $a = b = 0.2$, $c = 5.7$ |
| `y0` | initial state (real) | `[1, 1, 1]` |
| `T` | end of the integration interval (start at $t = 0$) | `5.0` |
| `N` | number of steps you intend to use | `100` |
| `N_LIST` | step counts averaged in the objective | `(N/2, N, 2N)` |
| `LEVELS` | family, see the table above | `"CCK65"` |
| `MAX_COEF` | bound on $\max\lvert a_{ij}\rvert$ and $\max\lvert b_i\rvert$ over all levels | `5.0` |
| `EMB_WEIGHT` | weight of the error of the embedded fifth-order solution in the objective (0 = ignore) | `0.0` |
| `ROUNDS` | number of descent rounds | `25` |

The right-hand side is evaluated at **complex** stage values. It must have real
coefficients and must accept complex `t` and `y`. Use NumPy functions (`np.sin`,
`np.exp`, …) and avoid `math.*`, `abs()` and branches on the sign of a component,
since these either fail on complex arguments or break analyticity.

For example, the Lorenz system:

```python
def f(t, y):
    return np.array([10.0 * (y[1] - y[0]),
                     y[0] * (28.0 - y[2]) - y[1],
                     y[0] * y[1] - 8.0 / 3.0 * y[2]])

y0 = np.array([1.0, 1.0, 20.0])
T = 0.5
N = 20
```

Keep `N_LIST` with several step counts. With a single step count the descent learns
to cancel the leading error term against the next one at this step count. This gives
large gains at that $N$ that disappear at $N/2$ and $2N$ (Section 7.7 of the paper).

## Output

The script prints:

* the number of unknowns and conditions and the residual of the order conditions at
  the starting tableau;
* the global errors of the starting tableau at the step counts of `N_LIST`, its
  $E_{p+1}$ and $\max\lvert\text{coef}\rvert$;
* one line per descent round: the objective (the error relative to the starting
  tableau), the dimension of the tangent space, $\max\lvert\text{coef}\rvert$ and the
  current step;
* the errors of the tuned tableau, the improvement factors, $E_{p+1}$ before and after,
  and the largest residual of the order conditions of every level;
* the tuned coefficients $a_{ij}$ and the weights $w_i^{(m)}$ of every level $m$, with
  18 digits.

The tuned tableau is also saved to `tuned_<LEVELS>.npy` as the packed vector of real
unknowns. To recover the matrix $A$ and the weights, import the script with the same
`LEVELS` and unpack the vector:

```python
import numpy as np
import tune_cck6 as tc            # LEVELS in USER INPUT must be the same as in the run
A, W = tc.unpack(np.load("tuned_CCK65.npy"))
```

## Results for the example

The table below shows the output for the Rössler example ($T = 5$, $N = 100$,
`MAX_COEF = 5`). It reproduces the tuning table of Section 7.7 of the paper. The
starting tableaux are CRK5-opt, CRK6-opt, CCK6(5) and CCK6-cascade from Appendix A of
the paper. The run times were measured on an Intel Core i7-10750H.

| `LEVELS` | improvement at $N = 50$ | $N = 100$ | $N = 200$ | $E_{p+1}$ before | $E_{p+1}$ after | time |
|---|---|---|---|---|---|---|
| `CCK65` | 3.34 | 27.57 | 3.56 | 7.61e-4 | 7.57e-4 | 10 s |
| `CCK6-cascade` | 5.32 | 7.78 | 5.84 | 9.16e-4 | 8.87e-4 | 17 s |
| `CRK5` | 7.02 | 20.07 | 13.37 | 1.39e-3 | 7.66e-3 | 17 s |
| `CRK6` | 6.33 | 8.69 | 8.17 | 3.22e-4 | 7.76e-4 | 19 s |

Note that the tuned CRK5 and CRK6 tableaux have larger $E_{p+1}$ than the starting
ones, while their error on the Rössler system is 6–20 times smaller. For CCK6(5) the
gain of 27.6 at $N = 100$ comes mostly from the cancellation described above. Its
asymptotic error constant on this problem improves only by 16%.

## How it works

1. **Unknowns.** The real and imaginary parts of $a_{ij}$ and of the weights of all
   levels. $\mathrm{Im}\, b_1$ of every level is dropped: the first stage
   $k_1 = f(t_n, y_n)$ is real, so this coefficient does not affect the projected
   solution.
2. **Order conditions.** For every level with $m$ stages and order $p$,
   $\mathrm{Re}(b^{\mathsf T} \Phi_t) = 1/\gamma(t)$ for all rooted trees $t$ with at
   most $p$ nodes. The elementary weights $\Phi_t$ and their exact derivatives with
   respect to $a_{ij}$ are computed recursively, which gives the exact Jacobian.
3. **Projection.** After every move the point is returned to the solution set by
   minimum-norm Newton iterations (least squares with the Jacobian).
4. **Tangent space.** The null space of the Jacobian, found from its singular value
   decomposition with the relative threshold 1e-10.
5. **Descent.** Central directional differences of the objective along a basis of the
   tangent space ($\varepsilon = 10^{-3}$, each perturbed point projected back), then a
   backtracking line search along the negative gradient. A trial point is accepted
   only if the residual of the order conditions is below 1e-12 and
   $\max\lvert\text{coef}\rvert \le$ `MAX_COEF`.
6. **Objective.** The geometric mean over `N_LIST` of the global error at $T$ divided
   by the error of the starting tableau. The reference solution is computed once by
   the starting tableau with $32N$ steps (16 times the largest step count of
   `N_LIST`).

## Limitations

* The integrator works in double precision. If the error at the working step count is
  below about 1e-13, round-off dominates the objective: use a larger step or rewrite
  `integrate` with `mpmath`.
* The descent finds a local minimum near the starting tableau.
* A tuned tableau is tuned for one problem, one interval and one range of step counts.
  It gives no guarantees for other problems, and its stability region differs from that
  of the starting tableau.
* For the pairs, all levels keep their order, but the quality of the error estimate is
  not optimized unless `EMB_WEIGHT > 0`.

## Citation

If you use this code, please cite our paper:

> [??: full reference of the paper: authors, title, journal, volume, pages, year, DOI]

```bibtex
@article{[??: key],
  author  = {Rybin, V. and Babkin, I. and Karimov, A. and Butusov, D.},
  title   = {The {Butcher} barrier is only real: complex {Runge--Kutta} methods of orders five and six and a complex {Cash--Karp} pair},
  journal = {[??]},
  volume  = {[??]},
  pages   = {[??]},
  year    = {[??]},
  doi     = {[??]}
}
```

## License

MIT, see [LICENSE](LICENSE).
