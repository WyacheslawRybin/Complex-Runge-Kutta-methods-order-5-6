"""Two-dimensional Brusselator with diffusion (BRUSS-2D), the large test problem of R1 (review, item 5.4).

  u_t = 1 + u^2 v - 4.4 u + alpha (u_xx + u_yy),   v_t = 3.4 u - u^2 v + alpha (v_xx + v_yy),
  (x, y) in [0, 1]^2, alpha = 2e-3, homogeneous Neumann conditions, u(x, y, 0) = 0.5 + y,
  v(x, y, 0) = 1 + 5x, t in [0, 11.5]  (problem BRUS of Hairer, Norsett, Wanner I, Sect. II.10).

Method of lines on the grid x_i = (i-1)/(N-1), y_j = (j-1)/(N-1), i, j = 1..N, N = 21, with the
five-point Laplacian and the Neumann condition imposed by reflection (u_0j = u_2j, u_{N+1}j = u_{N-1}j):
2 N^2 = 882 equations.  Errors are measured in the root-mean-square norm ||e||_2 / sqrt(882).

Everything runs in double precision:
  * reference solution: Vern6(5) with 16000 and 32000 fixed steps and compensated (Kahan) summation of
    the solution, cross-checked against DOP853 (scipy) with rtol = 1e-13;
  * fixed steps: all methods of the order study, N = N0 * 2^k, with compensated summation; the
    constants C = lim err / h^p by one Richardson step as in code/measure.py, using the step sizes at
    which the error is well above the round-off level;
  * variable steps: all pairs with the controller of code/adaptive.py (adaptive.solve).
Writes R1/data/bruss2d_ref.json and R1/data/bruss2d_r1.json.
Writes tables/bruss2d.tex and R1/data/bruss2d_costs.json (mode table).
Usage: python bruss2d_r1.py [ref|fixed|adaptive|all|table]
"""
import json
import math
import os
import sys
import time

import numpy as np

import r1common as R

NG = 21
ALPHA = 2e-3
TEND = 11.5
DIM = 2 * NG * NG
REF_FN = os.path.join(R.DATA, "bruss2d_ref.json")
OUT_FN = os.path.join(R.DATA, "bruss2d_r1.json")


def _lap(w):
    """Five-point Laplacian with reflecting (Neumann) boundaries; w has shape (NG, NG), axis 0 = x."""
    p = np.empty((NG + 2, NG + 2), w.dtype)
    p[1:-1, 1:-1] = w
    p[0, 1:-1] = w[1]
    p[-1, 1:-1] = w[-2]
    p[1:-1, 0] = w[:, 1]
    p[1:-1, -1] = w[:, -2]
    return (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4.0 * w) * (NG - 1) ** 2


def f_np(t, y):
    u = y[:NG * NG].reshape(NG, NG)
    v = y[NG * NG:].reshape(NG, NG)
    uuv = u * u * v
    du = 1.0 + uuv - 4.4 * u + ALPHA * _lap(u)
    dv = 3.4 * u - uuv + ALPHA * _lap(v)
    return np.concatenate([du.ravel(), dv.ravel()])


def y0():
    x = np.linspace(0.0, 1.0, NG)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.concatenate([(0.5 + Y).ravel(), (1.0 + 5.0 * X).ravel()])


def rms(e):
    return float(np.linalg.norm(e) / math.sqrt(DIM))


# ----------------------------------------------------------------------------- fixed-step integrator
def tableau_arrays(T, L=0):
    m, p, b = T.levels[L]
    A = np.array([[complex(T.A[i][j]) for j in range(m)] for i in range(m)])
    bb = np.array([complex(v) for v in b[:m]])
    if not T.projected:
        A, bb = A.real.copy(), bb.real.copy()
    return A, bb, p


def run_fixed(A, b, projected, N, T=TEND):
    """N uniform steps; the solution is accumulated with compensated summation."""
    s = len(b)
    c = A.sum(1)
    h = T / N
    y = y0()
    comp = np.zeros_like(y)
    dtype = complex if projected else float
    K = np.zeros((s, DIM), dtype)
    nz = [[j for j in range(i) if A[i, j] != 0] for i in range(s)]
    for n in range(N):
        t = n * h
        for i in range(s):
            Yi = y + h * sum(A[i, j] * K[j] for j in nz[i]) if i else y
            K[i] = f_np(t + c[i].real * h, Yi)
        inc = h * sum(b[i] * K[i] for i in range(s) if b[i] != 0)
        if projected:
            inc = inc.real
        # Kahan summation y <- y + inc
        z = inc - comp
        t2 = y + z
        comp = (t2 - y) - z
        y = t2
        if not np.all(np.isfinite(y)) or np.abs(y).max() > 1e8:
            return None
    return y


# ----------------------------------------------------------------------------- reference
def reference(recompute=False):
    if os.path.exists(REF_FN) and not recompute:
        return np.array(json.load(open(REF_FN))["y"])
    from tableaux import load_refined
    T = load_refined(os.path.join(R.DATA, "tableaux_r1.json"))["Vern6(5)"]
    A, b, _ = tableau_arrays(T)
    t0 = time.time()
    y1 = run_fixed(A, b, False, 16000)
    y2 = run_fixed(A, b, False, 32000)
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda t, y: f_np(t, y), (0.0, TEND), y0(), method="DOP853", rtol=1e-13, atol=1e-13)
    y3 = sol.y[:, -1]
    d12, d23 = rms(y1 - y2), rms(y2 - y3)
    print(f"reference: Vern6 16000 vs 32000 steps {d12:.2e}, Vern6 32000 vs DOP853 {d23:.2e} "
          f"({time.time() - t0:.0f}s); max|y| = {np.abs(y2).max():.3f}")
    json.dump(dict(y=[float(v) for v in y2], vern6_16000_vs_32000=d12, vern6_vs_dop853=d23,
                   note="Vern6(5), 32000 fixed steps, compensated summation, double precision"),
              open(REF_FN, "w"), indent=0)
    return y2


# ----------------------------------------------------------------------------- experiments
N0 = 50
KS = list(range(0, 8))           # N = 50 ... 6400


def _all_tableaux():
    from tableaux import load_refined
    TT = load_refined()
    TT.update(load_refined(os.path.join(R.DATA, "tableaux_r1.json")))
    return TT


def fixed_task(args):
    label, key, L = args
    TT = _all_tableaux()
    ref = reference()
    T = TT[key]
    A, b, p = tableau_arrays(T, L)
    rows = []
    for k in KS:
        N = N0 * 2 ** k
        y = run_fixed(A, b, T.projected, N)
        e = None if y is None else rms(y - ref)
        rows.append(dict(N=N, err=e, Ch=None if e is None else e / (TEND / N) ** p))
    return label, rows


def adaptive_task(name):
    import adaptive as AD
    import adaptive_new_r1 as AN
    P = AD.load_pairs()
    P.update(AN.new_pairs())
    ref = reference()
    rows = []
    for tol in AD.TOLS:
        out = AD.solve(P[name], f_np, y0(), TEND, tol)
        if out is None:
            rows.append(dict(tol=tol, err=None))
            continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=rms(y - ref), nfev=nf, nacc=na, nrej=nr))
    return name, rows


FIXED_METHODS = None  # set in main from measure.METHODS + the new pairs
PAIRS = ["CRK5(4)", "CRK6(4)", "CRK5-s(4)", "CRK6-s(4)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc",
         "DP5", "CK5", "Tsit5(4)", "BS5(4)", "DP6(5)", "Verner6(5)", "Vern6(5)"]


def main(what="all", nproc=6):
    from multiprocessing import Pool
    import measure as M
    reference()
    res = json.load(open(OUT_FN)) if os.path.exists(OUT_FN) else {}
    t0 = time.time()
    if what in ("fixed", "all"):
        methods = M.METHODS + [("Tsit5(4)", "Tsit5(4)", 0), ("BS5(4)", "BS5(4)", 0), ("Vern6(5)", "Vern6(5)", 0)]
        res["fixed"] = {}
        with Pool(nproc) as pool:
            for label, rows in pool.imap_unordered(fixed_task, methods):
                res["fixed"][label] = rows
                print(f"[{time.time() - t0:5.0f}s] fixed {label:12s} "
                      + " ".join("-" if r["err"] is None else f"{r['err']:.2e}" for r in rows), flush=True)
        json.dump(res, open(OUT_FN, "w"), indent=1)
    if what in ("adaptive", "all"):
        res["adaptive"] = {}
        with Pool(nproc) as pool:
            for name, rows in pool.imap_unordered(adaptive_task, PAIRS):
                res["adaptive"][name] = rows
                ok = [r for r in rows if r["err"] is not None]
                print(f"[{time.time() - t0:5.0f}s] adaptive {name:11s} tol 1e-13: err={ok[-1]['err']:.2e} "
                      f"nfev={ok[-1]['nfev']}  tol 1e-3: nfev={ok[0]['nfev']} rej={ok[0]['nrej']}", flush=True)
        json.dump(res, open(OUT_FN, "w"), indent=1)


# ----------------------------------------------------------------------------- table
ROWS5 = [("CRK5-opt", "CRK5(4)"), ("DP5", "DP5"), ("CK5", "CK5"), ("Tsit5(4)", "Tsit5(4)"), ("BS5(4)", "BS5(4)")]
ROWS6 = [("CRK6-opt", "CRK6(4)"), ("CCK6(5)", "CCK6(5)"), ("CCK6(5)-s", "CCK6(5)-s"), ("CCK6-casc", "CCK6-casc"),
         ("Butcher6", None), ("RK6-opt", "RK6-opt(4)"), ("DP6(5)", "DP6(5)"), ("Verner6(5)", "Verner6(5)"),
         ("Vern6(5)", "Vern6(5)")]
EPS = (1e-6, 1e-8, 1e-10)


def table():
    """tables/bruss2d.tex: right-hand side evaluations needed for the RMS accuracies 1e-6, 1e-8, 1e-10 at
    t = 11.5, with fixed steps (evaluations per step times N, interpolated along the work-precision curve)
    and with step-size control (including rejected steps); kappa = 1.  Green: the cheapest within each
    order and column.  Also prints the comparisons quoted in the text."""
    import r1patch
    MT = r1patch.apply()
    import figures_cost as FC
    res = json.load(open(OUT_FN))
    r64 = os.path.join(R.DATA, "adaptive_real64_r1.json")      # real64_r1.py
    if os.path.exists(r64):
        res["adaptive"]["RK6-opt(4)"] = json.load(open(r64))["RK6-opt(4)"]["Bruss2D"]
    ev = dict(FC.EVALS, **{"CCK6(5)-s": 6})

    def fixed_cost(m, eps):
        rows = [r for r in res["fixed"][m] if r["err"] is not None and r["err"] > 0]
        return FC.cost_at(np.array([ev[m] * r["N"] for r in rows], float), np.array([r["err"] for r in rows]), eps)

    def adapt_cost(m, eps):
        rows = [r for r in res["adaptive"][m] if r["err"] is not None and r["err"] > 0]
        return FC.cost_at(np.array([r["nfev"] for r in rows], float), np.array([r["err"] for r in rows]), eps)

    lines, num = [], {}
    for g, rows in enumerate((ROWS5, ROWS6)):
        cols = [[fixed_cost(m, e) for m, _ in rows] for e in EPS] + \
               [[adapt_cost(a, e) if a else None for _, a in rows] for e in EPS]
        cells = []
        for vals in cols:
            fin = [v if v is not None and math.isfinite(v) else None for v in vals]
            cells.append(MT.mark_best(fin, ["---" if v is None else f"{v:.0f}" for v in fin], rel=0))
        for i, (m, a) in enumerate(rows):
            name = MT.mname(m) + (f" / {MT.mname(a)}" if a and a != m else "")
            lines.append(name + " & " + " & ".join(c[i] for c in cells) + r"\\")
            num[m] = dict(fixed=[cols[k][i] for k in range(3)], adaptive=[cols[3 + k][i] for k in range(3)])
        if g == 0:
            lines.append(r"\midrule")
    open(os.path.join(R.TAB, "bruss2d.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    r = lambda a, b: a / b if a and b and math.isfinite(a) and math.isfinite(b) else float("nan")
    for k, e in enumerate(EPS):
        print(f"eps {e:g}: fixed CRK6-opt/Vern6 {r(num['CRK6-opt']['fixed'][k], num['Vern6(5)']['fixed'][k]):.3f}"
              f"  CRK6-opt/DP6 {r(num['CRK6-opt']['fixed'][k], num['DP6(5)']['fixed'][k]):.3f}"
              f"  CRK5-opt/BS5 {r(num['CRK5-opt']['fixed'][k], num['BS5(4)']['fixed'][k]):.3f}"
              f" | adaptive CRK6(4)/Vern6 {r(num['CRK6-opt']['adaptive'][k], num['Vern6(5)']['adaptive'][k]):.3f}"
              f"  CRK6(4)/DP6 {r(num['CRK6-opt']['adaptive'][k], num['DP6(5)']['adaptive'][k]):.3f}"
              f"  CRK5(4)/BS5 {r(num['CRK5-opt']['adaptive'][k], num['BS5(4)']['adaptive'][k]):.3f}")
    json.dump(num, open(os.path.join(R.DATA, "bruss2d_costs.json"), "w"), indent=1,
              default=lambda o: None if not math.isfinite(o) else o)


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w == "ref":
        reference(recompute=True)
    elif w == "table":
        table()
    else:
        main(w, int(sys.argv[2]) if len(sys.argv) > 2 else 6)
