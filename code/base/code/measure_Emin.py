"""
Measured error constants of the E-minimized tableaux (data/tableaux_Emin.json) on the
13 fixed-step problems, exactly as measure.py does for the tableaux of the paper
(34 digits, N = N0*2^k, k = -2..6, extended to k = 7, 8 where the extrapolation has
not converged to 1%).  Second part: the same runs in double precision for the
largest step counts, to see whether the large coefficients raise the round-off floor.
Writes ../data/measured_Emin.json.
Usage:  python measure_Emin.py
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
import mpmath as mp
import measure as M

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FN = os.path.join(DATA, "tableaux_Emin.json")

def _init():
    M._init()
    from tableaux import load_refined
    with mp.workdps(60):
        M._TAB.update(load_refined(FN))

def double_errors(args):
    """Global errors in double precision (complex128 stages, real projection) at N = N0*2^k."""
    key, prob, ks = args
    from problems import P
    T = M._TAB[key]
    m, p, b = T.levels[0]
    A = np.array([[complex(T.A[i][j]) for j in range(m)] for i in range(m)])
    bb = np.array([complex(v) for v in b]); c = A.sum(1)
    pr = P[prob]; f = pr["f_np"]; Tend = float(mp.mpf(pr["T"]))
    ref = np.array([float(v) for v in M._REF[prob]])
    out = []
    for k in ks:
        N = int(M.N0[prob] * 2 ** k); h = Tend / N
        y = np.array([float(mp.mpf(v)) for v in pr["y0"]]); t = 0.0
        for _ in range(N):
            K = np.zeros((m, len(y)), complex)
            for i in range(m):
                K[i] = f(t + c[i] * h, y + h * (A[i, :i] @ K[:i]) if i else y)
            y = (y + h * (bb @ K)).real
            t += h
        out.append(dict(N=N, err=float(np.linalg.norm(y - ref))))
    return key, prob, out

if __name__ == "__main__":
    from problems import ORDER
    names = list(json.load(open(FN)))
    ref_names = ["CRK5-opt", "CRK6-opt", "CCK6(5)"]
    tasks = [(n, n, 0, prob) for n in names for prob in ORDER]
    res = {}
    t0 = time.time()
    with Pool(11, initializer=_init) as pool:
        for label, prob, rows, dt in pool.imap_unordered(M.task, tasks):
            res.setdefault(label, {})[prob] = dict(rows=rows)
            print(f"[{time.time() - t0:6.0f}s] {label:12s} {prob:14s} " +
                  " ".join("-" if r["Ch"] is None else f"{r['Ch']:.4e}" for r in rows[-4:]), flush=True)
        # extension of slowly converging cells
        todo = []
        for lab in res:
            for prob, cell in res[lab].items():
                ok = [r for r in cell["rows"] if r["err"]]
                Ch = [r["Ch"] for r in ok]
                R = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
                conv = abs(R[-1] - R[-2]) / abs(R[-1]) if len(R) > 1 and R[-1] else 1
                if conv > 0.01:
                    todo.append((lab, lab, 0, prob, [7, 8]))
        print(len(todo), "cells to extend", flush=True)
        for label, prob, rows in pool.imap_unordered(M.task_ext, todo):
            res[label][prob]["rows"] += rows
        # double precision at the four largest step counts of every cell
        dtasks = [(n, prob, [3, 4, 5, 6]) for n in names + ref_names for prob in ORDER]
        dbl = {}
        for key, prob, rows in pool.imap_unordered(double_errors, dtasks):
            dbl.setdefault(key, {})[prob] = rows
    json.dump(dict(mp=res, double=dbl), open(os.path.join(DATA, "measured_Emin.json"), "w"), indent=1)
    print("done", time.time() - t0)
