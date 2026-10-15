"""Variable-step runs of Vern6(5) with split double-precision coefficients (internal review 5).

Rounded to double precision, the coefficients of Vern6(5) (up to 208 in modulus, with b_7 and b_8 nearly
cancelling) violate the condition of order two by 6.5e-12 (roundoff_r1.py), which gives a global error
term of first order in h and a floor near 1e-10 in the variable-step runs.  Here every coefficient is
stored as an unevaluated sum hi + lo of two doubles (hi = fl(a), lo = fl(a - hi)), and the stage sums
and the two solutions are formed as hi @ K + lo @ K.  This removes the systematic defect: the remaining
rounding errors of the arithmetic are not the same at every step.  The abscissae c are rounded from
their exact values.  Controller, starting step, tolerances (1e-3 ... 1e-15) and reference solutions are
those of code/adaptive.py and adaptive_tol15_r1.py.  Writes data/adaptive_vern6split_r1.json;
r1patch.build() uses it in place of the runs with plainly rounded coefficients.
Usage: python vern6_split_r1.py [processes]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import r1common as R
import adaptive as AD

TOLS = [10.0 ** (-k / 2) for k in range(6, 31)]          # 1e-3 ... 1e-15, 25 values
FN = os.path.join(R.DATA, "adaptive_vern6split_r1.json")
NAME = "Vern6(5)"


def split_pair():
    import mpmath as mp
    from tableaux import load_refined
    mp.mp.dps = 60
    T = load_refined(os.path.join(R.DATA, "tableaux_r1.json"))[NAME]
    s = T.s

    def hl(v):
        v = mp.re(v)
        hi = float(v)
        return hi, float(v - mp.mpf(hi))

    A = [[hl(T.A[i][j]) for j in range(s)] for i in range(s)]
    b = [hl(v) for v in T.levels[0][2]]
    bh = [hl(v) for v in T.levels[1][2]]
    c = [float(mp.re(sum(T.A[i][j] for j in range(s)))) for i in range(s)]
    arr = lambda M, k: np.array([[x[k] for x in row] for row in M])
    vec = lambda v, k: np.array([x[k] for x in v])
    return dict(Ah=arr(A, 0), Al=arr(A, 1), bh_=vec(b, 0), bl=vec(b, 1), eh=vec(bh, 0), el=vec(bh, 1),
                c=np.array(c), p=6, q=5)


def solve(P, f, y0, T, tol, max_steps=2_000_000):
    """code/adaptive.py: solve() for a real FSAL pair, with the stage sums formed as hi @ K + lo @ K."""
    Ah, Al, bh_, bl, eh, el, c, p, q = (P[k] for k in ("Ah", "Al", "bh_", "bl", "eh", "el", "c", "p", "q"))
    s = len(c)
    y = np.array(y0, float)
    t = 0.0
    sc = lambda u, v: tol + tol * np.maximum(np.abs(u), np.abs(v))
    nf = 0
    f0 = np.asarray(f(t, y), float)
    nf += 1
    d0 = np.sqrt(np.mean((y / sc(y, y)) ** 2))
    d1 = np.sqrt(np.mean((f0 / sc(y, y)) ** 2))
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    f1 = np.asarray(f(t + h0, y + h0 * f0), float)
    nf += 1
    d2 = np.sqrt(np.mean(((f1 - f0) / sc(y, y)) ** 2)) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1.0 / (p + 1))
    h = min(100 * h0, h1, T)
    k1 = f0
    nacc = nrej = 0
    facmax = 5.0
    while t < T * (1 - 1e-14):
        if t + h > T:
            h = T - t
        K = np.zeros((s, len(y)))
        K[0] = k1
        for i in range(1, s):
            K[i] = f(t + c[i] * h, y + h * (Ah[i, :i] @ K[:i] + Al[i, :i] @ K[:i]))
            nf += 1
        ynew = y + h * (bh_ @ K + bl @ K)
        yhat = y + h * (eh @ K + el @ K)
        err = np.sqrt(np.mean(((ynew - yhat) / sc(y, ynew)) ** 2))
        if not np.isfinite(err):
            return None
        if err <= 1.0:
            t += h
            y = ynew
            nacc += 1
            k1 = K[-1]                                  # FSAL
            fac = min(facmax, max(0.2, 0.9 * err ** (-1.0 / (q + 1)) if err > 0 else facmax))
            h *= fac
            facmax = 5.0
        else:
            nrej += 1
            h *= max(0.2, 0.9 * err ** (-1.0 / (q + 1)))
            facmax = 1.0
        if nacc + nrej > max_steps or h < 1e-14 * T:
            return None
    return y, nf, nacc, nrej


_P = None


def _init():
    global _P
    AD._init()
    _P = split_pair()


def task(prob):
    pr = AD._PROB[prob]
    T = float(pr["T"])
    y0 = [float(v) for v in pr["y0"]]
    rows = []
    for tol in TOLS:
        out = solve(_P, pr["f_np"], y0, T, tol)
        if out is None:
            rows.append(dict(tol=tol, err=None))
            continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=float(np.linalg.norm(y - AD._REF[prob])), nfev=nf, nacc=na, nrej=nr))
    return prob, rows


def main(nproc=6):
    from problems import ADAPTIVE
    res = {}
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for prob, rows in pool.imap_unordered(task, ADAPTIVE):
            res[prob] = rows
            r = rows[-1]
            print(f"[{time.time() - t0:5.0f}s] {NAME} split {prob:13s} tol 1e-15: "
                  + (f"err={r['err']:.2e} nfev={r['nfev']}" if r["err"] is not None else "FAILED"), flush=True)
    json.dump({NAME: res}, open(FN, "w"), indent=1)
    print("wrote", FN)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
