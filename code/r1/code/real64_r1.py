"""Embedded fourth-order formulas of the real seven-stage sixth-order tableaux (internal review 5).

The pair CRK6(4) uses a fourth-order estimate on its six complex stages.  The natural real counterpart is
a seven-stage sixth-order tableau with an embedded fourth-order formula, possibly with the FSAL stage
k_8 = f(t_{n+1}, y_{n+1}).  The fourth-order conditions are linear in the embedded weights: N(4) = 8
equations in 7 unknowns (8 with FSAL), and b itself is a solution, so a genuine embedded formula exists
if and only if the kernel of the 8 x 7 (8 x 8) matrix is nontrivial.  For the FSAL stage the elementary
weights are prod_q 1/gamma(u_q) for t = [u_1 ... u_k] (exact values, since y_{n+1} has order six).

If a kernel direction v exists, the pair b_hat = b + theta v is scaled as CRK6(4): A_hat = 1.183e-3
(the value of the fourth-order formula of DP5(4)).  Writes data/real64_r1.json; with 'run', the pairs
are run with the controller of code/adaptive.py on the ten problems (data/adaptive_real64_r1.json).
Usage: python real64_r1.py [run]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import mpmath as mp
import numpy as np

import r1common as R
import crk
from tableaux import load_refined

FN = os.path.join(R.DATA, "real64_r1.json")
FN_AD = os.path.join(R.DATA, "adaptive_real64_r1.json")
TOLS = [10.0 ** (-k / 2) for k in range(6, 31)]


def kernel(T, q=4, fsal=False):
    s = T.s
    ph = T.phi(q)
    rows = []
    for n in range(1, q + 1):
        for t in crk.trees_upto(n):
            if crk.order(t) != n:
                continue
            r = [mp.re(ph[t][i]) for i in range(s)]
            if fsal:
                w = mp.mpf(1)
                for u in t:
                    w /= crk.gamma(u)
                r.append(w)
            rows.append(r)
    M = mp.matrix(rows)
    U, S, V = mp.svd_r(M)
    n = M.cols
    sv = [S[i] for i in range(min(M.rows, n))] + [mp.mpf(0)] * max(0, n - M.rows)
    null = [k for k in range(n) if sv[k] < mp.mpf(10) ** -30 * sv[0]]
    return M, sv, [[V[k, j] for j in range(n)] for k in null]


def main(run=False):
    mp.mp.dps = 50
    TT = load_refined()
    out = {}
    for name in ("RK6-opt", "Butcher6"):
        T = TT[name]
        for fsal in (False, True):
            M, sv, null = kernel(T, 4, fsal)
            out[f"{name}|fsal={fsal}"] = dict(rows=M.rows, cols=M.cols, singular=[float(x) for x in sv],
                                             kernel_dim=len(null))
            print(f"{name:9s} FSAL={fsal!s:5s} {M.rows}x{M.cols}  singular values "
                  + " ".join(mp.nstr(x, 3) for x in sv) + f"  kernel {len(null)}")
    out["RK6-opt(4)"] = make_pair(TT["RK6-opt"])
    json.dump(out, open(FN, "w"), indent=1)
    if run:
        run_adaptive()


def make_pair(T, Ahat=1.1830e-3):
    """FSAL pair 6(4) on the seven stages of T plus k_8 = f(y_{n+1}): direction in the two-dimensional
    kernel minimizing C, scale giving A_hat (as for CRK6(4), code/embedded.py: make_pair)."""
    import scipy.linalg as sl
    import embedded as EM
    s = T.s
    ph = EM.stage_phi(T, 6, fsal=True)
    rows = [[mp.re(ph[t][i]) for i in range(s + 1)] for t in crk.trees_upto(4)]
    M = mp.matrix(rows)
    U, S, V = mp.svd_r(M, full_matrices=True)
    k = 2
    Vm = [[V[s + 1 - k + c, r] for c in range(k)] for r in range(s + 1)]
    b = [mp.re(v) for v in T.levels[0][2]] + [mp.mpf(0)]

    def errs(w, order):
        return np.array([float((mp.fsum(w[i] * mp.re(ph[t][i]) for i in range(s + 1)) - mp.mpf(1) / crk.gamma(t))
                               / crk.sigma(t)) for t in crk.TREES[order]])

    vec = lambda c: [Vm[r][c] for r in range(s + 1)]
    bh_of = lambda u: [b[r] + mp.fsum(Vm[r][c] * mp.mpf(float(u[c])) for c in range(k)) for r in range(s + 1)]
    W1 = np.array([errs([b[r] + vec(c)[r] for r in range(s + 1)], 5) for c in range(k)]).T
    W2 = np.array([errs([b[r] + vec(c)[r] for r in range(s + 1)], 6) for c in range(k)]).T
    w, Uu = sl.eigh(W2.T @ W2, W1.T @ W1)
    u = Uu[:, 0] / np.linalg.norm(W1 @ Uu[:, 0]) * Ahat
    bh = bh_of(u)
    res = max(abs(mp.fsum(bh[i] * mp.re(ph[t][i]) for i in range(s + 1)) - mp.mpf(1) / crk.gamma(t))
              for t in crk.trees_upto(4))
    t5, t6 = errs(bh, 5), errs(bh, 6)
    A1 = float(np.linalg.norm(t5))
    d = dict(base=T.name, q=4, kernel=k, fsal=True, resid=float(res), Ahat=A1, B=float(np.linalg.norm(t6)) / A1,
             C=float(np.linalg.norm(t6)) / A1, E=0.0, maxbh=float(max(abs(v) for v in bh)),
             A=[[mp.nstr(mp.re(T.A[i][j]), 40) for j in range(s)] for i in range(s)],
             b=[mp.nstr(v, 40) for v in b[:s]], bh=[mp.nstr(v, 40) for v in bh])
    print(f"RK6-opt(4): FSAL, kernel {k}, resid {d['resid']:.1e}, Ahat {A1:.4e}, B = C = {d['C']:.3f}, "
          f"max|bh| {d['maxbh']:.3f}")
    return d


def pair_arrays():
    d = json.load(open(FN))["RK6-opt(4)"]
    A7 = np.array([[float(x) for x in r] for r in d["A"]])
    b = np.array([float(x) for x in d["b"]])
    A = np.zeros((8, 8))
    A[:7, :7] = A7
    A[7, :7] = b
    return dict(A=A, b=np.append(b, 0.0), bh=np.array([float(x) for x in d["bh"]]), p=6, q=4,
                projected=False, fsal=True)


_P = None


def _init():
    global _P
    import adaptive as AD
    AD._init()
    _P = pair_arrays()


def task(prob):
    import adaptive as AD
    if prob == "Bruss2D":
        import bruss2d_r1 as BR
        f, y0, T, ref, norm, tols = BR.f_np, BR.y0(), BR.TEND, BR.reference(), BR.rms, AD.TOLS
    else:
        pr = AD._PROB[prob]
        f, y0, T, ref = pr["f_np"], [float(v) for v in pr["y0"]], float(pr["T"]), AD._REF[prob]
        norm, tols = np.linalg.norm, TOLS
    rows = []
    for tol in tols:
        out = AD.solve(_P, f, y0, T, tol)
        if out is None:
            rows.append(dict(tol=tol, err=None))
            continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=float(norm(y - ref)), nfev=nf, nacc=na, nrej=nr))
    return prob, rows


def run_adaptive(nproc=6):
    from problems import ADAPTIVE
    res = {}
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for prob, rows in pool.imap_unordered(task, list(ADAPTIVE) + ["Bruss2D"]):
            res[prob] = rows
            ok = [r for r in rows if r["err"] is not None]
            print(f"[{time.time() - t0:5.0f}s] RK6-opt(4) {prob:13s} last: err={ok[-1]['err']:.2e} "
                  f"nfev={ok[-1]['nfev']}", flush=True)
    json.dump({"RK6-opt(4)": res}, open(FN_AD, "w"), indent=1)
    print("wrote", FN_AD)


if __name__ == "__main__":
    main("run" in sys.argv)
