"""Embedded fifth-order formulas of six-stage sixth-order tableaux and the reality of c_6 (Section 3.4).

For a CRK6 tableau (A, b) let d = A^T b - b o (1 - c).  By the identity of D(1),
    Re(d^T Phi_t) = Re(b^T Phi_[t]) - Re(b^T Phi_t) + Re(b^T Phi_{t+}),
so for rho(t) <= 5 the sixth-order conditions of [t], t and t+ give Re(d^T Phi_t) = 0: for every real
theta, b + theta d satisfies the projected conditions of order five.  Since the last column of A vanishes,
d_6 = -b_6 (1 - c_6), and the sixth component of b + theta d vanishes iff theta (1 - c_6) = 1.  Hence a
tableau with real c_6 != 1 carries the fifth-order formula  bhat = b + d / (1 - c_6)  on its first five
stages, i.e. it is a CCK6 pair; conversely, where the fifth-order weight vectors on the six stages are
exactly the line b + theta d, such a formula exists only if c_6 is real.

1. Multiple precision (60 digits): at the CCK6 tableaux of the paper, bhat versus the fifth-order level
   b^(5), Im c_6 and the rank of M_5(A) on six stages; at all CRK6 and CCK6 tableaux, the residual of
   the fifth-order conditions of b + theta d.
2. Double precision: Im c_6 and the residual of b + theta d at the 460 CRK6 tableaux visited by the
   optimization runs, by the rank of the Jacobian of the 37 conditions.
3. Double precision: random walks along the four-dimensional CRK6 component through the sixth-order
   levels of CCK6(5) and CCK6(5)-s (37 conditions only): Im c_6 and the consistency of the fifth-order
   conditions on the first five stages.
Writes ../data/cck6_identity_r1.json.
"""
import json
import os
import sys

import mpmath as mp
import numpy as np

import r1common as R
import crk
from tableaux import load_refined

OUT = os.path.join(R.DATA, "cck6_identity_r1.json")
THETA = mp.mpf("0.7")
PAIRS = ["CCK6(5)", "CCK6(5)-s", "CCK6-casc"]
CRK6 = ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam"]


def dvec(A, b, s):
    c = [mp.fsum(A[i]) for i in range(s)]
    return [mp.fsum(A[i][j] * b[i] for i in range(s)) - b[j] * (1 - c[j]) for j in range(s)], c


def fifth_resid(T, w):
    ph = T.phi(5)
    return max(abs(mp.re(mp.fsum(w[i] * ph[t][i] for i in range(T.s))) - mp.mpf(1) / crk.gamma(t))
               for t in crk.trees_upto(5))


def rank_m5(T, m):
    ph = T.phi(5)
    rows = []
    for t in crk.trees_upto(5):
        v = [ph[t][i] for i in range(m)]
        rows.append([mp.re(x) for x in v] + [-mp.im(x) for x in v[1:]])
    M = mp.matrix(rows)
    sv = mp.svd_r(M, compute_uv=False)
    sv = sorted([sv[i] for i in range(len(sv))], reverse=True)
    return sum(1 for x in sv if x > sv[0] * mp.mpf(10) ** -40), [float(x / sv[0]) for x in sv[-2:]]


def part1():
    mp.mp.dps = 60
    TT = load_refined()
    out = {}
    for k in PAIRS + CRK6:
        T = TT[k]
        b = T.levels[0][2]
        d, c = dvec(T.A, b, T.s)
        row = dict(Im_c6=float(abs(mp.im(c[5]))), c6=float(mp.re(c[5])),
                   resid_b_theta_d=float(fifth_resid(T, [b[i] + THETA * d[i] for i in range(T.s)])),
                   max_d=float(max(abs(x) for x in d[1:])))
        rk, sv = rank_m5(T, 6)
        row.update(rank_M5_six_stages=rk, smallest_sv=sv)
        if k in PAIRS:
            bh = [b[i] + d[i] / (1 - c[5]) for i in range(T.s)]
            b5 = list(T.levels[1][2]) + [mp.mpc(0)] * (T.s - len(T.levels[1][2]))
            row["max_diff_bhat_b5"] = float(max(abs(bh[i] - b5[i]) for i in range(1, T.s)))
            row["diff_Re_bhat1"] = float(abs(mp.re(bh[0] - b5[0])))
            row["bhat6"] = float(abs(bh[5]))
        out[k] = row
        print(k, {q: (f"{v:.1e}" if isinstance(v, float) else v) for q, v in row.items()}, flush=True)
    return out


def part2():
    sys.path.insert(0, R.ORIG_CODE)
    from rank_survey import points
    from optimize_E import load_module
    TC = load_module("CRK6")
    ranks = [r["rank"] for r in json.load(open(os.path.join(R.ORIG_DATA, "rank_survey.json")))["CRK6"]]
    pts = points("CRK6")
    assert len(pts) == len(ranks)
    out = {}
    for (label, p), rk in zip(pts, ranks):
        A, W = TC.unpack(np.array(p["x"], float))
        b = W[0]
        c = A.sum(axis=1)
        d = A.T @ b - b * (1 - c)
        ph = TC.phi_all(A, 5, deriv=False)
        w = b + 0.7 * d
        res = max(abs((w @ ph[t][0]).real - 1 / TC.gamma(t)) for k in range(1, 6) for t in TC.TREES[k])
        o = out.setdefault(str(rk), dict(n=0, max_Im_c6=0.0, min_abs_c6_minus_1=1e9, max_resid=0.0))
        o["n"] += 1
        o["max_Im_c6"] = max(o["max_Im_c6"], float(abs(c[5].imag)))
        o["min_abs_c6_minus_1"] = min(o["min_abs_c6_minus_1"], float(abs(c[5] - 1)))
        o["max_resid"] = max(o["max_resid"], float(res))
    for k, v in out.items():
        print("rank", k, v, flush=True)
    return out


def part3(nwalks=12, steps=40, delta=0.02, tol=1e-9):
    from optimize_E import load_module, x_from_tableau
    TC = load_module("CRK6")
    rng = np.random.default_rng(1)

    def rank(x):
        sv = np.linalg.svd(TC.residual_and_jacobian(x)[1], compute_uv=False)
        return int((sv > tol * sv[0]).sum())

    def correct(x, r):
        for _ in range(10):
            res, J = TC.residual_and_jacobian(x)
            if np.abs(res).max() < 1e-15:
                break
            U, sv, Vt = np.linalg.svd(J, full_matrices=False)
            x = x - Vt[:r].T @ ((U[:, :r].T @ res) / sv[:r])
        return x, float(np.abs(TC.residual(x)).max())

    def five_stage(x, m=5):
        A, _ = TC.unpack(x)
        ph = TC.phi_all(A, 5, deriv=False)
        V = np.array([ph[t][0][:m] for k in range(1, 6) for t in TC.TREES[k]])
        r = np.array([1.0 / TC.gamma(t) for k in range(1, 6) for t in TC.TREES[k]])
        M = np.hstack([V.real, -V.imag])[:, [q for q in range(2 * m) if q != m]]
        sol = np.linalg.lstsq(M, r, rcond=None)[0]
        return float(np.abs(M @ sol - r).max())

    out = {}
    for name in ("CCK6(5)", "CCK6(5)-s", "CRK6-opt"):
        x0 = x_from_tableau(TC, name)
        r0 = rank(x0)
        x0, _ = correct(x0, r0)
        info = dict(rank=r0, dim=int(x0.size - r0), five_stage_inconsistency=five_stage(x0))
        if name != "CRK6-opt":
            npts, worst, worst_c6, dmax = 0, 0.0, 0.0, 0.0
            for _ in range(nwalks):
                x = x0.copy()
                Tb = TC.tangent_basis(x, tol=tol)
                t = Tb @ rng.standard_normal(Tb.shape[1])
                t /= np.linalg.norm(t)
                for _ in range(steps):
                    x, res = correct(x + delta * t, r0)
                    if res > 1e-11 or rank(x) != r0:
                        break
                    npts += 1
                    worst = max(worst, five_stage(x))
                    worst_c6 = max(worst_c6, float(abs(TC.unpack(x)[0][5].sum().imag)))
                    dmax = max(dmax, float(np.linalg.norm(x - x0)))
                    Tb = TC.tangent_basis(x, tol=tol)
                    t = Tb @ (Tb.T @ t)
                    t /= np.linalg.norm(t)
            info.update(walk_points=npts, max_distance=dmax, max_five_stage_inconsistency=worst,
                        max_Im_c6=worst_c6)
        out[name] = info
        print(name, info, flush=True)
    return out


def main():
    out = dict(exact=part1(), survey=part2(), walks=part3())
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
