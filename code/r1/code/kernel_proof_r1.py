"""The fifth-order formulas on six stages form exactly the line b + theta d (internal review 8).

Proposition 7(c) assumes that, up to the inert directions, the fifth-order weight vectors of a six-stage
sixth-order tableau are exactly b + theta d.  Since d is in the kernel of the homogeneous fifth-order
conditions (Proposition 7(a)), this holds iff the real 17 x 11 matrix
    M = [Re Phi_u(i) (i = 1..6) | -Im Phi_u(i) (i = 2..6)],   rho(u) <= 5,
has rank 10 (the column of Im w_1 is zero: inert).  Here M is enclosed in interval arithmetic (mpmath and
Arb) over the box of radius 1e-44 around the coefficient matrix of each CCK6 tableau, which contains the exact
tableaux of Theorem 1 and Propositions 8 and 10, and a 10 x 10 submatrix is certified nonsingular by
||I - Y M_sub|| < 1.  So rank M = 10 at these exact tableaux; since rank >= 10 is an open condition and the
components are connected real-analytic manifolds, it holds on them outside a proper analytic subset.
Writes data/kernel_proof_r1.json.  Usage: python kernel_proof_r1.py
"""
import json
import os

import numpy as np
import scipy.linalg as sl
import mpmath as mp

import r1common as R
import crk
import krawczyk_r1 as KR
import cascade_proof_r1 as CA
from tableaux import load_refined

mp.mp.dps = 90
BOX = mp.mpf(10) ** -44
OUT = os.path.join(R.DATA, "kernel_proof_r1.json")
TREES5 = crk.trees_upto(5)


def matrix(x, B):
    vals = [B["conv"](v) + B["conv"](0) for v in x]
    rad = CA.ivm([-BOX, BOX])
    vals = [v + rad for v in vals]
    A, b, _, zero = CA.build(vals, {})
    ph = CA.phis(A, zero, 5)
    return [[ph[t][i].re.v for i in range(6)] + [-ph[t][i].im.v for i in range(1, 6)] for t in TREES5]


def midf(z):
    return float(z.mid()) if callable(getattr(z, "mid", None)) else float(KR.mid(z))


def prove(name, T):
    x = list(T.get_x())
    out = {}
    Y = rows = cols = None
    for bname, arb in (("mpmath", False), ("Arb", True)):
        B = CA.backend(arb)
        CA.BK["iv"] = B["iv"]
        try:
            M = matrix(x, B)
            if Y is None:
                Mf = np.array([[midf(z) for z in row] for row in M])
                sv = np.linalg.svd(Mf, compute_uv=False)
                cols = sorted(int(q) for q in sl.qr(Mf, pivoting=True)[2][:10])
                rows = sorted(int(q) for q in sl.qr(Mf[:, cols].T, pivoting=True)[2][:10])
                Y = CA.point_inverse([[mp.mpf(Mf[r][k]) for k in cols] for r in rows])
            ok, nrm = CA.nonsing_cert([[M[r][k] for k in cols] for r in rows], Y, B)
        finally:
            CA.BK["iv"] = CA._iv
        out[bname] = dict(rank_ge_10=bool(ok), cert=nrm)
    out["sv_ratio_10_11"] = [float(sv[9] / sv[0]), float(sv[10] / sv[0])]
    print(f"{name:10s} rank >= 10 certified: mpmath {out['mpmath']['rank_ge_10']} ({out['mpmath']['cert']:.1e}), "
          f"Arb {out['Arb']['rank_ge_10']} ({out['Arb']['cert']:.1e}); sv10/sv1 {out['sv_ratio_10_11'][0]:.1e}, "
          f"sv11/sv1 {out['sv_ratio_10_11'][1]:.1e}", flush=True)
    return out


def main():
    tabs = load_refined()
    res = {}
    for nm in ("CCK6(5)", "CCK6(5)-s", "CCK6-casc"):
        T, _, _ = tabs[nm].level(0).refine(tol_digits=85, iters=10)
        res[nm] = prove(nm, T)
    d = json.load(open(os.path.join(R.DATA, "cck6v_r1.json")))["CCK6(5)-v"]
    cz = lambda v: mp.mpc(mp.mpf(v[0]), mp.mpf(v[1]))
    Tv = crk.Tableau("CCK6(5)-v", [[cz(d["A"][i][j]) for j in range(6)] for i in range(6)], [(6, 6, [cz(v) for v in d["b"][0]])])
    res["CCK6(5)-v"] = prove("CCK6(5)-v", Tv)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
