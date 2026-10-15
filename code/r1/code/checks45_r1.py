"""Checks with the full-precision (refined, 60-digit) coefficients (internal review 5).

1. Fourth-order formulas on the first five stages of the CCK6 tableaux: singular values of the 8 x 9 real
   matrix of the fourth-order conditions (Section 5.1); for comparison the CRK5 tableaux.
2. Smallest retained singular value of the Jacobian of the 37 sixth-order conditions (41 unknowns) at the
   CRK6 tableaux, which explains the size of the refinement shift of CRK6-0 (16 printed digits).
Writes data/checks45_r1.json.
"""
import json
import os

import mpmath as mp

import r1common as R
import crk
import krawczyk_r1 as KR
from tableaux import load_refined

mp.mp.dps = 60
OUT = os.path.join(R.DATA, "checks45_r1.json")


def m4_on_five(T):
    ph = T.phi(4)
    rows = [[mp.re(ph[t][i]) for i in range(5)] + [-mp.im(ph[t][i]) for i in range(1, 5)]
            for t in crk.trees_upto(4)]
    S = mp.svd_r(mp.matrix(rows), compute_uv=False)
    S = sorted([abs(x) for x in S], reverse=True)
    return [float(x / S[0]) for x in S]


def jac_sv(T):
    x = T.get_x()
    _, J = KR.evaluate(x, list(range(len(x))), T.s, T.p)
    S = mp.svd_r(mp.matrix([[mp.mpf(z.mid) for z in row] for row in J]), compute_uv=False)
    S = sorted([abs(v) for v in S], reverse=True)
    return [float(v / S[0]) for v in S]


def main():
    TT = load_refined()
    out = {"M4_five_stages": {}, "jacobian_crk6": {}}
    for name in ["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5", "CCK6(5)", "CCK6(5)-s", "CCK6-casc"]:
        sv = m4_on_five(TT[name])
        out["M4_five_stages"][name] = sv
        print(f"{name:10s} M_4 on five stages: smallest / largest singular value {sv[-1]:.2e} (next {sv[-2]:.2e})")
    for name in ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam"]:
        sv = jac_sv(TT[name])
        out["jacobian_crk6"][name] = sv
        print(f"{name:10s} Jacobian of 37 conditions: sigma_31/sigma_1 = {sv[30]:.2e}, sigma_32/sigma_1 = {sv[31]:.2e}")
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
