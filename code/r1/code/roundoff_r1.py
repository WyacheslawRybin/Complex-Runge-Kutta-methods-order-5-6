"""Order conditions of the tableaux with coefficients rounded to double precision (R1).

The variable-step runs and the timing use the coefficients rounded to double precision.  For every
tableau, the residual of its order conditions (all levels) is evaluated in 40-digit arithmetic with the
rounded coefficients.  For Vern6(5), whose weights b7 = -176.5, b8 = 172.4 and coefficients up to 208 in
modulus cancel strongly, the residual is 3.3e-11, while all other tableaux keep 1e-14 or better; this,
not the round-off of the arithmetic, limits the attainable accuracy of Vern6(5) in double precision.
The defects of Vern6(5) are also given by tree order: Sum b_i - 1 = 1.4e-14 (order one), but 6.5e-12 for
the order-two condition Sum b_i c_i = 1/2.  A defect delta_k of order k adds a global error of the size
delta_k h^(k-1), so the order-two defect gives a term of first order in h with a large coefficient, which
at the step sizes of the runs acts as a floor.
The second part shows it on the Kepler problem: N fixed steps in 30-digit arithmetic with the rounded
and with the exact coefficients; with the rounded ones the error halves with every doubling of N from
N = 2000 on (first order).
Writes R1/data/roundoff_r1.json.
"""
import json
import os

import mpmath as mp

import r1common as R
from crk import Tableau
from tableaux import load_refined

NAMES = ["CRK5-opt", "CRK5-s", "CRK6-opt", "CRK6-s", "CCK6(5)", "CCK6(5)-s", "CCK6-casc", "DP5", "CK5", "Tsit5(4)",
         "BS5(4)", "Butcher6", "RK6-opt", "DP6(5)", "Verner6(5)", "Vern6(5)"]


def rounded(T):
    rd = lambda z: mp.mpc(float(mp.re(z)), float(mp.im(z)))
    s = T.s
    A = [[rd(T.A[i][j]) for j in range(s)] for i in range(s)]
    return Tableau(T.name, A, [(m, p, [rd(v) for v in b]) for m, p, b in T.levels], T.projected, cost=T.cost)


def kepler(T, N, round_coef):
    import problems
    pr = problems.P["Kepler"]
    f = pr["f_mp"]
    Tend = mp.mpf(pr["T"])
    ref = [mp.mpf(v) for v in problems.references()["Kepler"]]
    U = rounded(T) if round_coef else T
    s = U.s
    A = [[mp.re(U.A[i][j]) for j in range(s)] for i in range(s)]
    b = [mp.re(v) for v in U.levels[0][2]]
    c = [mp.fsum(A[i]) for i in range(s)]
    h = Tend / N
    y = [mp.mpf(v) for v in pr["y0"]]
    for n in range(N):
        K = []
        for i in range(s):
            K.append(f(n * h + c[i] * h, [y[d] + h * mp.fsum(A[i][j] * K[j][d] for j in range(i)) for d in range(len(y))]))
        y = [y[d] + h * mp.fsum(b[i] * K[i][d] for i in range(s)) for d in range(len(y))]
    return float(mp.sqrt(mp.fsum((a - r) ** 2 for a, r in zip(y, ref))))


def main():
    mp.mp.dps = 40
    TT = load_refined()
    TT.update(load_refined(os.path.join(R.DATA, "tableaux_r1.json")))
    out = {"resid_double": {}, "kepler": {}, "vern6_defect_by_order": {}}
    for k in NAMES:
        r = max(abs(v) for lev in rounded(TT[k]).residuals() for v in lev)
        out["resid_double"][k] = float(r)
        print(f"{k:11s} residual of the order conditions with double coefficients: {float(r):.1e}", flush=True)
    U = rounded(TT["Vern6(5)"])
    ph = U.phi(6)
    b = U.levels[0][2]
    from crk import trees_upto, order, gamma
    for q in range(1, 7):
        r = max(abs(mp.re(mp.fsum(b[i] * ph[t][i] for i in range(U.s))) - mp.mpf(1) / gamma(t))
                for t in trees_upto(6) if order(t) == q)
        out["vern6_defect_by_order"][q] = float(r)
        print(f"Vern6(5), rounded coefficients: largest defect of order {q}: {float(r):.2e}", flush=True)
    mp.mp.dps = 30
    for N in (8000, 16000):
        a = kepler(TT["Vern6(5)"], N, True)
        out["kepler"][f"Vern6(5)|{N}"] = dict(double_coef=a)
        print(f"Kepler, Vern6(5), N = {N}: error {a:.2e} with double coefficients", flush=True)
    for k in ("Vern6(5)", "DP6(5)"):
        for N in (1000, 2000, 4000):
            a, b = kepler(TT[k], N, True), kepler(TT[k], N, False)
            out["kepler"][f"{k}|{N}"] = dict(double_coef=a, exact_coef=b)
            print(f"Kepler, {k}, N = {N}: error {a:.2e} with double coefficients, {b:.2e} with exact ones", flush=True)
    json.dump(out, open(os.path.join(R.DATA, "roundoff_r1.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
