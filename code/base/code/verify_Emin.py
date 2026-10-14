"""
Polish the best tableaux of optimize_E.py (more iterations), refine them in multiple
precision and recompute E_{p+1}, the residuals and max|coef| in mp.
Writes ../data/tableaux_Emin.json in the format of tableaux.json (45 digits).
Usage:  python verify_Emin.py [BOUND]
"""
import json, os, sys
import numpy as np
import mpmath as mp
import crk
import optimize_E as O

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NAMES = {"CRK5": "CRK5-E", "CRK6": "CRK6-E", "CCK65": "CCK6(5)-E"}

def tableau(TC, x, name):
    A, W = TC.unpack(np.array(x))
    Am = [[mp.mpc(complex(A[i, j])) for j in range(TC.S)] for i in range(TC.S)]
    lev = [(m, p, [mp.mpc(complex(v)) for v in W[L]]) for L, (m, p) in enumerate(TC.LEV)]
    return crk.Tableau(name, Am, lev)

def distinct(results, k=3):
    out = []
    for r in results:
        if all(abs(r["E"] - q["E"]) > 1e-4 * r["E"] or abs(r["maxcoef"] - q["maxcoef"]) > 1e-2 for q in out):
            out.append(r)
        if len(out) == k:
            break
    return out

def main(bound=100.0):
    mp.mp.dps = 60
    out = {}
    for fam, base in NAMES.items():
        d = json.load(open(os.path.join(DATA, f"Emin_{fam}_{bound:g}.json")))
        TC = O.load_module(fam)
        print(f"== {fam}: best local minima found")
        res_all = d["results"]
        small = [r for r in res_all if r["maxcoef"] <= 5.0 + 1e-9]
        cands = [(base, res_all[0], bound)] + ([(base + "5", small[0], 5.0)] if small else [])
        for name, r, bnd in cands:
            x, E, it = O.minimize(TC, np.array(r["x"]), bnd * 0.9999, rho=10.0, iters=3000)
            T = tableau(TC, x, name)
            T, shift, res = T.refine()
            E_mp = T.E()
            full = T.full_complex_residual()
            print(f"  {T.name:12s} bound {bnd:5g} start {r['start']:18s} it {it:4d} E(double) {E:.6e}  E(mp) {mp.nstr(E_mp, 8)}  "
                  f"max|coef| {mp.nstr(T.maxcoef(), 6)}  refine shift {mp.nstr(shift, 2)}  resid {mp.nstr(res, 2)}  "
                  f"c = {[mp.nstr(c, 4) for c in T.c]}", flush=True)
            out[T.name] = dict(s=T.s, projected=True, cost=T.s, levels=[(m, p) for m, p, _ in T.levels],
                               A=[[[mp.nstr(mp.re(T.A[i][j]), 45), mp.nstr(mp.im(T.A[i][j]), 45)]
                                   for j in range(T.s)] for i in range(T.s)],
                               b=[[[mp.nstr(mp.re(v), 45), mp.nstr(mp.im(v), 45)] for v in b] for _, _, b in T.levels],
                               E=[float(T.E(L)) for L in range(len(T.levels))], maxcoef=float(T.maxcoef()),
                               resid=float(res), full_complex_resid=float(full), start=r["start"], bound=bnd)
    json.dump(out, open(os.path.join(DATA, "tableaux_Emin.json"), "w"), indent=1)

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 100.0)
