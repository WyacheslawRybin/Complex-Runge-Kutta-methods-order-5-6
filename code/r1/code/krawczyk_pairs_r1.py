"""Computer-assisted existence proof of exact CCK6 pairs 6(5) (internal review 5).

By Proposition emb5(b), a six-stage sixth-order tableau with a real c_6 != 1 carries the fifth-order
formula b + d/(1 - c_6) on its first five stages, i.e., it is a CCK6 pair.  So an exact pair exists
near a numerical pair if the 37 projected sixth-order conditions together with Im c_6 = 0 (38 real
equations in 41 unknowns) have an exact zero there.  Where these 38 equations have full row rank, three
unknowns are fixed and the Krawczyk test of krawczyk_r1.py (interval arithmetic, 320 bits) is applied.
On the component of tangent dimension four (CCK6(5), CCK6(5)-s) Im c_6 appears to vanish identically
(Conjecture 1), the 38 equations are dependent and the test cannot succeed; on the component of tangent
dimension five, Im c_6 is a nonconstant function and the test applies.

Candidates: the 30 pairs of the descents of Section 4.2 with the bound B = 100 (code/base/data/
Emin_CCK65_100.json, double precision).  Writes data/krawczyk_pairs_r1.json.
Usage: python krawczyk_pairs_r1.py [max_candidates]
"""
import json
import os
import sys

import mpmath as mp

import r1common as R
import crk
import krawczyk_r1 as KR

FN = os.path.join(R.DATA, "krawczyk_pairs_r1.json")


def candidates():
    import numpy as np
    import optimize_E as OE
    d = json.load(open(os.path.join(R.ORIG_DATA, "Emin_CCK65_100.json")))
    TC = OE.load_module("CCK65")                     # layout of the stored vectors (code/tune_cck6.py)
    out, seen = [], set()
    for k, r in enumerate(d["results"]):
        key = round(r["E"], 9)
        if key in seen:                               # several descents end at the same tableau
            continue
        seen.add(key)
        A, W = TC.unpack(np.array(r["x"]))
        cz = lambda v: mp.mpc(float(v.real), float(v.imag))
        T = crk.Tableau(f"pair{k:02d}", [[cz(A[i, j]) for j in range(6)] for i in range(6)],
                        [(6, 6, [cz(v) for v in W[0]]), (5, 5, [cz(v) for v in W[1]])])
        out.append((T, r))
    return out


def main(nmax=30):
    mp.mp.dps = 90
    res = json.load(open(FN)) if os.path.exists(FN) else {}
    for T, r in candidates()[:nmax]:
        lev, shift, resid = T.refine(tol_digits=85, iters=14)
        if resid > mp.mpf(10) ** -60:
            print(f"{T.name}: refinement of the pair did not converge (residual {mp.nstr(resid, 3)})", flush=True)
            continue
        c6 = lev.c[5]
        info = KR.prove(T.name + "[6]+c6", {T.name: lev})
        info.update(E7=float(lev.E(0)), maxcoef=float(lev.maxcoef()), c6=[float(mp.re(c6)), float(mp.im(c6))],
                    start=r["start"])
        res[T.name] = info
        json.dump(res, open(FN, "w"), indent=1)
    ok = [k for k, v in res.items() if v["proved"]]
    print("proved:", ok)


def save_verified(name="pair20", label="CCK6(5)-v"):
    """Coefficients (45 digits) and invariants of the pair whose existence is proved, in the format of
    code/base/data/tableaux.json -> data/cck6v_r1.json (used by make_appendix_r1.py)."""
    mp.mp.dps = 90
    for T, r in candidates():
        if T.name != name:
            continue
        P, shift, resid = T.refine(tol_digits=85, iters=14)
        P = P.normalized()
        st = lambda z: [mp.nstr(mp.re(z), 45), mp.nstr(mp.im(z), 45)]
        res = [v for lev in P.residuals() for v in lev]
        out = {label: dict(s=6, cost=6, projected=True, levels=[[6, 6], [5, 5]],
                           resid_refined=float(max(abs(v) for v in res)), refine_shift=float(shift),
                           E=[float(P.E(0)), float(P.E(1))], maxcoef=float(P.maxcoef()),
                           c6=st(P.c[5]), A=[[st(P.A[i][j]) for j in range(6)] for i in range(6)],
                           b=[[st(v) for v in P.levels[0][2]], [st(v) for v in P.levels[1][2]]],
                           krawczyk=json.load(open(FN))[name], source=f"{name} of Emin_CCK65_100.json ({r['start']})")}
        json.dump(out, open(os.path.join(R.DATA, "cck6v_r1.json"), "w"), indent=1)
        d = out[label]
        print(f"{label}: E7 {d['E'][0]:.5e} E6 {d['E'][1]:.5e} max|coef| {d['maxcoef']:.6f} c6 {d['c6'][0][:12]} "
              f"resid {d['resid_refined']:.1e} shift {d['refine_shift']:.1e}")
        return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["save"]:
        save_verified()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
