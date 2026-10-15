"""Is the lower bound E_6 >= 1/720 attained in the CRK5 family? (internal review 8, item M5)

E_6 = ||e||_2 over the 20 trees of order six, and the tall tree has e = -1/720 in every five-stage method,
so E_6 = 1/720 iff the other 19 error coefficients vanish, i.e., iff the five-stage projected method
satisfies 36 real conditions (17 of order <= 5 and 19 of order six) in 29 real unknowns.  At CRK5-opt the
19 coefficients have the norm R = sqrt(E_6^2 - 1/720^2) = 1.1e-4.  Here R^2 is minimized on the CRK5
family (SLSQP with the 17 conditions as equality constraints and max|coef| <= Lambda) from the end points of
the descents of code/base/data/Emin_CRK5_100.json and from the named CRK5 tableaux, for Lambda = 5 and 100,
and the rank of the Jacobian of the 36 conditions is reported at the best point.
With the argument branch, the best point for Lambda = 100 is continued for Lambda up to 1e5.
Writes data/crk5_bound_r1.json.  Usage: python crk5_bound_r1.py [branch]
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

import r1common as R
import optimize_E as OE
import imag_estimate_r1 as IE

OUT = os.path.join(R.DATA, "crk5_bound_r1.json")
_TC = None


def tc():
    global _TC
    if _TC is None:
        _TC = OE.load_module("CRK5")
    return _TC


def tall_index(TC):
    def tall(t):
        return t == () or (len(t) == 1 and tall(t[0]))
    ks = [k for k, t in enumerate(TC.TREES[6]) if tall(t)]
    assert len(ks) == 1
    return ks[0]


def rest(x):
    TC = tc()
    e, J = OE.err_and_jac(TC, x)
    keep = [k for k in range(len(e)) if k != TALL]
    return e[keep], J[keep]


def descent(args):
    x0, lam = args
    TC = tc()
    IE.BOUND = lam
    x, ok = IE.project(TC, x0, 0)
    if not ok:
        return None
    f = lambda z: float(rest(z)[0] @ rest(z)[0])
    g = lambda z: 2 * rest(z)[1].T @ rest(z)[0]
    sol = minimize(f, x, jac=g, method="SLSQP",
                   constraints=[dict(type="eq", fun=lambda z: TC.residual_and_jacobian(z)[0],
                                     jac=lambda z: TC.residual_and_jacobian(z)[1]),
                                dict(type="ineq", fun=lambda z: IE.coef_ineq(TC, z)[0],
                                     jac=lambda z: IE.coef_ineq(TC, z)[1])],
                   options=dict(maxiter=1000, ftol=1e-24))
    x, ok = IE.project(TC, sol.x, 0)
    if not ok or IE.coef_ineq(TC, x)[0].min() < -1e-6:
        return None
    r, Jr = rest(x)
    _, Jc = TC.residual_and_jacobian(x)
    sv = np.linalg.svd(np.vstack([Jc, Jr]), compute_uv=False)
    E6 = float(np.linalg.norm(OE.err_and_jac(TC, x)[0]))
    A, W = TC.unpack(x)
    return dict(R=float(np.linalg.norm(r)), E6=E6, maxcoef=float(max(np.abs(A).max(), np.abs(W[0]).max())),
                rank36=int((sv > sv[0] * 1e-10).sum()), x=x.tolist())


TC0 = OE.load_module("CRK5")
TALL = tall_index(TC0)


def main():
    t0 = time.time()
    d = json.load(open(os.path.join(R.ORIG_DATA, "Emin_CRK5_100.json")))
    X = [np.array(r["x"]) for r in d["results"]]
    for nm in ("CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5"):
        try:
            X.append(OE.x_from_tableau(TC0, nm))
        except Exception:
            pass
    e, _ = OE.err_and_jac(TC0, OE.x_from_tableau(TC0, "CRK5-opt"))
    assert abs(abs(e[TALL]) - 1 / 720) < 1e-12, e[TALL]
    out = {}
    for lam in (5.0, 100.0):
        with Pool(4) as pool:
            res = [r for r in pool.map(descent, [(x, lam) for x in X]) if r is not None]
        res.sort(key=lambda r: r["R"])
        b = res[0]
        out[f"{lam:g}"] = dict(descents=len(res), best=b, R_sorted=[r["R"] for r in res[:10]])
        print(f"[{time.time() - t0:5.0f}s] Lambda = {lam:g}: {len(res)} descents, smallest R = "
              + " ".join(f"{r['R']:.3e}" for r in res[:6]) + f"; best: E6 = {b['E6']:.6e} "
              f"({100 * (b['E6'] * 720 - 1):.3f}% above 1/720), max|coef| {b['maxcoef']:.2f}, rank of the 36 "
              f"conditions {b['rank36']} of 29", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)




def branch(lams=(100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5)):
    """Continuation of the best point of Lambda = 100 along the degenerate branch for growing Lambda."""
    d = json.load(open(OUT))
    x = np.array(d["100"]["best"]["x"])
    rows = []
    for lam in lams:
        r = descent((x, lam))
        if r is None:
            print(f"Lambda = {lam:g}: descent failed", flush=True)
            continue
        x = np.array(r["x"])
        A, W = tc().unpack(x)
        c = A.sum(1)
        gap = float(min(abs(c[i] - c[j]) for i in range(5) for j in range(i)))
        rows.append(dict(Lambda=lam, R=r["R"], E6=r["E6"], maxcoef=r["maxcoef"], min_abscissa_gap=gap))
        print(f"Lambda = {lam:g}: R = {r['R']:.3e}, E6 = {r['E6']:.7e} ({100 * (r['E6'] * 720 - 1):.4f}% above "
              f"1/720), max|coef| {r['maxcoef']:.1f}, min |c_i - c_j| {gap:.2e}", flush=True)
    d["branch"] = rows
    json.dump(d, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["branch"]:
        branch()
    else:
        main()
