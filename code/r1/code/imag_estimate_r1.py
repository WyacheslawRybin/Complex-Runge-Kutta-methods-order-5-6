"""The imaginary part of the unprojected solution as a free error estimate: its price in E_{p+1} (internal review 8).

If, in addition to the projected conditions, Im(b^T Phi_t) = 0 for all trees with rho(t) <= q, then the
complex method (A, b) itself has order q, the unprojected solution y~ = y_n + h b^T K satisfies
Im y~ = O(h^{q+1}), and ||Im y~|| is a free estimate of the local error of an order-q formula, as for an
embedded pair p(q) -- without extra storage or evaluations.  The extra conditions are 1, 2, 4 and 8 real
equations for q = 1, ..., 4.  For CRK5 (family of dimension 12) and CRK6 (dimension 10 on the component of
D(1)), E_{p+1} is minimized under the bound max|coef| <= 5 subject to these equations (SLSQP from the
end points of the descents of code/base/data/Emin_CRK*_100.json, from the named tableaux of the family and
from points of random walks on the family started there, after a least-squares projection onto the equations),
and compared with the unconstrained minimum (q = 0).
Writes data/imag_estimate_r1.json.  Usage: python imag_estimate_r1.py [starts]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

import r1common as R
import optimize_E as OE

OUT = os.path.join(R.DATA, "imag_estimate_r1.json")
BOUND = 5.0


def imag_rows(TC, x, q):
    A, W = TC.unpack(x)
    m = TC.LEV[0][0]
    ph = TC.phi_all(A, max(q, 1))
    rows, Js = [], []
    for k in range(1, q + 1):
        for t in TC.TREES[k]:
            v, dv = ph[t]
            Jc = np.zeros(TC.NZ, complex)
            Jc[:TC.NA] = W[0] @ dv[:m]
            Jc[TC.WOFF[0]:TC.WOFF[0] + m] = v[:m]
            rows.append((W[0] @ v[:m]).imag)
            Js.append(np.concatenate([Jc.imag, Jc.real])[TC.KEEP])      # d Im / d(Re, Im)
    return np.array(rows), np.array(Js).reshape(len(rows), -1)


def cons(TC, x, q):
    r, J = TC.residual_and_jacobian(x)
    if q == 0:
        return r, J
    ri, Ji = imag_rows(TC, x, q)
    return np.concatenate([r, ri]), np.vstack([J, Ji])


def coef_ineq(TC, x):
    xx = np.zeros(2 * TC.NZ)
    xx[TC.KEEP] = x
    re, im = xx[:TC.NZ], xx[TC.NZ:]
    g = BOUND ** 2 - re ** 2 - im ** 2
    J = np.zeros((TC.NZ, 2 * TC.NZ))
    J[np.arange(TC.NZ), np.arange(TC.NZ)] = -2 * re
    J[np.arange(TC.NZ), TC.NZ + np.arange(TC.NZ)] = -2 * im
    return g, J[:, TC.KEEP]


def project(TC, x, q, iters=30):
    from scipy.optimize import least_squares
    if np.linalg.norm(cons(TC, x, q)[0]) > 1e-6:          # feasibility first: trust-region least squares
        sol = least_squares(lambda z: cons(TC, z, q)[0], x, jac=lambda z: cons(TC, z, q)[1], method="trf",
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200)
        x = sol.x
    for _ in range(iters):
        r, J = cons(TC, x, q)
        if np.linalg.norm(r) < 1e-13:
            return x, True
        x = x - np.linalg.lstsq(J, r, rcond=1e-12)[0]
    return x, np.linalg.norm(cons(TC, x, q)[0]) < 1e-11


def optimize(TC, x0, q):
    x, ok = project(TC, x0, q)
    if not ok:
        return None
    def f(x):
        e, Je = OE.err_and_jac(TC, x)
        return float(e @ e), 2 * Je.T @ e
    sol = minimize(lambda z: f(z)[0], x, jac=lambda z: f(z)[1], method="SLSQP",
                   constraints=[dict(type="eq", fun=lambda z: cons(TC, z, q)[0], jac=lambda z: cons(TC, z, q)[1]),
                                dict(type="ineq", fun=lambda z: coef_ineq(TC, z)[0], jac=lambda z: coef_ineq(TC, z)[1])],
                   options=dict(maxiter=400, ftol=1e-16))
    x, ok = project(TC, sol.x, q)
    if not ok or coef_ineq(TC, x)[0].min() < -1e-6:
        return None
    e, _ = OE.err_and_jac(TC, x)
    _, J = cons(TC, x, q)
    sv = np.linalg.svd(J, compute_uv=False)
    return float(np.linalg.norm(e)), x, int((sv > sv[0] * 1e-9).sum())


def main(starts=30):
    out = {}
    t0 = time.time()
    for fam in ("CRK5", "CRK6"):
        TC = OE.load_module(fam)
        d = json.load(open(os.path.join(R.ORIG_DATA, f"Emin_{fam}_100.json")))
        X = [np.array(r["x"]) for r in sorted(d["results"], key=lambda r: r["E"])]
        seen, starts_x = set(), []
        for x in X:
            k = round(float(np.linalg.norm(OE.err_and_jac(TC, x)[0])), 9)
            if k not in seen:
                seen.add(k)
                starts_x.append(x)
        starts_x = starts_x[:starts]
        named = (["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5"] if fam == "CRK5" else
                 ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam"])
        rng = np.random.default_rng(0)
        for nm in named:
            try:
                x = OE.x_from_tableau(TC, nm)
            except Exception:
                continue
            starts_x.append(x)
            for _ in range(5):                     # random walk on the family from each named tableau
                for _ in range(10):
                    Tb = TC.tangent_basis(x, tol=1e-9)
                    y = TC.newton_project(x + Tb @ rng.normal(size=Tb.shape[1]) * 0.05, iters=8)
                    if np.linalg.norm(TC.residual_and_jacobian(y)[0]) < 1e-12:
                        x = y
                starts_x.append(x.copy())
        out[fam] = {}
        for q in range(0, 5):
            best = None
            nok = 0
            for x0 in starts_x:
                r = optimize(TC, x0, q)
                if r is None:
                    continue
                nok += 1
                if best is None or r[0] < best[0]:
                    best = r
            if best is None:
                out[fam][q] = dict(E=None, feasible_starts=0)
                print(f"[{time.time() - t0:5.0f}s] {fam} q={q}: no feasible tableau found", flush=True)
                continue
            out[fam][q] = dict(E=best[0], feasible_starts=nok, rank=best[2], x=best[1].tolist())
            base = out[fam][0]["E"]
            print(f"[{time.time() - t0:5.0f}s] {fam} q={q} ({len(starts_x)} starts, {nok} feasible): min E = {best[0]:.4e} "
                  f"(x{best[0] / base:.2f}, effective x{(best[0] / base) ** (1 / (5 if fam == 'CRK5' else 6)):.3f} in h), "
                  f"rank {best[2]}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
