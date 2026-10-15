"""Numerical evidence for Conjecture 1 (internal review 6).

Random walk on the CRK6 component through the sixth-order level of CCK6(5) (only the 37 sixth-order
conditions are imposed; tangent steps of size 0.03, minimum-norm Newton back, the rank 37 is checked at every
point, distance from the start at most 1; double precision, code/base/code/tune_cck6.py).  On the sample:
  * the affine-linear functions of the 41 real unknowns that vanish (expected: Re sum b - 1 and Im c6);
  * the assumption (W) of code/conjecture_proof_r1.py: min over real beta of |w_1..5| and |b_6 w_6|;
  * the imaginary parts of all elementary weights Phi_t(i), t of order <= 6, of all b^T Phi_t, and of the
    residuals of C(2), C(3), D(1), D(2), reported if they vanish on all samples.
Writes data/conjecture_r1.json.  Usage: python conjecture_r1.py [points]
"""
import json
import os
import sys

import numpy as np

import r1common as R
import optimize_E as OE

OUT = os.path.join(R.DATA, "conjecture_r1.json")


def ts(t):
    return "t" if not t else "[" + ",".join(ts(c) for c in t) + "]"


def sample(TC, x0, npts, rng):
    def rank(x):
        _, J = TC.residual_and_jacobian(x)
        sv = np.linalg.svd(J, compute_uv=False)
        return int((sv > sv[0] * 1e-9).sum())
    r0 = rank(x0)
    pts, x = [x0], x0
    while len(pts) < npts:
        Tb = TC.tangent_basis(x, tol=1e-9)
        y = TC.newton_project(x + Tb @ rng.normal(size=Tb.shape[1]) * 0.03, iters=8)
        r, _ = TC.residual_and_jacobian(y)
        if np.linalg.norm(r) > 1e-12 or rank(y) != r0 or np.linalg.norm(y - x0) > 1.0:
            continue
        x = y
        pts.append(y)
    return np.array(pts), r0


def main(npts=1500):
    TC = OE.load_module("CRK6")
    d = json.load(open(os.path.join(R.ORIG_DATA, "tableaux.json")))["CCK6(5)"]
    A = np.zeros((6, 6), complex)
    for i in range(6):
        for j in range(i):
            A[i, j] = complex(float(d["A"][i][j][0]), float(d["A"][i][j][1]))
    b = np.array([complex(float(re), 0.0 if i == 0 else float(im)) for i, (re, im) in enumerate(d["b"][0])])
    x0 = TC.newton_project(TC.pack(A, [b]), iters=10)
    P, r0 = sample(TC, x0, npts, np.random.default_rng(1))
    dist = float(np.linalg.norm(P - x0, axis=1).max())
    F1 = np.hstack([np.ones((len(P), 1)), P])
    F1 = F1 / np.linalg.norm(F1, axis=0)
    sv = np.linalg.svd(F1, compute_uv=False)
    k = int((sv < 1e-9 * sv[0]).sum())
    trees = [t for kk in range(1, 7) for t in TC.TREES[kk]]
    vals = {}
    wmax, zmin, betas = 0.0, np.inf, []
    for x in P:
        A, W = TC.unpack(x)
        c = A.sum(1)
        u = c * (A @ c) - 3 * (A @ A @ c)
        v = A @ c ** 2 - 2 * (A @ A @ c)
        beta = -np.real(np.vdot(v[2:5], u[2:5])) / np.vdot(v[2:5], v[2:5]).real
        w = u + beta * v                              # assumption (W) of conjecture_proof_r1.py
        wmax, zmin = max(wmax, np.abs(w[:5]).max()), min(zmin, abs(W[0][5] * w[5]))
        betas.append(beta)
    for x in P:
        A, W = TC.unpack(x)
        b = W[0]
        ph = TC.phi_all(A, 6, deriv=False)
        c = A.sum(1)
        Q = {}
        for t in trees:
            v = ph[t][0]
            for i in range(1, 6):
                Q[f"Im Phi_{ts(t)}({i + 1})"] = v[i].imag
            Q[f"Im b^T Phi_{ts(t)}"] = (b @ v).imag
        for kk, vec in ((1, A.T @ b - b * (1 - c)), (2, A.T @ (b * c) - b * (1 - c ** 2) / 2)):
            for j in range(6):
                Q[f"Im D{kk}_{j + 1}"] = vec[j].imag
        for kk in (2, 3):
            xi = A @ c ** (kk - 1) - c ** kk / kk
            for i in range(1, 6):
                Q[f"Im C{kk}_{i + 1}"] = xi[i].imag
        for q, v in Q.items():
            vals[q] = max(vals.get(q, 0.0), abs(v))
    # quantities that vanish trivially for every explicit tableau are dropped (Phi_t(2) = 0 for non-bushy t, etc.)
    trivial = {q for q in vals if q.endswith("(2)") and "Phi_[" in q and not set(q.split("_", 1)[1][:-3]) <= set("[t,]")}
    vanish = {q: v for q, v in vals.items() if v < 1e-9 and v != 0.0}
    nonvan = sorted((v, q) for q, v in vals.items() if v >= 1e-9)[:5]
    out = dict(points=len(P), rank=r0, max_distance=dist, affine_nullity=k, vanishing=vanish,
               smallest_nonvanishing=[[q, v] for v, q in nonvan],
               assumption_W=dict(max_w_stages1to5=float(wmax), min_abs_b6w6=float(zmin),
                                 beta_range=[float(min(betas)), float(max(betas))]))
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{len(P)} points, rank {r0}, distance up to {dist:.2f}; vanishing affine functions: {k}")
    print("vanishing (nontrivial, < 1e-9):", ", ".join(f"{q} {v:.0e}" for q, v in sorted(vanish.items(), key=lambda z: z[1])))
    print(f"assumption (W): max |w_1..5| {wmax:.1e}, min |b6 w6| {zmin:.2e}, beta in [{min(betas):.3f}, {max(betas):.3f}]")
    print("smallest nonvanishing:", ", ".join(f"{q} {v:.1e}" for v, q in nonvan))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
