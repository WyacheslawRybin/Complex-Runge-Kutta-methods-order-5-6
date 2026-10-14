"""
Minimize the leading error constant E_{p+1} = ||e||_2 of the propagated level on a family
of projected tableaux under the bound max|coef| <= BOUND (all a_ij and all weights).

Families (the manifold machinery of tune_cck6.py is reused):
  CRK5          five stages, order 5                 -> E_6
  CRK6          six stages, order 6                  -> E_7
  CCK65         pair CCK6(5), levels (6,6), (5,5)    -> E_7 of the sixth-order level
Method: Levenberg-Marquardt (Gauss-Newton) on the residual vector
  [ e(x) ; rho * max(0, |z_k| - BOUND) ]
restricted to the tangent space of the solution set, with a minimum-norm Newton return
to the exact order conditions after every step.  Starts: the tableaux of the paper and
random tangent displacements of them.  Double precision; the results are refined and
re-evaluated in multiple precision by crk.Tableau.refine.
(The stored runs were made before the renaming CRK5-opt -> CRK5-s, CRK6-opt -> CRK6-s,
so their start labels use the old names.)
Usage:  python optimize_E.py FAMILY [BOUND] [NRANDOM]
Writes ../data/Emin_<FAMILY>_<BOUND>.json
"""
import importlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

STARTS = {"CRK5": ["CRK5-s", "CRK5-1", "CRK5-2", "GJM5"],
          "CRK6": ["CRK6-0", "CRK6-s", "CRK6-lam", "CCK6(5)", "CCK6(5)-s", "CCK6-casc"],
          "CCK65": ["CCK6(5)", "CCK6(5)-s", "CCK6-casc"]}

def load_module(levels):
    argv = sys.argv
    sys.argv = ["tune_cck6.py", levels]
    sys.modules.pop("tune_cck6", None)
    TC = importlib.import_module("tune_cck6")
    sys.argv = argv
    return TC

def x_from_tableau(TC, name):
    d = json.load(open(os.path.join(DATA, "tableaux.json")))[name]
    A = np.zeros((TC.S, TC.S), complex)
    for i in range(TC.S):
        for j in range(i):
            A[i, j] = complex(float(d["A"][i][j][0]), float(d["A"][i][j][1]))
    W = [np.array([complex(float(re), 0.0 if i == 0 else float(im)) for i, (re, im) in enumerate(d["b"][L][:m])])
         for L, (m, _) in enumerate(TC.LEV)]
    return TC.pack(A, W)

def err_and_jac(TC, x):
    """Error coefficients e(t) of level 0 over the trees of order p+1 and their Jacobian."""
    A, W = TC.unpack(x)
    m, p = TC.LEV[0]
    ph = TC.phi_all(A, p + 1)
    trees = TC.TREES[p + 1]
    e = np.empty(len(trees))
    Jc = np.zeros((len(trees), TC.NZ), complex)
    for q, t in enumerate(trees):
        v, dv = ph[t]
        s = TC.sigma(t)
        e[q] = ((W[0] @ v[:m]).real - 1.0 / TC.gamma(t)) / s
        Jc[q, :TC.NA] = (W[0] @ dv[:m]) / s
        Jc[q, TC.WOFF[0]:TC.WOFF[0] + m] = v[:m] / s
    return e, np.hstack([Jc.real, -Jc.imag])[:, TC.KEEP]

def pen_and_jac(TC, x, bound, rho):
    """Soft bound rho*max(0, |z_k| - bound) on every complex coefficient."""
    xx = np.zeros(2 * TC.NZ)
    xx[TC.KEEP] = x
    re, im = xx[:TC.NZ], xx[TC.NZ:]
    mod = np.hypot(re, im)
    act = np.where(mod > bound)[0]
    P = rho * (mod[act] - bound)
    J = np.zeros((len(act), 2 * TC.NZ))
    for r, k in enumerate(act):
        J[r, k] = rho * re[k] / mod[k]
        J[r, TC.NZ + k] = rho * im[k] / mod[k]
    return P, J[:, TC.KEEP]

def F(TC, x, bound, rho):
    e, _ = err_and_jac(TC, x)
    P, _ = pen_and_jac(TC, x, bound, rho)
    return float(e @ e + P @ P)

def minimize(TC, x, bound, rho=1.0, iters=400, verbose=False):
    x = TC.newton_project(x, iters=10)
    f = F(TC, x, bound, rho)
    mu = 1e-6
    for it in range(iters):
        T = TC.tangent_basis(x)
        e, Je = err_and_jac(TC, x)
        P, Jp = pen_and_jac(TC, x, bound, rho)
        R = np.concatenate([e, P])
        JT = np.vstack([Je, Jp]) @ T
        H, g = JT.T @ JT, JT.T @ R
        improved = False
        for _ in range(12):
            d = -np.linalg.solve(H + mu * np.eye(len(g)) * max(1e-30, np.trace(H) / len(g)), g)
            xc = TC.newton_project(x + T @ d, iters=8)
            if np.linalg.norm(TC.residual(xc)) < 1e-11:
                fc = F(TC, xc, bound, rho)
                if fc < f:
                    rel = (f - fc) / f
                    x, f, improved = xc, fc, True
                    mu = max(mu / 4, 1e-12)
                    break
            mu *= 8
        if verbose and it % 20 == 0:
            print(f"    it {it:3d}  E = {np.sqrt(f):.6e}  max|coef| = {TC.max_coef(x):.3f}  mu = {mu:.1e}", flush=True)
        if not improved or rel < 1e-12:
            break
    return x, np.sqrt(f), it

def main(family, bound=100.0, nrandom=12, seed=1):
    TC = load_module(family)
    rng = np.random.default_rng(seed)
    starts = []
    for name in STARTS[family]:
        x0 = TC.newton_project(x_from_tableau(TC, name), iters=10)
        starts.append((name, x0))
        T = TC.tangent_basis(x0)
        for k in range(nrandom):
            sig = [0.05, 0.15, 0.4][k % 3]
            xr = TC.newton_project(x0 + T @ (sig * rng.standard_normal(T.shape[1])), iters=12)
            if np.linalg.norm(TC.residual(xr)) < 1e-11:
                starts.append((f"{name}+rand{k}", xr))
    print(f"{family}: {len(starts)} starts, bound {bound}", flush=True)
    res = []
    t0 = time.time()
    for name, x0 in starts:
        E0 = TC.error_constant(x0)
        x1, E1, it = minimize(TC, x0, bound)
        r = np.linalg.norm(TC.residual(x1))
        res.append(dict(start=name, E_start=E0, E=TC.error_constant(x1), maxcoef=float(TC.max_coef(x1)),
                        resid=float(r), iters=it, x=x1.tolist()))
        print(f"  [{time.time() - t0:6.0f}s] {name:18s} E {E0:.4e} -> {res[-1]['E']:.6e}  "
              f"max|coef| {res[-1]['maxcoef']:8.3f}  resid {r:.1e}  ({it} it)", flush=True)
    res.sort(key=lambda d: d["E"])
    json.dump(dict(family=family, bound=bound, levels=TC.LEV, S=TC.S, results=res),
              open(os.path.join(DATA, f"Emin_{family}_{bound:g}.json"), "w"), indent=1)
    print("best:", res[0]["start"], f"E = {res[0]['E']:.6e}, max|coef| = {res[0]['maxcoef']:.3f}")

if __name__ == "__main__":
    fam = sys.argv[1]
    bound = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    nr = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    main(fam, bound, nr)
