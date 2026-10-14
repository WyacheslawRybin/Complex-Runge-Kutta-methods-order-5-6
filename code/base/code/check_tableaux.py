"""
Verification of every tableau of the paper in multiple precision:
  * Newton refinement of the printed coefficients (minimum-norm correction),
  * residual of the projected order conditions of every level,
  * residual of the full (unprojected) complex conditions,
  * leading error constant E_{p+1} of every level, max |coefficient|,
  * stability polynomial of every level against the truncated exponential,
  * rank of the Jacobian of all conditions and the local dimension of the solution set,
  * rank of the weight matrix M(A) of the propagated level.
Writes ../data/tableaux.json (all coefficients to 45 digits) and prints a summary.
"""
import json, os, sys, time
from math import factorial
import numpy as np
import mpmath as mp
import crk
from tableaux import all_tableaux

mp.mp.dps = 60
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

def jac_rank(T):
    """SVD of d(residuals)/d(free real vars) by forward differences in mp."""
    x = T.get_x()
    F = lambda xx: [v for lev in T.set_x(xx).residuals() for v in lev]
    r0 = F(x)
    h = mp.mpf(10) ** -30
    J = np.zeros((len(r0), len(x)))
    for k in range(len(x)):
        xp = x[:]; xp[k] += h
        J[:, k] = [float((a - b) / h) for a, b in zip(F(xp), r0)]
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-12 * sv[0]))
    return rank, len(x), sv

def weight_matrix_rank(T):
    m, p, b = T.levels[0]
    ph = T.phi(p)
    ts = crk.trees_upto(p)
    P = np.array([[float(mp.re(ph[t][i])) for i in range(m)] for t in ts])
    Q = np.array([[float(mp.im(ph[t][i])) for i in range(m)] for t in ts])
    M = np.hstack([P, -Q]) if T.projected else P
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > 1e-12 * sv[0])), M.shape, sv

def cstr(z, d=45):
    return [mp.nstr(mp.re(z), d), mp.nstr(mp.im(z), d)]

def main():
    res = {}
    for name, T in all_tableaux().items():
        t0 = time.time()
        r_printed = max(abs(v) for lev in T.residuals() for v in lev)
        exact = name in ("DP5", "CK5", "Butcher6", "DP6(5)", "Verner6(5)", "RK4")
        if exact:
            R, shift, r_ref = T, mp.mpf(0), r_printed
        else:
            R, shift, r_ref = T.refine()
        info = dict(s=R.s, cost=R.cost, projected=R.projected,
                    levels=[(m, p) for m, p, _ in R.levels],
                    resid_printed=float(r_printed), resid_refined=float(r_ref), refine_shift=float(shift),
                    E=[float(R.E(L)) for L in range(len(R.levels))],
                    maxcoef=float(R.maxcoef()),
                    full_complex_resid=float(R.full_complex_residual(0)) if R.projected else 0.0)
        # stability polynomial of every level vs T_m (meaningful when m = p)
        st = []
        for L, (m, p, _) in enumerate(R.levels):
            co = R.stab_coeffs(L)
            dev = [float(co[k] - mp.mpf(1) / factorial(k)) for k in range(len(co))]
            st.append(dict(coeffs=[float(v) for v in co], dev_from_exp=dev))
        info["stability"] = st
        rk, n, sv = jac_rank(R)
        info.update(jac_rank=rk, n_free=n, local_dim=n - rk, jac_sv=[float(v) for v in sv])
        mr, shape, msv = weight_matrix_rank(R)
        info.update(M_rank=mr, M_shape=list(shape), M_sv=[float(v) for v in msv])
        info["c"] = [cstr(v, 25) for v in R.c]
        info["A"] = [[cstr(R.A[i][j]) for j in range(i)] for i in range(R.s)]
        info["b"] = [[cstr(v) for v in b] for _, _, b in R.levels]
        res[name] = info
        print(f"{name:11s} printed {float(r_printed):9.2e} -> refined {float(r_ref):9.2e} (shift {float(shift):8.1e})"
              f"  E={info['E'][0]:.5e} max|c|={info['maxcoef']:.4f} full-C {info['full_complex_resid']:.3g}"
              f"  rank J={rk}/{n} dim={n - rk}  rank M={mr} {tuple(shape)}  [{time.time() - t0:.0f}s]", flush=True)
    json.dump(res, open(os.path.join(OUT, "tableaux.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
