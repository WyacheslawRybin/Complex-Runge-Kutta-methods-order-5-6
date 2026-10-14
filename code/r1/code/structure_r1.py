"""Evidence for the local dimensions of Section 3 (review items 3.1 and 3.6).

1. Continuation along every tangent direction.  At a tableau x0 with Jacobian rank r, each of the
   n - r right singular vectors of the null space is followed by a predictor--corrector walk:
   predictor x + delta*t, corrector = Gauss--Newton steps with the pseudo-inverse truncated at rank r.
   Recorded: the distance |x - x0| actually travelled, the residual after correction, the rank of the
   Jacobian along the walk, and the change of E_{p+1} and max|coef|.
2. Split of the tangent space of the CCK6 tableaux into directions that move A and directions that
   move only the weights (the cascade levels 2 and 3 have free weights for a fixed A).

Double precision with the analytic Jacobian of code/tune_cck6.py.  Writes ../data/structure_r1.json.
"""
import json, os
import numpy as np
import r1common as R
from optimize_E import load_module, x_from_tableau

OUT = os.path.join(R.DATA, "structure_r1.json")
TOL = 1e-9                     # relative rank threshold (the gaps are larger than 1e-6 .. 1e-14)
CASES = [("CRK6", "CRK6-opt"), ("CRK6", "CRK6-0"), ("CCK65", "CCK6(5)"), ("CCK65", "CCK6(5)-s"),
         ("CCK6-cascade", "CCK6-casc"), ("CRK5", "CRK5-opt")]


def rank_of(J):
    sv = np.linalg.svd(J, compute_uv=False)
    return int(np.sum(sv > TOL * sv[0])), sv


def correct(TC, x, r, iters=8):
    for _ in range(iters):
        res, J = TC.residual_and_jacobian(x)
        if np.abs(res).max() < 1e-15:
            break
        U, sv, Vt = np.linalg.svd(J, full_matrices=False)
        x = x - Vt[:r].T @ ((U[:, :r].T @ res) / sv[:r])
    return x, float(np.abs(TC.residual(x)).max())


def walk(TC, x0, r, t0, delta=0.02, dist=0.5, maxsteps=60):
    x, t = x0.copy(), t0.copy()
    path = []
    for step in range(maxsteps):
        x, res = correct(TC, x + delta * t, r)
        rk, sv = rank_of(TC.residual_and_jacobian(x)[1])
        path.append(dict(dist=float(np.linalg.norm(x - x0)), resid=res, rank=rk,
                         E=TC.error_constant(x), maxcoef=float(TC.max_coef(x))))
        if res > 1e-10 or rk != r or path[-1]["dist"] >= dist:
            break
        Tb = TC.tangent_basis(x, tol=TOL)
        t = Tb @ (Tb.T @ t)             # keep the orientation of the walk
        t /= np.linalg.norm(t)
    return path


def a_rows(TC):
    """Positions of Re a_ij and Im a_ij among the free real unknowns."""
    full = list(range(TC.NA)) + [TC.NZ + k for k in range(TC.NA)]
    pos = {q: i for i, q in enumerate(TC.KEEP)}
    return [pos[q] for q in full]


def main():
    out = {}
    for fam, name in CASES:
        TC = load_module(fam)
        x0 = x_from_tableau(TC, name)
        x0, res0 = correct(TC, x0, rank_of(TC.residual_and_jacobian(x0)[1])[0])
        J = TC.residual_and_jacobian(x0)[1]
        r, sv = rank_of(J)
        Tb = TC.tangent_basis(x0, tol=TOL)
        d = Tb.shape[1]
        rkA = np.linalg.matrix_rank(Tb[a_rows(TC), :], tol=1e-8)
        info = dict(family=fam, m=J.shape[0], n=J.shape[1], rank=r, dim=d, dim_A=int(rkA),
                    dim_weights_only=int(d - rkA), resid0=res0, E0=TC.error_constant(x0),
                    smallest_kept=float(sv[r - 1] / sv[0]),
                    largest_dropped=float(sv[r] / sv[0]) if r < len(sv) else 0.0)
        # free weights of every level for the fixed A (rank of the weight matrix M_p(A))
        A, W = TC.unpack(x0)
        ph = TC.phi_all(A, TC.PMAX, deriv=False)
        lev = []
        for L, (m, p) in enumerate(TC.LEV):
            V = np.array([ph[t][0][:m] for k in range(1, p + 1) for t in TC.TREES[k]])
            M = np.hstack([V.real, -V.imag])[:, [q for q in range(2 * m) if q != m]]
            rk = int(np.linalg.matrix_rank(M, tol=1e-9 * np.linalg.norm(M, 2)))
            lev.append(dict(m=m, p=p, unknowns=2 * m - 1, conditions=int(M.shape[0]), rank=rk,
                            free=2 * m - 1 - rk))
        info["levels"] = lev
        if fam != "CRK5":
            walks = []
            for k in range(d):
                p = walk(TC, x0, r, Tb[:, k])
                last = p[-1]
                walks.append(dict(direction=k, steps=len(p), dist=last["dist"],
                                  max_resid=max(q["resid"] for q in p), ranks=sorted({q["rank"] for q in p}),
                                  E_ratio=last["E"] / info["E0"], maxcoef=last["maxcoef"]))
                print(f"  {name} dir {k}: {len(p)} steps, distance {last['dist']:.3f}, "
                      f"max residual {walks[-1]['max_resid']:.1e}, ranks {walks[-1]['ranks']}, "
                      f"E ratio {walks[-1]['E_ratio']:.3f}", flush=True)
            info["walks"] = walks
        out[name] = info
        print(f"{name}: {info['m']}x{info['n']}, rank {r}, dim {d} = {rkA} (A) + {d - rkA} (weights only); "
              f"levels {[(q['m'], q['rank'], q['free']) for q in lev]}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
