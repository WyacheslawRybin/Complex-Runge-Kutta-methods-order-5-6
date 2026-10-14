"""
Existence of an embedded formula of order q on the stages of a fixed tableau.

The embedded weights solve the linear system M_q(A) x = r (x = Re bh_1..s, Im bh_2..s
for a projected method, Re bh_1..s for a real one, plus one real FSAL weight).
The propagated weights b are a solution, so a genuine embedded formula exists iff
ker M_q(A) is nontrivial.  All singular values are computed in multiple precision,
so that a zero singular value is distinguished from a small one by many digits.

Second part: local dimension of the set of six-stage sixth-order projected tableaux
that admit a fifth-order embedded formula on the same six stages, computed at the
CCK6(5) tableau from the Jacobian of the combined system
    Re(b^T Phi_t) = 1/gamma(t)  (rho(t) <= 6),   Re(bh^T Phi_t) = 1/gamma(t)  (rho(t) <= 5).
Usage:  python embedded_rank.py
"""
import mpmath as mp
import crk
from tableaux import load_refined
from embedded import stage_phi

mp.mp.dps = 50

def weight_matrix(T, q, fsal):
    ph = stage_phi(T, q, fsal)
    cols = [("re", i) for i in range(T.s)]
    if T.projected:
        cols += [("im", i) for i in range(1, T.s)]
    if fsal:
        cols.append(("re", T.s))
    rows = [[mp.re(ph[t][i]) if k == "re" else -mp.im(ph[t][i]) for k, i in cols] for t in crk.trees_upto(q)]
    return mp.matrix(rows)

def spectrum(M):
    return sorted(mp.svd_r(M, compute_uv=False), reverse=True)

def report(name, T, q):
    for fsal in (False, True):
        M = weight_matrix(T, q, fsal)
        sv = spectrum(M)
        ker = M.cols - sum(1 for v in sv if v > mp.mpf(10) ** -35)
        print(f"{name:10s} q={q} {'FSAL' if fsal else '    '} {M.rows}x{M.cols}  kernel {ker}   "
              f"smallest singular values: " + "  ".join(mp.nstr(v, 3) for v in sv[-3:]))

def combined_dimension(T, bh):
    """Local dimension of {(A, b, bh)} solving order 6 for b and order 5 for bh on all six stages."""
    s = T.s
    aslots = [(i, j) for i in range(s) for j in range(i)]
    def unpack(x):
        A = [[mp.mpc(0)] * s for _ in range(s)]
        k = 0
        for i, j in aslots:
            A[i][j] = mp.mpc(x[k], x[k + 1]); k += 2
        def w(k):
            v = [mp.mpc(x[k], 0)]
            k += 1
            for i in range(1, s):
                v.append(mp.mpc(x[k], x[k + 1])); k += 2
            return v, k
        b, k = w(k)
        c, k = w(k)
        return A, b, c
    def F(x):
        A, b, c = unpack(x)
        U = crk.Tableau("u", A, [(s, 6, b), (s, 5, c)])
        return [v for lev in U.residuals() for v in lev]
    x = []
    for i, j in aslots:
        x += [mp.re(T.A[i][j]), mp.im(T.A[i][j])]
    for v in (T.levels[0][2], bh):
        x.append(mp.re(v[0]))
        for i in range(1, s):
            x += [mp.re(v[i]), mp.im(v[i])]
    r0 = F(x)
    h = mp.mpf(10) ** -25
    J = mp.matrix(len(r0), len(x))
    for k in range(len(x)):
        xp = x[:]; xp[k] += h
        rp = F(xp)
        for q in range(len(r0)):
            J[q, k] = (rp[q] - r0[q]) / h
    sv = spectrum(J)
    rank = sum(1 for v in sv if v > mp.mpf(10) ** -18)
    gap = (mp.nstr(sv[rank - 1], 3), mp.nstr(sv[rank], 3) if rank < len(sv) else "-")
    print(f"combined system at {T.name}: {J.rows} equations, {J.cols} unknowns, max residual "
          f"{mp.nstr(max(abs(v) for v in r0), 3)}, rank {rank} (singular values {gap[0]} | {gap[1]}), "
          f"solution set of dimension {J.cols - rank}")
    return J.cols - rank

if __name__ == "__main__":
    T = load_refined()
    print("--- complex (projected) tableaux")
    report("CRK5-opt", T["CRK5-opt"], 4)
    for k in ("CRK6-0", "CRK6-opt", "CRK6-lam"):
        report(k, T[k], 5)
    for k in ("CRK6-0", "CRK6-opt"):
        report(k, T[k], 4)
    for k in ("CCK6(5)", "CCK6(5)-s", "CCK6-casc"):
        U = T[k]
        report(k, crk.Tableau(k, U.A, [U.levels[0]]), 5)
    print("--- real tableaux (weights real)")
    for k in ("Butcher6", "RK6-opt", "DP6(5)", "Verner6(5)"):
        U = T[k]
        report(k, crk.Tableau(k, U.A, [U.levels[0]], projected=False), 5)
    print("--- size of the set of CRK6 tableaux with a fifth-order embedded formula")
    U = T["CCK6(5)"]
    bh = U.levels[1][2] + [mp.mpc(0)]
    combined_dimension(crk.Tableau("CCK6(5)", U.A, [U.levels[0]]), bh)
