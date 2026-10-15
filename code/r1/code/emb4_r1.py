"""Fourth-order formulas of six-stage sixth-order tableaux and the cascades of the CCK6 pairs (internal review 6).

For a tableau (A, b) that satisfies the 37 projected conditions of order six, let
    d      = A^T b - b o (1 - c)               (D(1) residual, Proposition emb5),
    delta2 = A^T (b o c) - b o (1 - c^2)/2     (D(2) residual),
    d2     = A^T d - d o (1 - c).
By the identities of the type (d1), Re(w^T Phi_u) = 0 for rho(u) <= 4 and w = d, delta2, d2, so
b + t1 d + t2 delta2 + t3 d2 has order four for real t.  The last column of A vanishes, so the sixth components
are -b6(1-c6), -b6(1-c6^2)/2 and b6(1-c6)^2; for a real c6 they are real multiples of b6.  Zero fifth (complex)
and sixth (real) weights are three real equations in t, so a fourth-order formula on four stages exists if the
3 x 3 matrix is nonsingular; the levels on three, two and one stage are underdetermined linear systems.
This script checks the identities, the rank of (d, delta2, d2), the 3 x 3 systems and the lower levels at the
CCK6 tableaux and at CCK6(5)-v (60 digits).  Writes data/emb4_r1.json.
"""
import json
import os

import mpmath as mp

import r1common as R
import crk
from tableaux import load_refined

mp.mp.dps = 60
OUT = os.path.join(R.DATA, "emb4_r1.json")


def vectors(T):
    A, c, b = T.A, T.c, T.levels[0][2]
    n = T.s
    d = [mp.fsum(b[i] * A[i][j] for i in range(n)) - b[j] * (1 - c[j]) for j in range(n)]
    dl2 = [mp.fsum(b[i] * c[i] * A[i][j] for i in range(n)) - b[j] * (1 - c[j] ** 2) / 2 for j in range(n)]
    dd2 = [mp.fsum(d[i] * A[i][j] for i in range(n)) - d[j] * (1 - c[j]) for j in range(n)]
    return d, dl2, dd2


def level_lsq(T, m, q):
    ph = T.phi(q)
    rows, rhs = [], []
    for t in crk.trees_upto(q):
        rows.append([mp.re(ph[t][i]) for i in range(m)] + [-mp.im(ph[t][i]) for i in range(1, m)])
        rhs.append(mp.mpf(1) / crk.gamma(t))
    M, r = mp.matrix(rows), mp.matrix(rhs)
    U, S, Vt = mp.svd_r(M)
    k = sum(1 for s in S if s > S[0] * mp.mpf(10) ** -40)
    x = mp.matrix(M.cols, 1)
    for i in range(k):
        coef = mp.fsum(U[j, i] * r[j] for j in range(M.rows)) / S[i]
        for l in range(M.cols):
            x[l] += coef * Vt[i, l]
    return float(mp.norm(M * x - r)), k


def check(name, T):
    b = T.levels[0][2]
    W = vectors(T)
    ph = T.phi(4)
    ident = max(abs(mp.re(mp.fsum(w[i] * ph[t][i] for i in range(6)))) for w in W for t in crk.trees_upto(4))
    realvec = lambda v: [mp.re(x) for x in v] + [mp.im(x) for x in v[1:]]
    sv = mp.svd_r(mp.matrix([realvec(w) for w in W]), compute_uv=False)
    M = mp.matrix([[mp.re(w[4]) for w in W], [mp.im(w[4]) for w in W], [mp.re(w[5]) for w in W]])
    cond = mp.norm(M) * mp.norm(mp.inverse(M))
    th = mp.lu_solve(M, mp.matrix([-mp.re(b[4]), -mp.im(b[4]), -mp.re(b[5])]))
    b4 = [b[i] + mp.fsum(th[k] * W[k][i] for k in range(3)) for i in range(6)]
    r4 = max(abs(mp.re(mp.fsum(b4[i] * ph[t][i] for i in range(6))) - mp.mpf(1) / crk.gamma(t)) for t in crk.trees_upto(4))
    out = dict(identity_residual=float(ident), sv_d_delta2_d2=[float(s) for s in sv], cond3=float(cond),
               b4_resid=float(r4), b4_tail=[float(abs(b4[4])), float(abs(b4[5]))],
               imag_c6=float(abs(mp.im(T.c[5]))))
    for m in (3, 2, 1):
        out[f"level{m}"] = level_lsq(T, m, m)
    print(f"{name:10s} identities {out['identity_residual']:.1e}; singular values of (d, delta2, d2) "
          + " ".join(f"{s:.2e}" for s in out["sv_d_delta2_d2"]) + f"; 3x3 cond {out['cond3']:.2e}; "
          f"4-stage level residual {out['b4_resid']:.1e}; lower levels "
          + ", ".join(f"{m}: rank {out[f'level{m}'][1]} resid {out[f'level{m}'][0]:.0e}" for m in (3, 2, 1)))
    return out


def main():
    TT = load_refined()
    V = json.load(open(os.path.join(R.DATA, "cck6v_r1.json")))["CCK6(5)-v"]
    cz = lambda p: mp.mpc(mp.mpf(p[0]), mp.mpf(p[1]))
    TV = crk.Tableau("CCK6(5)-v", [[cz(V["A"][i][j]) for j in range(6)] for i in range(6)],
                     [(6, 6, [cz(v) for v in V["b"][0]])])
    res = {nm: check(nm, T) for nm, T in (("CCK6(5)", TT["CCK6(5)"].level(0)), ("CCK6(5)-s", TT["CCK6(5)-s"].level(0)),
                                          ("CCK6-casc", TT["CCK6-casc"].level(0)), ("CCK6(5)-v", TV))}
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
