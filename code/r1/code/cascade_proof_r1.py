"""Existence of an exact cascade near CCK6-cascade (internal review 7).

At the sixth-order level of CCK6-cascade the 37 conditions have rank 36, so the test of Proposition 8 (37
conditions and Im c_6 = 0) cannot succeed.  The dependency is explained by the assumption of
code/conjecture_proof_r1.py in its strong form (W0): for a real beta,
    w = c o Ac + beta A c^2 - (3 + 2 beta) A^2 c = 0      on all six stages
(stages 1 and 2 hold for every explicit tableau).  Then P w = 0 for all eight matrices P, so all eight
combinations G_P vanish identically, and the conditions of the eight trees t_1 follow from the others.
The system E0 = (W0) (8 real equations) + the 29 other conditions has 37 equations in the 41 unknowns and
beta (dimension five, the tangent dimension of the component); with Im c_6 = 0 added, 38 equations.

Proof: with four unknowns fixed, the Krawczyk test (mpmath intervals and Arb) for E0 + {Im c_6 = 0} proves an
exact zero; it satisfies the 37 conditions and has a real c_6 != 1, so by Proposition 7(b) it carries the
exact fifth-order formula b + d/(1 - c_6).  Over the box, the 3 x 3 matrix of Proposition emb4(c) is
nonsingular (certificate ||I - Y M|| < 1), which gives the exact fourth-order formula on four stages, and the
real matrices of the conditions of the levels on three, two and one stage have full row rank (certificate for
a square submatrix), so these levels exist as well.  In addition, the 37 equations E0 are tested alone
(four unknowns fixed and beta free gives 38 unknowns against 37 equations, so five are fixed): if the
Krawczyk test succeeds and a 36 x 36 submatrix of the Jacobian of the 37 conditions is nonsingular, then near
the point the solution set of the 37 conditions is a five-dimensional manifold on which (W0) holds.
Writes data/cascade_proof_r1.json.  Usage: python cascade_proof_r1.py
"""
import json
import os
import time

import numpy as np
import scipy.linalg as sl
import mpmath as mp
from mpmath import iv as _iv

import r1common as R
import crk
import krawczyk_r1 as KR
import conjecture_proof_r1 as CP
from krawczyk_r1 import D, Cx
from tableaux import load_refined

mp.mp.dps = 90
_iv.prec = 320
RHO = KR.RHO
OUT = os.path.join(R.DATA, "cascade_proof_r1.json")
S = 6
TREES = crk.trees_upto(6)
DROP8 = [CP.TRIPLES[P][0] for P in CP.TRIPLES]
BK = {"iv": _iv}          # current interval backend


def ivm(v):
    return BK["iv"].mpf(v)


def build(vals, free_pos):
    zero = D(ivm(0))
    var = lambda q: D(vals[q], {free_pos[q]: ivm(1)} if q in free_pos else {})
    A, q = [[None] * S for _ in range(S)], 0
    for i in range(S):
        for j in range(i):
            A[i][j] = Cx(var(q), var(q + 1))
            q += 2
    b = []
    for i in range(S):
        if i == 0:
            b.append(Cx(var(q), zero)); q += 1
        else:
            b.append(Cx(var(q), var(q + 1))); q += 2
    beta = var(q) if q < len(vals) else None
    return A, b, beta, zero


def mv(A, v, zero):
    w = [Cx(zero, zero)]
    for i in range(1, S):
        acc = A[i][0] * v[0]
        for j in range(1, i):
            acc = acc + A[i][j] * v[j]
        w.append(acc)
    return w


def phis(A, zero, pmax=6):
    cache = {}

    def Phi(t):
        if t not in cache:
            v = [Cx(D(ivm(1)), zero) for _ in range(S)]
            for u in t:
                pu = mv(A, Phi(u), zero)
                v = [v[i] * pu[i] for i in range(S)]
            cache[t] = v
        return cache[t]
    for t in crk.trees_upto(pmax):
        Phi(t)
    return cache


def system(vals, free_pos, mode):
    """mode 'E0c6': (W0) + 29 conditions + Im c6;  'E0': (W0) + 29 conditions;  '37': all 37 conditions."""
    A, b, beta, zero = build(vals, free_pos)
    ph = phis(A, zero)
    out = []
    for t in TREES:
        if mode != "37" and CP.ts(t) in DROP8:
            continue
        v = ph[t]
        acc = b[0] * v[0]
        for i in range(1, S):
            acc = acc + b[i] * v[i]
        out.append(acc.re - D(ivm(1) / crk.gamma(t)))
    if mode == "37":
        return out
    c = ph[((),)]
    Ac = mv(A, c, zero)
    r1 = [c[i] * Ac[i] for i in range(S)]
    r2 = mv(A, [c[i] * c[i] for i in range(S)], zero)
    r3 = mv(A, Ac, zero)
    k3 = D(ivm(3)) + D(ivm(2)) * beta
    for i in (2, 3, 4, 5):
        out += [r1[i].re + beta * r2[i].re - k3 * r3[i].re, r1[i].im + beta * r2[i].im - k3 * r3[i].im]
    if mode == "E0c6":
        acc = A[5][0].im
        for j in range(1, 5):
            acc = acc + A[5][j].im
        out.append(acc)
    return out


def evaluate(x, free, mode, box=None):
    free_pos = {q: k for k, q in enumerate(free)}
    vals = [ivm(v) for v in x]
    if box is not None:
        for q in free:
            vals[q] = vals[q] + ivm([-box, box])
    res = system(vals, free_pos, mode)
    J = [[r.g.get(k, ivm(0)) for k in range(len(free))] for r in res]
    return [r.v for r in res], J


mid = KR.mid


def backend(arb):
    if arb:
        import flint
        import krawczyk_arb_r1 as KA
        flint.ctx.prec = 320
        rho = KA.to_arb(KR.rounded(RHO, 200))
        return dict(iv=KA.ArbIV, rad=flint.arb(0, rho.mid()), conv=KA.to_arb,
                    inside=lambda v: bool(abs(v) < rho),
                    upper=lambda v: float(abs(v).mid() + abs(v).rad()),
                    absv=lambda v: abs(v), lt1=lambda a: bool(a < 1))
    return dict(iv=_iv, rad=_iv.mpf([-RHO, RHO]), conv=_iv.mpf,
                inside=lambda v: bool(-RHO < mp.mpf(v.a)) and bool(mp.mpf(v.b) < RHO),
                upper=lambda v: float(max(abs(mp.mpf(v.a)), abs(mp.mpf(v.b)))),
                absv=lambda v: _iv.mpf(max(abs(mp.mpf(v.a)), abs(mp.mpf(v.b)))),
                lt1=lambda a: bool(mp.mpf(a.b) < 1))


def nonsing_cert(Mi, Y, B):
    """||I - Y M||_inf for an interval matrix M and a point matrix Y."""
    n = len(Mi)
    Yi = [[B["conv"](Y[i][j]) for j in range(n)] for i in range(n)]
    ok, nrm = True, 0.0
    for i in range(n):
        acc = B["conv"](0)
        for j in range(n):
            e = B["conv"](1 if i == j else 0) - sum((Yi[i][l] * Mi[l][j] for l in range(n)), B["conv"](0))
            acc = acc + B["absv"](e)
        nrm = max(nrm, B["upper"](acc))
        ok = ok and B["lt1"](acc)
    return ok, nrm


def point_inverse(Mmid):
    Y = mp.inverse(mp.matrix(Mmid))
    return [[KR.rounded(Y[i, j], 200) for j in range(Y.cols)] for i in range(Y.rows)]


def solve_point(x, n_eq, mode, nfix_extra=0, nb=41):
    """Choose free unknowns (beta always free), round, Newton; returns x, free, fixed, C."""
    x = list(x)
    _, J0 = evaluate(x, list(range(len(x))), mode)
    Jf = np.array([[float(mid(z)) for z in row] for row in J0])
    Jp = Jf.copy()
    Jp[:, nb] *= 1e6
    piv = sl.qr(Jp, pivoting=True)[2]
    free, fixed = sorted(piv[:n_eq]), sorted(piv[n_eq:])
    assert nb in free
    sv = np.linalg.svd(Jf[:, free], compute_uv=False)
    for q in fixed:
        x[q] = KR.rounded(x[q], KR.FIX_BITS)
    for _ in range(10):
        F, J = evaluate(x, free, mode)
        Fm = mp.matrix([mid(z) for z in F])
        if mp.norm(Fm, mp.inf) < mp.mpf(10) ** -85:
            break
        dx = mp.lu_solve(mp.matrix([[mid(z) for z in row] for row in J]), Fm)
        for k, q in enumerate(free):
            x[q] -= dx[k]
    for q in free:
        x[q] = KR.rounded(x[q], 280)
    _, J = evaluate(x, free, mode)
    C = point_inverse([[mid(z) for z in row] for row in J])
    return x, free, fixed, C, float(sv[0] / sv[-1])


def krawczyk(x, free, mode, C, B):
    m = len(free)
    F, _ = evaluate(x, free, mode)
    _, JX = evaluate(x, free, mode, box=RHO)
    Ci = [[B["conv"](C[i][j]) for j in range(m)] for i in range(m)]
    worst, ok = 0.0, True
    for i in range(m):
        kk = -sum((Ci[i][j] * F[j] for j in range(m)), B["conv"](0))
        for j in range(m):
            e = B["conv"](1 if i == j else 0) - sum((Ci[i][l] * JX[l][j] for l in range(m)), B["conv"](0))
            kk += e * B["rad"]
        worst = max(worst, B["upper"](kk))
        ok = ok and B["inside"](kk)
    return ok, worst


def box_vals(x, free, B):
    vals = [B["conv"](v) for v in x]
    for q in free:
        vals[q] = vals[q] + B["rad"]
    return vals


def cascade_checks(x, free, B, Ys=None):
    """Over the box: c6 enclosure, the 3 x 3 matrix of Proposition emb4(c), and the lower levels."""
    vals = box_vals(x, free, B)
    A, b, _, zero = build(vals, {})
    ph = phis(A, zero, 3)
    cc = ph[((),)]
    re = lambda z: z.re.v
    im = lambda z: z.im.v
    # d, delta2, d2 (complex, as pairs of intervals)
    one = B["conv"](1)
    half = B["conv"](1) / 2

    def cmul(a, b_):
        return (a[0] * b_[0] - a[1] * b_[1], a[0] * b_[1] + a[1] * b_[0])
    Ac_ = [[(re(A[i][j]), im(A[i][j])) if j < i else (B["conv"](0), B["conv"](0)) for j in range(S)] for i in range(S)]
    bb = [(re(b[i]), im(b[i])) for i in range(S)]
    cv = [(re(cc[i]), im(cc[i])) for i in range(S)]

    def AT(v):
        out = []
        for j in range(S):
            sr, si = B["conv"](0), B["conv"](0)
            for i in range(j + 1, S):
                pr, pi = cmul(v[i], Ac_[i][j])
                sr, si = sr + pr, si + pi
            out.append((sr, si))
        return out
    omc = [(one - cv[j][0], -cv[j][1]) for j in range(S)]
    d = [(a[0] - p[0], a[1] - p[1]) for a, p in zip(AT(bb), [cmul(bb[j], omc[j]) for j in range(S)])]
    bc = [cmul(bb[i], cv[i]) for i in range(S)]
    c2 = [cmul(cv[i], cv[i]) for i in range(S)]
    omc2 = [((one - c2[j][0]) * half, (-c2[j][1]) * half) for j in range(S)]
    dl = [(a[0] - p[0], a[1] - p[1]) for a, p in zip(AT(bc), [cmul(bb[j], omc2[j]) for j in range(S)])]
    d2 = [(a[0] - p[0], a[1] - p[1]) for a, p in zip(AT(d), [cmul(d[j], omc[j]) for j in range(S)])]
    W = [d, dl, d2]
    M = [[w[4][0] for w in W], [w[4][1] for w in W], [w[5][0] for w in W]]
    Mmid = [[mid(z) if not hasattr(z, "mid") or not callable(getattr(z, "mid")) else mp.mpf(float(z.mid())) for z in row] for row in M]
    out = {}
    Y3 = Ys["M3"] if Ys else point_inverse(Mmid)
    out["M3"] = nonsing_cert(M, Y3, B)
    # lower levels: real matrices [Re Phi_t(i) | -Im Phi_t(i) (i >= 2)] for rho(t) <= m on stages 1..m
    lev_Y = {}
    for m in (3, 2, 1):
        rows = []
        for t in crk.trees_upto(m):
            rows.append([re(ph[t][i]) for i in range(m)] + [-im(ph[t][i]) for i in range(1, m)])
        nr = len(rows)
        Rm = np.array([[float(mid(z)) if not callable(getattr(z, "mid", None)) else float(z.mid()) for z in row] for row in rows])
        cols = sorted(int(q) for q in sl.qr(Rm, pivoting=True)[2][:nr])
        sub = [[row[k] for k in cols] for row in rows]
        Y = Ys[f"L{m}"] if Ys else point_inverse([[mp.mpf(Rm[i][k]) for k in cols] for i in range(nr)])
        lev_Y[f"L{m}"] = Y
        out[f"level{m}"] = nonsing_cert(sub, Y, B) + (nr, len(rows[0]))
    c6 = (re(cc[5]), im(cc[5]))
    return out, dict(M3=Y3, **lev_Y), c6


def run_cascade(T, st, B_list):
    x = list(T.get_x()) + [st["beta"]]
    x_ref = list(x)
    res = {}
    # (i) E0 + Im c6 = 0: existence of an exact sixth-order level with a real c6 (pair, cascade)
    xs, free, fixed, C, cond = solve_point(x, 38, "E0c6")
    shift = max(abs(a - b) for a, b in zip(xs[:41], x_ref[:41]))
    Ys = None
    for name, arb in B_list:
        B = backend(arb)
        BK["iv"] = B["iv"]
        try:
            ok, worst = krawczyk(xs, free, "E0c6", C, B)
            casc, Ys, c6 = cascade_checks(xs, free, B, Ys)
            c6re = B["upper"](c6[0])
        finally:
            BK["iv"] = _iv
        res[f"E0c6|{name}"] = dict(proved=ok, K_radius=worst, fixed=[int(q) for q in fixed], cond=cond, shift=float(shift),
                                   M3_cert=casc["M3"], levels={k: v for k, v in casc.items() if k.startswith("level")},
                                   c6_upper=c6re)
        print(f"  E0 + Im c6 [{name}]: {'PROVED' if ok else 'not proved'}, K - y~ within {worst:.1e}, cond {cond:.1e}, "
              f"shift {float(shift):.1e}; 3x3 cert {casc['M3'][1]:.1e} ({casc['M3'][0]}); levels "
              + ", ".join(f"{m}: {casc[f'level{m}'][1]:.1e} ({casc[f'level{m}'][0]}, {casc[f'level{m}'][2]}x{casc[f'level{m}'][3]})"
                          for m in (3, 2, 1)), flush=True)
    # (ii) E0 alone (37 equations, five unknowns fixed): the component is the zero set of E0
    _, J0 = evaluate(x, list(range(42)), "E0")
    Jf = np.array([[float(mid(z)) for z in row] for row in J0])
    rk = int((np.linalg.svd(Jf, compute_uv=False) > 1e-30).sum())
    xs2, free2, fixed2, C2, cond2 = solve_point(x, 37, "E0")
    _, J37 = evaluate(xs2[:41], list(range(41)), "37")
    J37f = np.array([[float(mid(z)) for z in row] for row in J37])
    sv37 = np.linalg.svd(J37f, compute_uv=False)
    rows = sorted(int(q) for q in sl.qr(J37f.T, pivoting=True)[2][:36])
    cols = sorted(int(q) for q in sl.qr(J37f[rows], pivoting=True)[2][:36])
    Y = point_inverse([[mp.mpf(J37f[r][k]) for k in cols] for r in rows])
    for name, arb in B_list:
        B = backend(arb)
        BK["iv"] = B["iv"]
        try:
            ok, worst = krawczyk(xs2, free2, "E0", C2, B)
            vals = box_vals(xs2[:41], [q for q in free2 if q < 41], B)
            free_pos = {q: k for k, q in enumerate(range(41))}
            res37 = system(vals, free_pos, "37")
            JI = [[res37[r].g.get(k, B["conv"](0)) for k in cols] for r in rows]
            cert = nonsing_cert(JI, Y, B)
        finally:
            BK["iv"] = _iv
        res[f"E0|{name}"] = dict(proved=ok, K_radius=worst, rank_E0=rk, fixed=[int(q) for q in fixed2], cond=cond2,
                                 sub36_cert=cert, rank37_numeric=int((sv37 > sv37[0] * 1e-30).sum()))
        print(f"  E0 alone [{name}]: rank {rk}, {'PROVED' if ok else 'not proved'}, K - y~ within {worst:.1e}, "
              f"36x36 cert {cert[1]:.1e} ({cert[0]})", flush=True)
    return res


def main():
    tabs = load_refined()
    T, _, _ = tabs["CCK6-casc"].level(0).refine(tol_digits=85, iters=10)
    st = CP.structure(T)
    w = st["w6"]
    print(f"CCK6-cascade: beta {float(st['beta']):.6f}, |w_1..5| {float(st['w15']):.1e}, |w_6| {float(abs(w)):.1e}, "
          f"c6 {mp.nstr(st['c6'], 12)}", flush=True)
    out = dict(beta=float(st["beta"]), w15=float(st["w15"]), w6=float(abs(w)), c6=float(mp.re(st["c6"])))
    out.update(run_cascade(T, st, [("mpmath", False), ("Arb", True)]))
    json.dump(out, open(OUT, "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
