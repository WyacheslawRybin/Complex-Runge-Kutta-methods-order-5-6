"""Proof of Conjecture 1 (internal review 7): Im c_6 and Im (Ac)_6 vanish on the CRK6 component through the
sixth-order levels of CCK6(5) and CCK6(5)-s.

Structure.  For a real beta let
    w(beta) = c o Ac + beta A c^2 - (3 + 2 beta) A^2 c     (complex 6-vector),
a real combination of the stage weights of the three trees [[t],t], [[t,t]], [[[t]]] that vanishes for the
exact weights c^3/2, c^3/3, c^3/6.  Always w_1 = w_2 = 0.  Assumption (W): w_3 = w_4 = w_5 = 0 for some real
beta, so that w = w_6 e_6.  For each P in {I, C, A, C^2, diag(Ac), CA, AC, A^2} the combination
    G_P = g_{P[[t],t]} + beta g_{P[[t,t]]} - (3 + 2 beta) g_{P[[[t]]]},   g_t = Re(b^T Phi_t) - 1/gamma(t),
equals Re(b^T P w) identically (the constants cancel, checked below in rational arithmetic).  Under (W),
A w = 0, so G_A = G_CA = G_AC = G_AA = 0 identically, and with z = b_6 w_6:
    G_I = Re z,  G_C = Re(z c_6),  G_CC = Re(z c_6^2),  G_Ac = Re(z (Ac)_6).
Hence if z != 0, the conditions of order <= 6 force z in iR and Im c_6 = Im (Ac)_6 = 0, and conversely the
system E = (W) + the 32 conditions other than g_t for the first trees of the triples P = A, CA, AC, A^2, C^2
implies all 37 conditions.

Proof near a tableau (computer-assisted, as Proposition 2(a)): E has 38 equations in the 41 unknowns and
beta.  With four unknowns fixed (column-pivoted QR), the Krawczyk test (code/krawczyk_r1.py, mpmath
intervals, 320 bits) proves an exact zero of E with a nonsingular Jacobian, so near it the zero set M_E of E
is a four-dimensional graph over the fixed unknowns; |z| > 0 on the box, so M_E lies in the solution set V
of the 37 conditions.  A 37 x 37 submatrix of the interval Jacobian of the 37 conditions over the box is
nonsingular, so V is a four-dimensional manifold there; by the invariance of domain M_E = V near the
point, i.e., (W), Im c_6 = 0 and Im (Ac)_6 = 0 hold on a neighbourhood in V, hence (real analyticity) on
the whole connected component of regular points of V.
Writes data/conjecture_proof_r1.json.  Usage: python conjecture_proof_r1.py
"""
import json
import os
import time
from fractions import Fraction

import numpy as np
import scipy.linalg as sl
import mpmath as mp
from mpmath import iv

import r1common as R
import crk
import krawczyk_r1 as KR
from krawczyk_r1 import D, Cx
from tableaux import load_refined

mp.mp.dps = 90
iv.prec = 320
RHO = mp.mpf(10) ** -50
OUT = os.path.join(R.DATA, "conjecture_proof_r1.json")
S = 6


def ts(t):
    return "t" if not t else "[" + ",".join(ts(c) for c in t) + "]"


TRIPLES = {          # P -> trees P[[t],t], P[[t,t]], P[[[t]]]
    "I": ("[[t],t]", "[[t,t]]", "[[[t]]]"),
    "C": ("[[t],t,t]", "[[t,t],t]", "[[[t]],t]"),
    "A": ("[[[t],t]]", "[[[t,t]]]", "[[[[t]]]]"),
    "CC": ("[[t],t,t,t]", "[[t,t],t,t]", "[[[t]],t,t]"),
    "Ac": ("[[t],[t],t]", "[[t,t],[t]]", "[[[t]],[t]]"),
    "CA": ("[[[t],t],t]", "[[[t,t]],t]", "[[[[t]]],t]"),
    "AC": ("[[[t],t,t]]", "[[[t,t],t]]", "[[[[t]],t]]"),
    "AA": ("[[[[t],t]]]", "[[[[t,t]]]]", "[[[[[t]]]]]"),
}
DROP = [TRIPLES[P][0] for P in ("A", "CA", "AC", "AA", "CC")]
TREES = crk.trees_upto(6)


def check_constants():
    """1/gamma of the three trees of every triple satisfy g1 + beta g2 - (3 + 2 beta) g3 = 0 for all beta."""
    G = {ts(t): Fraction(1, crk.gamma(t)) for t in TREES}
    for P, (t1, t2, t3) in TRIPLES.items():
        assert G[t1] - 3 * G[t3] == 0 and G[t2] - 2 * G[t3] == 0, P
    return True


# ---------------------------------------------------------------- structure at the refined tableaux (mp)
def structure(T):
    A, b = T.A, T.levels[0][2]
    c = T.c
    mv = lambda M, v: [mp.fsum(M[i][j] * v[j] for j in range(i)) for i in range(S)]
    Ac = mv(A, c)
    r1 = [c[i] * Ac[i] for i in range(S)]
    r2 = mv(A, [x ** 2 for x in c])
    r3 = mv(A, Ac)
    u = [r1[i] - 3 * r3[i] for i in range(S)]
    v = [r2[i] - 2 * r3[i] for i in range(S)]
    num = mp.fsum(mp.re(u[i] * mp.conj(v[i])) for i in (2, 3, 4))
    den = mp.fsum(abs(v[i]) ** 2 for i in (2, 3, 4))
    beta = -num / den
    w = [u[i] + beta * v[i] for i in range(S)]
    return dict(beta=beta, w15=max(abs(w[i]) for i in range(5)), w6=w[5], z=b[5] * w[5], c6=c[5],
                imc6=mp.im(c[5]), imac6=mp.im(Ac[5]))


# ---------------------------------------------------------------- the system E in interval arithmetic
def system(vals, free_pos, nb, full37=False):
    """vals: interval values of the 41 real unknowns (layout of crk.Tableau.get_x) followed by beta."""
    zero = D(iv.mpf(0))
    var = lambda q: D(vals[q], {free_pos[q]: iv.mpf(1)} if q in free_pos else {})
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
    beta = var(q)
    cache = {}

    def mv(v):
        w = [Cx(zero, zero)]
        for i in range(1, S):
            acc = A[i][0] * v[0]
            for j in range(1, i):
                acc = acc + A[i][j] * v[j]
            w.append(acc)
        return w

    def Phi(t):
        if t not in cache:
            v = [Cx(D(iv.mpf(1)), zero) for _ in range(S)]
            for u in t:
                pu = mv(Phi(u))
                v = [v[i] * pu[i] for i in range(S)]
            cache[t] = v
        return cache[t]

    out = []
    for t in TREES:
        if not full37 and ts(t) in DROP:
            continue
        ph = Phi(t)
        acc = b[0] * ph[0]
        for i in range(1, S):
            acc = acc + b[i] * ph[i]
        out.append(acc.re - D(iv.mpf(1) / crk.gamma(t)))
    if full37:
        return out
    c = Phi(((),))                                  # [t]: c
    Ac = mv(c)
    c2 = [c[i] * c[i] for i in range(S)]
    r1 = [c[i] * Ac[i] for i in range(S)]
    r2 = mv(c2)
    r3 = mv(Ac)
    k3 = D(iv.mpf(3)) + D(iv.mpf(2)) * beta
    neg = lambda d: zero - d
    for i in (2, 3, 4):
        wi_re = r1[i].re + beta * r2[i].re - k3 * r3[i].re
        wi_im = r1[i].im + beta * r2[i].im - k3 * r3[i].im
        out += [wi_re, wi_im]
    # z = b_6 w_6 (returned separately)
    w6 = Cx(r1[5].re + beta * r2[5].re - k3 * r3[5].re, r1[5].im + beta * r2[5].im - k3 * r3[5].im)
    z = b[5] * w6
    return out, z


def evaluate(xvals, free, box=None, full37=False):
    free_pos = {q: k for k, q in enumerate(free)}
    vals = [iv.mpf(v) for v in xvals]
    if box is not None:
        for q in free:
            vals[q] = vals[q] + iv.mpf([-box, box])
    if full37:
        res = system(vals, free_pos, None, True)
        z = None
    else:
        res, z = system(vals, free_pos, None)
    J = [[r.g.get(k, iv.mpf(0)) for k in range(len(free))] for r in res]
    return [r.v for r in res], J, z


mid = KR.mid


def certify(x, free, nb, C, sub, Y, arb=False):
    """Krawczyk test for E, |z| > 0 and ||I - Y J_37|| < 1 over the box, in mpmath intervals or in Arb."""
    global iv
    m = len(free)
    saved = iv
    if arb:
        import flint
        import krawczyk_arb_r1 as KA
        flint.ctx.prec = 320
        iv = KA.ArbIV
        rho = KA.to_arb(KR.rounded(RHO, 200))
        rad = flint.arb(0, rho.mid())
        inside = lambda v: bool(abs(v) < rho)
        upper = lambda v: float(abs(v).mid() + abs(v).rad())
        nonzero = lambda v: bool(v > 0) or bool(v < 0)
        lower = lambda v: max(0.0, float(abs(v).mid() - abs(v).rad()))
        conv = KA.to_arb
    else:
        rad = iv.mpf([-RHO, RHO])
        inside = lambda v: bool(-RHO < mp.mpf(v.a)) and bool(mp.mpf(v.b) < RHO)
        upper = lambda v: float(max(abs(mp.mpf(v.a)), abs(mp.mpf(v.b))))
        nonzero = lambda v: bool(v.a > 0) or bool(v.b < 0)
        lower = lambda v: float(min(abs(mp.mpf(v.a)), abs(mp.mpf(v.b)))) if nonzero(v) else 0.0
        conv = iv.mpf
    try:
        F, _, _ = evaluate(x, free)
        _, JX, zX = evaluate(x, free, box=RHO)
        Ci = [[conv(C[i][j]) for j in range(m)] for i in range(m)]
        worst, ok = 0.0, True
        for i in range(m):
            kk = -sum((Ci[i][j] * F[j] for j in range(m)), conv(0))
            for j in range(m):
                e = conv(1 if i == j else 0) - sum((Ci[i][l] * JX[l][j] for l in range(m)), conv(0))
                kk += e * rad
            worst = max(worst, upper(kk))
            ok = ok and inside(kk)
        zre, zim = zX.re.v, zX.im.v
        z_ok = nonzero(zre) or nonzero(zim)
        zlow = max(lower(zre), lower(zim))
        # interval Jacobian of the 37 conditions over the same box (x part), columns sub
        cols41 = list(range(nb))
        free_pos = {q: k for k, q in enumerate(cols41)}
        vals = [iv.mpf(v) for v in x[:nb]] + [iv.mpf(x[nb])]
        for q in free:
            if q != nb:
                vals[q] = vals[q] + rad
        res37 = system(vals, free_pos, None, True)
        JI = [[r.g.get(k, conv(0)) for k in sub] for r in res37]
        Yi = [[conv(Y[i][j]) for j in range(37)] for i in range(37)]
        nrm_ok, nrm = True, 0.0
        for i in range(37):
            acc = conv(0)
            for j in range(37):
                e = conv(1 if i == j else 0) - sum((Yi[i][l] * JI[l][j] for l in range(37)), conv(0))
                acc = acc + (abs(e) if arb else iv.mpf(max(abs(mp.mpf(e.a)), abs(mp.mpf(e.b)))))
            nrm = max(nrm, upper(acc))
            nrm_ok = nrm_ok and (bool(acc < 1) if arb else bool(mp.mpf(acc.b) < 1))
    finally:
        iv = saved
    return dict(proved=bool(ok), K_radius=worst, z_nonzero=bool(z_ok), z_lower_bound=zlow,
                sub37_norm=nrm, sub37_nonsingular=bool(nrm_ok))


def prove(name, T):
    t0 = time.time()
    st = structure(T)
    x = list(T.get_x()) + [st["beta"]]
    x_ref = list(x)
    n = len(x)
    nb = n - 1
    F0, J0, _ = evaluate(x, list(range(n)))
    Jf = np.array([[float(mid(z)) for z in row] for row in J0])
    m = Jf.shape[0]
    Jp = Jf.copy()
    Jp[:, nb] *= 1e6                                 # keep beta free
    piv = sl.qr(Jp, pivoting=True)[2]
    free, fixed = sorted(piv[:m]), sorted(piv[m:])
    assert nb in free
    sv = np.linalg.svd(Jf[:, free], compute_uv=False)
    for q in fixed:
        x[q] = KR.rounded(x[q], KR.FIX_BITS)
    for it in range(10):
        F, J, _ = evaluate(x, free)
        Fm = mp.matrix([mid(z) for z in F])
        if mp.norm(Fm, mp.inf) < mp.mpf(10) ** -85:
            break
        dx = mp.lu_solve(mp.matrix([[mid(z) for z in row] for row in J]), Fm)
        for k, q in enumerate(free):
            x[q] -= dx[k]
    for q in free:
        x[q] = KR.rounded(x[q], 280)
    shift = max(abs(x[q] - x_ref[q]) for q in range(nb))       # distance of y~ from the refined tableau
    F, J, _ = evaluate(x, free)
    Fmax = max(max(abs(mp.mpf(z.a)), abs(mp.mpf(z.b))) for z in F)
    C = mp.inverse(mp.matrix([[mid(z) for z in row] for row in J]))
    C = [[KR.rounded(C[i, j], 200) for j in range(m)] for i in range(m)]
    _, J37, _ = evaluate(x, list(range(nb)), full37=True)
    J37f = np.array([[float(mid(z)) for z in row] for row in J37])
    sub = sorted(int(q) for q in sl.qr(J37f, pivoting=True)[2][:37])
    Y = mp.inverse(mp.matrix([[mid(J37[r][k]) for k in sub] for r in range(37)]))
    Y = [[KR.rounded(Y[i, j], 200) for j in range(37)] for i in range(37)]
    cm = certify(x, free, nb, C, sub, Y, arb=False)
    ca = certify(x, free, nb, C, sub, Y, arb=True)
    info = dict(beta=float(st["beta"]), w_stages1to5=float(st["w15"]), w6=[float(mp.re(st["w6"])), float(mp.im(st["w6"]))],
                z=[float(mp.re(st["z"])), float(mp.im(st["z"]))], c6=float(mp.re(st["c6"])), im_c6=float(st["imc6"]), im_Ac6=float(st["imac6"]),
                equations=m, unknowns=n, fixed=[int(q) for q in fixed], F_enclosure=float(Fmax), shift=float(shift),
                cond_square=float(sv[0] / sv[-1]), mpmath=cm, arb=ca, seconds=round(time.time() - t0))
    for lab, c_ in (("mpmath", cm), ("Arb", ca)):
        print(f"{name:14s} {lab:6s} shift {info['shift']:.1e}, beta {info['beta']:.6f}, |w_1..5| {info['w_stages1to5']:.1e}, "
              f"z {info['z'][0]:+.1e}{info['z'][1]:+.3e}i: E {m} eq / {n} unknowns "
              f"{'PROVED' if c_['proved'] else 'not proved'}, K - y~ within {c_['K_radius']:.1e}, "
              f"|z| >= {c_['z_lower_bound']:.2e} ({c_['z_nonzero']}), ||I - YJ_37|| <= {c_['sub37_norm']:.1e} "
              f"({c_['sub37_nonsingular']}), cond {info['cond_square']:.1e}", flush=True)
    return info


def main():
    check_constants()
    tabs = load_refined()
    res = {}
    for name in ("CCK6-casc", "CRK6-s", "CRK6-opt"):    # other components: (W) fails, or holds with w_6 = 0
        if name not in tabs:
            continue
        T, _, _ = tabs[name].level(0).refine(tol_digits=85, iters=10)
        st = structure(T)
        res[name] = dict(beta=float(st["beta"]), w_stages1to5=float(st["w15"]), abs_w6=float(abs(st["w6"])),
                         im_c6=float(st["imc6"]), im_Ac6=float(st["imac6"]))
        print(f"{name:14s} best beta {float(st['beta']):.4f}: min |w_3..5| = {float(st['w15']):.2e}, "
              f"|w_6| = {float(abs(st['w6'])):.1e}, Im (Ac)_6 = {float(st['imac6']):.1e}", flush=True)
    d = json.load(open(os.path.join(R.DATA, "cck6v_r1.json")))["CCK6(5)-v"]
    cz = lambda v: mp.mpc(mp.mpf(v[0]), mp.mpf(v[1]))
    Tv = crk.Tableau("CCK6(5)-v", [[cz(d["A"][i][j]) for j in range(S)] for i in range(S)], [(6, 6, [cz(v) for v in d["b"][0]])])
    st = structure(Tv)
    res["CCK6(5)-v"] = dict(beta=float(st["beta"]), w_stages1to5=float(st["w15"]), im_c6=float(st["imc6"]), im_Ac6=float(st["imac6"]))
    print(f"{'CCK6(5)-v':14s} best beta {float(st['beta']):.4f}: min |w_3..5| = {float(st['w15']):.2e}, "
          f"Im (Ac)_6 = {float(st['imac6']):.1e}", flush=True)
    for name in ("CCK6(5)", "CCK6(5)-s"):
        T, _, _ = tabs[name].level(0).refine(tol_digits=85, iters=10)
        res[name] = prove(name, T)
        json.dump(res, open(OUT, "w"), indent=1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
