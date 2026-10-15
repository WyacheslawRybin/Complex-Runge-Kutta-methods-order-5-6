"""Computer-assisted existence proof (Krawczyk test) for projected tableaux at points of full rank.

Review item 3.3: at a point where the Jacobian of the projected order conditions has full row rank N
(N conditions, n > N free real unknowns), n - N unknowns can be fixed and the remaining square system
F(y) = 0 is nonsingular.  If the Krawczyk operator
    K(X) = y~ - C F(y~) + (I - C J(X)) (X - y~),      C ~ J(y~)^{-1},
computed in interval arithmetic, satisfies K(X) in int X for a box X around y~, then F has a unique
zero in X (Krawczyk 1969, Moore 1977, Rump 2010).  F and its Jacobian over X are evaluated by
forward-mode differentiation in mpmath interval arithmetic (outward rounding, 320 bits), so the test
is rigorous up to the correctness of mpmath's interval arithmetic.

The fixed unknowns are rounded to 100-bit binary numbers and the free ones to 280-bit numbers, so
every input is exactly representable.  Writes ../data/krawczyk.json.
Usage:  python krawczyk_r1.py [name ...]   (name[6] = sixth-order level of a CCK6 pair; name[6]+c6 = the
        same with the additional equation Im c_6 = 0, which by Proposition emb5(b) proves an exact 6(5) pair)
"""
import json, os, sys, time
import numpy as np
import scipy.linalg as sl
import mpmath as mp
from mpmath import iv
import r1common as R
import crk
from tableaux import load_refined

mp.mp.dps = 90
iv.prec = 320
RHO = mp.mpf(10) ** -50          # radius of the box X
OUT = os.path.join(R.DATA, "krawczyk.json")
NAMES = ["CCK6(5)[6]", "CCK6(5)-s[6]", "CRK5-opt", "CRK5-s"]


class D:
    """Value and sparse gradient {free index: derivative}, all entries interval numbers."""
    __slots__ = ("v", "g")

    def __init__(self, v, g=None):
        self.v, self.g = v, (g if g is not None else {})

    def __add__(a, b):
        g = dict(a.g)
        for k, x in b.g.items():
            g[k] = g[k] + x if k in g else x
        return D(a.v + b.v, g)

    def __sub__(a, b):
        g = dict(a.g)
        for k, x in b.g.items():
            g[k] = g[k] - x if k in g else -x
        return D(a.v - b.v, g)

    def __mul__(a, b):
        g = {k: x * b.v for k, x in a.g.items()}
        for k, x in b.g.items():
            g[k] = g[k] + a.v * x if k in g else a.v * x
        return D(a.v * b.v, g)


class Cx:
    __slots__ = ("re", "im")

    def __init__(self, re, im):
        self.re, self.im = re, im

    def __add__(a, b):
        return Cx(a.re + b.re, a.im + b.im)

    def __mul__(a, b):
        return Cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)


def conditions(vals, free_pos, s, p, imc6=False):
    """Projected order conditions Re(b^T Phi_t) - 1/gamma(t), rho(t) <= p, with gradients.
    vals: interval values of the real unknowns in the order of crk.Tableau.get_x()."""
    zero = D(iv.mpf(0))
    var = lambda q: D(vals[q], {free_pos[q]: iv.mpf(1)} if q in free_pos else {})
    A, q = [[None] * s for _ in range(s)], 0
    for i in range(s):
        for j in range(i):
            A[i][j] = Cx(var(q), var(q + 1))
            q += 2
    b = []
    for i in range(s):
        if i == 0:
            b.append(Cx(var(q), zero))
            q += 1
        else:
            b.append(Cx(var(q), var(q + 1)))
            q += 2
    assert q == len(vals)
    cache = {}

    def Phi(t):
        if t not in cache:
            v = [Cx(D(iv.mpf(1)), zero) for _ in range(s)]
            for u in t:
                pu = Phi(u)
                w = [Cx(zero, zero)]
                for i in range(1, s):
                    acc = A[i][0] * pu[0]
                    for j in range(1, i):
                        acc = acc + A[i][j] * pu[j]
                    w.append(acc)
                v = [v[i] * w[i] for i in range(s)]
            cache[t] = v
        return cache[t]

    out = []
    for t in crk.trees_upto(p):
        ph = Phi(t)
        acc = b[0] * ph[0]
        for i in range(1, s):
            acc = acc + b[i] * ph[i]
        out.append(acc.re - D(iv.mpf(1) / crk.gamma(t)))
    if imc6:                                    # Im c_s = sum_j Im a_sj = 0 (Proposition emb5(b))
        acc = A[s - 1][0].im
        for j in range(1, s - 1):
            acc = acc + A[s - 1][j].im
        out.append(acc)
    return out


def evaluate(xvals, free, s, p, box=None, imc6=False):
    """Residual enclosures and Jacobian enclosure (rows: conditions, columns: free unknowns)."""
    free_pos = {q: k for k, q in enumerate(free)}
    vals = [iv.mpf(v) for v in xvals]
    if box is not None:
        for q in free:
            vals[q] = vals[q] + iv.mpf([-box, box])
    res = conditions(vals, free_pos, s, p, imc6)
    J = [[r.g.get(k, iv.mpf(0)) for k in range(len(free))] for r in res]
    return [r.v for r in res], J


def mid(z):
    return mp.mpf(z.mid)


def rounded(v, bits):
    with mp.workprec(bits):
        return +mp.mpf(v)


def prove(name, tabs):
    t0 = time.time()
    imc6 = name.endswith("+c6")
    base = name[:-3] if imc6 else name
    T = tabs[base[:-3]].level(0) if base.endswith("[6]") else tabs[base]
    T, _, _ = T.refine(tol_digits=85, iters=10)
    s, p = T.s, T.p
    x = T.get_x()
    n = len(x)
    # choice of the unknowns kept free: column-pivoted QR of the Jacobian at the point
    _, J0 = evaluate(x, list(range(n)), s, p, imc6=imc6)
    Jf = np.array([[float(mid(z)) for z in row] for row in J0])
    m = Jf.shape[0]
    piv = sl.qr(Jf, pivoting=True)[2]
    free, fixed = sorted(piv[:m]), sorted(piv[m:])
    sv = np.linalg.svd(Jf[:, free], compute_uv=False)
    # fix the others at 100-bit values and re-solve the square system (Newton in 90 digits)
    x = list(x)
    for q in fixed:
        x[q] = rounded(x[q], 100)
    for it in range(8):
        F, J = evaluate(x, free, s, p, imc6=imc6)
        Fm = mp.matrix([mid(z) for z in F])
        if mp.norm(Fm, mp.inf) < mp.mpf(10) ** -85:
            break
        dx = mp.lu_solve(mp.matrix([[mid(z) for z in row] for row in J]), Fm)
        for k, q in enumerate(free):
            x[q] -= dx[k]
    for q in free:
        x[q] = rounded(x[q], 280)
    # Krawczyk test
    F, J = evaluate(x, free, s, p, imc6=imc6)
    C = mp.inverse(mp.matrix([[mid(z) for z in row] for row in J]))
    _, JX = evaluate(x, free, s, p, box=RHO, imc6=imc6)
    Ci = [[iv.mpf(C[i, j]) for j in range(m)] for i in range(m)]
    rad = iv.mpf([-RHO, RHO])
    worst = mp.mpf(0)
    ok = True
    for i in range(m):
        cf = iv.mpf(0)
        for j in range(m):
            cf += Ci[i][j] * F[j]
        kk = -cf
        for j in range(m):
            e = iv.mpf(1 if i == j else 0)
            for l in range(m):
                e -= Ci[i][l] * JX[l][j]
            kk += e * rad
        lo, hi = mp.mpf(kk.a), mp.mpf(kk.b)           # K_i - y~_i
        worst = max(worst, abs(lo), abs(hi))
        ok = ok and (-RHO < lo) and (hi < RHO)
    Fmax = max(max(abs(mp.mpf(z.a)), abs(mp.mpf(z.b))) for z in F)
    info = dict(conditions=m, unknowns=n, fixed=[int(q) for q in fixed], proved=bool(ok),
                box_radius=float(RHO), K_radius=float(worst), F_enclosure=float(Fmax),
                cond_square=float(sv[0] / sv[-1]), seconds=round(time.time() - t0))
    print(f"{name:12s} {m} conditions, {n} unknowns, fixed {info['fixed']}: "
          f"{'PROVED' if ok else 'not proved'}  |F(y~)| <= {float(Fmax):.1e}, "
          f"K - y~ within {float(worst):.1e} < {float(RHO):.0e}, cond {info['cond_square']:.1e}  "
          f"[{info['seconds']}s]", flush=True)
    return info


def main(names):
    tabs = load_refined()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in names:
        res[name] = prove(name, tabs)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES)
