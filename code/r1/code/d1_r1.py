"""The simplifying assumption D(1) on the CRK6 component of tangent dimension ten (review R1b, Q2/C6).

For an explicit tableau (A, b) with c = A 1 let  d = b^T A - (b o (1 - c))^T.  For every tree u and
u+ = u with one more leaf at the root,
    Re(b^T A Phi_u) = Re(b^T Phi_u) - Re(b^T Phi_{u+}) + Re(d^T Phi_u),
    1/gamma([u])   = 1/gamma(u)     - 1/gamma(u+),
so if Re(d^T Phi_u) = 0 for all u of order <= p - 1, the condition of every tree [u] (root degree one)
follows from those of u and u+, and by induction all conditions follow from the 20 conditions of the
trees whose root degree is not one (orders 1..6: 1, 0, 1, 2, 5, 11).  Since Phi_u(1) = 0 for u != tau
and Phi_u(2) = c_2^k for u = [tau^k] (0 otherwise), Re(d^T Phi_u) = 0 holds in two cases:
    D1   : d_2 = ... = d_6 = 0 and Re d_1 = 0                                       (11 equations)
    D1c2 : d_3 = ... = d_6 = 0, Im c_2 = 0, Re d_2 = 0 and Re d_1 = 0                (11 equations)
Together with the 20 conditions, 31 equations on the 41 free real unknowns: dimension 10.  As
d_6 = -b_6 (1 - c_6) and b_6 != 0 (tall tree), D(1) forces c_6 = 1.

1. Survey: d at all 460 CRK6 tableaux visited by the optimization runs (code/rank_survey.py), against
   the rank of the Jacobian of all 37 conditions recorded in data/rank_survey.json.
2. Krawczyk test (as in krawczyk_r1.py) of the 31 equations at the four CRK6 tableaux of the paper,
   with 10 unknowns fixed: existence of an exact D(1) tableau, hence of a CRK6 tableau, nearby, and a
   smooth ten-dimensional manifold of such tableaux through it.  Then the interval Jacobian of all 37
   conditions over the same box is shown to have rank >= 31 (a 31 x 31 submatrix with
   ||I - C J|| < 1), so that near the exact tableau the solution set of the 37 conditions is that
   manifold: local dimension exactly ten.
Writes ../data/d1_r1.json.
"""
import json, os, sys, time, collections
import numpy as np
import scipy.linalg as sl
import mpmath as mp
from mpmath import iv
import r1common as R
import crk
from tableaux import load_refined
from krawczyk_r1 import D, Cx, conditions, mid, rounded

mp.mp.dps = 90
iv.prec = 320
RHO = mp.mpf(10) ** -50
OUT = os.path.join(R.DATA, "d1_r1.json")
NAMES = {"CRK6-opt": "D1", "CRK6-s": "D1", "CRK6-0": "D1", "CRK6-lam": "D1c2"}


# ---------------------------------------------------------------- 1. survey (double precision)
def survey():
    sys.path.insert(0, os.path.join(R.ORIG, "code"))
    from rank_survey import points
    from optimize_E import load_module
    TC = load_module("CRK6")
    ranks = [r["rank"] for r in json.load(open(os.path.join(R.ORIG_DATA, "rank_survey.json")))["CRK6"]]
    pts = points("CRK6")
    assert len(pts) == len(ranks)
    rows = []
    for (label, p), rk in zip(pts, ranks):
        A, W = TC.unpack(np.array(p["x"], float))
        b = W[0]
        c = A.sum(axis=1)
        d = b @ A - b * (1 - c)
        sc = max(1.0, np.abs(b).max() * np.abs(A).max())
        d1 = max(np.abs(d[1:]).max(), abs(d[0].real)) / sc                       # D1
        d1c2 = max(np.abs(d[2:]).max(), abs(d[0].real), abs(d[1].real), abs(c[1].imag)) / sc   # D1c2
        rows.append(dict(label=label, rank=rk, D1=float(d1), D1c2=float(d1c2), c6=[c[5].real, c[5].imag],
                         c2_imag=float(c[1].imag), d2=float(abs(d[1]))))
    print("survey of the CRK6 tableaux visited by the optimization runs:")
    for rk in sorted(set(ranks)):
        g = [r for r in rows if r["rank"] == rk]
        dist = [min(r["D1"], r["D1c2"]) for r in g]
        n1 = sum(1 for r in g if r["D1"] < 1e-9)
        n2 = sum(1 for r in g if r["D1c2"] < 1e-9 and r["D1"] >= 1e-9)
        print(f"  rank {rk}: {len(g):3d} tableaux, D1 holds at {n1}, D1c2 (only) at {n2};  "
              f"defect min(D1, D1c2) in [{min(dist):.1e}, {max(dist):.1e}],  "
              f"|c6 - 1| in [{min(abs(complex(*r['c6']) - 1) for r in g):.1e}, "
              f"{max(abs(complex(*r['c6']) - 1) for r in g):.1e}]")
    return rows


# ---------------------------------------------------------------- 2. Krawczyk test of the D(1) system
def d1_system(vals, free_pos, s, p, variant):
    """The 11 D(1) equations and the 20 conditions of the trees with root degree != 1, with gradients."""
    zero = D(iv.mpf(0))
    one = D(iv.mpf(1))
    var = lambda q: D(vals[q], {free_pos[q]: iv.mpf(1)} if q in free_pos else {})
    A, q = [[Cx(zero, zero)] * s for _ in range(s)], 0
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
    c = []
    for i in range(s):
        acc = Cx(zero, zero)
        for j in range(i):
            acc = acc + A[i][j]
        c.append(acc)
    d = []
    for j in range(s):
        acc = Cx(zero - b[j].re, zero - b[j].im) * Cx(one - c[j].re, zero - c[j].im)
        for i in range(j + 1, s):
            acc = acc + b[i] * A[i][j]
        d.append(acc)
    if variant == "D1":
        eqs = [d[0].re] + [x for j in range(1, s) for x in (d[j].re, d[j].im)]
    else:
        eqs = [d[0].re, d[1].re, c[1].im] + [x for j in range(2, s) for x in (d[j].re, d[j].im)]
    full = conditions(vals, free_pos, s, p)
    trees = crk.trees_upto(p)
    eqs += [f for t, f in zip(trees, full) if len(t) != 1]
    return eqs


def evaluate(fun, xvals, free, s, p, box=None, **kw):
    free_pos = {q: k for k, q in enumerate(free)}
    vals = [iv.mpf(v) for v in xvals]
    if box is not None:
        for q in free:
            vals[q] = vals[q] + iv.mpf([-box, box])
    res = fun(vals, free_pos, s, p, **kw)
    J = [[r.g.get(k, iv.mpf(0)) for k in range(len(free))] for r in res]
    return [r.v for r in res], J


def interval_nonsingular(JX, rows, m):
    """True if every matrix in the interval enclosure JX[rows][:, :m] is nonsingular: ||I - C JX||_inf < 1."""
    Jm = mp.matrix([[mid(JX[r][j]) for j in range(m)] for r in rows])
    C = mp.inverse(Jm)
    Ci = [[iv.mpf(C[i, j]) for j in range(m)] for i in range(m)]
    norm = mp.mpf(0)
    for i in range(m):
        acc = mp.mpf(0)
        for j in range(m):
            e = iv.mpf(1 if i == j else 0)
            for l in range(m):
                e -= Ci[i][l] * JX[rows[l]][j]
            acc += max(abs(mp.mpf(e.a)), abs(mp.mpf(e.b)))
        norm = max(norm, acc)
    return norm


def prove(name, variant, tabs):
    t0 = time.time()
    T, _, _ = tabs[name].refine(tol_digits=85, iters=10)
    s, p = T.s, T.p
    x = list(T.get_x())
    n = len(x)
    fun = lambda vals, fp, s_, p_: d1_system(vals, fp, s_, p_, variant)
    G0, J0 = evaluate(fun, x, list(range(n)), s, p)
    m = len(G0)
    Jf = np.array([[float(mid(z)) for z in row] for row in J0])
    sv_all = np.linalg.svd(Jf, compute_uv=False)
    piv = sl.qr(Jf, pivoting=True)[2]
    free, fixed = sorted(piv[:m]), sorted(piv[m:])
    for q in fixed:
        x[q] = rounded(x[q], 100)
    for it in range(10):
        G, J = evaluate(fun, x, free, s, p)
        Gm = mp.matrix([mid(z) for z in G])
        if mp.norm(Gm, mp.inf) < mp.mpf(10) ** -85:
            break
        dx = mp.lu_solve(mp.matrix([[mid(z) for z in row] for row in J]), Gm)
        for k, q in enumerate(free):
            x[q] -= dx[k]
    for q in free:
        x[q] = rounded(x[q], 280)
    G, J = evaluate(fun, x, free, s, p)
    C = mp.inverse(mp.matrix([[mid(z) for z in row] for row in J]))
    _, JX = evaluate(fun, x, free, s, p, box=RHO)
    Ci = [[iv.mpf(C[i, j]) for j in range(m)] for i in range(m)]
    rad = iv.mpf([-RHO, RHO])
    worst, ok = mp.mpf(0), True
    for i in range(m):
        cf = iv.mpf(0)
        for j in range(m):
            cf += Ci[i][j] * G[j]
        kk = -cf
        for j in range(m):
            e = iv.mpf(1 if i == j else 0)
            for l in range(m):
                e -= Ci[i][l] * JX[l][j]
            kk += e * rad
        lo, hi = mp.mpf(kk.a), mp.mpf(kk.b)
        worst = max(worst, abs(lo), abs(hi))
        ok = ok and (-RHO < lo) and (hi < RHO)
    Gmax = max(max(abs(mp.mpf(z.a)), abs(mp.mpf(z.b))) for z in G)
    # all 37 conditions over the same box: residual enclosure and a nonsingular 31 x 31 submatrix
    F, _ = evaluate(conditions, x, free, s, p)
    _, FX = evaluate(conditions, x, free, s, p, box=RHO)
    Fmax = max(max(abs(mp.mpf(z.a)), abs(mp.mpf(z.b))) for z in F)
    Fm = np.array([[float(mid(z)) for z in row] for row in FX])
    rows = sorted(sl.qr(Fm.T, pivoting=True)[2][:m])
    nrm = interval_nonsingular(FX, rows, m)
    info = dict(variant=variant, equations=m, unknowns=n, fixed=[int(q) for q in fixed], proved=bool(ok),
                box_radius=float(RHO), K_radius=float(worst), G_enclosure=float(Gmax),
                F37_enclosure_at_point=float(Fmax), rank37_cert_norm=float(nrm), rank37_ge_31=bool(nrm < 1),
                cond_square=float(sv_all[0] / sv_all[-1]), seconds=round(time.time() - t0))
    print(f"{name:9s} {variant:5s} {m} equations, {n} unknowns: {'PROVED' if ok else 'not proved'}  "
          f"|G(y~)| <= {float(Gmax):.1e}, K - y~ within {float(worst):.1e};  all 37 conditions "
          f"|F(y~)| <= {float(Fmax):.1e};  rank >= 31 certificate ||I - CJ|| = {float(nrm):.2e}  "
          f"cond {info['cond_square']:.1e} [{info['seconds']}s]", flush=True)
    return info


def main():
    out = dict(survey=survey())
    tabs = load_refined()
    out["krawczyk"] = {name: prove(name, v, tabs) for name, v in NAMES.items()}
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
