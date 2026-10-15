"""Independent check of the Krawczyk tests in Arb (python-flint) (internal review 5).

krawczyk_r1.py evaluates the Krawczyk operator in the interval arithmetic of mpmath (mpmath.iv).  Here the
same test is repeated with the ball arithmetic of Arb (FLINT), whose operations are rigorous: the
residuals F(y~) and the Jacobian over the box X are evaluated by the same forward-mode code
(krawczyk_r1.conditions) with every interval number replaced by an Arb ball; all inputs (the point y~, the
fixed unknowns and the entries of the preconditioner C) are binary numbers and are converted to balls
exactly; the inclusion K(X) - y~ in (-rho, rho)^m is decided by Arb's certified comparisons.
The point y~, the fixed unknowns and C are those of krawczyk_r1.prove (recomputed here).
Writes data/krawczyk_arb_r1.json.  Usage: python krawczyk_arb_r1.py [name ...]
"""
import json
import os
import sys
import time

import flint
import mpmath as mp
import numpy as np
import scipy.linalg as sl

import r1common as R
import krawczyk_r1 as KR
from tableaux import load_refined

flint.ctx.prec = 320
OUT = os.path.join(R.DATA, "krawczyk_arb_r1.json")


def to_arb(x):
    """Exact conversion of an mpf (or int) to an Arb ball of radius zero."""
    if isinstance(x, int):
        return flint.arb(x)
    x = mp.mpf(x)
    man, exp = mp.frexp(x)                 # x = man * 2^exp, 0.5 <= |man| < 1
    m = int(mp.ldexp(man, 400))            # exact: man has at most 400 bits here
    assert mp.ldexp(mp.mpf(m), exp - 400) == x
    return flint.arb(m) * flint.arb(2) ** (exp - 400)


class ArbIV:
    """The part of mpmath.iv used by krawczyk_r1.conditions/evaluate, with Arb balls."""
    @staticmethod
    def mpf(v):
        if isinstance(v, (list, tuple)):    # symmetric interval [-r, r]
            r = to_arb(mp.mpf(v[1]))
            return flint.arb(0, r.mid())
        return to_arb(v)


def prove_arb(name, tabs):
    t0 = time.time()
    mp.mp.dps = 90
    imc6 = name.endswith("+c6")
    base = name[:-3] if imc6 else name
    T = tabs[base[:-3]].level(0) if base.endswith("[6]") else tabs[base]
    T, _, _ = T.refine(tol_digits=85, iters=10)
    s, p = T.s, T.p
    x = T.get_x()
    n = len(x)
    # point, fixed unknowns and preconditioner exactly as in krawczyk_r1.prove (mpmath part)
    _, J0 = KR.evaluate(x, list(range(n)), s, p, imc6=imc6)
    Jf = np.array([[float(KR.mid(z)) for z in row] for row in J0])
    m = Jf.shape[0]
    piv = sl.qr(Jf, pivoting=True)[2]
    free, fixed = sorted(piv[:m]), sorted(piv[m:])
    x = list(x)
    for q in fixed:
        x[q] = KR.rounded(x[q], 100)
    for _ in range(8):
        F, J = KR.evaluate(x, free, s, p, imc6=imc6)
        Fm = mp.matrix([KR.mid(z) for z in F])
        if mp.norm(Fm, mp.inf) < mp.mpf(10) ** -85:
            break
        dx = mp.lu_solve(mp.matrix([[KR.mid(z) for z in row] for row in J]), Fm)
        for k, q in enumerate(free):
            x[q] -= dx[k]
    for q in free:
        x[q] = KR.rounded(x[q], 280)
    F, J = KR.evaluate(x, free, s, p, imc6=imc6)
    C = mp.inverse(mp.matrix([[KR.mid(z) for z in row] for row in J]))
    C = [[KR.rounded(C[i, j], 200) for j in range(m)] for i in range(m)]
    # Arb part
    saved = KR.iv
    KR.iv = ArbIV
    try:
        Fa, _ = KR.evaluate(x, free, s, p, imc6=imc6)
        _, JX = KR.evaluate(x, free, s, p, box=KR.RHO, imc6=imc6)
    finally:
        KR.iv = saved
    rho = to_arb(KR.rounded(KR.RHO, 200))
    rad = flint.arb(0, rho.mid())
    Ca = [[to_arb(C[i][j]) for j in range(m)] for i in range(m)]
    ok, worst = True, flint.arb(0)
    for i in range(m):
        kk = -sum((Ca[i][j] * Fa[j] for j in range(m)), flint.arb(0))
        for j in range(m):
            e = flint.arb(1 if i == j else 0) - sum((Ca[i][l] * JX[l][j] for l in range(m)), flint.arb(0))
            kk += e * rad
        ok = ok and bool(abs(kk) < rho)
        worst = max(worst, abs(kk).mid() + abs(kk).rad(), key=lambda a: float(a))
    info = dict(conditions=m, unknowns=n, fixed=[int(q) for q in fixed], proved=bool(ok), box_radius=float(KR.RHO),
                K_radius_upper=float(worst), prec_bits=flint.ctx.prec, seconds=round(time.time() - t0))
    print(f"{name:16s} Arb: {'PROVED' if ok else 'not proved'}  K - y~ within {float(worst):.1e} "
          f"< {float(KR.RHO):.0e}  [{info['seconds']}s]", flush=True)
    return info


def prove_d1_arb(name, variant, tabs):
    """Proposition 6(a): the Krawczyk test of the 31 equations of Proposition 5 (d1_r1.prove) and the
    certificate ||I - C J|| < 1 for a 31 x 31 submatrix of the Jacobian of all 37 conditions, in Arb."""
    import d1_r1 as D1
    t0 = time.time()
    mp.mp.dps = 90
    T, _, _ = tabs[name].refine(tol_digits=85, iters=10)
    s, p = T.s, T.p
    x = list(T.get_x())
    n = len(x)
    fun = lambda vals, fp, s_, p_: D1.d1_system(vals, fp, s_, p_, variant)
    G0, J0 = D1.evaluate(fun, x, list(range(n)), s, p)
    m = len(G0)
    Jf = np.array([[float(KR.mid(z)) for z in row] for row in J0])
    piv = sl.qr(Jf, pivoting=True)[2]
    free, fixed = sorted(piv[:m]), sorted(piv[m:])
    for q in fixed:
        x[q] = KR.rounded(x[q], 100)
    for _ in range(10):
        G, J = D1.evaluate(fun, x, free, s, p)
        Gm = mp.matrix([KR.mid(z) for z in G])
        if mp.norm(Gm, mp.inf) < mp.mpf(10) ** -85:
            break
        dx = mp.lu_solve(mp.matrix([[KR.mid(z) for z in row] for row in J]), Gm)
        for k, q in enumerate(free):
            x[q] -= dx[k]
    for q in free:
        x[q] = KR.rounded(x[q], 280)
    _, J = D1.evaluate(fun, x, free, s, p)
    C = mp.inverse(mp.matrix([[KR.mid(z) for z in row] for row in J]))
    C = [[KR.rounded(C[i, j], 200) for j in range(m)] for i in range(m)]
    _, FX0 = D1.evaluate(KR.conditions, x, free, s, p, box=KR.RHO)
    Fm = np.array([[float(KR.mid(z)) for z in row] for row in FX0])
    rows = sorted(sl.qr(Fm.T, pivoting=True)[2][:m])
    C2 = mp.inverse(mp.matrix([[KR.mid(FX0[r][j]) for j in range(m)] for r in rows]))
    C2 = [[KR.rounded(C2[i, j], 200) for j in range(m)] for i in range(m)]
    saved, saved1 = KR.iv, D1.iv
    KR.iv = D1.iv = ArbIV
    try:
        Ga, _ = D1.evaluate(fun, x, free, s, p)
        _, JX = D1.evaluate(fun, x, free, s, p, box=KR.RHO)
        _, FX = D1.evaluate(KR.conditions, x, free, s, p, box=KR.RHO)
    finally:
        KR.iv, D1.iv = saved, saved1
    rho = to_arb(KR.rounded(KR.RHO, 200))
    rad = flint.arb(0, rho.mid())
    Ca = [[to_arb(C[i][j]) for j in range(m)] for i in range(m)]
    ok, worst = True, 0.0
    for i in range(m):
        kk = -sum((Ca[i][j] * Ga[j] for j in range(m)), flint.arb(0))
        for j in range(m):
            e = flint.arb(1 if i == j else 0) - sum((Ca[i][l] * JX[l][j] for l in range(m)), flint.arb(0))
            kk += e * rad
        ok = ok and bool(abs(kk) < rho)
        worst = max(worst, float(abs(kk).mid() + abs(kk).rad()))
    C2a = [[to_arb(C2[i][j]) for j in range(m)] for i in range(m)]
    nrm = flint.arb(0)
    for i in range(m):
        acc = flint.arb(0)
        for j in range(m):
            e = flint.arb(1 if i == j else 0) - sum((C2a[i][l] * FX[rows[l]][j] for l in range(m)), flint.arb(0))
            acc += abs(e)
        nrm = max(nrm, acc, key=lambda a: float(a.mid() + a.rad()))
    cert = bool(nrm < 1)
    info = dict(variant=variant, equations=m, unknowns=n, proved=bool(ok), K_radius_upper=worst,
                rank37_cert_upper=float(nrm.mid() + nrm.rad()), rank37_ge_31=cert, seconds=round(time.time() - t0))
    print(f"{name:16s} Arb D(1) {variant}: {'PROVED' if ok else 'not proved'}  K - y~ within {worst:.1e};  "
          f"rank >= 31 certificate ||I - CJ|| <= {info['rank37_cert_upper']:.1e} ({cert})  [{info['seconds']}s]",
          flush=True)
    return info


def main(names):
    tabs = load_refined()
    extra = {}
    if any(nm.startswith("pair") for nm in names):
        import krawczyk_pairs_r1 as KP
        mp.mp.dps = 90
        for T, _ in KP.candidates():
            if T.name in [nm.split("[")[0] for nm in names]:
                lev, _, _ = T.refine(tol_digits=85, iters=14)
                extra[T.name] = lev
    tabs.update(extra)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if names == ["d1"]:
        import d1_r1 as D1
        for nm, var in D1.NAMES.items():
            res[nm + "|D1"] = prove_d1_arb(nm, var, tabs)
            json.dump(res, open(OUT, "w"), indent=1)
        return
    for nm in names:
        res[nm] = prove_arb(nm, tabs)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["CRK5-opt", "CRK5-s", "CCK6(5)[6]", "CCK6(5)-s[6]", "pair20[6]+c6"])
