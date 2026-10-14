"""Singular-value spectra of the Jacobian of all projected order conditions, in multiple precision.

R1 replacement of the spectra of Figure 2 (review, Section 4, item 2): the original panel (a) was a
double-precision SVD of a forward-difference Jacobian, so its "zero" singular values sat at the
finite-difference noise floor.  Here every tableau is first refined to 80 digits, the Jacobian with
respect to all free real unknowns (Im b_1 of every level excluded) is formed by central differences
with h = 1e-30 in 90-digit arithmetic (truncation and rounding errors about 1e-60), and its singular
values are computed in 90 digits.

Writes ../data/spectra_mp.json.  Usage:  python spectra_r1.py [name ...]
"""
import json, os, sys, time
import mpmath as mp
import r1common as R
from tableaux import load_refined

mp.mp.dps = 90
H = mp.mpf(10) ** -30
NAMES = ["CRK5-opt", "CRK5-s", "CRK6-opt", "CRK6-0", "CCK6(5)[6]", "CCK6(5)", "CCK6(5)-s", "CCK6-casc",
         "Butcher6", "RK6-opt"]
OUT = os.path.join(R.DATA, "spectra_mp.json")


def tableau(name, tabs):
    if name.endswith("[6]"):                     # sixth-order level of a CCK6 pair as a CRK6 tableau
        return tabs[name[:-3]].level(0)
    return tabs[name]


def spectrum(T):
    x = T.get_x()
    F = lambda xx: [v for lev in T.set_x(xx).residuals() for v in lev]
    m, n = len(F(x)), len(x)
    J = mp.matrix(m, n)
    for k in range(n):
        xp, xm = x[:], x[:]
        xp[k] += H
        xm[k] -= H
        for i, (a, b) in enumerate(zip(F(xp), F(xm))):
            J[i, k] = (a - b) / (2 * H)
    sv = sorted((abs(v) for v in mp.svd_r(J, compute_uv=False)), reverse=True)
    return m, n, sv


def main(names):
    tabs = load_refined()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in names:
        t0 = time.time()
        T = tableau(name, tabs)
        if T.projected:
            T, shift, resid = T.refine(tol_digits=80, iters=10)
        else:
            resid = max(abs(v) for lev in T.residuals() for v in lev)
        m, n, sv = spectrum(T)
        # numerical rank: the largest relative gap between consecutive singular values
        rel = [v / sv[0] for v in sv]
        gaps = [(mp.log10(rel[k]) - mp.log10(max(rel[k + 1], mp.mpf(10) ** -88)), k + 1) for k in range(len(sv) - 1)]
        g, rank = max(gaps)
        if rel[-1] > mp.mpf(10) ** -30:          # no small singular value at all: full rank
            g, rank = mp.mpf(0), len(sv)
        res[name] = dict(m=m, n=n, rank=rank, dim=n - rank, resid=float(resid),
                         smallest_kept=float(rel[rank - 1]),
                         largest_dropped=float(rel[rank]) if rank < len(sv) else 0.0,
                         gap_decades=float(g), sv=[mp.nstr(v, 6) for v in sv])
        print(f"{name:11s} {m}x{n}: rank {rank}, dim {n - rank}, residual {float(resid):.1e}, "
              f"smallest kept {float(rel[rank - 1]):.2e}, largest dropped "
              f"{res[name]['largest_dropped']:.2e}  [{time.time() - t0:.0f}s]", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES)
