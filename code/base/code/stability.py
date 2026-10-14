"""
Linear stability of every method.

For a projected method applied to a real linear system y' = J y the step is
y_{n+1} = P(hJ) y_n with the REAL polynomial P(z) = 1 + sum_k Re(b^T A^{k-1} 1) z^k,
so the eigenvalues of the step matrix are P(h lambda) for the (possibly complex)
eigenvalues lambda of J.  The stability region is {z : |P(z)| <= 1}.
We check this directly on 2x2 real matrices with eigenvalues sigma +- i omega, and
compute the real and imaginary stability intervals, the area of the region (left
half-plane part and total), and the same quantities per function evaluation.
Writes ../data/stability.json.
"""
import json, os, sys
from math import factorial
import numpy as np
import mpmath as mp
from tableaux import load_refined

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def poly(T, L=0):
    return np.array([float(v) for v in T.stab_coeffs(L)])

def Tm(m):
    return np.array([1.0 / factorial(k) for k in range(m + 1)])

def real_interval(co):
    """Left end of the stability interval on the negative real axis (connected to 0)."""
    P = np.polynomial.Polynomial(co)
    x = np.linspace(0, -12, 1_200_001)
    bad = np.abs(P(x)) > 1 + 1e-13
    bad[0] = False
    i = np.argmax(bad[1:]) + 1 if bad[1:].any() else len(x) - 1
    # refine the crossing by bisection on |P| - 1
    a, b = x[i - 1], x[i]
    for _ in range(80):
        m = 0.5 * (a + b)
        if abs(P(m)) > 1:
            b = m
        else:
            a = m
    return a

def imag_interval(co):
    """Largest y with |P(i s)| <= 1 for 0 <= s <= y.  g(y) = |P(iy)|^2 - 1 is an even
    polynomial; if its lowest non-vanishing coefficient is positive the axis is left
    at once (interval 0), otherwise the interval ends at the first positive root of g."""
    P = np.polynomial.Polynomial(co)
    # P(iy) = Re + i Im as polynomials in y
    n = len(co)
    re = np.zeros(n); im = np.zeros(n)
    for k, a in enumerate(co):
        ik = (1j) ** k
        re[k] += a * ik.real; im[k] += a * ik.imag
    g = (np.polynomial.Polynomial(re) ** 2 + np.polynomial.Polynomial(im) ** 2 - 1).coef
    g[np.abs(g) < 1e-14 * max(1.0, np.abs(g).max())] = 0.0
    nz = np.nonzero(g)[0]
    if len(nz) == 0 or g[nz[0]] > 0:
        return 0.0
    roots = np.polynomial.Polynomial(g).roots()
    pos = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 1e-9)
    return pos[0] if pos else np.inf

def area(co, box=(-7.0, 2.0, -6.0, 6.0), n=1801):
    P = np.polynomial.Polynomial(co)
    X, Y = np.meshgrid(np.linspace(box[0], box[1], n), np.linspace(box[2], box[3], n))
    inside = np.abs(P(X + 1j * Y)) <= 1
    dA = (box[1] - box[0]) / (n - 1) * (box[3] - box[2]) / (n - 1)
    return float(inside.sum() * dA), float((inside & (X <= 0)).sum() * dA)

def check_2x2(T, L=0, samples=((-1.0, 0.7), (-2.5, 1.3), (-0.3, 2.9), (-3.1, 0.2)), h=1.0):
    """Spectral radius of one projected step on real 2x2 systems vs |P(h lambda)|."""
    m, p, b = T.levels[L]
    co = poly(T, L)
    worst = 0.0
    for sg, om in samples:
        J = [[mp.mpf(sg), mp.mpf(om)], [mp.mpf(-om), mp.mpf(sg)]]
        f = lambda t, y: [J[0][0] * y[0] + J[0][1] * y[1], J[1][0] * y[0] + J[1][1] * y[1]]
        S = []
        for e in ([1, 0], [0, 1]):
            y = [mp.mpf(e[0]), mp.mpf(e[1])]
            K = []
            for i in range(m):
                yi = [y[q] + h * mp.fsum(T.A[i][j] * K[j][q] for j in range(i)) for q in range(2)]
                K.append(f(0, yi))
            S.append([mp.re(y[q] + h * mp.fsum(b[i] * K[i][q] for i in range(m))) for q in range(2)])
        M = np.array([[float(S[0][0]), float(S[1][0])], [float(S[0][1]), float(S[1][1])]])
        rho = max(abs(np.linalg.eigvals(M)))
        val = abs(np.polynomial.Polynomial(co)(complex(sg, om) * h))
        worst = max(worst, abs(rho - val))
    return worst

ROWS = [
    # label, tableau, level, evaluations per step
    ("CRK5 family ($T_5$)", "CRK5-opt", 0, 5),
    ("CRK6 / CCK6 ($T_6$)", "CRK6-opt", 0, 6),
    ("CCK6 embedded ($T_5$)", "CCK6(5)", 1, 5),
    ("DP5", "DP5", 0, 6),
    ("CK5", "CK5", 0, 6),
    ("RK4", "RK4", 0, 4),
    ("Butcher6", "Butcher6", 0, 7),
    ("RK6-opt", "RK6-opt", 0, 7),
    ("DP6(5)", "DP6(5)", 0, 8),
    ("Verner6(5)", "Verner6(5)", 0, 8),
]

def main():
    mp.mp.dps = 40
    TT = load_refined()
    out = {}
    for lab, key, L, ev in ROWS:
        T = TT[key]
        co = poly(T, L)
        r, i = real_interval(co), imag_interval(co)
        A_tot, A_left = area(co)
        out[lab] = dict(key=key, level=L, evals=ev, coeffs=co.tolist(), real=r, imag=i,
                        real_per_eval=-r / ev, imag_per_eval=i / ev, area=A_tot, area_left=A_left,
                        area_left_per_eval2=A_left / ev ** 2, check2x2=check_2x2(T, L))
        print(f"{lab:24s} real [{r:.4f}, 0]  imag {i:.4f}  per eval {-r / ev:.3f}  area(left) {A_left:.3f}"
              f"  2x2 check {out[lab]['check2x2']:.1e}")
    # every level of every complex tableau: deviation of P from T_m
    dev = {}
    for key in ["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5", "CRK6-0", "CRK6-opt", "CRK6-s", "CRK6-lam",
                "CCK6(5)", "CCK6(5)-s", "CCK6-casc"]:
        T = TT[key]
        dev[key] = [float(np.max(np.abs(poly(T, L) - Tm(T.levels[L][0])))) for L in range(len(T.levels))]
    out["_Tm_deviation"] = dev
    for m in range(1, 7):
        co = Tm(m)
        out[f"_T{m}"] = dict(real=real_interval(co), imag=imag_interval(co), area_left=area(co)[1])
    print("max deviation of P from T_m over all levels:", max(max(v) for v in dev.values()))
    print({k: (round(v["real"], 4), round(v["imag"], 4)) for k, v in out.items() if k.startswith("_T") and "real" in v})
    json.dump(out, open(os.path.join(DATA, "stability.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
