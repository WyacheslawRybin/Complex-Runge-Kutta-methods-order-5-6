"""Real parallel RK methods with two chains of five stages: does order five exist? (internal review 8)

A method with two independent chains (A_1, b_1), (A_2, b_2), both explicit with five stages and real
coefficients, advances y_{n+1} = y_n + h (b_1^T K_1 + b_2^T K_2); its elementary weights are
b_1^T Phi_t(A_1) + b_2^T Phi_t(A_2), so order five means the 17 real equations
    b_1^T Phi_t(A_1) + b_2^T Phi_t(A_2) = 1/gamma(t),   rho(t) <= 5,
in 30 real unknowns.  The projected methods are the special case A_2 = conj(A_1), b_2 = conj(b_1) of the
complexified equations.  Multistart damped minimum-norm Newton (exact Jacobian by the complex step) from random starts.
With TWO_CHAIN_S=6 the same for two chains of six stages and order six (37 equations, 42 unknowns).
Writes data/two_chain{S}_r1.json.  With the argument emin, E_{p+1} is minimized under max|coef| <= 5 from the solutions found.
With the argument prove, the Krawczyk test proves an exact method near the best one.
Usage: [TWO_CHAIN_S=6] python two_chain_r1.py [starts | emin | prove]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import r1common as R
import crk

S = int(os.environ.get("TWO_CHAIN_S", "5"))       # stages per chain = order
TREES = crk.trees_upto(S)
RHS = np.array([1.0 / crk.gamma(t) for t in TREES])
IDX = [(i, j) for i in range(S) for j in range(i)]
NA = S * (S - 1) // 2
NC = NA + S                                       # unknowns per chain
OUT = os.path.join(R.DATA, f"two_chain{S}_r1.json")


def unpack(x):
    out = []
    for k in range(2):
        z = x[NC * k:NC * (k + 1)]
        A = np.zeros((S, S), dtype=z.dtype)
        for q, (i, j) in enumerate(IDX):
            A[i, j] = z[q]
        out.append((A, z[NA:NC]))
    return out


def phi(A):
    cache = {}

    def P(t):
        if t not in cache:
            v = np.ones(S, dtype=A.dtype)
            for u in t:
                v = v * (A @ P(u))
            cache[t] = v
        return cache[t]
    return [P(t) for t in TREES]


def residual(x):
    r = -RHS.astype(x.dtype)
    for A, b in unpack(x):
        r = r + np.array([b @ v for v in phi(A)])
    return r


def jac(x):
    """Exact Jacobian by the complex step (the residual is a real polynomial)."""
    h = 1e-30
    J = np.empty((len(TREES), len(x)))
    for k in range(len(x)):
        z = x.astype(complex)
        z[k] += 1j * h
        J[:, k] = residual(z).imag / h
    return J


def newton(x, iters=200):
    """Damped minimum-norm Newton for the underdetermined system."""
    r = residual(x)
    for _ in range(iters):
        nr = np.linalg.norm(r)
        if nr < 1e-14 or not np.isfinite(nr):
            break
        dx = np.linalg.lstsq(jac(x), r, rcond=1e-12)[0]
        lam = 1.0
        while lam > 1e-4:
            y = x - lam * dx
            ry = residual(y)
            if np.linalg.norm(ry) < nr:
                break
            lam /= 2
        x, r = y, ry
    return x, float(np.linalg.norm(r))


def run(seed):
    rng = np.random.default_rng(seed)
    best = None
    for scale in (0.5, 1.0, 2.0):
        x, n = newton(rng.normal(size=2 * NC) * scale)
        if np.isfinite(n) and (best is None or n < best[0]):
            best = (n, x.tolist())
    return best


def main(starts=2000):
    t0 = time.time()
    with Pool(4) as pool:
        res = [r for r in pool.imap_unordered(run, range(starts), chunksize=10) if r is not None]
    res.sort(key=lambda z: z[0])
    norms = np.array([r[0] for r in res])
    print(f"{len(res)} starts x 3 scales in {time.time() - t0:.0f}s; smallest residuals:",
          " ".join(f"{v:.2e}" for v in norms[:10]))
    print("quantiles of the best residual per start: ", np.quantile(norms, [0.01, 0.1, 0.5]))
    json.dump(dict(starts=starts, best=res[:60], quantiles=np.quantile(norms, [0.0, 0.01, 0.1, 0.5]).tolist()),
              open(OUT, "w"), indent=1)




# ---------------------------------------------------------------- error-constant minimization
TREES_NEXT = crk.trees_upto(S + 1)[len(TREES):]
SIG = np.array([float(crk.sigma(t)) for t in TREES_NEXT])
RHS_NEXT = np.array([1.0 / crk.gamma(t) for t in TREES_NEXT])
BOUND = 5.0


def err_vec(x):
    e = -RHS_NEXT.astype(x.dtype)
    for A, b in unpack(x):
        cache = {}

        def P(t):
            if t not in cache:
                v = np.ones(S, dtype=A.dtype)
                for u in t:
                    v = v * (A @ P(u))
                cache[t] = v
            return cache[t]
        e = e + np.array([b @ P(t) for t in TREES_NEXT])
    return e / SIG


def err_jac(x):
    """Exact Jacobian of err_vec by the complex step."""
    h = 1e-30
    J = np.empty((len(TREES_NEXT), len(x)))
    for k in range(len(x)):
        z = x.astype(complex)
        z[k] += 1j * h
        J[:, k] = err_vec(z).imag / h
    return J


def num_jac(fun, x, h=1e-7):
    f0 = fun(x)
    J = np.zeros((len(f0), len(x)))
    for k in range(len(x)):
        d = np.zeros(len(x)); d[k] = h
        J[:, k] = (fun(x + d) - fun(x - d)) / (2 * h)
    return J


def emin_one(x0):
    from scipy.optimize import minimize
    obj = lambda z: float(err_vec(z) @ err_vec(z))
    sol = minimize(obj, x0, jac=lambda z: 2 * err_jac(z).T @ err_vec(z), method="SLSQP",
                   constraints=[dict(type="eq", fun=residual, jac=jac),
                                dict(type="ineq", fun=lambda z: BOUND - np.abs(z)), ],
                   options=dict(maxiter=500, ftol=1e-18))
    x = sol.x
    for _ in range(20):                         # Newton back onto the conditions
        r = residual(x)
        if np.linalg.norm(r) < 1e-14:
            break
        x = x - np.linalg.lstsq(jac(x), r, rcond=1e-12)[0]
    if np.linalg.norm(residual(x)) > 1e-12 or np.abs(x).max() > BOUND + 1e-6:
        return None
    return float(np.linalg.norm(err_vec(x))), x.tolist()


def emin(nstarts=40):
    d = json.load(open(OUT))
    starts = [np.array(x) for _, x in d["best"]][:nstarts]
    with Pool(4) as pool:
        res = [r for r in pool.map(emin_one, starts) if r is not None]
    res.sort(key=lambda z: z[0])
    print(f"two chains of {S} stages, order {S}: smallest E_{S + 1} under max|coef| <= {BOUND:g} over {len(res)} "
          f"descents: " + " ".join(f"{r[0]:.3e}" for r in res[:5]))
    d["emin"] = dict(bound=BOUND, E=[r[0] for r in res], best_x=res[0][1] if res else None)
    json.dump(d, open(OUT, "w"), indent=1)



# ---------------------------------------------------------------- existence proof (Krawczyk) for the best tableau
def prove():
    """Krawczyk test (mpmath intervals and Arb) for the 17 conditions at the best tableau of the descents, with
    13 unknowns fixed: an exact real two-chain method of order five exists within 1e-50 of it."""
    import mpmath as mp
    import scipy.linalg as sl
    import krawczyk_r1 as KR
    import cascade_proof_r1 as CA
    from krawczyk_r1 import D
    mp.mp.dps = 90
    d = json.load(open(OUT))
    x = [mp.mpf(v) for v in d["emin"]["best_x"]]

    def system(xv, free, box=None):
        fp = {q: k for k, q in enumerate(free)}
        vals = [CA.ivm(v) for v in xv]
        if box is not None:
            for q in free:
                vals[q] = vals[q] + CA.ivm([-box, box])
        var = lambda q: D(vals[q], {fp[q]: CA.ivm(1)} if q in fp else {})
        zero = D(CA.ivm(0))
        out = [D(-CA.ivm(1) / crk.gamma(t)) for t in TREES]
        for k in range(2):
            o = NC * k
            A = [[None] * S for _ in range(S)]
            for q, (i, j) in enumerate(IDX):
                A[i][j] = var(o + q)
            b = [var(o + NA + i) for i in range(S)]
            cache = {}

            def P(t):
                if t not in cache:
                    v = [D(CA.ivm(1)) for _ in range(S)]
                    for u in t:
                        pu = P(u)
                        w = [zero] + [sum((A[i][j] * pu[j] for j in range(1, i)), A[i][0] * pu[0]) for i in range(1, S)]
                        v = [v[i] * w[i] for i in range(S)]
                    cache[t] = v
                return cache[t]
            for n, t in enumerate(TREES):
                pv = P(t)
                acc = b[0] * pv[0]
                for i in range(1, S):
                    acc = acc + b[i] * pv[i]
                out[n] = out[n] + acc
        return [r.v for r in out], [[r.g.get(k, CA.ivm(0)) for k in range(len(free))] for r in out]

    m = len(TREES)
    _, J0 = system(x, list(range(len(x))))
    Jf = np.array([[float(KR.mid(z)) for z in row] for row in J0])
    piv = sl.qr(Jf, pivoting=True)[2]
    free, fixed = sorted(piv[:m]), sorted(piv[m:])
    for q in fixed:
        x[q] = KR.rounded(x[q], KR.FIX_BITS)
    for _ in range(12):
        F, J = system(x, free)
        Fm = mp.matrix([KR.mid(z) for z in F])
        if mp.norm(Fm, mp.inf) < mp.mpf(10) ** -85:
            break
        dx = mp.lu_solve(mp.matrix([[KR.mid(z) for z in row] for row in J]), Fm)
        for k, q in enumerate(free):
            x[q] -= dx[k]
    for q in free:
        x[q] = KR.rounded(x[q], 280)
    _, J = system(x, free)
    C = CA.point_inverse([[KR.mid(z) for z in row] for row in J])
    out = {}
    for name, arb in (("mpmath", False), ("Arb", True)):
        B = CA.backend(arb)
        CA.BK["iv"] = B["iv"]
        try:
            F, _ = system(x, free)
            _, JX = system(x, free, box=KR.RHO)
            Ci = [[B["conv"](C[i][j]) for j in range(m)] for i in range(m)]
            ok, worst = True, 0.0
            for i in range(m):
                kk = -sum((Ci[i][j] * F[j] for j in range(m)), B["conv"](0))
                for j in range(m):
                    e = B["conv"](1 if i == j else 0) - sum((Ci[i][l] * JX[l][j] for l in range(m)), B["conv"](0))
                    kk += e * B["rad"]
                worst = max(worst, B["upper"](kk))
                ok = ok and B["inside"](kk)
        finally:
            CA.BK["iv"] = CA._iv
        out[name] = dict(proved=bool(ok), K_radius=worst)
        print(f"two chains of {S} stages, order {S}, E = {d['emin']['E'][0]:.4e}: Krawczyk [{name}] "
              f"{'PROVED' if ok else 'not proved'}, K - y~ within {worst:.1e}", flush=True)
    d["krawczyk"] = dict(fixed=[int(q) for q in fixed], **out)
    json.dump(d, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["emin"]:
        emin()
    elif sys.argv[1:2] == ["prove"]:
        prove()

    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
