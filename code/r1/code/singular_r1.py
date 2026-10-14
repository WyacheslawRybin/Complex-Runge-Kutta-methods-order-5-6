"""Distance of the complex stage values to the singularities of f (review, Section 7, question 6).

The Kepler and Arenstorf right-hand sides contain (x^2 + y^2)^{3/2} (for Arenstorf about two centers),
evaluated in complex arithmetic with the principal branch.  For a complex position Q = X + iH
(X, H real 2-vectors) relative to a center, r^2 = |X|^2 - |H|^2 + 2i X.H, so r^2 lies on the branch cut
(-inf, 0] or at the singularity r^2 = 0 only if |H| >= |X|.  The ratio

    rho = |Im Q| / |Re Q - center|

is the imaginary displacement of a stage relative to the distance of the real point to the nearest
complex singularity; rho < 1 guarantees that the principal branch is never left.

For every projected pair and tolerance of the variable-step experiments (all attempted steps,
including rejected ones), and for the fixed-step runs with the coarsest step counts, the script
records the maximum of rho over all stages, over the stages of accepted steps only, and the
99th percentile, together with the global error at T.
Writes data/singular.json.
"""
import json
import os

import numpy as np

import r1common as R
import adaptive as AD
from problems import P

MU = 0.012277471
CENTERS = {"Kepler": [(0.0, 0.0)], "Arenstorf": [(-MU, 0.0), (1 - MU, 0.0)]}
PAIRS = ["CRK5(4)", "CRK6(4)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc"]


def rho_of(Y, centers):
    X, H = Y[:2].real, Y[:2].imag
    return max(np.linalg.norm(H) / np.linalg.norm(X - np.array(c)) for c in centers)


def solve_track(pair, f, y0, T, tol, centers, max_steps=2_000_000):
    """adaptive.solve with the stage values recorded (same controller, same starting step)."""
    A, b, bh, q = pair["A"], pair["b"], pair["bh"], pair["q"]
    s = len(b); c = A.sum(1); p = pair["p"]
    y = np.array(y0, float); t = 0.0
    sc = lambda u, v: tol + tol * np.maximum(np.abs(u), np.abs(v))
    f0 = np.real(f(t, y)) * 1.0
    d0 = np.sqrt(np.mean((y / sc(y, y)) ** 2)); d1 = np.sqrt(np.mean((f0 / sc(y, y)) ** 2))
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    f1 = np.real(f(t + h0, y + h0 * f0))
    d2 = np.sqrt(np.mean(((f1 - f0) / sc(y, y)) ** 2)) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1.0 / (p + 1))
    h = min(100 * h0, h1, T)
    k1 = f0; nacc = nrej = 0; facmax = 5.0
    rhos = []; acc = []
    while t < T * (1 - 1e-14):
        if t + h > T:
            h = T - t
        K = np.zeros((s, len(y)), complex)
        K[0] = k1
        step = []
        for i in range(1, s):
            Yi = y + h * (A[i, :i] @ K[:i])
            step.append(rho_of(Yi, centers))
            K[i] = f(t + c[i] * h, Yi)
        rhos += step
        ynew = (y + h * (b @ K)).real; yhat = (y + h * (bh @ K)).real
        err = np.sqrt(np.mean(((ynew - yhat) / sc(y, ynew)) ** 2))
        if not np.isfinite(err):
            return None
        if err <= 1.0:
            acc += step
            t += h; y = ynew; nacc += 1
            k1 = np.real(f(t, y)) * 1.0
            h *= min(facmax, max(0.2, 0.9 * err ** (-1.0 / (q + 1)) if err > 0 else facmax)); facmax = 5.0
        else:
            nrej += 1
            h *= max(0.2, 0.9 * err ** (-1.0 / (q + 1))); facmax = 1.0
        if nacc + nrej > max_steps or h < 1e-14 * T:
            return None
    return y, np.array(rhos), nacc, nrej, np.array(acc)


def fixed_track(A, b, f, y0, T, N, centers):
    s = len(b); c = A.sum(1); h = T / N
    y = np.array(y0, float); t = 0.0; rhos = []
    for n in range(N):
        K = np.zeros((s, len(y)), complex)
        K[0] = f(t, y)
        for i in range(1, s):
            Yi = y + h * (A[i, :i] @ K[:i])
            rhos.append(rho_of(Yi, centers))
            K[i] = f(t + c[i] * h, Yi)
        y = (y + h * (b @ K)).real; t += h
    return y, np.array(rhos)


def main():
    global REF
    from problems import references
    REF = {k: np.array([float(v) for v in vals]) for k, vals in references().items()}
    pairs = AD.load_pairs()
    out = {"adaptive": {}, "fixed": {}}
    for prob in ("Kepler", "Arenstorf"):
        pr = P[prob]; T = float(pr["T"]); y0 = [float(v) for v in pr["y0"]]
        for name in PAIRS:
            rows = []
            for tol in AD.TOLS:
                r = solve_track(pairs[name], pr["f_np"], y0, T, tol, CENTERS[prob])
                if r is None:
                    rows.append(dict(tol=tol, failed=True)); continue
                y, rh, na, nr, ra = r
                err = float(np.linalg.norm(y - REF[prob]))
                rows.append(dict(tol=tol, err=err, rho_max_accepted=float(ra.max()), rho_max=float(rh.max()), rho_99=float(np.percentile(rh, 99)),
                                 rho_med=float(np.median(rh)), n_ge_1=int((rh >= 1).sum()), nacc=na, nrej=nr))
            out["adaptive"].setdefault(prob, {})[name] = rows
            ok = [x for x in rows if not x.get("failed")]
            print(f"{prob:9s} {name:10s} rho_max over tol: {max(x['rho_max'] for x in ok):.3f} "
                  f"(tol 1e-3: {ok[0]['rho_max']:.3f}, tol 1e-6: {[x for x in ok if abs(x['tol'] - 1e-6) < 1e-12][0]['rho_max']:.3f}, "
                  f"tol 1e-13: {ok[-1]['rho_max']:.3f}); failed: {len(rows) - len(ok)}", flush=True)
    # fixed steps: the coarsest step counts of the fixed-step experiments (Kepler, N0 = 200, k = -2, -1, 0)
    pr = P["Kepler"]; T = float(pr["T"]); y0 = [float(v) for v in pr["y0"]]
    for name in ("CRK5(4)", "CRK6(4)", "CCK6(5)"):
        for N in (50, 100, 200):
            y, rh = fixed_track(pairs[name]["A"], pairs[name]["b"], pr["f_np"], y0, T, N, CENTERS["Kepler"])
            out["fixed"].setdefault(name, {})[N] = dict(rho_max=float(rh.max()), rho_99=float(np.percentile(rh, 99)))
            print(f"fixed Kepler {name:8s} N={N:4d}: rho_max={rh.max():.3f}", flush=True)
    json.dump(out, open(os.path.join(R.DATA, "singular.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
