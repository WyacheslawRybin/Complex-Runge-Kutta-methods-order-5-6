"""Long-time behaviour on conservative problems (internal review 8).

1. Harmonic oscillator q'' = -q, fixed step h = 0.5, 10^5 steps: the projected step of a linear problem is
   T_p(hJ) (Section 6), so the amplitude grows by |T_p(ih)| per step; measured against the prediction.
2. Kepler problem, e = 0.5, 1000 periods, fixed step with 600 right-hand side evaluations per period for
   every method: global error and energy error at the end of periods 1, 2, 5, 10, ..., 1000, and the slopes
   of log error versus log t over periods 10-1000.
Writes data/longtime_r1.json.  Usage: python longtime_r1.py
"""
import json
import os

import numpy as np

import r1common as R
import crk
import r1patch
from tableaux import load_refined

OUT = os.path.join(R.DATA, "longtime_r1.json")
METHODS = ["CRK5-opt", "CRK5-s", "DP5", "CRK6-opt", "CRK6-s", "Butcher6", "RK6-opt"]
CHECK = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def stepper(A, b, proj, f):
    s = len(b)
    c = A.sum(1)
    dt = complex if proj else float

    def step(t, y, h):
        K = np.zeros((s, len(y)), dt)
        for i in range(s):
            K[i] = f(t + c[i] * h, y + h * (A[i, :i] @ K[:i]))
        y1 = y + h * (b @ K)
        return y1.real if proj else y1
    return step


def kepler(t, y):
    q = y[:2]
    r3 = (q[0] * q[0] + q[1] * q[1]) ** 1.5
    return np.array([y[2], y[3], -q[0] / r3, -q[1] / r3])


def energy(y):
    return 0.5 * (y[2] ** 2 + y[3] ** 2) - 1.0 / np.sqrt(y[0] ** 2 + y[1] ** 2)


def main():
    r1patch.apply()
    T = load_refined()
    res = {}
    e = 0.5
    y0 = np.array([1 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))])
    H0 = energy(y0)
    for m in METHODS:
        A, b, _ = crk.as_numpy(T[m])
        proj = bool(T[m].projected)
        if not proj:
            A, b = A.real, b.real
        s = len(b)
        # 1. harmonic oscillator
        osc = stepper(A, b, proj, lambda t, y: np.array([y[1], -y[0]]))
        y = np.array([1.0, 0.0])
        h = 0.5
        for _ in range(100000):
            y = osc(0.0, y, h)
        amp = float(np.hypot(*y))
        # stability polynomial on the imaginary axis (real part of the complex one for projected methods)
        Rz = lambda z: 1 + z * (b @ np.linalg.solve(np.eye(s) - z * A, np.ones(s)))
        if proj:
            Rr = 0.5 * (Rz(1j * h) + np.conj(Rz(-1j * h)))       # T_p(ih)
        else:
            Rr = Rz(1j * h)
        pred = float(abs(Rr) ** 100000)
        # 2. Kepler
        n = round(600 / s)
        hk = 2 * np.pi / n
        stp = stepper(A, b, proj, kepler)
        y = y0.copy()
        rows = []
        for k in range(1, CHECK[-1] + 1):
            for j in range(n):
                y = stp(0.0, y, hk)
            if k in CHECK:
                rows.append(dict(period=k, err=float(np.linalg.norm(y - y0)), dH=float(abs(energy(y) - H0))))
        sel = [r for r in rows if r["period"] >= 10]
        lt = np.log10([r["period"] for r in sel])
        slope_err = float(np.polyfit(lt, np.log10([r["err"] for r in sel]), 1)[0])
        slope_H = float(np.polyfit(lt, np.log10([r["dH"] for r in sel]), 1)[0])
        res[m] = dict(stages=s, osc_amplitude=amp, osc_predicted=pred, kepler_steps_per_period=n, kepler=rows,
                      slope_err=slope_err, slope_energy=slope_H)
        print(f"{m:9s} oscillator: amplitude after 1e5 steps {amp:.4f} (predicted {pred:.4f}); Kepler ({n} steps/period): "
              f"err at 10/100/1000 periods {rows[3]['err']:.1e}/{rows[6]['err']:.1e}/{rows[9]['err']:.1e}, "
              f"|dH| {rows[3]['dH']:.1e}/{rows[6]['dH']:.1e}/{rows[9]['dH']:.1e}, slopes err {slope_err:.2f}, "
              f"energy {slope_H:.2f}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
