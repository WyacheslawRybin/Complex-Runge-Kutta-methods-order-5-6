"""Variable-step comparison on two N-body problems (internal review 8).

  * Pleiades: seven bodies in the plane, masses m_i = i, G = 1, t in [0, 3], 28 equations
    ([HNW, Sect. II.10]); close encounters make the step size vary over several decades.
  * Outer solar system: the Sun (with the inner planets), Jupiter, Saturn, Uranus, Neptune and Pluto in
    three dimensions, 36 equations, data of [HLW, Sect. I.2.3], t in [0, T] days.
The right-hand sides are written with complex-safe operations ((r^2)^(-3/2) with the principal branch),
which are holomorphic near the real trajectory.  All pairs of Section 7.5 are run with the controller of
code/adaptive.py (code/traj_r1.solve_record), tolerances 10^(-3) ... 10^(-15) in steps of 10^(-1/2); the
error at T is measured against DOP853 (scipy) with rtol = 2.3e-14, whose accuracy is estimated by a second
reference with rtol = 1e-13.  The costs to reach 1e-6 and 1e-8 are interpolated as in Section 7.5.
Writes data/nbody_r1.json and tables/nbody.tex.  Usage: python nbody_r1.py [processes]
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp

import r1common as R
import traj_r1 as TR

FN = os.path.join(R.DATA, "nbody_r1.json")
TOLS = [10.0 ** (-k / 2) for k in range(6, 31)]
EPS = (1e-6, 1e-8)
REAL5 = ["DP5", "CK5", "Tsit5(4)", "BS5(4)"]
REAL6 = TR.REAL6

# ---------------------------------------------------------------- Pleiades
PL_M = np.arange(1.0, 8.0)
PL_Y0 = np.array([3, 3, -1, -3, 2, -2, 2, 3, -3, 2, 0, 0, -4, 4,
                  0, 0, 0, 0, 0, 1.75, -1.5, 0, 0, 0, -1.25, 1, 0, 0], float)


def pleiades(t, y):
    x, yy = y[0:7], y[7:14]
    dx = x[None, :] - x[:, None]               # x_j - x_i
    dy = yy[None, :] - yy[:, None]
    r2 = dx * dx + dy * dy
    np.fill_diagonal(r2, 1.0)
    w = PL_M[None, :] * r2 ** (-1.5)
    np.fill_diagonal(w, 0.0)
    return np.concatenate([y[14:21], y[21:28], (w * dx).sum(1), (w * dy).sum(1)])


# ---------------------------------------------------------------- outer solar system [HLW, Sect. I.2.3]
SS_G = 2.95912208286e-4
SS_M = np.array([1.00000597682, 0.000954786104043, 0.000285583733151, 0.0000437273164546,
                 0.0000517759138449, 1.0 / 1.3e8])
SS_Q = np.array([[0, 0, 0],
                 [-3.5023653, -3.8169847, -1.5507963],
                 [9.0755314, -3.0458353, -1.6483708],
                 [8.3101420, -16.2901086, -7.2521278],
                 [11.4707666, -25.7294829, -10.8169456],
                 [-15.5387357, -25.2225594, -3.1902382]])
SS_V = np.array([[0, 0, 0],
                 [0.00565429, -0.00412490, -0.00190589],
                 [0.00168318, 0.00483525, 0.00192462],
                 [0.00354178, 0.00137102, 0.00055029],
                 [0.00288930, 0.00114527, 0.00039677],
                 [0.00276725, -0.00170702, -0.00136504]])
SS_Y0 = np.concatenate([SS_Q.ravel(), SS_V.ravel()])
SS_T = 200000.0


def solar(t, y):
    q = y[:18].reshape(6, 3)
    d = q[None, :, :] - q[:, None, :]          # q_j - q_i
    r2 = (d * d).sum(2)
    np.fill_diagonal(r2, 1.0)
    w = SS_G * SS_M[None, :] * r2 ** (-1.5)
    np.fill_diagonal(w, 0.0)
    acc = (w[:, :, None] * d).sum(1)
    return np.concatenate([y[18:], acc.ravel()])


PROBLEMS = {"Pleiades": (pleiades, PL_Y0, 3.0), "OuterSolar": (solar, SS_Y0, SS_T)}


def reference(f, y0, T, rtol):
    sol = solve_ivp(f, (0.0, T), y0, method="DOP853", rtol=rtol, atol=rtol * 1e-2)
    assert sol.status == 0
    return sol.y[:, -1]


def task(args):
    prob, name = args
    f, y0, T = PROBLEMS[prob]
    rows = []
    for tol in TOLS:
        out = TR.solve_record(None if name == "Vern6(5)" else TR._P[name], f, y0, T, tol,
                              split=TR._VS if name == "Vern6(5)" else None, max_steps=400_000)
        if out is None:
            rows.append(dict(tol=tol, err=None))
            continue
        ts, ys, nf = out
        rows.append(dict(tol=tol, nfev=int(nf), y=ys[-1].tolist()))
    return prob, name, rows


def cost(rows, eps):
    import figures_cost as FC
    ok = [r for r in rows if r.get("err") is not None and r["err"] > 0]
    if not ok:
        return math.inf
    return float(FC.cost_at(np.array([r["nfev"] for r in ok], float), np.array([r["err"] for r in ok]), eps))


def main(nproc=4):
    import r1patch
    r1patch.apply()
    t0 = time.time()
    refs = {}
    for p, (f, y0, T) in PROBLEMS.items():
        y1 = reference(f, y0, T, 2.3e-14)
        y2 = reference(f, y0, T, 1e-13)
        refs[p] = dict(y=y1.tolist(), check=float(np.linalg.norm(y1 - y2)))
        print(f"[{time.time() - t0:5.0f}s] {p}: reference, difference of the two DOP853 runs {refs[p]['check']:.1e}", flush=True)
    tasks = [(p, m) for p in PROBLEMS for m in TR.PAIRS]
    data = {p: {} for p in PROBLEMS}
    with Pool(nproc, initializer=TR._init) as pool:
        for n, (p, m, rows) in enumerate(pool.imap_unordered(task, tasks), 1):
            yr = np.array(refs[p]["y"])
            for r in rows:
                if "y" in r:
                    r["err"] = float(np.linalg.norm(np.array(r.pop("y")) - yr))
            data[p][m] = rows
            print(f"[{time.time() - t0:5.0f}s] {n}/{len(tasks)} {p} {m}", flush=True)
    costs = {p: {m: {f"{e:g}": cost(data[p][m], e) for e in EPS} for m in TR.PAIRS} for p in PROBLEMS}
    for p in PROBLEMS:
        for e in EPS:
            k = f"{e:g}"
            c = costs[p]
            b5 = min(c[m][k] for m in REAL5)
            b6 = min(c[m][k] for m in REAL6)
            print(f"{p} eps {k}: CRK5(4) {c['CRK5(4)'][k]:.0f} vs best real 5(4) {b5:.0f}; CRK6(4) {c['CRK6(4)'][k]:.0f}, "
                  f"CCK6(5) {c['CCK6(5)'][k]:.0f} vs best real 6 {b6:.0f} "
                  f"({min(REAL6, key=lambda m: c[m][k])})", flush=True)
    json.dump(dict(references={p: dict(check=v["check"]) for p, v in refs.items()}, runs=data, costs=costs,
                   T={p: v[2] for p, v in PROBLEMS.items()}), open(FN, "w"), indent=1)
    write_table(costs)
    print("wrote", FN)


def write_table(costs):
    import r1patch
    MT = r1patch.apply()
    cols5 = ["CRK5(4)", "DP5", "CK5", "Tsit5(4)", "BS5(4)"]
    cols6 = ["CRK6(4)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc", "RK6-opt(4)", "DP6(5)", "Verner6(5)", "Vern6(5)"]
    name = lambda m: MT.mname(m) if m in MT.MNAME else m
    lines = ["problem & $\\varepsilon$ & " + " & ".join(MT.rot(name(m)) for m in cols5 + cols6) + r"\\", r"\midrule"]
    for p, lab in (("Pleiades", "Pleiades"), ("OuterSolar", "solar system")):
        for e in EPS:
            k = f"{e:g}"
            cells = []
            for grp in (cols5, cols6):
                vals = [costs[p][m][k] for m in grp]
                cells += MT.mark_best(vals, ["---" if not math.isfinite(v) else f"{v:.0f}" for v in vals], rel=0)
            lines.append(f"{lab} & $10^{{{int(round(math.log10(e)))}}}$ & " + " & ".join(cells) + r"\\")
    open(os.path.join(R.TAB, "nbody.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
