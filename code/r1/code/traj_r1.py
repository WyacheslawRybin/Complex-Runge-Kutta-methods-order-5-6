"""Variable-step comparison with the error measured along the whole trajectory and on perturbed initial data
(internal review 6, item C4 of the independent review).

The costs of Section 7.5 use the error at the final time T only.  Here every pair is run with the same controller,
starting step and tolerances (1e-3 ... 1e-15), and the error is measured at every accepted step,
    err_traj = max_n || y_n - y(t_n) ||_2,
which does not change the step-size sequence.  The reference y(t) on [0, T] is the dense output of DOP853
(scipy) with rtol = atol = 1e-14; its accuracy is checked against the Taylor references at T.  Besides the
initial data of Appendix B, five perturbed initial data per problem are used, y0 * (1 + 0.01 xi) with xi
uniform in [-1, 1] componentwise (fixed seeds), with their own DOP853 references.
For every problem and initial datum, the cost to reach 1e-6 and 1e-8 in err_traj and in the error at T is
interpolated as in Section 7.5, and CRK6(4) is compared with the best real order-six pair.
Writes data/traj_r1.json.  Usage: python traj_r1.py [processes]
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
import adaptive as AD
import adaptive_new_r1 as AN
import vern6_split_r1 as VS
import real64_r1 as R64

TOLS = [10.0 ** (-k / 2) for k in range(6, 31)]
FN = os.path.join(R.DATA, "traj_r1.json")
PAIRS = ["CRK5(4)", "DP5", "CK5", "Tsit5(4)", "BS5(4)", "CRK6(4)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc",
         "RK6-opt(4)", "DP6(5)", "Verner6(5)", "Vern6(5)"]
REAL6 = ["RK6-opt(4)", "DP6(5)", "Verner6(5)", "Vern6(5)"]
NPERT = 5
_P = _VS = None


def solve_record(pair, f, y0, T, tol, split=None, max_steps=2_000_000):
    """code/adaptive.py: solve() (or vern6_split_r1.solve for split coefficients) recording (t_n, y_n)."""
    if split is not None:
        Ah, Al, bh_, bl, eh, el, c = (split[k] for k in ("Ah", "Al", "bh_", "bl", "eh", "el", "c"))
        A, b, bh = Ah, bh_, eh
        p, q, proj, fsal = 6, 5, False, True
    else:
        A, b, bh, q, proj, fsal = pair["A"], pair["b"], pair["bh"], pair["q"], pair["projected"], pair["fsal"]
        p = pair["p"]
        c = A.sum(1)
    s = len(b)
    y = np.array(y0, float)
    t = 0.0
    sc = lambda u, v: tol + tol * np.maximum(np.abs(u), np.abs(v))
    nf = 0
    f0 = np.real(f(t, y)) * 1.0
    nf += 1
    d0 = np.sqrt(np.mean((y / sc(y, y)) ** 2))
    d1 = np.sqrt(np.mean((f0 / sc(y, y)) ** 2))
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    f1 = np.real(f(t + h0, y + h0 * f0))
    nf += 1
    d2 = np.sqrt(np.mean(((f1 - f0) / sc(y, y)) ** 2)) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1.0 / (p + 1))
    h = min(100 * h0, h1, T)
    k1 = f0
    nacc = nrej = 0
    facmax = 5.0
    dtype = complex if proj else float
    ts, ys = [0.0], [y.copy()]
    while t < T * (1 - 1e-14):
        if t + h > T:
            h = T - t
        K = np.zeros((s, len(y)), dtype)
        K[0] = k1
        for i in range(1, s):
            if split is not None:
                Yi = y + h * (Ah[i, :i] @ K[:i] + Al[i, :i] @ K[:i])
            else:
                Yi = y + h * (A[i, :i] @ K[:i])
            K[i] = f(t + c[i] * h, Yi)
            nf += 1
        if split is not None:
            ynew = y + h * (bh_ @ K + bl @ K)
            yhat = y + h * (eh @ K + el @ K)
        else:
            ynew = y + h * (b @ K)
            yhat = y + h * (bh @ K)
        if proj:
            ynew = ynew.real
            yhat = yhat.real
        err = np.sqrt(np.mean(((ynew - yhat) / sc(y, ynew)) ** 2))
        if not np.isfinite(err):
            return None
        if err <= 1.0:
            t += h
            y = ynew
            nacc += 1
            ts.append(t)
            ys.append(y.copy())
            if fsal:
                k1 = K[-1].real
            else:
                k1 = np.real(f(t, y)) * 1.0
                nf += 1
            fac = min(facmax, max(0.2, 0.9 * err ** (-1.0 / (q + 1)) if err > 0 else facmax))
            h *= fac
            facmax = 5.0
        else:
            nrej += 1
            h *= max(0.2, 0.9 * err ** (-1.0 / (q + 1)))
            facmax = 1.0
        if nacc + nrej > max_steps or h < 1e-14 * T:
            return None
    if not fsal:
        nf -= 1
    return np.array(ts), np.array(ys), nf


def reference(f, y0, T):
    sol = solve_ivp(f, (0.0, T), y0, method="DOP853", rtol=2.3e-14, atol=1e-14, dense_output=True)
    return sol if sol.status == 0 else None


def _init():
    global _P, _VS
    AD._init()
    _P = dict(AD._P)
    _P.update(AN.new_pairs())
    _P["RK6-opt(4)"] = R64.pair_arrays()
    _VS = VS.split_pair()


def initial_data(prob):
    pr = AD._PROB[prob]
    y0 = np.array([float(v) for v in pr["y0"]])
    out = [y0]
    rng = np.random.default_rng(sum(map(ord, prob)))
    eps = 1e-5 if prob == "Arenstorf" else 1e-2     # a 1% change sends the Arenstorf orbit into the moon
    for _ in range(NPERT):
        out.append(y0 * (1 + eps * rng.uniform(-1, 1, size=y0.size)))
    return out


def task(args):
    prob, k = args
    pr = AD._PROB[prob]
    f, T = pr["f_np"], float(pr["T"])
    y0 = initial_data(prob)[k]
    ref = reference(f, y0, T)
    if ref is None:
        return prob, k, None, None
    ref_check = float(np.linalg.norm(ref.y[:, -1] - AD._REF[prob])) if k == 0 else None
    res = {}
    for name in PAIRS:
        rows = []
        for tol in TOLS:
            out = solve_record(None if name == "Vern6(5)" else _P[name], f, y0, T, tol,
                               split=_VS if name == "Vern6(5)" else None, max_steps=200_000)
            if out is None:
                rows.append(dict(tol=tol, err=None))
                continue
            ts, ys, nf = out
            e = np.linalg.norm(ys - ref.sol(ts).T, axis=1)
            rows.append(dict(tol=tol, nfev=nf, err_T=float(e[-1]), err_traj=float(e.max())))
        res[name] = rows
    return prob, k, res, ref_check


def cost(rows, key, eps):
    import figures_cost as FC
    ok = [r for r in rows if r.get(key) is not None and r[key] > 0]
    if not ok:
        return math.inf
    return FC.cost_at(np.array([r["nfev"] for r in ok], float), np.array([r[key] for r in ok]), eps)


def analyse(data):
    import r1patch
    r1patch.apply()
    out = {}
    for eps in (1e-6, 1e-8):
        for key in ("err_T", "err_traj"):
            wins, total, savings = 0, 0, []
            for prob, runs in data.items():
                for k, res in runs.items():
                    c6 = cost(res["CRK6(4)"], key, eps)
                    br = min(cost(res[m], key, eps) for m in REAL6)
                    if not (math.isfinite(c6) and math.isfinite(br)):
                        continue
                    total += 1
                    wins += int(c6 < br)
                    savings.append(1 - c6 / br)
            out[f"{key}|{eps:g}"] = dict(cheaper=wins, cases=total, median_saving=float(np.median(savings)))
            print(f"{key:8s} eps {eps:g}: CRK6(4) cheaper than the best real order-six pair in {wins} of {total} "
                  f"(problem, initial datum) cases, median saving {np.median(savings):+.1%}")
        # unperturbed initial data only
        for key in ("err_T", "err_traj"):
            wins = [p for p, runs in data.items() if cost(runs["0"]["CRK6(4)"], key, eps) <
                    min(cost(runs["0"][m], key, eps) for m in REAL6)]
            out[f"{key}|{eps:g}|unperturbed"] = wins
            print(f"   unperturbed, {key}: cheaper on {len(wins)} problems: {wins}")
    return out


def main(nproc=6):
    from problems import ADAPTIVE
    tasks = [(p, k) for p in ADAPTIVE for k in range(NPERT + 1)]
    data, checks = {}, {}
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for n, (prob, k, res, chk) in enumerate(pool.imap_unordered(task, tasks), 1):
            if res is None:
                print(f"[{time.time() - t0:5.0f}s] {n}/{len(tasks)} {prob} datum {k}: reference failed, skipped", flush=True)
                continue
            data.setdefault(prob, {})[str(k)] = res
            msg = ""
            if chk is not None:
                checks[prob] = chk
                msg = f", DOP853 vs Taylor at T {chk:.1e}"
            print(f"[{time.time() - t0:5.0f}s] {n}/{len(tasks)} {prob} datum {k}{msg}", flush=True)
    stats = analyse(data)
    json.dump(dict(runs=data, reference_check=checks, stats=stats), open(FN, "w"), indent=1)
    print("wrote", FN)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
