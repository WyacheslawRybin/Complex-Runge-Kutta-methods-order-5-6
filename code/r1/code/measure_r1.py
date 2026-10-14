"""Measured error constants of the real pairs added in R1 (Tsit5(4), BS5(4), Vern6(5)).

Same procedure as code/measure.py of the original submission: 34-digit fixed-step runs on
N = N0 * 2^k, k = -2..6, on the thirteen problems of the order study, one Richardson step on
C(h) = err / h^p, and an extension to k = 7, 8 for the cells whose two last extrapolated values
differ by more than 1%.  The propagated (higher-order) solution of each pair is used.
Writes R1/data/measured_r1.json in the format of data/measured.json; the original file is not
modified.  Usage: python measure_r1.py [processes]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import mpmath as mp

import r1common as R
import measure as M

METHODS = [("Tsit5(4)", "Tsit5(4)", 0), ("BS5(4)", "BS5(4)", 0), ("Vern6(5)", "Vern6(5)", 0)]
FN = os.path.join(R.DATA, "measured_r1.json")


def _init():
    M._init()
    from tableaux import load_refined
    with mp.workdps(60):
        M._TAB.update(load_refined(os.path.join(R.DATA, "tableaux_r1.json")))


def _conv(rows):
    Ch = [r["Ch"] for r in rows if r["err"]]
    Rk = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
    return abs(Rk[-1] - Rk[-2]) / abs(Rk[-1]) if len(Rk) > 1 and Rk[-1] else 1.0


def main(nproc=6):
    from problems import ORDER
    res = json.load(open(FN)) if os.path.exists(FN) else {}
    tasks = [(lab, key, L, prob) for (lab, key, L) in METHODS for prob in ORDER
             if prob not in res.get(lab, {})]
    tasks.sort(key=lambda a: -M.N0[a[3]])
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for label, prob, rows, dt in pool.imap_unordered(M.task, tasks):
            res.setdefault(label, {})[prob] = dict(rows=rows)
            print(f"[{time.time() - t0:6.0f}s] {label:9s} {prob:14s} ({dt:5.0f}s) "
                  + " ".join("-" if r["Ch"] is None else f"{r['Ch']:.4e}" for r in rows[-4:]), flush=True)
            json.dump(res, open(FN, "w"), indent=1)
        ks = (7, 8)
        todo = []
        for (lab, key, L) in METHODS:
            for prob, cell in res[lab].items():
                have = max(r["N"] for r in cell["rows"])
                if _conv(cell["rows"]) > 0.01 and have < M.N0[prob] * 2 ** max(ks):
                    todo.append((lab, key, L, prob, [k for k in ks if M.N0[prob] * 2 ** k > have]))
        print(len(todo), "cells to extend:", [(t[0], t[3]) for t in todo], flush=True)
        for label, prob, rows in pool.imap_unordered(M.task_ext, todo):
            res[label][prob]["rows"] += rows
            print(f"[{time.time() - t0:6.0f}s] {label:9s} {prob:14s} extended "
                  + " ".join(f"{r['Ch']:.4e}" for r in res[label][prob]["rows"][-4:]), flush=True)
            json.dump(res, open(FN, "w"), indent=1)
    for (lab, key, L) in METHODS:
        for prob in ORDER:
            s = M.summarize(res[lab][prob]["rows"], None)
            print(f"{lab:9s} {prob:14s} C = {s['C']:.4e}  conv {s['conv']:.1e}  "
                  f"order {s.get('order_last', float('nan')):.2f}")


P_OF = {"Tsit5(4)": 5, "BS5(4)": 5, "Vern6(5)": 6}
DPS_EXT = 44                      # round-off of the longest runs (4e5 steps) stays below 1e-3 of the error


def _task_more(args):
    label, key, L, prob, N = args
    mp.mp.dps = DPS_EXT
    T = M._TAB[key]
    p = T.levels[L][1]
    Tend = mp.mpf(__import__("problems").P[prob]["T"])
    e = M._run(T, L, prob, N)
    return label, prob, dict(N=N, err=None if e is None else float(e),
                             Ch=None if e is None else float(e / (Tend / N) ** p))


def extend_more(nproc=6, maxk=11):
    """Halve the step further (k = 9..maxk, one level at a time) for the cells that are not
    classified as converged by results.classify; Vern6(5) has a very small sixth-order constant,
    so its error is dominated by the h^7 term down to small steps."""
    from results import classify
    res = json.load(open(FN))
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for k in range(9, maxk + 1):
            todo = []
            for (lab, key, L) in METHODS:
                for prob, cell in res[lab].items():
                    N = M.N0[prob] * 2 ** k
                    if classify(cell["rows"], P_OF[lab])["cls"] != "ok" and max(r["N"] for r in cell["rows"]) < N:
                        todo.append((lab, key, L, prob, N))
            print(f"k={k}:", [(t[0], t[3]) for t in todo], flush=True)
            for label, prob, row in pool.imap_unordered(_task_more, todo):
                res[label][prob]["rows"].append(row)
                c = classify(res[label][prob]["rows"], P_OF[label])
                print(f"[{time.time() - t0:6.0f}s] {label:9s} {prob:14s} N={row['N']:7d} Ch={row['Ch']:.5e} "
                      f"C={c['C']:.4e} conv={c['conv']:.1e} order={c['order']:.2f} {c['cls']}", flush=True)
                json.dump(res, open(FN, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["more"]:
        extend_more(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
