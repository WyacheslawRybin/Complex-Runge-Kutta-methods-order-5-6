"""
Front of the error constant against the coefficient bound, E_min(B), on the families
CRK5, CRK6 and CCK6(5), and the behaviour of the minimizers for large B.

Chains of constrained minimizations (optimize_E.minimize, soft bound with rho = 10):
  * descending chains: start at a local minimum of the multistart with bound 100
    (data/Emin_<FAM>_100.json) and lower the bound step by step from 1e4 to 0.45,
    each minimization warm-started from the previous one;
  * ascending chains: start at the tableaux of the paper and raise the bound from
    0.45 to 1e4.
E_min(B) is the smallest E over all chains that satisfy the bound.  For every point
the smallest distance between two abscissae (including c_1 = 0) and the largest
moduli of A and of b are recorded: a minimizer whose coefficients grow without
bound while the abscissae coalesce is a degenerate (confluent) limit, not a tableau.
Writes ../data/front_E.json.
(The stored runs were made before the renaming CRK5-opt -> CRK5-s, CRK6-opt -> CRK6-s,
so their start labels use the old names.)
Usage:  python front_E.py [NPROC]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BOUNDS = [1e4, 3e3, 1e3, 300, 100, 50, 30, 20, 15, 10, 7, 5, 4, 3, 2.5, 2, 1.5, 1.25, 1.0,
          0.8, 0.7, 0.6, 0.55, 0.5, 0.45]
PAPER = {"CRK5": ["CRK5-s", "CRK5-1", "CRK5-2", "GJM5"],
         "CRK6": ["CRK6-0", "CRK6-s", "CRK6-lam", "CCK6(5)", "CCK6(5)-s", "CCK6-casc"],
         "CCK65": ["CCK6(5)", "CCK6(5)-s", "CCK6-casc"]}

_MOD = {}
def module(fam):
    if fam not in _MOD:
        import optimize_E as O
        _MOD[fam] = (O, O.load_module(fam))
    return _MOD[fam]

def info(TC, x):
    A, W = TC.unpack(x)
    c = A.sum(1)
    gap = min(abs(c[i] - c[j]) for i in range(TC.S) for j in range(i))
    return dict(gap=float(gap), maxA=float(np.abs(A).max()), maxb=float(max(np.abs(w).max() for w in W)),
                c=[[float(v.real), float(v.imag)] for v in c])

def chain(args):
    fam, label, x0, bounds = args
    O, TC = module(fam)
    x = np.array(x0)
    out = []
    t0 = time.time()
    for B in bounds:
        x1, E, it = O.minimize(TC, x, B, rho=10.0, iters=800)
        r = float(np.linalg.norm(TC.residual(x1)))
        ok = r < 1e-10 and np.all(np.isfinite(x1))
        if ok:
            x = x1
        rec = dict(B=B, E=TC.error_constant(x1) if ok else None, maxcoef=float(TC.max_coef(x1)) if ok else None,
                   resid=r, iters=it, x=x1.tolist() if ok else None)
        if ok:
            rec.update(info(TC, x1))
            if len(TC.LEV) > 1:
                rec["E_emb"] = TC.error_constant(x1, 1)
        out.append(rec)
    return fam, label, out, time.time() - t0

def distinct(results, k):
    sel = []
    for r in results:
        if r["resid"] > 1e-11:
            continue
        if all(abs(r["E"] - q["E"]) > 2e-5 * r["E"] or abs(r["maxcoef"] - q["maxcoef"]) > 0.05 * r["maxcoef"]
               for q in sel):
            sel.append(r)
        if len(sel) == k:
            break
    return sel

if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    import optimize_E as O
    tasks = []
    for fam, k in [("CRK5", 7), ("CRK6", 9), ("CCK65", 6)]:
        d = json.load(open(os.path.join(DATA, f"Emin_{fam}_100.json")))
        for r in distinct(d["results"], k):
            tasks.append((fam, "down:" + r["start"], r["x"], BOUNDS))
        TC = O.load_module(fam)
        for name in PAPER[fam]:
            x0 = TC.newton_project(O.x_from_tableau(TC, name), iters=10)
            tasks.append((fam, "up:" + name, x0.tolist(), BOUNDS[::-1]))
    print(len(tasks), "chains", flush=True)
    res = []
    t0 = time.time()
    with Pool(nproc) as pool:
        for fam, label, out, dt in pool.imap_unordered(chain, tasks):
            res.append(dict(family=fam, chain=label, points=out))
            pts = {p["B"]: p for p in out}
            show = " ".join(f"B{B:g}:" + ("-" if pts[B]["E"] is None else f"{pts[B]['E']:.4e}/{pts[B]['maxcoef']:.3g}")
                            for B in (1e4, 100, 5, 2, 1, 0.5))
            print(f"[{time.time() - t0:5.0f}s] {fam:5s} {label:24s} ({dt:4.0f}s) {show}", flush=True)
            json.dump(res, open(os.path.join(DATA, "front_E.json"), "w"))
    print("done", time.time() - t0)
