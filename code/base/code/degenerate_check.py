"""
Do the minimizers of E_{p+1} that reached the bound max|coef| = 100 (CRK5-E, CCK6(5)-E of
verify_Emin.py) lie on a degenerate branch?  Unconstrained descent (no bound) from these
points with a long iteration budget; along the path we record E, max|coef|, the smallest
distance between two abscissae, and the norm of the projected gradient of E^2.
A degenerate (confluent) branch shows max|coef| growing without bound, the gap shrinking
like 1/max|coef| and E decreasing towards a limit that no tableau attains.
Writes ../data/degenerate_check.json.
Usage:  python degenerate_check.py
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def gap_of(TC, x):
    A, W = TC.unpack(x)
    c = A.sum(1)
    return float(min(abs(c[i] - c[j]) for i in range(TC.S) for j in range(i)))

def run(args):
    fam, name, x0, blocks, per = args
    import optimize_E as O
    TC = O.load_module(fam)
    x = np.array(x0)
    path = []
    for k in range(blocks):
        e, Je = O.err_and_jac(TC, x)
        g = np.linalg.norm((Je @ TC.tangent_basis(x)).T @ e)
        path.append(dict(it=k * per, E=TC.error_constant(x), maxcoef=float(TC.max_coef(x)), gap=gap_of(TC, x),
                         grad=float(g), resid=float(np.linalg.norm(TC.residual(x)))))
        x, E, it = O.minimize(TC, x, 1e12, rho=1.0, iters=per)
        if it < per - 1:                      # converged: no further decrease possible
            e, Je = O.err_and_jac(TC, x)
            g = np.linalg.norm((Je @ TC.tangent_basis(x)).T @ e)
            path.append(dict(it=k * per + it + 1, E=TC.error_constant(x), maxcoef=float(TC.max_coef(x)),
                             gap=gap_of(TC, x), grad=float(g), resid=float(np.linalg.norm(TC.residual(x))),
                             converged=True))
            break
    A, W = TC.unpack(x)
    return fam, name, path, [[float(v.real), float(v.imag)] for v in A.sum(1)]

if __name__ == "__main__":
    import optimize_E as O
    tasks = []
    for fam, base in [("CRK5", "CRK5-E"), ("CRK6", "CRK6-E"), ("CCK65", "CCK6(5)-E")]:
        d = json.load(open(os.path.join(DATA, f"Emin_{fam}_100.json")))
        seen = []
        for r in d["results"]:
            if r["maxcoef"] < 8 or any(abs(r["E"] - q) < 1e-5 * r["E"] for q in seen):
                continue
            seen.append(r["E"])
            tasks.append((fam, r["start"], r["x"], 60, 200))
            if len(seen) == 4:
                break
    print(len(tasks), "tasks", flush=True)
    out = []
    with Pool(min(11, len(tasks))) as pool:
        for fam, name, path, c in pool.imap_unordered(run, tasks):
            out.append(dict(family=fam, start=name, path=path, c=c))
            print(f"== {fam} {name}")
            for p in path[::5] + ([path[-1]] if len(path) % 5 != 1 else []):
                print(f"   it {p['it']:6d}  E {p['E']:.7e}  max|coef| {p['maxcoef']:10.3f}  gap {p['gap']:.3e}  "
                      f"|grad| {p['grad']:.2e}  resid {p['resid']:.1e}" + ("  converged" if p.get("converged") else ""),
                      flush=True)
            json.dump(out, open(os.path.join(DATA, "degenerate_check.json"), "w"), indent=1)
