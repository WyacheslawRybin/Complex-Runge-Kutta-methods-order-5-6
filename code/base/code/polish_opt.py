"""
Final search for CRK5-opt and CRK6-opt: minimum of E_{p+1} under max|coef| <= 5.
Starts: the best points of front_E.py at B = 5 (distinct local minima) and random tangent
displacements of the best of them; minimization with the soft bound (rho = 10) and a long
iteration budget, then refinement in multiple precision (60 digits) and recomputation of
E_{p+1}, the residuals and max|coef| in mp.
Writes ../data/opt_search.json (all runs) and ../data/tableaux_opt.json (CRK5-opt, CRK6-opt,
format of tableaux.json, 45 digits).
Usage:  python polish_opt.py [NPROC]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BOUND = 5.0

def run(args):
    fam, label, x0 = args
    import optimize_E as O
    TC = O.load_module(fam)
    x, E, it = O.minimize(TC, np.array(x0), BOUND, rho=10.0, iters=2500)
    r = float(np.linalg.norm(TC.residual(x)))
    return fam, label, dict(start=label, E=TC.error_constant(x), maxcoef=float(TC.max_coef(x)), resid=r, iters=it,
                            x=x.tolist())

def starts(fam, rng, nrand=24, k=5):
    import optimize_E as O
    TC = O.load_module(fam)
    d = json.load(open(os.path.join(DATA, "front_E.json")))
    pts = sorted([(p["E"], ch["chain"], p["x"]) for ch in d if ch["family"] == fam for p in ch["points"]
                  if p["E"] is not None and p["maxcoef"] <= BOUND * 1.001], key=lambda t: t[0])
    sel = []
    for E, lab, x in pts:
        if all(abs(E - q[0]) > 1e-5 * E for q in sel):
            sel.append((E, lab, x))
        if len(sel) == k:
            break
    out = [(fam, f"{lab}@{E:.6e}", x) for E, lab, x in sel]
    x0 = np.array(sel[0][2])
    T = TC.tangent_basis(x0)
    for j in range(nrand):
        sig = [0.03, 0.1, 0.3][j % 3]
        xr = TC.newton_project(x0 + T @ (sig * rng.standard_normal(T.shape[1])), iters=12)
        if np.linalg.norm(TC.residual(xr)) < 1e-11:
            out.append((fam, f"best+rand{j}", xr.tolist()))
    return out

def finalize(best):
    import mpmath as mp
    import crk
    import optimize_E as O
    from verify_Emin import tableau
    mp.mp.dps = 60
    out = {}
    for fam, name in [("CRK5", "CRK5-opt"), ("CRK6", "CRK6-opt")]:
        TC = O.load_module(fam)
        r = best[fam]
        T = tableau(TC, r["x"], name)
        T, shift, res = T.refine()
        out[name] = dict(s=T.s, projected=True, cost=T.s, levels=[(m, p) for m, p, _ in T.levels],
                         A=[[[mp.nstr(mp.re(T.A[i][j]), 45), mp.nstr(mp.im(T.A[i][j]), 45)] for j in range(T.s)]
                            for i in range(T.s)],
                         b=[[[mp.nstr(mp.re(v), 45), mp.nstr(mp.im(v), 45)] for v in b] for _, _, b in T.levels],
                         E=[float(T.E(L)) for L in range(len(T.levels))], maxcoef=float(T.maxcoef()),
                         resid=float(res), full_complex_resid=float(T.full_complex_residual()), start=r["start"],
                         bound=BOUND)
        print(f"{name}: E(mp) {mp.nstr(T.E(), 10)}  max|coef| {mp.nstr(T.maxcoef(), 6)}  refine shift "
              f"{mp.nstr(shift, 2)}  resid {mp.nstr(res, 2)}  c = {[mp.nstr(c, 5) for c in T.c]}", flush=True)
    json.dump(out, open(os.path.join(DATA, "tableaux_opt.json"), "w"), indent=1)

if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    rng = np.random.default_rng(7)
    tasks = starts("CRK5", rng) + starts("CRK6", rng)
    print(len(tasks), "starts", flush=True)
    res = {"CRK5": [], "CRK6": []}
    t0 = time.time()
    with Pool(nproc) as pool:
        for fam, label, r in pool.imap_unordered(run, tasks):
            res[fam].append(r)
            print(f"[{time.time() - t0:5.0f}s] {fam} {label:34s} E {r['E']:.7e}  max|coef| {r['maxcoef']:.4f}  "
                  f"resid {r['resid']:.1e}  ({r['iters']} it)", flush=True)
    best = {}
    for fam in res:
        ok = [r for r in res[fam] if r["resid"] < 1e-11 and r["maxcoef"] <= BOUND * 1.001]
        ok.sort(key=lambda r: r["E"])
        best[fam] = ok[0]
        print(fam, "best:", ok[0]["start"], f"E {ok[0]['E']:.7e} max|coef| {ok[0]['maxcoef']:.4f}")
    json.dump(res, open(os.path.join(DATA, "opt_search.json"), "w"))
    finalize(best)
