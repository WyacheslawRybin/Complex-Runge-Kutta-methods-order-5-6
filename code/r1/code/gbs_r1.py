"""GBS(2,4,6) as a variable-step pair 6(4) and its comparison with CRK6(4) at equal latency (internal review 5).

GBS(2,4,6) is the extrapolated Gragg midpoint method with the step sequence 2, 4, 6 and no smoothing,
written as a real ten-stage RK tableau (gbs246_r1.py); the extrapolation of the sequences 2 and 4 (T_32,
order four) is its embedded formula.  On two processors, one computes the sequence n = 6 and the other the
sequences n = 2 and n = 4 one after the other, so a step takes six sequential evaluations of f, as a step
of CRK6(4) does.  Every midpoint update adds one product and one addition to the critical path, so a step
of GBS(2,4,6) takes 6 l_R in the notation of latency_r1.py, and a step of a projected six-stage method
takes 6 kappa_lat l_R.

1. Variable steps with the controller of code/adaptive.py (exponent 1/5, no FSAL) on the ten problems
   and on BRUSS-2D; tolerances 1e-3 ... 1e-15 (BRUSS-2D: 1e-3 ... 1e-13, as bruss2d_r1.py).
   Writes data/adaptive_gbs_r1.json.
2. tables/gbs.tex: evaluations of GBS(2,4,6) and CRK6(4) at the accuracy 1e-8 and the latency ratio
   (steps x latency per step) for kappa_lat at nu = 1/4 and nu = 1, on the polynomial problems; numbers of
   real multiplications for one evaluation of f (two real units for GBS, one complex unit for CRK6(4)).
Usage: python gbs_r1.py [run|table|all]
"""
import json
import math
import os
import sys
import time
from fractions import Fraction as F
from multiprocessing import Pool

import numpy as np

import r1common as R
import adaptive as AD

FN = os.path.join(R.DATA, "adaptive_gbs_r1.json")
TOLS = [10.0 ** (-k / 2) for k in range(6, 31)]
NAME = "GBS(2,4,6)"


def gbs_tableau(seq=(2, 4, 6)):
    """Exact tableau (A, b, bh) of the extrapolated midpoint method; bh: extrapolation of seq[:-1]."""
    stages = [dict()]
    out = {}
    for n in seq:
        d = F(1, n)
        z_prev, z = {}, {0: d}
        for _ in range(1, n):
            stages.append(dict(z))
            ki = len(stages) - 1
            z_new = dict(z_prev)
            z_new[ki] = z_new.get(ki, 0) + 2 * d
            z_prev, z = z, z_new
        out[n] = z

    def weights(ns):
        w = {}
        for nj in ns:
            wj = F(1)
            for ni in ns:
                if ni != nj:
                    wj *= F(nj * nj, nj * nj - ni * ni)
            w[nj] = wj
        return w

    s = len(stages)
    A = [[F(0)] * s for _ in range(s)]
    for i, st in enumerate(stages):
        for k, v in st.items():
            A[i][k] = v
    vec = lambda w: [sum((w[n] * out[n].get(k, 0) for n in w), F(0)) for k in range(s)]
    return A, vec(weights(seq)), vec(weights(seq[:-1]))


def euler_tableau(seq=(1, 2, 3, 4, 5)):
    """Exact tableau of the extrapolated explicit Euler method (harmonic sequence, polynomial
    extrapolation in h, order len(seq)); bh: extrapolation of seq[:-1] (order len(seq) - 1)."""
    stages = [dict()]
    out = {}
    for n in seq:
        d = F(1, n)
        z = {0: d}                                   # z_1 = y0 + d k0
        for _ in range(1, n):
            stages.append(dict(z))
            z = dict(z)
            ki = len(stages) - 1
            z[ki] = z.get(ki, 0) + d
        out[n] = z

    def weights(ns):
        w = {}
        for nj in ns:
            wj = F(1)
            for ni in ns:
                if ni != nj:
                    wj *= F(nj, nj - ni)
            w[nj] = wj
        return w

    s = len(stages)
    A = [[F(0)] * s for _ in range(s)]
    for i, st in enumerate(stages):
        for k, v in st.items():
            A[i][k] = v
    vec = lambda w: [sum((w[n] * out[n].get(k, 0) for n in w), F(0)) for k in range(s)]
    return A, vec(weights(seq)), vec(weights(seq[:-1]))


METHODS = {   # name: (tableau, order, embedded order, depth, processors, complex counterpart)
    "GBS(2,4,6)": (gbs_tableau, 6, 4, 6, 2, "CRK6(4)"),
    "EX5": (euler_tableau, 5, 4, 5, 3, "CRK5(4)"),
}


def pair(name=NAME):
    fun, p, q = METHODS[name][:3]
    A, b, bh = fun()
    return dict(A=np.array([[float(x) for x in r] for r in A]), b=np.array([float(x) for x in b]),
                bh=np.array([float(x) for x in bh]), p=p, q=q, projected=False, fsal=False)


_P = None


def _init():
    global _P
    AD._init()
    _P = {n: pair(n) for n in METHODS}


def task(args):
    name, prob = args
    if prob == "Bruss2D":
        import bruss2d_r1 as BR
        f, y0, T, ref, norm, tols = BR.f_np, BR.y0(), BR.TEND, BR.reference(), BR.rms, AD.TOLS
    else:
        pr = AD._PROB[prob]
        f, y0, T, ref = pr["f_np"], [float(v) for v in pr["y0"]], float(pr["T"]), AD._REF[prob]
        norm, tols = np.linalg.norm, TOLS
    rows = []
    for tol in tols:
        out = AD.solve(_P[name], f, y0, T, tol)
        if out is None:
            rows.append(dict(tol=tol, err=None))
            continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=float(norm(y - ref)), nfev=nf, nacc=na, nrej=nr))
    return name, prob, rows


def run(nproc=6):
    from problems import ADAPTIVE
    res = {}
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        tasks = [(n, p) for n in METHODS for p in list(ADAPTIVE) + ["Bruss2D"]]
        for name, prob, rows in pool.imap_unordered(task, tasks):
            res.setdefault(name, {})[prob] = rows
            ok = [r for r in rows if r["err"] is not None]
            print(f"[{time.time() - t0:5.0f}s] {name} {prob:13s} last: err={ok[-1]['err']:.2e} nfev={ok[-1]['nfev']}",
                  flush=True)
    json.dump(res, open(FN, "w"), indent=1)
    print("wrote", FN)


def table(eps=1e-8):
    """tables/gbs.tex: per polynomial problem, evaluations to reach eps and latency ratio complex/real for
    GBS(2,4,6) vs CRK6(4) and EX5 vs CRK5(4) at nu = 1/4 and nu = 1; data/gbs_r1.json."""
    import r1patch
    MT = r1patch.apply()
    import figures_cost as FC
    from problems import ADAPTIVE
    g = json.load(open(FN))
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    lat = json.load(open(os.path.join(R.DATA, "latency_r1.json")))
    br = json.load(open(os.path.join(R.DATA, "bruss2d_r1.json")))["adaptive"]

    def cost(rows, e):
        ok = [r for r in rows if r["err"] is not None and r["err"] > 0]
        return FC.cost_at(np.array([r["nfev"] for r in ok], float), np.array([r["err"] for r in ok]), e)

    def ccost(m, p, e):
        return cost(br[m], e) if p == "Bruss2D" else FC.cost_at(*FC.adapt_curve(ad, m, p)[:2], e)

    cell = lambda v: MT.best(f"${v:.2f}$") if v < 0.995 else f"${v:.2f}$"
    lines, num = [], {}
    for p, d in lat.items():
        row = [MT.PNAME.get(p, "BRUSS-2D").replace(", $T=5$", "")]
        for name, (_, order, _, depth, nproc, cm) in METHODS.items():
            evals = len(gbs_tableau()[0]) if name == NAME else len(euler_tableau()[0])
            cg, cc = cost(g[name][p], eps), ccost(cm, p, eps)
            lr = cg * depth / evals                     # latency of the real method in units of l_R
            r = {nu: cc * d["kappa_lat"][f"{order}|{nu:g}"] / lr for nu in (0.25, 1)}
            num.setdefault(name, {})[p] = dict(real=cg, cplx=cc, lat_real=lr, ratio_nu025=r[0.25], ratio_nu1=r[1],
                                               mult_real=nproc * d["real_mult"], mult_cplx=d["cplx_mult"])
            row += [f"{cg:.0f}", f"{cc:.0f}", cell(r[0.25]), cell(r[1])]
            print(f"{p:12s} {name:10s} {cg:7.0f}  {cm} {cc:7.0f}  latency complex/real nu=1/4 {r[0.25]:.2f}  "
                  f"nu=1 {r[1]:.2f}  mult real {nproc * d['real_mult']} complex {d['cplx_mult']}")
        lines.append(" & ".join(row) + r"\\")
    open(os.path.join(R.TAB, "gbs.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # evaluations on all ten problems (kappa = 1)
    for name, (_, _, _, depth, _, cm) in METHODS.items():
        for e in (1e-6, 1e-8, 1e-10):
            rr = [(q, cost(g[name][q], e), ccost(cm, q, e)) for q in ADAPTIVE]
            num.setdefault(name + "|all", {})[f"{e:g}"] = {q: [a, b] for q, a, b in rr}
            print(f"{name} eps {e:g}: " + "  ".join(f"{q[:6]} {a:.0f}/{b:.0f}" for q, a, b in rr))
    json.dump(num, open(os.path.join(R.DATA, "gbs_r1.json"), "w"), indent=1,
              default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else o)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("run", "all"):
        run()
    if what in ("table", "all"):
        table()
