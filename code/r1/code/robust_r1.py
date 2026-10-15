"""Noise level of the variable-step cost comparisons (internal review 5).

The costs of Tables adaptive_6/8/10 are interpolated along work-precision curves sampled at 25 tolerances
10^{-k/2}.  A single run in which the error at T happens to be small can make a pair accidentally cheap.
To estimate this noise, every pair is run a second time on the interleaved grid 10^{-k/2 - 1/4}
(the same controller, starting step and reference solutions), and the cost to reach 1e-6, 1e-8 and 1e-10
is computed from each grid separately.  The relative differences of the two estimates over all pairs and
problems give the noise level, and the comparison of CRK6(4) with the best real order-six pair is
repeated on the second grid.  Writes data/robust_r1.json.
Usage: python robust_r1.py [processes]
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import r1common as R
import adaptive as AD
import adaptive_new_r1 as AN
import vern6_split_r1 as VS
import real64_r1 as R64

TOLS_OFF = [10.0 ** (-k / 2 - 0.25) for k in range(6, 30)]      # 10^-3.25 ... 10^-14.75
FN = os.path.join(R.DATA, "robust_r1.json")
PAIRS = ["CRK5(4)", "DP5", "CK5", "Tsit5(4)", "BS5(4)", "CRK6(4)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc",
         "RK6-opt(4)", "DP6(5)", "Verner6(5)", "Vern6(5)"]
_P = _VS = None


def _init():
    global _P, _VS
    AD._init()
    _P = dict(AD._P)
    _P.update(AN.new_pairs())
    _P["RK6-opt(4)"] = R64.pair_arrays()
    _VS = VS.split_pair()


def task(args):
    name, prob = args
    pr = AD._PROB[prob]
    T, y0 = float(pr["T"]), [float(v) for v in pr["y0"]]
    rows = []
    for tol in TOLS_OFF:
        out = (VS.solve(_VS, pr["f_np"], y0, T, tol) if name == "Vern6(5)"
               else AD.solve(_P[name], pr["f_np"], y0, T, tol))
        if out is None:
            rows.append(dict(tol=tol, err=None))
            continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=float(np.linalg.norm(y - AD._REF[prob])), nfev=nf, nacc=na, nrej=nr))
    return name, prob, rows


def cost(rows, eps):
    import figures_cost as FC
    ok = [r for r in rows if r["err"] is not None and r["err"] > 0]
    return FC.cost_at(np.array([r["nfev"] for r in ok], float), np.array([r["err"] for r in ok]), eps)


def analyse(off):
    import r1patch
    r1patch.apply()
    from problems import ADAPTIVE
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    out = {}
    for eps in (1e-6, 1e-8, 1e-10):
        diffs, wins = [], {}
        for g, data in (("main", ad), ("offset", off)):
            n = 0
            for p in ADAPTIVE:
                c = {m: cost(data[m][p], eps) for m in PAIRS}
                br = min(c[m] for m in ("RK6-opt(4)", "DP6(5)", "Verner6(5)", "Vern6(5)"))
                n += c["CRK6(4)"] < br
            wins[g] = int(n)
        for p in ADAPTIVE:
            for m in PAIRS:
                a, b = cost(ad[m][p], eps), cost(off[m][p], eps)
                if math.isfinite(a) and math.isfinite(b):
                    diffs.append(abs(a - b) / min(a, b))
        diffs = np.array(diffs)
        out[f"{eps:g}"] = dict(median=float(np.median(diffs)), q90=float(np.quantile(diffs, 0.9)),
                               max=float(diffs.max()), n=int(diffs.size), crk6_wins=wins)
        print(f"eps {eps:g}: |cost(main) - cost(offset)| / min: median {np.median(diffs):.1%}, "
              f"90% quantile {np.quantile(diffs, 0.9):.1%}, max {diffs.max():.0%} over {diffs.size}; "
              f"CRK6(4) cheaper than the best real order-6 pair on {wins['main']} (main) / {wins['offset']} (offset)")
    return out


def main(nproc=6):
    from problems import ADAPTIVE
    off = {}
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for name, prob, rows in pool.imap_unordered(task, [(m, p) for m in PAIRS for p in ADAPTIVE]):
            off.setdefault(name, {})[prob] = rows
    print(f"runs done in {time.time() - t0:.0f}s")
    res = dict(offset_runs=off, stats=analyse(off))
    json.dump(res, open(FN, "w"), indent=1)
    print("wrote", FN)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
