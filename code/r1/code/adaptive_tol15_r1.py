"""Variable-step runs at the tolerances 10^-13.5 ... 10^-15 (review R1b, item C4).

The original runs (code/adaptive.py, adaptive_new_r1.py) stop at tol = 1e-13, so that some pairs did not
reach the accuracy 1e-10 within the tolerance range.  This script runs all fourteen pairs with the same
controller, starting step and reference solutions at the four additional tolerances
10^-13.5, 10^-14, 10^-14.5 and 10^-15 (the same spacing of half a decade) and writes only the new rows to
data/adaptive_tol15_r1.json; r1patch.build() appends them to the merged data.
Usage: python adaptive_tol15_r1.py [processes]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import r1common as R
import adaptive as AD
import adaptive_new_r1 as AN

TOLS_EXT = [10.0 ** (-k / 2) for k in range(27, 31)]          # 10^-13.5 ... 1e-15
NAMES = ["CRK5(4)", "CRK6(4)", "CRK5-s(4)", "CRK6-s(4)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc", "DP5", "CK5",
         "DP6(5)", "Verner6(5)", "Tsit5(4)", "BS5(4)", "Vern6(5)"]
FN = os.path.join(R.DATA, "adaptive_tol15_r1.json")


def _init():
    AD._init()
    AD._P.update(AN.new_pairs())


def task(args):
    name, prob = args
    pr = AD._PROB[prob]
    T = float(pr["T"])
    y0 = [float(v) for v in pr["y0"]]
    rows = []
    for tol in TOLS_EXT:
        out = AD.solve(AD._P[name], pr["f_np"], y0, T, tol)
        if out is None:
            rows.append(dict(tol=tol, err=None))
            continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=float(np.linalg.norm(y - AD._REF[prob])), nfev=nf, nacc=na, nrej=nr))
    return name, prob, rows


def main(nproc=6):
    from problems import ADAPTIVE
    tasks = [(n, p) for n in NAMES for p in ADAPTIVE]
    res = {}
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for name, prob, rows in pool.imap_unordered(task, tasks):
            res.setdefault(name, {})[prob] = rows
            r = rows[-1]
            print(f"[{time.time() - t0:5.0f}s] {name:11s} {prob:13s} tol 1e-15: "
                  + (f"err={r['err']:.2e} nfev={r['nfev']}" if r["err"] is not None else "FAILED"), flush=True)
    json.dump(res, open(FN, "w"), indent=1)
    print("wrote", FN)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
