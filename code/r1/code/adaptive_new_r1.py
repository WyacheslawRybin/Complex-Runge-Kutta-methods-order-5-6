"""Variable-step runs of the real pairs added in R1 (Tsit5(4), BS5(4), Vern6(5)) on the ten problems.

Same controller, tolerances and reference solutions as code/adaptive.py of the original submission;
the FSAL stage of all three pairs is reused after an acceptance.  The results are merged with the
original data/adaptive.json (which is not modified) into R1/data/adaptive_r1.json.
Usage: python adaptive_new_r1.py [processes]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import r1common as R
import adaptive as AD

NEW = ["Tsit5(4)", "BS5(4)", "Vern6(5)"]
FN = os.path.join(R.DATA, "adaptive_r1.json")


def new_pairs():
    from tableaux import load_refined
    TT = load_refined(os.path.join(R.DATA, "tableaux_r1.json"))
    out = {}
    for name in NEW:
        T = TT[name]
        s = T.s
        A = np.array([[float(T.A[i][j].real) for j in range(s)] for i in range(s)])
        b = np.array([float(v.real) for v in T.levels[0][2]])
        bh = np.array([float(v.real) for v in T.levels[1][2]])
        fsal = bool(np.allclose(A[-1, :-1], b[:-1], atol=1e-15) and abs(b[-1]) < 1e-15)
        assert fsal, name
        out[name] = dict(A=A, b=b, bh=bh, p=T.levels[0][1], q=T.levels[1][1], projected=False, fsal=fsal)
    return out


def _init():
    AD._init()
    AD._P.update(new_pairs())


def main(nproc=6):
    from problems import ADAPTIVE
    res = json.load(open(os.path.join(R.ORIG_DATA, "adaptive.json")))
    tasks = [(n, p) for n in NEW for p in ADAPTIVE]
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for name, prob, rows in pool.imap_unordered(AD.task, tasks):
            res.setdefault(name, {})[prob] = rows
            ok = [r for r in rows if r["err"] is not None]
            print(f"[{time.time() - t0:5.0f}s] {name:9s} {prob:13s} tol 1e-13: err={ok[-1]['err']:.2e} "
                  f"nfev={ok[-1]['nfev']}" if ok else f"{name} {prob} FAILED", flush=True)
    json.dump(res, open(FN, "w"), indent=1)
    print("wrote", FN)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
