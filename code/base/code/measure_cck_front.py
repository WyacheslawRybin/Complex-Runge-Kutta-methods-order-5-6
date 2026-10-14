"""
Measured constants of two CCK6(5) pairs from the front of front_E.py: the constrained minimum
of E_7 at B = 5 and the interior local minimum at max|coef| = 19.1 (the smallest E_7 found with
bounded coefficients).  Same procedure as measure.py.  Writes ../data/measured_cck_front.json.
Usage:  python measure_cck_front.py
"""
import json, os
from multiprocessing import Pool
import numpy as np
import mpmath as mp
import measure as M

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
_T = {}

def pick():
    d = json.load(open(os.path.join(DATA, "front_E.json")))
    pts = [p for ch in d if ch["family"] == "CCK65" for p in ch["points"] if p["E"] is not None]
    b5 = min((p for p in pts if p["B"] == 5 and p["maxcoef"] <= 5.005), key=lambda p: p["E"])
    loc = min((p for p in pts if 18 < p["maxcoef"] < 20 and p["B"] >= 20), key=lambda p: p["E"])
    return {"CCK6(5)-B5": b5["x"], "CCK6(5)-L19": loc["x"]}

def _init():
    M._init()
    import optimize_E as O
    from verify_Emin import tableau
    TC = O.load_module("CCK65")
    with mp.workdps(60):
        for name, x in pick().items():
            T, shift, res = tableau(TC, x, name).refine()
            M._TAB[name] = T

if __name__ == "__main__":
    from problems import ORDER
    names = list(pick())
    tasks = [(n, n, 0, q) for n in names for q in ORDER]
    res = {}
    with Pool(11, initializer=_init) as pool:
        for label, q, rows, dt in pool.imap_unordered(M.task, tasks):
            res.setdefault(label, {})[q] = dict(rows=rows)
        todo = []
        for lab in res:
            for q, cell in res[lab].items():
                Ch = [r["Ch"] for r in cell["rows"] if r["err"]]
                R = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
                if len(R) > 1 and abs(R[-1] - R[-2]) > 0.01 * abs(R[-1]):
                    todo.append((lab, lab, 0, q, [7, 8]))
        for label, q, rows in pool.imap_unordered(M.task_ext, todo):
            res[label][q]["rows"] += rows
    with mp.workdps(60):
        import optimize_E as O
        TC = O.load_module("CCK65")
        info = {n: dict(E=TC.error_constant(np.array(x)), E_emb=TC.error_constant(np.array(x), 1),
                        maxcoef=float(TC.max_coef(np.array(x)))) for n, x in pick().items()}
    json.dump(dict(mp=res, info=info), open(os.path.join(DATA, "measured_cck_front.json"), "w"), indent=1)
    print(info)
