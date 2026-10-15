"""Linear stability for the compressed Section 6 of R1 (review, item 1.6), with the new real pairs.

One table per method: evaluations per step s, real stability interval and its length per evaluation,
imaginary stability interval [0, y_max], area of the stability region in the left half-plane per
evaluation (area / s^2, i.e., in the variable z/s) and area of the preference region for tol = 1 per
evaluation.  Also the coefficient of z^{p+1} of the stability polynomial, which fixes the leading
error on linear problems, relative to 1/(p+1)! (the value of the projected families).
Same numerical procedures as code/stability.py and code/prefregions.py of the original submission.
Writes R1/tables/stability.tex and R1/data/stability_r1.json.
"""
import json
import os
from math import factorial

import mpmath as mp

import r1common as R
import stability as S
import prefregions as PR
from tableaux import load_refined

ROWS = [  # label, tableau, evaluations per step; the first row of each order is T_p
    [(r"$T_5$: CRK5", "CRK5-opt", 5), ("DP5(4)", "DP5", 6), ("CK5(4)", "CK5", 6),
     ("Tsit5(4)", "Tsit5(4)", 6), ("BS5(4)", "BS5(4)", 7)],
    [(r"$T_6$: CRK6, CCK6", "CRK6-opt", 6), ("Butcher6", "Butcher6", 7), ("RK6-opt", "RK6-opt", 7),
     ("DP6(5)", "DP6(5)", 8), ("Verner6(5)", "Verner6(5)", 8), ("Vern6(5)", "Vern6(5)", 8)],
]


def best(vals, strs):
    m = max(v for v in vals if v is not None)
    return [("\\best{" + s + "}") if v is not None and abs(v - m) < 1e-12 * max(1, abs(m)) else s
            for v, s in zip(vals, strs)]


def main():
    mp.mp.dps = 40
    TT = load_refined()
    TT.update(load_refined(os.path.join(R.DATA, "tableaux_r1.json")))
    X, Y, Z, dA = PR.grid()
    out, lines = {}, []
    for group in ROWS:
        D = []
        for lab, key, ev in group:
            T = TT[key]
            co = S.poly(T, 0)
            p = T.levels[0][1]
            r, i = S.real_interval(co), S.imag_interval(co)
            _, A_left = S.area(co)
            a = PR.areas(co, Z, dA)
            cp1 = float(co[p + 1]) * factorial(p + 1) if len(co) > p + 1 else 0.0
            d = dict(key=key, evals=ev, p=p, coeffs=co.tolist(), real=r, imag=i, real_per_eval=-r / ev,
                     area_left=A_left, area_left_per_eval2=A_left / ev ** 2, pref=a["pref"],
                     pref_per_eval2=a["pref"] / ev ** 2, cp1_times_fact=cp1,
                     # linear error of the family divided by that of the method (leading term)
                     linear_error_gain=(1.0 / abs(1.0 - cp1)) if abs(1.0 - cp1) > 1e-12 else float("inf"))
            out[lab] = d
            D.append(d)
            print(f"{lab:24s} s={ev} real [{r:.4f},0] per eval {-r / ev:.3f} imag {i:.4f} "
                  f"area/s^2 {A_left / ev ** 2:.3f} pref/s^2 {a['pref'] / ev ** 2:.3f} "
                  f"(p+1)! c_(p+1) = {cp1:.6f} linear-error gain {d['linear_error_gain']:.3g}")
        cols = [best([-d["real"] for d in D], [f"$[{d['real']:.4f},\\,0]$" for d in D]),
                best([d["real_per_eval"] for d in D], [f"${d['real_per_eval']:.3f}$" for d in D]),
                best([d["imag"] if d["imag"] > 0 else None for d in D], [f"${d['imag']:.4f}$" if d["imag"] > 0 else "$0$" for d in D]),
                best([d["area_left_per_eval2"] for d in D], [f"${d['area_left_per_eval2']:.3f}$" for d in D]),
                best([d["pref_per_eval2"] for d in D], [f"${d['pref_per_eval2']:.3f}$" for d in D])]
        for k, ((lab, key, ev), d) in enumerate(zip(group, D)):
            lines.append(f"{lab} & {ev} & " + " & ".join(c[k] for c in cols) + r"\\")
        lines.append(r"\midrule")
    lines.pop()
    open(os.path.join(R.TAB, "stability.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(R.DATA, "stability_r1.json"), "w"), indent=1)
    print("wrote tables/stability.tex")


if __name__ == "__main__":
    main()
