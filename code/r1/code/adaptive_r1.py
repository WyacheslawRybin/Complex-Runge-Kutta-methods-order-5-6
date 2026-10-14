"""Variable-step costs for R1 (review items 5.1, 5.2, 5.4).

1. tables/adaptive_6.tex, adaptive_8.tex, adaptive_10.tex, adaptive_stats.tex: evaluations needed for
   the accuracies 1e-6, 1e-8 and 1e-10 and the controller statistics, with the new real pairs Tsit5(4),
   BS5(4) and Vern6(5) (code/r1patch.py, data/adaptive_r1.json).
2. data/crk64_vs_real.json and a printout: the single pair CRK6(4), chosen in advance (it is built from
   the error-optimized CRK6-opt), against the best of the real 6(5) pairs DP6(5), Verner6(5) and Vern6(5)
   on each problem, at 1e-6, 1e-8 and 1e-10; likewise CRK5(4) against the best real 5(4) pair.
"""
import json
import math
import os

import r1patch
from r1common import DATA, TAB

MT = r1patch.apply()
from figures_cost import adapt_curve, cost_at, ADAPT, ADAPT5, ADAPT6, PREF_ADAPT  # noqa: E402

MT.MNAME.update({k: k for k in r1patch.NEW})
REAL65 = ("DP6(5)", "Verner6(5)", "Vern6(5)")
REAL54 = ("DP5", "CK5", "Tsit5(4)", "BS5(4)")
CPLX6 = ("CRK6(4)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc")


def table(ad, eps):
    lines = ["problem & " + " & ".join(MT.rot(MT.mname(m)) for m in ADAPT5 + ADAPT6) + r"\\", r"\midrule"]
    for p in PREF_ADAPT:
        res = {m: cost_at(*adapt_curve(ad, m, p)[:2], eps) for m in ADAPT}
        cells = []
        for group in (ADAPT5, ADAPT6):
            vals = [res[m] for m in group]
            cells += MT.mark_best(vals, ["---" if not math.isfinite(v) else f"{v:.0f}" for v in vals], rel=0)
        lines.append(MT.PNAME[p].replace(" orbit", "") + " & " + " & ".join(cells) + r"\\")
    return "\n".join(lines) + "\n"


def stats_table(ad):
    """tables/adaptive_stats.tex as in make_tables.table_adaptive, over the tolerances of the original runs
    (tol >= 1e-13; slope for 1e-13 <= tol <= 1e-5): below 1e-13 the error of every pair approaches the
    round-off level, which would flatten the fit."""
    import numpy as np
    lines = []
    for m in ADAPT:
        rej, slopes = [], []
        for p in PREF_ADAPT:
            rows = [r for r in ad[m][p] if r["err"] and r["tol"] > 10 ** -13.25]
            rej += [r["nrej"] / max(1, r["nacc"]) for r in rows]
            sel = [r for r in rows if r["tol"] <= 1e-5]
            if len(sel) > 3:
                slopes.append(np.polyfit(np.log10([r["tol"] for r in sel]), np.log10([r["err"] for r in sel]), 1)[0])
        lines.append(f"{MT.mname(m)} & ${np.median(slopes):.2f}$ & ${np.min(slopes):.2f}$--${np.max(slopes):.2f}$ & "
                     f"${100 * np.median(rej):.0f}\\%$ & ${100 * np.max(rej):.0f}\\%$" + r"\\")
    open(os.path.join(TAB, "adaptive_stats.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    MT.table_adaptive()                      # adaptive_6, adaptive_10, adaptive_stats (MT.TAB = R1/tables)
    stats_table(ad)
    for eps in (1e-6, 1e-8, 1e-10):
        open(os.path.join(TAB, f"adaptive_{int(-math.log10(eps))}.tex"), "w", encoding="utf-8").write(table(ad, eps))
    c = lambda m, p, eps: cost_at(*adapt_curve(ad, m, p)[:2], eps)
    out = {}
    for eps in (1e-6, 1e-8, 1e-10):
        rows = []
        for p in PREF_ADAPT:
            x6, r6 = c("CRK6(4)", p, eps), min(c(m, p, eps) for m in REAL65)
            bx6 = min(c(m, p, eps) for m in CPLX6)
            x5, r5 = c("CRK5(4)", p, eps), min(c(m, p, eps) for m in REAL54)
            sv = lambda a, b: (1 - a / b) if math.isfinite(a) and math.isfinite(b) else None
            rows.append(dict(problem=p, crk64=x6, best_real65=r6, saving=sv(x6, r6), saving_best_complex=sv(bx6, r6),
                             best_real65_name=min(REAL65, key=lambda m: c(m, p, eps)),
                             crk54=x5, best_real54=r5, saving54=sv(x5, r5),
                             best_real54_name=min(REAL54, key=lambda m: c(m, p, eps))))
        out[f"{eps:g}"] = rows
        print(f"eps = {eps:g}")
        f = lambda v: "   ---" if v is None else f"{100 * v:+6.1f}%"
        for r in rows:
            print(f"  {r['problem']:14s} CRK6(4) {r['crk64']:8.0f}  best real 6(5) {r['best_real65']:8.0f} "
                  f"({r['best_real65_name']:10s}) saving {f(r['saving'])} (best complex {f(r['saving_best_complex'])})"
                  f" | CRK5(4) {r['crk54']:8.0f} best real 5(4) {r['best_real54']:8.0f} ({r['best_real54_name']:8s})"
                  f" {f(r['saving54'])}")
        n = sum(1 for r in rows if r["saving"] is not None and r["saving"] > 0)
        print(f"  CRK6(4) cheaper than the best real 6(5) pair on {n} of {len(rows)} problems")
    json.dump(out, open(os.path.join(DATA, "crk64_vs_real.json"), "w"), indent=1,
              default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else str(o))


if __name__ == "__main__":
    main()
