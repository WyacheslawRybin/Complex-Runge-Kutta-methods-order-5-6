"""Tables of the R1 revision that do not depend on CPU times, with the new real pairs Tsit5(4), BS5(4)
and Vern6(5) (code/r1patch.py; review, Section 5, item 4).

tableaux.tex:    invariants of the tableaux, with the effective error constant s_eff * E_{p+1}^{1/p}
                 (s_eff = right-hand side evaluations per step) after E_{p+1} (review, Section 5, item 6);
                 the source of CRK5-2 is "this work".
pairs.tex:       Dormand--Prince measures of the embedded pairs.
measured5/6.tex: measured error constants (code/make_tables.py, table_measured, with the new columns).
kappa_star.tex:  break-even cost ratios; columns: CRK5-opt against the four real 5(4) pairs, CRK6-opt
                 and CCK6(5) against the four real sixth-order methods RK6-opt, DP6(5), Verner6(5),
                 Vern6(5).
Writes R1/tables/*.tex and R1/data/numbers_r1.json (scalars quoted in the text).
"""
import json
import math
import os

import r1common as R
import r1patch

MT = r1patch.apply()
from results import classify  # noqa: E402
import figures_cost as FC  # noqa: E402

MT.MNAME.update({"Tsit5(4)": "Tsit5(4)", "BS5(4)": "BS5(4)", "Vern6(5)": "Vern6(5)"})
mname, rot, best, sci, fix3, mark_best = MT.mname, MT.rot, MT.best, MT.sci, MT.fix3, MT.mark_best
NUM = {}

M5 = ["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5", "CCK6(5)/5", "CCK6-casc/5", "DP5", "CK5", "Tsit5(4)", "BS5(4)"]
M6 = ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam", "CCK6(5)", "CCK6(5)-s", "CCK6-casc", "Butcher6", "RK6-opt",
      "DP6(5)", "Verner6(5)", "Vern6(5)"]
REAL5 = ["DP5", "CK5", "Tsit5(4)", "BS5(4)"]
REAL6 = ["RK6-opt", "DP6(5)", "Verner6(5)", "Vern6(5)"]
KAPPA_COLS = [("CRK5-opt", r) for r in REAL5] + [("CRK6-opt", r) for r in REAL6] + [("CCK6(5)", r) for r in REAL6]
EV = dict(FC.EVALS, **{"CCK6(5)/5": 6, "CCK6-casc/5": 6})


def write(name, lines):
    open(os.path.join(R.TAB, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def table_measured():
    meas = json.load(open(os.path.join(r1patch.MERGED, "measured.json")))
    S = {m: {p: classify(meas[m][p]["rows"], MT.ORDER_OF(m)) for p in meas[m]} for m in M5 + M6}
    NUM["measured"] = S
    head = {"CCK6(5)/5": "CCK6(5), emb.", "CCK6-casc/5": "CCK6-cascade, emb."}
    for name, methods in (("measured5", M5), ("measured6", M6)):
        lines = ["problem & unit & " + " & ".join(rot(head.get(m, mname(m))) for m in methods) + r"\\", r"\midrule"]
        for p in MT.PROBS:
            vals, kinds = [], []
            for m in methods:
                v = S[m][p]
                if m == "DP6(5)" and p in ("Dahlquist", "pendulum"):
                    vals.append(None); kinds.append(r"ord.~7")
                elif m == "Vern6(5)" and p == "nonautonomous":
                    # C(h) = err/h^6 keeps decreasing down to the round-off level of the 34-digit runs: the
                    # leading term nearly vanishes; upper bound = largest C(h) of the last five step sizes
                    rows = sorted(meas[m][p]["rows"], key=lambda r: r["N"])
                    vals.append(max(abs(r["Ch"]) for r in rows[-5:])); kinds.append("bound")
                elif v["cls"] == "deg":
                    vals.append(None); kinds.append(r"deg.")
                else:
                    vals.append(v["C"]); kinds.append("approx" if v["cls"] == "approx" else "ok")
            k = math.floor(math.log10(max(v for v in vals if v is not None))) - 1
            cells = [kd if v is None else
                     (f"$<${math.ceil(100 * v / 10 ** k) / 100:g}" if kd == "bound" else fix3(v / 10 ** k, kd == "approx"))
                     for v, kd in zip(vals, kinds)]
            lines.append(MT.PNAME[p] + f" & $10^{{{k}}}$ & " + " & ".join(mark_best(vals, cells)) + r"\\")
            if p == "pendulum":
                lines.append(r"\midrule")
        write(name + ".tex", lines)
    worst = {}
    for m in S:
        p = MT.ORDER_OF(m)
        dev = [abs(S[m][q]["order"] - p) for q in S[m] if S[m][q]["cls"] == "ok"]
        worst[m] = max(dev)
    NUM["order_dev_max"] = worst
    # break-even ratios against the real methods of the same order
    kap = {}
    pairs = [(m, r) for m in ("CRK5-opt", "CRK5-s") for r in REAL5]
    pairs += [(m, r) for m in ("CRK6-opt", "CRK6-s", "CRK6-0", "CCK6(5)", "CCK6-casc") for r in REAL6 + ["Butcher6"]]
    for m, ref in pairs:
        p = MT.ORDER_OF(m)
        for q in MT.PROBS:
            a, b = S[m][q], S[ref][q]
            if a["cls"] == "deg" or b["cls"] == "deg" or a["C"] <= 0 or b["C"] <= 0:
                kap[(m, ref, q)] = None
            else:
                kap[(m, ref, q)] = (EV[ref] / EV[m]) * (b["C"] / a["C"]) ** (1 / p)
    NUM["kappa_star"] = {f"{a}|{b}|{c}": v for (a, b, c), v in kap.items()}
    lines = ["problem & " + " & ".join(rot(mname(r)) for _, r in KAPPA_COLS) + r"\\", r"\midrule"]
    for q in MT.PROBS:
        vals = [kap[(c, r, q)] for c, r in KAPPA_COLS]
        lines.append(MT.PNAME[q] + " & " + " & ".join("---" if v is None else
                     (best(f"${v:.2f}$") if v >= 1.005 else f"${v:.2f}$") for v in vals) + r"\\")
        if q == "pendulum":
            lines.append(r"\midrule")
    write("kappa_star.tex", lines)


def table_tableaux():
    T = json.load(open(os.path.join(r1patch.MERGED, "tableaux.json")))
    src = {"GJM5": r"\cite[Table~4]{GJM}", "DP5": r"\cite{DP}", "CK5": r"\cite{CK}", "Butcher6": r"\cite{Butcher64}",
           "DP6(5)": r"\cite{PD81}", "Verner6(5)": r"\cite{Verner78}"}
    src.update(r1patch.CITE)
    groups = [["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5", "DP5", "CK5", "Tsit5(4)", "BS5(4)"],
              ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam", "CCK6(5)", "CCK6(5)-s", "CCK6-casc", "Butcher6", "RK6-opt",
               "DP6(5)", "Verner6(5)", "Vern6(5)"]]
    eff = lambda t: t["cost"] * t["E"][0] ** (1.0 / t["levels"][0][1])
    lines = []
    for g, rows in enumerate(groups):
        Ec = mark_best([T[k]["E"][0] for k in rows], [sci(T[k]["E"][0], 4) for k in rows])
        Fc = mark_best([eff(T[k]) for k in rows], ["$%.3f$" % eff(T[k]) for k in rows], rel=0)
        Mc = mark_best([T[k]["maxcoef"] for k in rows], [f"${T[k]['maxcoef']:.3f}$" for k in rows])
        for k, e, f, mc in zip(rows, Ec, Fc, Mc):
            t = T[k]
            lv = ",".join(f"{p}" for m, p in t["levels"])
            lines.append(f"{mname(k)} & {src.get(k, 'this work')} & {t['s']} & {t['cost']} & {lv} & {e} & {f} & {mc} & "
                         f"{sci(t['resid_printed'], 2)} & {sci(t['resid_refined'], 2)} & "
                         + (f"{sci(t['full_complex_resid'], 2)}" if t["projected"] else "---")
                         + f" & {t['local_dim']}" + r"\\")
        if g == 0:
            lines.append(r"\midrule")
    write("tableaux.tex", lines)
    NUM["tableaux"] = {k: dict(E=T[k]["E"][0], eff=eff(T[k]), maxcoef=T[k]["maxcoef"], dim=T[k]["local_dim"])
                       for k in sum(groups, [])}


def table_pairs():
    E = json.load(open(os.path.join(r1patch.MERGED, "embedded.json")))
    # RK6-opt(4): real FSAL pair 6(4) on the seven stages of RK6-opt (real64_r1.py)
    r64 = json.load(open(os.path.join(R.DATA, "real64_r1.json")))["RK6-opt(4)"]
    E["RK6-opt(4)"] = {k: r64[k] for k in ("Ahat", "B", "C", "E", "maxbh")}
    rows = [("CRK5(4)", 5, 4, 5), ("CRK6(4)", 6, 4, 6), ("CCK6(5)", 6, 5, 6), ("CCK6(5)-s", 6, 5, 6),
            ("CCK6-casc", 6, 5, 6), ("DP5", 6, 4, 7), ("CK5", 6, 4, 6), ("Tsit5(4)", 6, 4, 7), ("BS5(4)", 7, 4, 8),
            ("RK6-opt(4)", 7, 4, 8), ("DP6(5)", 8, 5, 8), ("Verner6(5)", 8, 5, 8), ("Vern6(5)", 8, 5, 9)]
    lines = []
    for k, ev, q, s in rows:
        d = E[k]
        # CRK6(4): tau^(6) = 0 exactly (propagated order six); the computed value is round-off
        e = "$0$" if d["E"] < 1e-10 else (f"${d['E']:.2f}$" if d["E"] >= 0.01 else sci(d["E"], 2))
        lines.append(f"{mname(k)} & {s} & {ev} & {q} & {sci(d['Ahat'], 3)} & ${d['B']:.2f}$ & ${d['C']:.2f}$ & "
                     f"{e} & ${d['maxbh']:.3f}$" + r"\\")
        if k == "CCK6-casc":
            lines.append(r"\midrule")
    write("pairs.tex", lines)
    NUM["pairs"] = {k: E[k] for k, *_ in rows}


def main():
    table_measured(); table_tableaux(); table_pairs()
    json.dump(NUM, open(os.path.join(R.DATA, "numbers_r1.json"), "w"), indent=1,
              default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else str(o))
    for f in ("tableaux", "pairs", "measured5", "measured6", "kappa_star"):
        print("==", f)
        print(open(os.path.join(R.TAB, f + ".tex"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
