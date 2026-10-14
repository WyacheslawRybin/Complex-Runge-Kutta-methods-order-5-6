"""
LaTeX table bodies generated from the data files (no number is typed by hand).
Writes ../tables/*.tex and ../tables/numbers.json (scalars quoted in the text).
"""
import json, math, os
import numpy as np
from results import classify, ORDER_OF
from figures_cost import (fixed_curve, adapt_curve, cost_at, bench, BENCH_KEY, BENCH_PROB, EVALS,
                          FIXED, ADAPT, PREF_FIXED, PREF_ADAPT, ADAPT5, ADAPT6, FIXED5, FIXED6)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(TAB, exist_ok=True)
NUM = {}

def sci(x, d=3, approx=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    s = f"{m:.{d - 1}f}"
    if s.startswith("10"):
        e += 1; s = f"{m / 10:.{d - 1}f}"
    pre = r"{\sim}" if approx else ""
    if -2 <= e <= 2:
        return f"${pre}{x:.{max(0, d - 1 - e)}f}$"
    return f"${pre}{s}{{\\cdot}}10^{{{e}}}$"

def fix3(x, approx=False):
    """x in [0.001, 100) with three significant digits in fixed-point notation."""
    d = max(0, 2 - math.floor(math.log10(abs(x))))
    return "$" + (r"{\sim}" if approx else "") + f"{x:.{d}f}" + "$"

def rot(s):
    """Rotated column header (\\rot is defined in the preamble)."""
    return "\\rot{" + s + "}"

def best(s):
    """Green cell: the best value within the comparison group (\\best is defined in the preamble)."""
    return "\\best{" + s + "}"

def mark_best(vals, cells, better=min, rel=5e-3):
    """Wrap the cells whose value is the best of vals (ties within rel) in \\best; None values are skipped."""
    ok = [v for v in vals if v is not None and math.isfinite(v)]
    if not ok:
        return cells
    b = better(ok)
    return [best(c) if v is not None and math.isfinite(v) and abs(v - b) <= rel * abs(b) else c
            for v, c in zip(vals, cells)]

PNAME = {"Dahlquist": "Dahlquist", "pendulum": "linear pendulum", "quadratic": r"$y'=-y^{2}$",
         "exponential": r"$y'=-e^{y}$", "nonautonomous": r"$y'=4y\sin^{3}t\cos t$", "VanderPol": "Van der Pol",
         "Rossler": r"R\"ossler", "Bouali": "Bouali", "Lorenz05": r"Lorenz, $T=0.5$", "Lorenz5": r"Lorenz, $T=5$",
         "Kepler": r"Kepler, $e=0.5$", "RigidBody": "rigid body", "Brusselator": "Brusselator",
         "Arenstorf": "Arenstorf orbit"}
# one name per method throughout the paper (internal keys on the left)
MNAME = {"CRK6-lam": r"CRK6-$\lambda$", "CCK6-casc": "CCK6-cascade", "CCK6(5)/5": "CCK6(5)",
         "CCK6-casc/5": "CCK6-cascade", "CCK6(5)-s/5": "CCK6(5)-s", "DP5": "DP5(4)", "CK5": "CK5(4)"}
mname = lambda m: MNAME.get(m, m)
PROBS = ["Dahlquist", "pendulum", "quadratic", "exponential", "nonautonomous", "VanderPol", "Rossler", "Bouali",
         "Lorenz05", "Lorenz5", "Kepler", "RigidBody", "Brusselator"]

def table_measured():
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    S = {m: {p: classify(meas[m][p]["rows"], ORDER_OF(m)) for p in meas[m]} for m in meas}
    NUM["measured"] = S
    head = {"CCK6(5)/5": "CCK6(5), emb.", "CCK6-casc/5": "CCK6-cascade, emb."}
    for name, methods in [("measured5", ["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5", "CCK6(5)/5", "CCK6-casc/5", "DP5", "CK5"]),
                          ("measured6", ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam", "CCK6(5)", "CCK6(5)-s", "CCK6-casc",
                                         "Butcher6", "RK6-opt", "DP6(5)", "Verner6(5)"])]:
        # each row in units of a power of ten (second column), so that ten columns fit the page
        lines = ["problem & unit & " + " & ".join(rot(head.get(m, mname(m))) for m in methods) + r"\\", r"\midrule"]
        for p in PROBS:
            vals, kinds = [], []
            for m in methods:
                v = S[m][p]
                if m == "DP6(5)" and p in ("Dahlquist", "pendulum"):
                    vals.append(None); kinds.append(r"ord.~7")
                elif v["cls"] == "deg":
                    vals.append(None); kinds.append(r"deg.")
                else:
                    vals.append(v["C"]); kinds.append("approx" if v["cls"] == "approx" else "ok")
            k = math.floor(math.log10(max(v for v in vals if v is not None))) - 1
            cells = [kd if v is None else fix3(v / 10 ** k, kd == "approx") for v, kd in zip(vals, kinds)]
            lines.append(PNAME[p] + f" & $10^{{{k}}}$ & " + " & ".join(mark_best(vals, cells)) + r"\\")
            if p in ("pendulum",):
                lines.append(r"\midrule")
        open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # orders
    worst = {}
    for m in S:
        p = ORDER_OF(m)
        dev = [abs(S[m][q]["order"] - p) for q in S[m] if S[m][q]["cls"] == "ok"]
        worst[m] = max(dev)
    NUM["order_dev_max"] = worst
    # cost-normalized break-even ratios against the real methods of the same order
    kap = {}
    pairs = [(m, r, 5, 6, 5) for m in ("CRK5-opt", "CRK5-s") for r in ("DP5", "CK5")]
    pairs += [(m, r, 6, {"RK6-opt": 7, "Butcher6": 7}.get(r, 8), 6)
              for m in ("CRK6-opt", "CRK6-s", "CRK6-0", "CCK6(5)", "CCK6-casc")
              for r in ("RK6-opt", "DP6(5)", "Verner6(5)", "Butcher6")]
    for m, ref, s_c, s_r, p in pairs:
        for q in PROBS:
            a, b = S[m][q], S[ref][q]
            if a["cls"] == "deg" or b["cls"] == "deg" or a["C"] <= 0 or b["C"] <= 0:
                kap[(m, ref, q)] = None
            else:
                kap[(m, ref, q)] = (s_r / s_c) * (b["C"] / a["C"]) ** (1 / p)
    NUM["kappa_star"] = {f"{a}|{b}|{c}": v for (a, b, c), v in kap.items()}
    # columns: complex method (header row in the paper) against the real method named here
    cols = [("CRK5-opt", "DP5"), ("CRK5-opt", "CK5"), ("CRK6-opt", "RK6-opt"), ("CRK6-opt", "DP6(5)"),
            ("CRK6-opt", "Verner6(5)"), ("CCK6(5)", "RK6-opt"), ("CCK6(5)", "DP6(5)"), ("CCK6(5)", "Verner6(5)"),
            ("CCK6-casc", "RK6-opt")]
    lines = ["problem & " + " & ".join(mname(r) for _, r in cols) + r"\\", r"\midrule"]
    for q in PROBS:
        vals = [kap[(c, r, q)] for c, r in cols]
        lines.append(PNAME[q] + " & " + " & ".join("---" if v is None else
                     (best(f"${v:.2f}$") if v >= 1.005 else f"${v:.2f}$") for v in vals) + r"\\")
        if q == "pendulum":
            lines.append(r"\midrule")
    open(os.path.join(TAB, "kappa_star.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

def table_tableaux():
    T = json.load(open(os.path.join(DATA, "tableaux.json")))
    # grouped by order: complex tableaux first, then the real methods of the same order
    groups = [[("CRK5-opt", "this work"), ("CRK5-s", "this work"), ("CRK5-1", "this work"), ("CRK5-2", "independent search"),
               ("GJM5", r"\cite[Table~4]{GJM}"), ("DP5", r"\cite{DP}"), ("CK5", r"\cite{CK}")],
              [("CRK6-opt", "this work"), ("CRK6-s", "this work"), ("CRK6-0", "this work"), ("CRK6-lam", "this work"),
               ("CCK6(5)", "this work"), ("CCK6(5)-s", "this work"), ("CCK6-casc", "this work"),
               ("Butcher6", r"\cite{Butcher64}"), ("RK6-opt", "this work"), ("DP6(5)", r"\cite{PD81}"),
               ("Verner6(5)", r"\cite{Verner78}")]]
    lines = []
    for g, rows in enumerate(groups):
        Ecells = mark_best([T[k]["E"][0] for k, _ in rows], [sci(T[k]["E"][0], 4) for k, _ in rows])
        Mcells = mark_best([T[k]["maxcoef"] for k, _ in rows], [f"${T[k]['maxcoef']:.3f}$" for k, _ in rows])
        for (k, src), Ec, Mc in zip(rows, Ecells, Mcells):
            t = T[k]
            lv = ",".join(f"{p}" for m, p in t["levels"])
            lines.append(f"{mname(k)} & {src} & {t['s']} & {t['cost']} & {lv} & {Ec} & {Mc} & "
                         f"{sci(t['resid_printed'], 2)} & {sci(t['resid_refined'], 2)} & "
                         + (f"{sci(t['full_complex_resid'], 2)}" if t["projected"] else "---")
                         + f" & {t['local_dim']}" + r"\\")
        if g == 0:
            lines.append(r"\midrule")
    open(os.path.join(TAB, "tableaux.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    NUM["tableaux"] = {k: dict(E=T[k]["E"], maxcoef=T[k]["maxcoef"], dim=T[k]["local_dim"], rankJ=T[k]["jac_rank"],
                               nfree=T[k]["n_free"], Mrank=T[k]["M_rank"], shift=T[k]["refine_shift"],
                               resid=T[k]["resid_refined"], c=T[k]["c"]) for k in T}

def table_pairs():
    E = json.load(open(os.path.join(DATA, "embedded.json")))
    rows = [("CRK5(4)", 5, 4, 5), ("CRK6(4)", 6, 4, 6), ("CCK6(5)", 6, 5, 6), ("CCK6(5)-s", 6, 5, 6),
            ("CCK6-casc", 6, 5, 6), ("DP5", 6, 4, 7), ("CK5", 6, 4, 6), ("DP6(5)", 8, 5, 8), ("Verner6(5)", 8, 5, 8)]
    lines = []
    for k, ev, q, s in rows:
        d = E[k]
        nm = mname(k)
        lines.append(f"{nm} & {s} & {ev} & {q} & {sci(d['Ahat'], 3)} & ${d['B']:.2f}$ & ${d['C']:.2f}$ & "
                     f"${d['E']:.2f}$ & ${d['maxbh']:.3f}$" + r"\\")
        if k == "CCK6-casc":
            lines.append(r"\midrule")
    open(os.path.join(TAB, "pairs.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    NUM["pairs"] = E

def table_stability():
    S = json.load(open(os.path.join(DATA, "stability.json")))
    # per evaluation: lengths divided by s, areas by s^2 (the region in the variable z/s)
    groups = [[("CRK5 family ($T_5$)", r"$T_5$: CRK5, CCK6 emb."), ("DP5", "DP5(4)"), ("CK5", "CK5(4)")],
              [("CRK6 / CCK6 ($T_6$)", r"$T_6$: CRK6, CCK6"), ("Butcher6", "Butcher6"), ("RK6-opt", "RK6-opt"),
               ("DP6(5)", "DP6(5)"), ("Verner6(5)", "Verner6(5)")]]
    lines = []
    for rows in groups:
        D = [S[k] for k, _ in rows]
        cols = [mark_best([-d["real"] for d in D], [f"$[{d['real']:.4f},\\,0]$" for d in D], max),
                mark_best([d["real_per_eval"] for d in D], [f"${d['real_per_eval']:.3f}$" for d in D], max),
                mark_best([d["imag"] if d["imag"] > 0 else None for d in D], [f"${d['imag']:.4f}$" for d in D], max),
                mark_best([d["area_left"] for d in D], [f"${d['area_left']:.2f}$" for d in D], max),
                mark_best([d["area_left"] / d["evals"] ** 2 for d in D],
                          [f"${d['area_left'] / d['evals'] ** 2:.3f}$" for d in D], max)]
        for i, ((k, nm), d) in enumerate(zip(rows, D)):
            lines.append(f"{nm} & {d['evals']} & " + " & ".join(c[i] for c in cols) + r"\\")
        lines.append(r"\midrule")
    for m in range(1, 5):
        d = S[f"_T{m}"]
        lines.append(f"cascade level $T_{m}$ & {m} & $[{d['real']:.4f},\\,0]$ & ${-d['real'] / m:.3f}$ & "
                     f"${d['imag']:.4f}$ & ${d['area_left']:.2f}$ & ${d['area_left'] / m ** 2:.3f}$" + r"\\")
    open(os.path.join(TAB, "stability.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    NUM["stability"] = S

def table_adaptive():
    ad = json.load(open(os.path.join(DATA, "adaptive.json")))
    B = bench()
    probs = PREF_ADAPT
    res = {}
    pairname = mname
    for eps in (1e-6, 1e-10):
        # problems as rows, pairs as columns; the cheapest pair of each order is marked per problem
        lines = ["problem & " + " & ".join(rot(pairname(m)) for m in ADAPT5 + ADAPT6) + r"\\", r"\midrule"]
        for p in probs:
            res[(eps, p)] = {m: cost_at(*adapt_curve(ad, m, p)[:2], eps) for m in ADAPT}
            cells = []
            for group in (ADAPT5, ADAPT6):
                vals = [res[(eps, p)][m] for m in group]
                cells += mark_best(vals, ["---" if not math.isfinite(v) else f"{v:.0f}" for v in vals], rel=0)
            lines.append(PNAME[p].replace(" orbit", "") + " & " + " & ".join(cells) + r"\\")
        open(os.path.join(TAB, f"adaptive_{int(-math.log10(eps))}.tex"), "w", encoding="utf-8").write(chr(10).join(lines) + chr(10))
    NUM["adaptive_cost"] = {f"{e}|{p}|{m}": v for (e, p), d in res.items() for m, v in d.items()}
    # rejected-step ratio and tolerance proportionality slope
    stats = {}
    for m in ADAPT:
        rej = []; slopes = []
        for p in probs:
            rows = [r for r in ad[m][p] if r["err"]]
            rej += [r["nrej"] / max(1, r["nacc"]) for r in rows]
            t = np.log10([r["tol"] for r in rows if r["tol"] <= 1e-5]); e = np.log10([r["err"] for r in rows if r["tol"] <= 1e-5])
            if len(t) > 3:
                slopes.append(np.polyfit(t, e, 1)[0])
        stats[m] = dict(rej_median=float(np.median(rej)), rej_max=float(np.max(rej)),
                        slope_median=float(np.median(slopes)), slope_min=float(np.min(slopes)), slope_max=float(np.max(slopes)))
    NUM["adaptive_stats"] = stats
    lines = []
    for m in ADAPT:
        s = stats[m]
        nm = mname(m)
        lines.append(f"{nm} & ${s['slope_median']:.2f}$ & ${s['slope_min']:.2f}$--${s['slope_max']:.2f}$ & "
                     f"${100 * s['rej_median']:.0f}\\%$ & ${100 * s['rej_max']:.0f}\\%$" + r"\\")
    open(os.path.join(TAB, "adaptive_stats.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

def table_bench():
    B = bench()
    probs = ["exponential", "nonautonomous", "VanderPol", "Rossler", "Lorenz", "Bouali", "Kepler", "RigidBody",
             "Brusselator", "Arenstorf"]
    methods = ["CRK5-opt", "CRK6-opt", "CCK6(5)", "DP5", "CK5", "Butcher6", "RK6-opt", "DP6(5)", "Verner6(5)"]
    lines = ["problem & " + " & ".join(rot(mname(m)) for m in methods) + r" & $\kappa_{\mathrm{CPU}}$\\", r"\midrule"]
    kap = {}
    for p in probs:
        k = (B[(p, "CCK6(5)")]["ns"] / 6) / (B[(p, "RK6-opt")]["ns"] / 7)
        kap[p] = k
        name = PNAME.get(p, p) if p != "Lorenz" else "Lorenz"
        lines.append(name + " & " + " & ".join(f"${B[(p, m)]['ns']:.0f}$" for m in methods) + f" & ${k:.2f}$" + r"\\")
    open(os.path.join(TAB, "bench.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    NUM["kappa_cpu"] = kap

def table_cost_ratio():
    """Best complex / best real cost at eps, within each order; kappa = 1 and measured CPU time."""
    ad = json.load(open(os.path.join(DATA, "adaptive.json")))
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    B = bench()
    out = {}
    lines = []
    for p in PREF_ADAPT:
        cells = []
        for mode, groups in [("fixed", (FIXED5, FIXED6)), ("adaptive", (ADAPT5, ADAPT6))]:
            if mode == "fixed" and p not in PREF_FIXED:
                cells += ["---"] * 4; continue
            for group in groups:
                for cm in ("kappa1", "cpu"):
                    vals = {}
                    for m in group:
                        if m in ("GJM5", "CRK5-s", "CRK5-1", "CRK5-2", "CRK6-s", "CRK6-0", "CRK6-lam", "CCK6(5)-s"):
                            continue
                        if mode == "fixed":
                            N, e = fixed_curve(meas, m, p)
                            c = EVALS[m] * N if cm == "kappa1" else N * B[(BENCH_PROB.get(p, p), BENCH_KEY[m])]["ns"]
                        else:
                            n, e, _ = adapt_curve(ad, m, p)
                            bm = B[(BENCH_PROB.get(p, p), BENCH_KEY[m])]
                            c = n if cm == "kappa1" else n * bm["ns"] / bm["evals"]
                        vals[m] = cost_at(c, e, 1e-8)
                    cx = min(v for m, v in vals.items() if m.startswith(("CRK", "CCK")))
                    rl = min(v for m, v in vals.items() if not m.startswith(("CRK", "CCK")))
                    r = cx / rl if math.isfinite(cx) and math.isfinite(rl) else None
                    out[f"{mode}|{p}|{'5' if group in (FIXED5, ADAPT5) else '6'}|{cm}"] = r
                    cells.append("---" if r is None else (best(f"${r:.2f}$") if r < 0.995 else f"${r:.2f}$"))
        lines.append(PNAME[p] + " & " + " & ".join(cells) + r"\\")
    open(os.path.join(TAB, "cost_ratio.tex"), "w", encoding="utf-8").write(chr(10).join(lines) + chr(10))
    NUM["cost_ratio"] = out
    # cross-order (discussion): best order-6 pair / best 5(4) pair, kappa = 1
    cross = {}
    for p in PREF_ADAPT:
        for eps in (1e-4, 1e-6, 1e-8, 1e-10):
            b5 = min(cost_at(*adapt_curve(ad, m, p)[:2], eps) for m in ADAPT5)
            b6 = min(cost_at(*adapt_curve(ad, m, p)[:2], eps) for m in ADAPT6)
            cross[f"{p}|{eps:g}"] = b6 / b5 if math.isfinite(b5) and math.isfinite(b6) else None
    NUM["cross_order"] = cross

def table_prefregions():
    R = json.load(open(os.path.join(DATA, "prefregions.json")))
    lines = []
    keys = list(R)
    for g in (keys[:3], keys[3:]):                      # order five, order six
        V = [R[k] for k in g]
        cols = [mark_best([v[c] for v in V], [f"${v[c]:.{d}f}$" for v in V], max)
                for c, d in (("pref", 2), ("pref_per_eval", 3))]
        for i, (k, v) in enumerate(zip(g, V)):
            lines.append(f"{k} & {v['evals']} & " + " & ".join(c[i] for c in cols) + r"\\")
        if g == keys[:3]:
            lines.append(r"\midrule")
    open(os.path.join(TAB, "prefregions.tex"), "w", encoding="utf-8").write(chr(10).join(lines) + chr(10))
    NUM["prefregions"] = R

def table_tuning():
    tu = json.load(open(os.path.join(DATA, "tuned.json")))
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    start = {"CRK5": "CRK5-opt", "CRK6": "CRK6-opt", "CCK65": "CCK6(5)", "CCK6-cascade": "CCK6-casc"}
    logs = {}
    for k in start:
        txt = open(os.path.join(HERE, "tune_runs", f"tune_{k}.log"), encoding="utf-8").read()
        f = [l for l in txt.splitlines() if l.startswith("improvement factors")][0].split(":")[1].split()
        E0 = [l for l in txt.splitlines() if l.startswith("errors of the starting")][0]
        logs[k] = [float(x) for x in f]
    lines = []
    for k, m in start.items():
        C0 = classify(meas[m]["Rossler"]["rows"], 5 if k == "CRK5" else 6)["C"]
        C1 = tu[k]["C"]
        T = json.load(open(os.path.join(DATA, "tableaux.json")))[m]
        lines.append(f"{mname(m)} & " + " & ".join(f"${v:.2f}$" for v in logs[k]) +
                     f" & {sci(C0)} & {sci(C1)} & ${C0 / C1:.2f}$ & {sci(T['E'][0])} & {sci(tu[k]['E'])} & "
                     f"${T['maxcoef']:.2f}\\to{tu[k]['maxcoef']:.2f}$" + r"\\")
    open(os.path.join(TAB, "tuning.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    NUM["tuning"] = {k: dict(factors=logs[k], C_tuned=tu[k]["C"], E_tuned=tu[k]["E"]) for k in start}

PAIR_OF = {"CRK5-opt": "CRK5(4)", "CRK6-opt": "CRK6(4)", "CCK6(5)": "CCK6(5)", "CCK6(5)-s": "CCK6(5)-s",
           "CCK6-casc": "CCK6-casc", "DP5": "DP5", "CK5": "CK5", "DP6(5)": "DP6(5)", "Verner6(5)": "Verner6(5)"}

def gmean(v):
    return float(np.exp(np.mean(np.log(v)))) if len(v) else None

def table_summary(eps=1e-8):
    """One row per method, within each order: invariants, fixed-step cost from the measured constants,
    variable-step cost of its pair at accuracy eps; costs relative to DP5(4) / DP6(5), geometric means
    over the problems, for kappa = 1 and for the measured CPU time."""
    T = json.load(open(os.path.join(DATA, "tableaux.json")))
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    ad = json.load(open(os.path.join(DATA, "adaptive.json")))
    B = bench()
    S = {m: {q: classify(meas[m][q]["rows"], ORDER_OF(m)) for q in meas[m]} for m in FIXED5 + FIXED6}
    nonlin = PROBS[2:]
    lines, out = [], {}
    for p, methods, ref in [(5, FIXED5, "DP5"), (6, FIXED6, "DP6(5)")]:
        def wf(m, q, cpu):
            v = S[m][q]
            if v["cls"] == "deg" or v["C"] <= 0:
                return None
            if cpu:
                key = (BENCH_PROB.get(q, q), BENCH_KEY[m])
                if key not in B:
                    return None
                return B[key]["ns"] * v["C"] ** (1 / p)
            return EVALS[m] * v["C"] ** (1 / p)
        def wa(m, q, cpu):
            P = PAIR_OF.get(m)
            if P is None:
                return None
            n, e, _ = adapt_curve(ad, P, q)
            if cpu:
                bm = B[(BENCH_PROB.get(q, q), BENCH_KEY[P])]
                n = n * bm["ns"] / bm["evals"]
            c = cost_at(n, e, eps)
            return c if math.isfinite(c) else None
        pairs = [m for m in methods if m in PAIR_OF]
        probs_a = [q for q in PREF_ADAPT if all(wa(m, q, False) is not None for m in pairs)]
        cols = {}
        for key, fun, probs in [("fix1", lambda m, q: wf(m, q, False), nonlin),
                                ("fixcpu", lambda m, q: wf(m, q, True), nonlin),
                                ("ad1", lambda m, q: wa(m, q, False), probs_a),
                                ("adcpu", lambda m, q: wa(m, q, True), probs_a)]:
            rel, wins = {}, {}
            for m in methods:
                r = [fun(m, q) / fun(ref, q) for q in probs
                     if fun(m, q) is not None and fun(ref, q) is not None]
                rel[m] = gmean(r) if r else None
            for q in probs:
                vals = {m: fun(m, q) for m in methods if fun(m, q) is not None}
                if vals:
                    w = min(vals, key=vals.get)
                    wins[w] = wins.get(w, 0) + 1
            cols[key] = (rel, wins, probs)
        out[p] = {k: dict(rel=v[0], wins=v[1], problems=v[2]) for k, v in cols.items()}
        cell = lambda v, d=2: "---" if v is None else f"${v:.{d}f}$"
        rel_cells = {k: mark_best([cols[k][0][m] for m in methods], [cell(cols[k][0][m]) for m in methods])
                     for k in cols}
        Ecells = mark_best([T[m]["E"][0] for m in methods], [sci(T[m]["E"][0], 3) for m in methods])
        Mcells = mark_best([T[m]["maxcoef"] for m in methods], [f"${T[m]['maxcoef']:.2f}$" for m in methods])
        win_cells = {k: [f"{cols[k][1].get(m, 0)}" if (k == "fix1" or m in PAIR_OF) else "---" for m in methods]
                     for k in ("fix1", "ad1")}
        for i, m in enumerate(methods):
            lines.append(f"{mname(m)} & {EVALS[m]} & {Ecells[i]} & {Mcells[i]} & {rel_cells['fix1'][i]} & "
                         f"{win_cells['fix1'][i]} & {rel_cells['fixcpu'][i]} & "
                         f"{rel_cells['ad1'][i]} & {win_cells['ad1'][i]} & {rel_cells['adcpu'][i]}" + r"\\")
        if p == 5:
            lines.append(r"\midrule")
    open(os.path.join(TAB, "summary.tex"), "w", encoding="utf-8").write(chr(10).join(lines) + chr(10))
    NUM["summary"] = out

if __name__ == "__main__":
    table_measured(); table_tableaux(); table_pairs(); table_stability(); table_adaptive(); table_bench()
    table_cost_ratio(); table_prefregions(); table_tuning(); table_summary()
    json.dump(NUM, open(os.path.join(TAB, "numbers.json"), "w"), indent=1, default=str)
    for f in sorted(os.listdir(TAB)):
        print("==", f)
        if f.endswith(".tex"):
            print(open(os.path.join(TAB, f), encoding="utf-8").read())
