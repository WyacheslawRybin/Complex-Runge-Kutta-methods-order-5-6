"""CPU timings of the R1 revision and the tables that depend on them (review, Section 5, items 3 and 4).

In the original submission, Verner6(5), CRK5-opt and CRK6-opt were timed in later runs of the benchmark
and scaled to the reference run (code/bench/merge_bench.py).  In R1 all fourteen methods, including the
new real pairs Tsit5(4), BS5(4) and Vern6(5), are timed in the same runs on eleven problems (BRUSS-2D
included): code/bench/run_bench_r1.sh runs bench.exe five times in a row on an otherwise idle machine
(data/bench_r1/run1.csv ... run5.csv; each entry of a run is already the best of five repetitions).
For every problem and method the minimum over the runs is taken; the spread over the runs (max/min) is
printed.

Writes data/bench.csv (same format as the original data/bench.csv), tables/bench.tex,
tables/cost_ratio.tex, tables/summary.tex and data/numbers_cpu.json (the scalars quoted in the text).
The kappa = 1 columns of cost_ratio.tex and summary.tex do not depend on the timings.  Without
benchmark runs, the original timings (which do not contain the new pairs) are used, and every CPU entry
that needs a missing timing is written as "---".
"""
import csv
import glob
import json
import math
import os

import numpy as np

import r1common as R
import r1patch

MT = r1patch.apply()
import figures_cost as FC  # noqa: E402
from figures_cost import adapt_curve, cost_at, fixed_curve  # noqa: E402
from results import classify  # noqa: E402

MT.MNAME.update({k: k for k in r1patch.NEW})
RUNS = sorted(glob.glob(os.path.join(R.DATA, "bench_r1", "run*.csv")))
NUM = {}
BENCH_METHODS = ["CRK5-opt", "CRK6-opt", "CCK6(5)", "DP5", "CK5", "Tsit5(4)", "BS5(4)", "Butcher6", "RK6-opt",
                 "DP6(5)", "Verner6(5)", "Vern6(5)"]
BENCH_PROBS = ["exponential", "nonautonomous", "VanderPol", "Rossler", "Lorenz", "Bouali", "Kepler", "RigidBody",
               "Brusselator", "Arenstorf", "Bruss2D"]
SKIP = ("GJM5", "CRK5-s", "CRK5-1", "CRK5-2", "CRK6-s", "CRK6-0", "CRK6-lam", "CCK6(5)-s")


def merge():
    runs = [list(csv.DictReader(open(fn))) for fn in RUNS]
    runs = [r for r in runs if r]
    assert runs, "no benchmark runs"
    keys = [(r["problem"], r["method"]) for r in runs[0]]
    t = {k: [] for k in keys}
    for run in runs:
        assert [(r["problem"], r["method"]) for r in run] == keys, "runs differ in their rows"
        for r in run:
            t[(r["problem"], r["method"])].append(float(r["ns_per_step"]))
    out = [dict(r, ns_per_step=f"{min(t[(r['problem'], r['method'])]):.3f}") for r in runs[0]]
    spread = {k: max(v) / min(v) for k, v in t.items()}
    with open(os.path.join(R.DATA, "bench.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0][0]))
        w.writeheader(); w.writerows(out)
    s = np.array(list(spread.values()))
    print(f"{len(runs)} runs, {len(keys)} entries; spread max/min over runs: median {np.median(s):.3f}, "
          f"90% {np.percentile(s, 90):.3f}, max {s.max():.3f} ({max(spread, key=spread.get)})")
    # same structure, different coefficients: these must take the same time in an undisturbed run
    for a, b in (("CRK5-opt", "CRK5-s"), ("CRK6-opt", "CRK6-s")):
        r = [float(x["ns_per_step"]) / float(y["ns_per_step"]) for x in out for y in out
             if x["problem"] == y["problem"] and x["method"] == a and y["method"] == b]
        print(f"{a}/{b}: {min(r):.3f}--{max(r):.3f}")
        if max(abs(math.log(v)) for v in r) > math.log(1.05):
            print("WARNING: tableaux of the same structure differ by more than 5%; the machine was probably loaded")
    return len(runs), spread


def bench(fn=None):
    if fn is None:
        fn = os.path.join(R.DATA, "bench.csv")
        if not os.path.exists(fn):
            fn = os.path.join(R.ORIG_DATA, "bench.csv")
    rows = [l.strip().split(",") for l in open(fn, encoding="ascii") if l.strip()][1:]
    return {(r[0], r[1]): dict(stages=int(r[2]), evals=int(r[3]), ns=float(r[4])) for r in rows}


def timing(B, prob, m):
    """Time per step of method m (or of the tableau that the pair m is timed with) on prob, or None."""
    return B.get((FC.BENCH_PROB.get(prob, prob), FC.BENCH_KEY.get(m, m)))


def compare_with_original(B):
    """Ratios that the text quotes, new against the original data/bench.csv."""
    old = bench(os.path.join(R.ORIG_DATA, "bench.csv"))
    print("\nproblem         kappa_CPU old -> new   CRK5-opt/CRK5-s  CRK6-opt/CRK6-s  Verner/DP6(5)")
    for p in sorted({p for p, _ in B if (p, "CCK6(5)") in old}):
        k = lambda D: (D[(p, "CCK6(5)")]["ns"] / 6) / (D[(p, "RK6-opt")]["ns"] / 7)
        r = lambda a, b: B[(p, a)]["ns"] / B[(p, b)]["ns"]
        print(f"{p:15s} {k(old):5.2f} -> {k(B):5.2f}        {r('CRK5-opt', 'CRK5-s'):.3f}            "
              f"{r('CRK6-opt', 'CRK6-s'):.3f}            {r('Verner6(5)', 'DP6(5)'):.3f}")


COLS = [("CRK5-opt", r) for r in ("DP5", "CK5", "Tsit5(4)", "BS5(4)")] + \
       [(c, r) for c in ("CRK6-opt", "CCK6(5)") for r in ("RK6-opt", "DP6(5)", "Verner6(5)", "Vern6(5)")]


def cpu_breakeven(B):
    """Fixed steps on a CPU: the complex method is cheaper iff (t_r / t_c) (C_r / C_c)^(1/p) > 1, t = time per
    step, i.e. iff its measured cost ratio per stage is below the break-even ratio kappa* of Table kappastar.
    Returns the cases (columns of Table kappastar, problems with a timing) in which the complex method wins."""
    meas = json.load(open(os.path.join(r1patch.MERGED, "measured.json")))
    wins, allv = [], {}
    for c, r in COLS:
        p = MT.ORDER_OF(c)
        for q in meas[c]:
            tc, tr = timing(B, q, c), timing(B, q, r)
            if tc is None or tr is None:
                continue
            a, b = classify(meas[c][q]["rows"], p), classify(meas[r][q]["rows"], p)
            if a["cls"] == "deg" or b["cls"] == "deg" or a["C"] <= 0 or b["C"] <= 0:
                continue
            v = tr["ns"] / tc["ns"] * (b["C"] / a["C"]) ** (1 / p)
            allv[f"{c}|{r}|{q}"] = v
            if v > 1:
                wins.append((c, r, q, v))
    print("\nfixed steps, CPU time: complex cheaper than real in", len(wins), "of", len(allv), "cases")
    for w in wins:
        print("   %-10s vs %-11s %-14s time ratio real/complex x (C_r/C_c)^(1/p) = %.3f" % w)
    return wins, allv


def table_bench(B):
    probs = [p for p in BENCH_PROBS if (p, "CCK6(5)") in B]
    lines = ["problem & " + " & ".join(MT.rot(MT.mname(m)) for m in BENCH_METHODS) + r" & $\kappa_{\mathrm{CPU}}$\\",
             r"\midrule"]
    kap = {}
    for p in probs:
        k = (B[(p, "CCK6(5)")]["ns"] / 6) / (B[(p, "RK6-opt")]["ns"] / 7)
        kap[p] = k
        name = {"Lorenz": "Lorenz", "Bruss2D": "BRUSS-2D"}.get(p, MT.PNAME.get(p, p))
        cells = ["---" if (p, m) not in B else f"${B[(p, m)]['ns']:.0f}$" for m in BENCH_METHODS]
        lines.append(name + " & " + " & ".join(cells) + f" & ${k:.2f}$" + r"\\")
    open(os.path.join(R.TAB, "bench.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    NUM["kappa_cpu"] = kap


def table_cost_ratio(B, eps=1e-8):
    """Best complex / best real cost at eps, within each order; kappa = 1 and measured CPU time
    (make_tables.table_cost_ratio with the new pairs; a CPU entry needs the timings of all methods)."""
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    meas = json.load(open(os.path.join(r1patch.MERGED, "measured.json")))
    out, lines = {}, []
    for p in FC.PREF_ADAPT:
        cells = []
        for mode, groups in (("fixed", (FC.FIXED5, FC.FIXED6)), ("adaptive", (FC.ADAPT5, FC.ADAPT6))):
            if mode == "fixed" and p not in FC.PREF_FIXED:
                cells += ["---"] * 4
                continue
            for group in groups:
                for cm in ("kappa1", "cpu"):
                    vals, missing = {}, False
                    for m in group:
                        if m in SKIP:
                            continue
                        bm = timing(B, p, m)
                        if cm == "cpu" and bm is None:
                            missing = True
                            break
                        if mode == "fixed":
                            N, e = fixed_curve(meas, m, p)
                            c = FC.EVALS[m] * N if cm == "kappa1" else N * bm["ns"]
                        else:
                            n, e, _ = adapt_curve(ad, m, p)
                            c = n if cm == "kappa1" else n * bm["ns"] / bm["evals"]
                        vals[m] = cost_at(c, e, eps)
                    r = None
                    if not missing:
                        cx = min(v for m, v in vals.items() if m.startswith(("CRK", "CCK")))
                        rl = min(v for m, v in vals.items() if not m.startswith(("CRK", "CCK")))
                        r = cx / rl if math.isfinite(cx) and math.isfinite(rl) else None
                    out[f"{mode}|{p}|{'5' if group in (FC.FIXED5, FC.ADAPT5) else '6'}|{cm}"] = r
                    cells.append("---" if r is None else (MT.best(f"${r:.2f}$") if r < 0.995 else f"${r:.2f}$"))
        lines.append(MT.PNAME[p] + " & " + " & ".join(cells) + r"\\")
    open(os.path.join(R.TAB, "cost_ratio.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    NUM["cost_ratio"] = out
    # cross-order (discussion): best order-6 pair / best 5(4) pair, kappa = 1
    cross = {}
    for p in FC.PREF_ADAPT:
        for e in (1e-4, 1e-6, 1e-8, 1e-10):
            b5 = min(cost_at(*adapt_curve(ad, m, p)[:2], e) for m in FC.ADAPT5)
            b6 = min(cost_at(*adapt_curve(ad, m, p)[:2], e) for m in FC.ADAPT6)
            cross[f"{p}|{e:g}"] = b6 / b5 if math.isfinite(b5) and math.isfinite(b6) else None
    NUM["cross_order"] = cross


def table_summary(B, eps=1e-8):
    """make_tables.table_summary with the new pairs; CPU entries without a timing are written as ---."""
    T = json.load(open(os.path.join(r1patch.MERGED, "tableaux.json")))
    meas = json.load(open(os.path.join(r1patch.MERGED, "measured.json")))
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    S = {m: {q: classify(meas[m][q]["rows"], MT.ORDER_OF(m)) for q in meas[m]} for m in FC.FIXED5 + FC.FIXED6}
    nonlin = MT.PROBS[2:]
    lines, out = [], {}
    for p, methods, ref in ((5, FC.FIXED5, "DP5"), (6, FC.FIXED6, "DP6(5)")):
        def wf(m, q, cpu):
            v = dict(S[m][q])
            if (m, q) == ("Vern6(5)", "nonautonomous"):
                # C(h) keeps decreasing (tables_r1.table_measured): its upper bound, a conservative cost
                rows = sorted(meas[m][q]["rows"], key=lambda r: r["N"])
                v.update(cls="ok", C=max(abs(r["Ch"]) for r in rows[-5:]))
            if v["cls"] == "deg" or v["C"] <= 0:
                return None
            if cpu:
                bm = timing(B, q, m)
                return None if bm is None else bm["ns"] * v["C"] ** (1 / p)
            return FC.EVALS[m] * v["C"] ** (1 / p)

        def wa(m, q, cpu):
            P = MT.PAIR_OF.get(m)
            if P is None:
                return None
            n, e, _ = adapt_curve(ad, P, q)
            if cpu:
                bm = timing(B, q, P)
                if bm is None:
                    return None
                n = n * bm["ns"] / bm["evals"]
            c = cost_at(n, e, eps)
            return c if math.isfinite(c) else None

        pairs = [m for m in methods if m in MT.PAIR_OF]
        probs_a = [q for q in FC.PREF_ADAPT if all(wa(m, q, False) is not None for m in pairs)]
        cols = {}
        for key, fun, probs in (("fix1", lambda m, q: wf(m, q, False), nonlin),
                                ("fixcpu", lambda m, q: wf(m, q, True), nonlin),
                                ("ad1", lambda m, q: wa(m, q, False), probs_a),
                                ("adcpu", lambda m, q: wa(m, q, True), probs_a)):
            rel, wins = {}, {}
            for m in methods:
                r = [fun(m, q) / fun(ref, q) for q in probs if fun(m, q) is not None and fun(ref, q) is not None]
                rel[m] = MT.gmean(r) if r else None
            for q in probs:
                vals = {m: fun(m, q) for m in methods if fun(m, q) is not None}
                if vals:
                    w = min(vals, key=vals.get)
                    wins[w] = wins.get(w, 0) + 1
            cols[key] = (rel, wins, probs)
        out[p] = {k: dict(rel=v[0], wins=v[1], problems=v[2]) for k, v in cols.items()}
        cell = lambda v: "---" if v is None else f"${v:.2f}$"
        rel_cells = {k: MT.mark_best([cols[k][0][m] for m in methods], [cell(cols[k][0][m]) for m in methods])
                     for k in cols}
        Ec = MT.mark_best([T[m]["E"][0] for m in methods], [MT.sci(T[m]["E"][0], 3) for m in methods])
        Mc = MT.mark_best([T[m]["maxcoef"] for m in methods], [f"${T[m]['maxcoef']:.2f}$" for m in methods])
        win_cells = {k: [f"{cols[k][1].get(m, 0)}" if (k == "fix1" or m in MT.PAIR_OF) else "---" for m in methods]
                     for k in ("fix1", "ad1")}
        for i, m in enumerate(methods):
            lines.append(f"{MT.mname(m)} & {FC.EVALS[m]} & {Ec[i]} & {Mc[i]} & {rel_cells['fix1'][i]} & "
                         f"{win_cells['fix1'][i]} & {rel_cells['fixcpu'][i]} & "
                         f"{rel_cells['ad1'][i]} & {win_cells['ad1'][i]} & {rel_cells['adcpu'][i]}" + r"\\")
        if p == 5:
            lines.append(r"\midrule")
    open(os.path.join(R.TAB, "summary.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    NUM["summary"] = out


def main():
    nruns, spread = merge() if RUNS else (0, {})
    if not RUNS:
        print("no benchmark runs in data/bench_r1: original timings, CPU entries of the new pairs left empty")
    B = bench()
    if RUNS:
        compare_with_original(B)
    wins, allv = cpu_breakeven(B)
    table_bench(B); table_cost_ratio(B); table_summary(B)
    num = dict(NUM, cpu_fixed_wins=[list(w) for w in wins], cpu_fixed_gain=allv, runs=nruns,
               spread_max=max(spread.values()) if spread else None)
    json.dump(num, open(os.path.join(R.DATA, "numbers_cpu.json"), "w"), indent=1,
              default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else str(o))
    k = num["kappa_cpu"]
    print("\nkappa_CPU (CCK6(5) stage / RK6-opt stage):", ", ".join(f"{p} {v:.2f}" for p, v in k.items()),
          f"\nrange {min(k.values()):.2f}--{max(k.values()):.2f}")
    for f in ("bench", "cost_ratio", "summary"):
        print("==", f)
        print(open(os.path.join(R.TAB, f + ".tex"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
