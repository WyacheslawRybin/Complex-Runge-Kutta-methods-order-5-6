"""
Timings of data/bench.csv.  The reference run (data/bench_runs/bench_ref.csv) is kept for all
methods it contains; its tableaux "CRK5-opt" and "CRK6-opt" are the small-coefficient tableaux
now called CRK5-s and CRK6-s.  Methods added later were timed in later runs of the full benchmark,
which were a few percent slower overall (thermal state of the laptop), and are scaled to the
reference run through a method of the same structure timed in the same runs:
    t(new) = t_ref(base) * min_runs t(new) / min_runs t(base),
    Verner6(5) <- DP6(5) (bench_new*.csv),  CRK5-opt <- CRK5-s,  CRK6-opt <- CRK6-s (bench_opt*.csv).
Writes data/bench.csv and prints the within-run ratios.
"""
import csv, glob, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
R = os.path.join(D, "bench_runs")
RENAME = {"CRK5-opt": "CRK5-s", "CRK6-opt": "CRK6-s"}
ref = [dict(r, method=RENAME.get(r["method"], r["method"])) for r in csv.DictReader(open(os.path.join(R, "bench_ref.csv")))]
load = lambda fns: [{(r["problem"], r["method"]): r for r in csv.DictReader(open(fn))} for fn in fns]
ADDED = [("Verner6(5)", "DP6(5)", load(sorted(glob.glob(os.path.join(R, "bench_new*.csv"))))),
         ("CRK5-opt", "CRK5-s", load(sorted(glob.glob(os.path.join(R, "bench_opt*.csv"))))),
         ("CRK6-opt", "CRK6-s", load(sorted(glob.glob(os.path.join(R, "bench_opt*.csv")))))]
t = lambda run, p, m: float(run[(p, m)]["ns_per_step"])
out = []
for p in dict.fromkeys(r["problem"] for r in ref):
    rows = [r for r in ref if r["problem"] == p]
    for new, base, runs in ADDED:
        if not runs:
            continue
        b = [r for r in rows if r["method"] == base][0]
        ratio = min(t(n, p, new) for n in runs) / min(t(n, p, base) for n in runs)
        rows.append(dict(b, method=new, ns_per_step=f"{float(b['ns_per_step']) * ratio:.3f}"))
        print(f"{p:14s} {new}/{base} = {ratio:.3f}")
    out += rows
with open(os.path.join(D, "bench.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ref[0]))
    w.writeheader(); w.writerows(out)
print("wrote bench.csv,", len(out), "rows")
