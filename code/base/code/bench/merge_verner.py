"""
Verner6(5) was added to the benchmark after the reference run (data/bench_runs/bench_ref.csv).
It was timed in three later runs of the full benchmark (bench_new*.csv), which were 4-8%
slower overall than the reference run (thermal state of the laptop).  To keep the reference
timings of all other methods, the Verner6(5) entries are scaled to the reference run:
    t(Verner6(5)) = t_ref(DP6(5)) * min_runs t(Verner6(5)) / min_runs t(DP6(5)),
both methods having eight real stages.  Writes data/bench.csv.
"""
import csv, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
R = os.path.join(D, "bench_runs")
ref = list(csv.DictReader(open(os.path.join(R, "bench_ref.csv"))))
new = [{(r["problem"], r["method"]): r for r in csv.DictReader(open(os.path.join(R, fn)))}
       for fn in ("bench_new.csv", "bench_new2.csv", "bench_new3.csv")]
t = lambda run, p, m: float(run[(p, m)]["ns_per_step"])
out = []
for p in dict.fromkeys(r["problem"] for r in ref):
    rows = [r for r in ref if r["problem"] == p]
    out += rows
    dp = [r for r in rows if r["method"] == "DP6(5)"][0]
    ratio = min(t(n, p, "Verner6(5)") for n in new) / min(t(n, p, "DP6(5)") for n in new)
    v = dict(dp, method="Verner6(5)", ns_per_step=f"{float(dp['ns_per_step']) * ratio:.3f}")
    out.append(v)
    print(f"{p:14s} Verner6(5)/DP6(5) = {ratio:.3f}")
with open(os.path.join(D, "bench.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ref[0]))
    w.writeheader(); w.writerows(out)
print("wrote bench.csv,", len(out), "rows")
