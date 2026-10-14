"""
Summary of the measured constants: final value, convergence class, empirical
order, cost-normalized constants and break-even cost ratios.  Writes
../data/summary.json and prints the tables used in the paper.
"""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

ORDER_OF = lambda lab: 4 if lab == "RK4" else 5 if any(k in lab for k in ("CRK5", "GJM5", "DP5", "CK5", "/5")) else 6
COST = {"CRK5-opt": 5, "CRK5-s": 5, "CRK6-s": 6, "CRK5-1": 5, "CRK5-2": 5, "GJM5": 5, "DP5": 6, "CK5": 6, "CCK6(5)/5": 5, "CCK6(5)-s/5": 5,
        "CCK6-casc/5": 5, "CRK6-0": 6, "CRK6-opt": 6, "CRK6-lam": 6, "CCK6(5)": 6, "CCK6(5)-s": 6, "CCK6-casc": 6,
        "Butcher6": 7, "RK6-opt": 7, "DP6(5)": 8, "Verner6(5)": 8, "RK4": 4}
COMPLEX = lambda lab: lab.startswith(("CRK", "CCK", "GJM"))

def classify(rows, p):
    ok = [r for r in sorted(rows, key=lambda r: r["N"]) if r["err"] is not None and r["err"] > 0]
    Ch = [r["Ch"] for r in ok]
    R = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
    order = math.log2(ok[-2]["err"] / ok[-1]["err"])
    C = R[-1]
    conv = abs(R[-1] - R[-2]) / abs(R[-1]) if R[-1] != 0 else math.inf
    if C <= 0 or (order > p + 0.5) or (order < p - 0.5):
        cls = "deg"
    elif conv <= 0.01:
        cls = "ok"
    elif conv <= 0.15:
        cls = "approx"
    else:
        cls = "deg"
    return dict(C=C, conv=conv, order=order, cls=cls, Nmax=ok[-1]["N"], err_last=ok[-1]["err"])

def main():
    d = json.load(open(os.path.join(DATA, "measured.json")))
    out = {}
    for lab, cells in d.items():
        p = ORDER_OF(lab)
        out[lab] = {prob: classify(c["rows"], p) for prob, c in cells.items()}
    json.dump(out, open(os.path.join(DATA, "summary.json"), "w"), indent=1)
    probs = ["Dahlquist", "pendulum", "quadratic", "exponential", "nonautonomous", "VanderPol", "Rossler",
             "Bouali", "Lorenz05", "Lorenz5", "Kepler", "RigidBody", "Brusselator"]
    for lab in out:
        bad = [(pr, f"{v['C']:.3e}", f"{v['conv']:.1%}", f"{v['order']:.2f}") for pr, v in out[lab].items() if v["cls"] != "ok"]
        if bad:
            print(lab, bad)
    return out

if __name__ == "__main__":
    main()
