"""Asymptotic check of the tuned tableaux: refine in mp, then measure C on the tuning problem."""
import importlib, json, os, sys
import numpy as np
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crk
from problems import P, references

def load(levels):
    sys.argv = ["tune_cck6.py", levels]
    if "tune_cck6" in sys.modules:
        del sys.modules["tune_cck6"]
    TC = importlib.import_module("tune_cck6")
    x = np.load(os.path.join(HERE, "tune_runs", f"tuned_{levels}.npy"))
    A, W = TC.unpack(x)
    lev = [(m, p, [mp.mpc(complex(v)) for v in W[L]]) for L, (m, p) in enumerate(TC.LEV)]
    T = crk.Tableau(f"tuned-{levels}", [[mp.mpc(complex(A[i, j])) for j in range(TC.S)] for i in range(TC.S)], lev)
    return T

def errors(T, prob, Ns, L=0):
    pr = P[prob]; ref = references()[prob]
    out = []
    for N in Ns:
        y = crk.integrate_mp(T, pr["f_mp"], pr["y0"], 0, mp.mpf(pr["T"]), N, L)
        out.append(float(mp.sqrt(mp.fsum((a - b) ** 2 for a, b in zip(y, ref)))))
    return out

if __name__ == "__main__":
    mp.mp.dps = 34
    res = {}
    for levels in ["CRK5", "CRK6", "CCK65", "CCK6-cascade"]:
        with mp.workdps(60):
            T = load(levels)
            T, shift, r = T.refine()
        p = T.levels[0][1]
        Ns = [50, 100, 200, 400, 800, 1600, 3200]
        e = errors(T, "Rossler", Ns)
        Ch = [v / (5.0 / N) ** p for v, N in zip(e, Ns)]
        R = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
        res[levels] = dict(Ns=Ns, err=e, Ch=Ch, C=R[-1], conv=abs(R[-1] - R[-2]) / abs(R[-1]),
                           E=float(T.E()), maxcoef=float(T.maxcoef()), shift=float(shift),
                           A=[[[mp.nstr(mp.re(T.A[i][j]), 25), mp.nstr(mp.im(T.A[i][j]), 25)] for j in range(i)] for i in range(T.s)],
                           b=[[[mp.nstr(mp.re(v), 25), mp.nstr(mp.im(v), 25)] for v in b] for _, _, b in T.levels])
        print(levels, "shift", f"{float(shift):.1e}", "errors", " ".join(f"{v:.3e}" for v in e),
              f"C = {R[-1]:.4e} ({res[levels]['conv']:.1%})", flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "data", "tuned.json"), "w"), indent=1)
