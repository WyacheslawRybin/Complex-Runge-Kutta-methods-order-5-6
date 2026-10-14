"""
Cost-based figures: fixed-step and adaptive work-precision, tolerance
proportionality, preference regions (kappa = 1 and measured CPU time), and the
C(h) curves on linear and nonlinear problems.
Usage:  python figures_cost.py [name ...]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from figures import ST, lbl, save, DATA

PROB_T = {"exponential": 5, "nonautonomous": 5, "VanderPol": 5, "Rossler": 5, "Bouali": 5, "Lorenz05": 0.5,
          "Lorenz5": 5, "Kepler": 20, "RigidBody": 12, "Brusselator": 20, "quadratic": 5,
          "Arenstorf": 17.0652165601579625588917206249, "Dahlquist": 5, "pendulum": 5}
PTITLE = {"exponential": r"$y'=-e^{y}$", "nonautonomous": r"$y'=4y\sin^3t\cos t$", "VanderPol": "Van der Pol",
          "Rossler": "Rössler", "Bouali": "Bouali", "Lorenz05": r"Lorenz, $T=0.5$", "Lorenz5": r"Lorenz, $T=5$",
          "Kepler": r"Kepler, $e=0.5$", "RigidBody": "rigid body", "Brusselator": "Brusselator",
          "quadratic": r"$y'=-y^2$", "Arenstorf": "Arenstorf orbit", "Dahlquist": r"Dahlquist $y'=-y$",
          "pendulum": "linear pendulum"}
FIXED = ["CRK5-opt", "DP5", "CK5", "CRK6-opt", "CCK6(5)", "CCK6-casc", "Butcher6", "RK6-opt", "DP6(5)"]
EVALS = {"CRK5-opt": 5, "DP5": 6, "CK5": 6, "CRK6-opt": 6, "CCK6(5)": 6, "CCK6-casc": 6, "Butcher6": 7,
         "RK6-opt": 7, "DP6(5)": 8, "Verner6(5)": 8, "CRK6-0": 6, "CRK6-lam": 6, "CCK6(5)-s": 6, "CRK5-1": 5, "CRK5-2": 5, "GJM5": 5,
         "CRK5-s": 5, "CRK6-s": 6}
ADAPT = ["CRK5(4)", "CRK6(4)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc", "DP5", "CK5", "DP6(5)", "Verner6(5)"]
BENCH_KEY = {"CRK5(4)": "CRK5-opt", "CRK6(4)": "CRK6-opt", "CCK6(5)-s": "CCK6(5)", "CCK6-casc": "CCK6(5)",
             "Verner6(5)": "Verner6(5)", "CRK5-opt": "CRK5-opt", "CRK6-opt": "CRK6-opt", "CCK6(5)": "CCK6(5)",
             "DP5": "DP5", "CK5": "CK5", "Butcher6": "Butcher6", "RK6-opt": "RK6-opt", "DP6(5)": "DP6(5)"}
# the time per step depends on the stage count and the arithmetic, not on the coefficient values
BENCH_KEY.update({k: "CRK5-opt" for k in ("CRK5-s", "CRK5-1", "CRK5-2", "GJM5")})
BENCH_KEY.update({k: "CRK6-opt" for k in ("CRK6-s", "CRK6-0", "CRK6-lam")})
BENCH_PROB = {"Lorenz05": "Lorenz", "Lorenz5": "Lorenz"}
PREF_FIXED = ["exponential", "nonautonomous", "VanderPol", "Rossler", "Bouali", "Lorenz05", "Lorenz5",
              "Kepler", "RigidBody", "Brusselator"]
PREF_ADAPT = ["exponential", "nonautonomous", "VanderPol", "Rossler", "Bouali", "Lorenz5",
              "Kepler", "RigidBody", "Brusselator", "Arenstorf"]

def bench():
    rows = [l.strip().split(",") for l in open(os.path.join(DATA, "bench.csv"), encoding="ascii") if l.strip()][1:]
    return {(r[0], r[1]): dict(stages=int(r[2]), evals=int(r[3]), ns=float(r[4])) for r in rows}

def fixed_curve(meas, m, prob):
    rows = sorted([r for r in meas[m][prob]["rows"] if r["err"] is not None and r["err"] > 0], key=lambda r: r["N"])
    return np.array([r["N"] for r in rows], float), np.array([r["err"] for r in rows])

def adapt_curve(ad, m, prob):
    rows = [r for r in ad[m][prob] if r["err"] is not None and r["err"] > 0]
    return (np.array([r["nfev"] for r in rows], float), np.array([r["err"] for r in rows]),
            np.array([r["tol"] for r in rows]))

def cost_at(cost, err, eps):
    """Smallest cost reaching accuracy eps: log-log interpolation along the lower envelope."""
    o = np.argsort(cost)
    c, e = cost[o], np.minimum.accumulate(err[o])
    if eps < e[-1]:
        return np.inf
    if eps >= e[0]:
        return c[0]
    k = int(np.argmax(e <= eps))
    y0, y1 = np.log(e[k - 1]), np.log(e[k])
    if y1 == y0:
        return c[k]
    x0, x1 = np.log(c[k - 1]), np.log(c[k])
    return float(np.exp(x0 + (np.log(eps) - y0) * (x1 - x0) / (y1 - y0)))

def costs(meas, ad, B, mode, costmode, p, methods):
    out = {}
    for m in methods:
        if mode == "fixed":
            N, e = fixed_curve(meas, m, p)
            if costmode == "kappa1":
                c = EVALS[m] * N
            else:
                c = N * B[(BENCH_PROB.get(p, p), BENCH_KEY[m])]["ns"] * 1e-9
        else:
            n, e, _ = adapt_curve(ad, m, p)
            if costmode == "kappa1":
                c = n
            else:
                bm = B[(BENCH_PROB.get(p, p), BENCH_KEY[m])]
                c = n * bm["ns"] / bm["evals"] * 1e-9
        out[m] = (c, e)
    return out

FIXED5 = ["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5", "DP5", "CK5"]
FIXED6 = ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam", "CCK6(5)", "CCK6(5)-s", "CCK6-casc", "Butcher6", "RK6-opt", "DP6(5)",
          "Verner6(5)"]
ADAPT5 = ["CRK5(4)", "DP5", "CK5"]
ADAPT6 = ["CRK6(4)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc", "DP6(5)", "Verner6(5)"]
PAIRNAME = {"DP5": "DP5(4)", "CK5": "CK5(4)"}

FIG5 = ["CRK5-opt", "CRK5-s", "GJM5", "DP5", "CK5"]             # variants CRK5-1/-2 only in the tables
FIG6 = ["CRK6-opt", "CRK6-s", "CCK6(5)", "CCK6-casc", "RK6-opt", "DP6(5)", "Verner6(5)"]   # likewise CRK6-0/-lam, CCK6(5)-s, Butcher6
REF = {5: "DP5", 6: "DP6(5)"}

def ratio_curve(c, e, cr, er, lo=1e-14, hi=1e-2, n=160):
    """cost(eps) / cost_ref(eps) along the lower envelopes of two work-precision curves."""
    env = lambda cc, ee: np.minimum.accumulate(ee[np.argsort(cc)])
    a, b = env(c, e), env(cr, er)
    emax, emin = min(a[0], b[0], hi), max(a[-1], b[-1], lo)
    if emax <= emin:
        return np.array([]), np.array([])
    eps = np.geomspace(emax, emin, n)
    return eps, np.array([cost_at(c, e, x) / cost_at(cr, er, x) for x in eps])

def _wp_block(sf, curves, methods, ref, probs, title, name=lambda m: lbl(m), xlab="RHS evaluations",
              ylim=None, ncol=6):
    """One order: work-precision row on top, cost relative to the reference method below."""
    axs = sf.subplots(2, len(probs), gridspec_kw=dict(height_ratios=[1.45, 1.0]))
    for j, p in enumerate(probs):
        a, b = axs[0, j], axs[1, j]
        for m in methods:
            c, e = curves(m, p)
            col, mk, ls = ST[m]
            a.loglog(c, e, ls=ls, marker=mk, ms=2.4, color=col, mfc="none", label=name(m), lw=1.0)
            if m != ref:
                cr, er = curves(ref, p)
                x, r = ratio_curve(c, e, cr, er)
                b.semilogx(x, r, ls=ls, color=col, lw=1.2)
        b.axhline(1, color=ST[ref][0], ls=ST[ref][2], lw=1.0)
        b.invert_xaxis()
        a.set_title(f"{title}: {PTITLE[p]}", fontsize=7.5)
        if ylim:
            a.set_ylim(*ylim)
        a.set_xlabel(xlab, fontsize=7, labelpad=1)
        b.set_xlabel(r"error at $T$", fontsize=7, labelpad=1)
    axs[0, 0].set_ylabel(r"error at $T$")
    axs[1, 0].set_ylabel(f"cost / cost of {name(ref)}", fontsize=7)
    h, l = axs[0, 0].get_legend_handles_labels()
    sf.legend(h, l, loc="outside lower center", ncol=ncol, fontsize=6.6, columnspacing=1.0, handlelength=2.4)
    return axs

def fig_wp_fixed():
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    probs = ["VanderPol", "Bouali", "Lorenz5", "Kepler"]
    def curves(m, p):
        N, e = fixed_curve(meas, m, p)
        ok = e < 1
        return EVALS[m] * N[ok], e[ok]
    fig = plt.figure(figsize=(7.4, 8.4), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.04)
    _wp_block(sfs[0], curves, FIG5, REF[5], probs, "order 5", ylim=(1e-16, 1), ncol=5)
    _wp_block(sfs[1], curves, FIG6, REF[6], probs, "order 6", ylim=(1e-16, 1), ncol=7)
    save(fig, "fig_wp_fixed")

def fig_adaptive():
    ad = json.load(open(os.path.join(DATA, "adaptive.json")))
    probs = ["Arenstorf", "Kepler", "Lorenz5", "Brusselator"]
    curves = lambda m, p: adapt_curve(ad, m, p)[:2]
    name = lambda m: PAIRNAME.get(m, lbl(m))
    fig = plt.figure(figsize=(7.4, 8.4), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.04)
    _wp_block(sfs[0], curves, ADAPT5, REF[5], probs, "5(4) pairs", name, "RHS evaluations incl. rejections")
    _wp_block(sfs[1], curves, ADAPT6, REF[6], probs, "order-6 pairs", name, "RHS evaluations incl. rejections")
    save(fig, "fig_wp_adaptive")
    # tolerance proportionality, one row per order
    fig = plt.figure(figsize=(7.2, 4.9), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.04)
    for sf, methods, title in [(sfs[0], ADAPT5, "5(4) pairs"), (sfs[1], ADAPT6, "order-6 pairs")]:
        axs = sf.subplots(1, 3)
        for a, p in zip(axs, ["Arenstorf", "Kepler", "VanderPol"]):
            a.loglog([1e-13, 1e-3], [1e-13, 1e-3], color="#999999", lw=0.6, label="error = tol")
            for m in methods:
                _, e, tol = adapt_curve(ad, m, p)
                col, mk, ls = ST[m]
                a.loglog(tol, e, ls=ls, marker=mk, ms=2.2, color=col, label=name(m), mfc="none", lw=1.0)
            a.set_title(f"{title}: {PTITLE[p]}", fontsize=7.5); a.set_xlabel("tolerance", fontsize=7, labelpad=1)
        axs[0].set_ylabel(r"error at $T$")
        h, l = axs[0].get_legend_handles_labels()
        sf.legend(h[1:] + h[:1], l[1:] + l[:1], loc="outside lower center", ncol=7, fontsize=6.6,
                  columnspacing=1.0, handlelength=2.4)
    save(fig, "fig_tolerance")


def fig_preference():
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    ad = json.load(open(os.path.join(DATA, "adaptive.json")))
    B = bench()
    eps = np.geomspace(1e-3, 1e-12, 181)
    out = {"eps": eps.tolist()}
    fig, axs = plt.subplots(2, 2, figsize=(7.3, 5.3), sharex=True)
    for ci, mode in enumerate(["fixed", "adaptive"]):
        methods = FIXED if mode == "fixed" else ADAPT
        PREF_PROBS = PREF_FIXED if mode == "fixed" else PREF_ADAPT
        cmap = ListedColormap([ST[m][0] for m in methods] + ["#f2f2f2"])
        for ri, cm in enumerate(["kappa1", "cpu"]):
            G = np.zeros((len(PREF_PROBS), len(eps)), int)
            for r, p in enumerate(PREF_PROBS):
                cur = costs(meas, ad, B, mode, cm, p, methods)
                for j, ep in enumerate(eps):
                    cs = [cost_at(*cur[m], ep) for m in methods]
                    G[r, j] = int(np.argmin(cs)) if np.isfinite(min(cs)) else len(methods)
                out[f"{mode}|{cm}|{p}"] = [methods[g] if g < len(methods) else None for g in G[r]]
            a = axs[ri, ci]
            a.imshow(G, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(methods) + 0.5, interpolation="nearest",
                     extent=(-3, -12, len(PREF_PROBS) - 0.5, -0.5))
            a.set_yticks(range(len(PREF_PROBS))); a.set_yticklabels([PTITLE[p] for p in PREF_PROBS], fontsize=6.5)
            a.grid(False)
            tag = "abcd"[2 * ri + ci]
            a.set_title(f"({tag}) {'fixed' if ci == 0 else 'adaptive'} step, "
                        + (r"$\kappa=1$" if ri == 0 else "CPU time"))
        handles = [Line2D([], [], marker="s", ls="", color=ST[m][0], ms=6, label=lbl(m)) for m in methods]
        axs[1, ci].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.3)
    for a in axs[1]:
        a.set_xlabel(r"$\log_{10}\varepsilon$ (required accuracy)")
    fig.tight_layout(h_pad=1.0)
    save(fig, "fig_preference")
    json.dump(out, open(os.path.join(DATA, "preference.json"), "w"))

def fig_blind():
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55))
    family = ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam", "CCK6(5)", "CCK6-casc"]
    for a, p in zip(axs, ["Dahlquist", "pendulum", "VanderPol"]):
        for m in family + ["Butcher6", "RK6-opt"]:
            rows = sorted([r for r in meas[m][p]["rows"] if r["err"]], key=lambda r: r["N"])
            h = np.array([PROB_T[p] / r["N"] for r in rows]); C = np.array([r["Ch"] for r in rows])
            ok = (h < 0.3) & (C > 0)
            col, mk, ls = ST[m]
            a.loglog(h[ok], C[ok], ls=ls, marker=mk, ms=3, color=col, label=lbl(m), mfc="none")
        a.set_title(PTITLE[p]); a.set_xlabel(r"$h$")
        if p != "VanderPol":
            # the five complex tableaux share P = T_6 and therefore one curve
            rows = sorted([r for r in meas["CCK6(5)"][p]["rows"] if r["err"]], key=lambda r: r["N"])
            y = rows[len(rows) // 2]["Ch"]
            a.annotate("all CRK6 and CCK6\n" + r"tableaux ($P=T_6$)", xy=(0.5, y), xycoords=("axes fraction", "data"),
                       xytext=(0, -6), textcoords="offset points", ha="center", va="top", fontsize=6.3)
    axs[0].set_ylabel(r"$C(h)=\|y_{T/h}-y(T)\|_2/h^{6}$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_blind")


ALL = {"wp_fixed": fig_wp_fixed, "adaptive": fig_adaptive, "preference": fig_preference, "blind": fig_blind}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(ALL)):
        ALL[n]()
